import fitz  # PyMuPDF
import re
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple
from src.models import Paper, slugify
from src.parsers.base import BaseParser
from src import console as con

# Regex patterns
DOI_REGEX = re.compile(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', re.IGNORECASE)
YEAR_REGEX = re.compile(r'\b(19\d{2}|20[0-2]\d)\b')
REF_SPLIT_REGEX = re.compile(r'\[\d+\]\s+|\n(?=\d+\.\s+)|^\d+\.\s+', re.MULTILINE)

class PDFParser(BaseParser):
    def parse(self, source: str) -> Tuple[Paper, List[str], str]:
        """
        Parses a PDF file and extracts metadata, references, and full text.
        Returns:
            paper (Paper): The parsed paper domain model
            references (List[str]): List of citation strings extracted from references
            full_text (str): The entire text of the document
        """
        if not os.path.exists(source):
            raise FileNotFoundError(f"PDF file not found: {source}")

        doc = fitz.open(source)
        
        # 1. Full text extraction
        full_text_list = []
        for page in doc:
            full_text_list.append(page.get_text())
        full_text = "\n".join(full_text_list)
        
        # First page text for heuristics
        first_page_text = doc[0].get_text() if len(doc) > 0 else ""
        
        # 2. Extract Title
        title = doc.metadata.get("title", "")
        if not title or title.lower().strip() in ["untitled", "layout 1", "microsoft word", "manuscript", "pdf", ""] or ".pdf" in title.lower():
            # Fallback to first page lines
            lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
            if lines:
                # Find the first substantial line that doesn't look like journal headers
                title_candidates = []
                for line in lines[:5]:
                    if len(line) > 15 and not any(w in line.lower() for w in ["arxiv", "preprint", "journal", "proceedings", "vol.", "no.", "issn", "http", "permission", "google", "grants", "copyright", "license"]):
                        title_candidates.append(line)
                title = " ".join(title_candidates[:2]) if title_candidates else lines[0]
            else:
                title = os.path.basename(source).replace(".pdf", "")
        
        # 3. Extract Authors
        author_meta = doc.metadata.get("author", "")
        authors = []
        
        # PDF metadata author field — split by comma/semicolon/and
        if author_meta:
            raw = [a.strip() for a in re.split(r',|;|\band\b', author_meta, flags=re.IGNORECASE) if a.strip()]
            # Accept only plausible names: 2+ words, no digits, reasonable length
            authors = [a for a in raw if 2 <= len(a.split()) <= 5 and not any(c.isdigit() for c in a) and len(a) < 60]
        
        if not authors:
            # Heuristic: scan first-page lines for a comma-separated author block.
            lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
            
            # Find where title ends on first page
            title_idx = -1
            for idx, line in enumerate(lines[:10]):
                if len(title) > 8 and (title[:15].lower() in line.lower() or line.lower() in title.lower()):
                    title_idx = idx
                    break

            AUTHOR_LINE_RE = re.compile(
                r'^([A-Z][a-zé-]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-zé-]+){0,3}[,\*\d]*'
                r'(?:\s*,\s*[A-Z][a-zé-]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-zé-]+){0,3}[,\*\d]*)*)$'
            )
            NAME_TOKEN_RE = re.compile(r'[A-Z][a-zé-]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-zé-]+){0,2}')
            
            search_start = max(0, title_idx)
            for line in lines[search_start:search_start + 10]:
                clean_line = re.sub(r'[\d\*†‡§]+', '', line).strip()
                if len(clean_line) < 5 or len(clean_line) > 300:
                    continue
                if any(w in clean_line.lower() for w in ["abstract", "introduction", "arxiv", "http", "@", "university", "google", "deepmind"]):
                    continue
                tokens = NAME_TOKEN_RE.findall(clean_line)
                if len(tokens) >= 2:
                    authors = [t.strip() for t in tokens if 1 < len(t.split()) <= 5]
                    if len(authors) >= 2:
                        break
            
            # Last resort: run NER on the first 2000 chars of the first page
            if not authors:
                try:
                    from src.ner_engine import extract_persons_from_text
                    ner_names = extract_persons_from_text(first_page_text[:2000])
                    authors = [n for n in ner_names if 1 < len(n.split()) <= 5][:15]
                except Exception:
                    pass
        
        # Final cleanup
        cleaned = []
        seen = set()
        for a in authors:
            a = re.sub(r'[\d\*†‡§,]+$', '', a).strip()
            if a and a.lower() not in seen and len(a) > 3:
                seen.add(a.lower())
                cleaned.append(a)
        authors = cleaned

        # 4. Extract Year
        year = None
        creation_date = doc.metadata.get("creationDate", "")
        if creation_date and len(creation_date) >= 6:
            match = re.search(r'D:(\d{4})', creation_date)
            if match:
                year = int(match.group(1))
        
        if not year:
            matches = YEAR_REGEX.findall(first_page_text)
            if matches:
                year = int(matches[0])
            else:
                year = 2026

        # 5. Extract DOI
        doi = None
        doi_matches = DOI_REGEX.findall(full_text[:5000])
        if doi_matches:
            doi = doi_matches[0].rstrip(".,;)")
            if doi.lower().startswith("doi:"):
                doi = doi[4:]

        # 6. Extract Abstract
        abstract = None
        abstract_match = re.search(r'(?:abstract|summary)[:\-\s]+(.*?)(?:\n\s*(?:1\s+)?introduction|\n\s*(?:i\.\s+)?introduction|\n\s*background|\n\s*ii\.)', full_text, re.IGNORECASE | re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(1).strip()
        else:
            idx = full_text.lower().find("abstract")
            if idx != -1:
                abstract = full_text[idx+8:idx+1200].strip()
            else:
                abstract = full_text[:1000].strip()

        # 7. Extract References
        references = []
        ref_idx = -1
        for ref_keyword in ["references", "bibliography", "литература", "список литературы"]:
            idx = full_text.lower().rfind(ref_keyword)
            if idx > ref_idx:
                line_start = full_text.rfind("\n", 0, idx)
                line_end = full_text.find("\n", idx)
                line = full_text[line_start:line_end].strip()
                if len(line) < 30:
                    ref_idx = idx
        
        if ref_idx != -1:
            ref_section = full_text[ref_idx:]
            raw_refs = REF_SPLIT_REGEX.split(ref_section)
            for r in raw_refs:
                if not r:
                    continue
                r_clean = r.strip().replace("\n", " ")
                if len(r_clean) > 20 and not any(kw in r_clean.lower() for kw in ["references", "bibliography", "page"]):
                    r_clean = re.sub(r'\s+', ' ', r_clean)
                    references.append(r_clean)
        
        paper_id = doi if doi else slugify(title)
        
        paper = Paper(
            id=paper_id,
            title=title.strip(),
            authors=authors,
            year=year,
            doi=doi,
            abstract=abstract,
            file_path=source
        )
        
        return paper, references, full_text

    @staticmethod
    def compress_and_save_pdf(
        input_path: str,
        output_path: str,
        dpi_threshold: int,
        dpi_target: int,
        quality: int,
    ) -> None:
        """Recompresses images in a PDF using PyMuPDF and saves the result."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(input_path)
        actual_threshold = max(dpi_threshold, dpi_target + 1)

        try:
            doc.rewrite_images(
                dpi_threshold=actual_threshold,
                dpi_target=dpi_target,
                quality=quality,
                lossy=True,
                lossless=True,
            )
        except Exception as e:
            con.warning(f"Failed to rewrite images in PDF: {e}")

        if Path(input_path).resolve() == Path(output_path).resolve():
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_name = tmp.name
            try:
                doc.save(tmp_name, garbage=4, deflate=True)
                doc.close()
                shutil.move(tmp_name, output_path)
            except Exception as e:
                if os.path.exists(tmp_name):
                    try:
                        os.remove(tmp_name)
                    except Exception:
                        pass
                raise e
        else:
            doc.save(output_path, garbage=4, deflate=True)
            doc.close()
