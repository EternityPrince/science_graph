import fitz  # PyMuPDF
import re
import os
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from src.models import Paper

# Regex patterns
DOI_REGEX = re.compile(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', re.IGNORECASE)
YEAR_REGEX = re.compile(r'\b(19\d{2}|20[0-2]\d)\b')
REF_SPLIT_REGEX = re.compile(r'\[\d+\]\s+|\n(?=\d+\.\s+)|^\d+\.\s+', re.MULTILINE)

class PDFParser:
    @staticmethod
    def extract_text_and_metadata(file_path: str) -> Tuple[Paper, List[str], str]:
        """
        Parses a PDF file and extracts metadata, references, and full text.
        Returns:
            paper (Paper): The parsed paper domain model
            references (List[str]): List of citation strings extracted from references
            full_text (str): The entire text of the document
        """
        doc = fitz.open(file_path)
        
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
                title = os.path.basename(file_path).replace(".pdf", "")
        
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
            # Academic papers typically have authors right after the title.
            # Pattern: "Firstname Lastname, Firstname Lastname" etc.
            # We look for lines that look like lists of names (>=2 capitalized words per segment)
            lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
            
            # Find where title ends on first page
            title_idx = -1
            for idx, line in enumerate(lines[:10]):
                if len(title) > 8 and (title[:15].lower() in line.lower() or line.lower() in title.lower()):
                    title_idx = idx
                    break

            # Check lines after title for author-looking content
            # Typical: "Ashish Vaswani1, Noam Shazeer1, Niki Parmar1, ..."
            AUTHOR_LINE_RE = re.compile(
                r'^([A-Z][a-zé-]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-zé-]+){0,3}[,\*\d]*'
                r'(?:\s*,\s*[A-Z][a-zé-]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-zé-]+){0,3}[,\*\d]*)*)$'
            )
            NAME_TOKEN_RE = re.compile(r'[A-Z][a-zé-]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-zé-]+){0,2}')
            
            search_start = max(0, title_idx)
            for line in lines[search_start:search_start + 10]:
                # Strip trailing superscript digits/symbols before checking
                clean_line = re.sub(r'[\d\*†‡§]+', '', line).strip()
                if len(clean_line) < 5 or len(clean_line) > 300:
                    continue
                if any(w in clean_line.lower() for w in ["abstract", "introduction", "arxiv", "http", "@", "university", "google", "deepmind"]):
                    continue
                # Count how many name-like tokens we find
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
                    # Keep only names, not institution/venue names
                    authors = [n for n in ner_names if 1 < len(n.split()) <= 5][:15]
                except Exception:
                    pass
        
        # Final cleanup: strip trailing digits/punctuation, deduplicate
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
            # D:20241012...
            match = re.search(r'D:(\d{4})', creation_date)
            if match:
                year = int(match.group(1))
        
        if not year:
            # Heuristic: search first page for years
            matches = YEAR_REGEX.findall(first_page_text)
            if matches:
                # Get the most common or first year
                year = int(matches[0])
            else:
                year = 2026 # Default backup

        # 5. Extract DOI
        doi = None
        doi_matches = DOI_REGEX.findall(full_text[:5000]) # Look in first 5000 chars
        if doi_matches:
            doi = doi_matches[0].rstrip(".,;)")
            # Clean up DOI
            if doi.lower().startswith("doi:"):
                doi = doi[4:]

        # 6. Extract Abstract
        abstract = None
        abstract_match = re.search(r'(?:abstract|summary)[:\-\s]+(.*?)(?:\n\s*(?:1\s+)?introduction|\n\s*(?:i\.\s+)?introduction|\n\s*background|\n\s*ii\.)', full_text, re.IGNORECASE | re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(1).strip()
        else:
            # Fallback: search for "Abstract" and take first 1000 chars
            idx = full_text.lower().find("abstract")
            if idx != -1:
                abstract = full_text[idx+8:idx+1200].strip()
            else:
                # Take first 1000 characters of the document
                abstract = full_text[:1000].strip()

        # 7. Extract References
        references = []
        ref_idx = -1
        for ref_keyword in ["references", "bibliography", "литература", "список литературы"]:
            # Find the last occurrence of these section headers
            idx = full_text.lower().rfind(ref_keyword)
            if idx > ref_idx:
                # Ensure the line is actually a heading (brief line)
                line_start = full_text.rfind("\n", 0, idx)
                line_end = full_text.find("\n", idx)
                line = full_text[line_start:line_end].strip()
                if len(line) < 30:
                    ref_idx = idx
        
        if ref_idx != -1:
            ref_section = full_text[ref_idx:]
            # Attempt to split by typical reference numbering like [1], [2] or 1., 2.
            raw_refs = REF_SPLIT_REGEX.split(ref_section)
            for r in raw_refs:
                if not r:
                    continue
                r_clean = r.strip().replace("\n", " ")
                if len(r_clean) > 20 and not any(kw in r_clean.lower() for kw in ["references", "bibliography", "page"]):
                    # Clean multiple spaces
                    r_clean = re.sub(r'\s+', ' ', r_clean)
                    references.append(r_clean)
        
        paper_id = doi if doi else hashlib.md5(title.encode('utf-8')).hexdigest()
        
        paper = Paper(
            id=paper_id,
            title=title.strip(),
            authors=authors,
            year=year,
            doi=doi,
            abstract=abstract,
            file_path=file_path
        )
        
        return paper, references, full_text
