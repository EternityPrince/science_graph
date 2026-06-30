import os
import re
import logging

# Monkeypatch huggingface_hub.dataclasses.strict and mock missing transformers module
# to ensure compatibility between marker-pdf 0.1.3 and transformers v5
import sys
import types
if "transformers.utils.model_parallel_utils" not in sys.modules:
    mod = types.ModuleType("transformers.utils.model_parallel_utils")
    mod.get_device_map = lambda *a, **k: None
    mod.assert_device_map = lambda *a, **k: None
    sys.modules["transformers.utils.model_parallel_utils"] = mod

try:
    import huggingface_hub.dataclasses
    huggingface_hub.dataclasses.strict = lambda cls=None, *args, **kwargs: (lambda c: c) if cls is None else cls
except ImportError:
    pass

# Patch torchvision functional normalize to accept numpy arrays (resolving torchvision v2 / transformers TypeError)
try:
    import torchvision.transforms.v2.functional as tvF2
    original_normalize2 = tvF2.normalize
    def patched_normalize2(inpt, *args, **kwargs):
        import torch
        import numpy as np
        if isinstance(inpt, np.ndarray):
            tensor_inpt = torch.from_numpy(inpt)
            out_tensor = original_normalize2(tensor_inpt, *args, **kwargs)
            return out_tensor.numpy()
        return original_normalize2(inpt, *args, **kwargs)
    tvF2.normalize = patched_normalize2
except Exception:
    pass

try:
    import torchvision.transforms.functional as tvF
    original_normalize = tvF.normalize
    def patched_normalize(inpt, *args, **kwargs):
        import torch
        import numpy as np
        if isinstance(inpt, np.ndarray):
            tensor_inpt = torch.from_numpy(inpt)
            out_tensor = original_normalize(tensor_inpt, *args, **kwargs)
            return out_tensor.numpy()
        return original_normalize(inpt, *args, **kwargs)
    tvF.normalize = patched_normalize
except Exception:
    pass

# Suppress noisy debug logs from third-party libraries used by Marker
logging.getLogger("ocrmypdf").setLevel(logging.WARNING)
logging.getLogger("pdfminer").setLevel(logging.WARNING)

from typing import List, Tuple

from src.models import Paper, slugify
from src.parsers.base import BaseParser
from src.parsers.pdf_parser import PDFParser
from src import console as con

logger = logging.getLogger(__name__)

# Cached marker models loaded on demand
_marker_models = None

def get_marker_models():
    global _marker_models
    if _marker_models is None:
        con.info("Loading Marker OCR/Layout models into memory (Mac Mini M4)...")
        from marker.models import load_all_models
        _marker_models = load_all_models()
        con.success("Marker models loaded successfully.")
    return _marker_models

class MarkerPDFParser(BaseParser):
    def parse(self, source: str) -> Tuple[Paper, List[str], str]:
        """
        Parses a PDF file using the Marker engine to produce high-quality Markdown text,
        falling back to Fitz if Marker fails or has issues.
        """
        if not os.path.exists(source):
            raise FileNotFoundError(f"PDF file not found: {source}")

        # 1. Use the legacy Fitz-based parser first to get clean metadata
        legacy_parser = PDFParser()
        try:
            paper, references, legacy_text = legacy_parser.parse(source)
        except Exception as e:
            logger.warning(f"Legacy parser failed to extract metadata from {source}: {e}")
            paper = Paper(
                id=slugify(os.path.basename(source)),
                title=os.path.basename(source).replace(".pdf", ""),
                authors=[],
                year=2026,
                doi=None,
                abstract="",
                file_path=source
            )
            references = []
            legacy_text = ""

        # 2. Extract rich Markdown text using Marker
        try:
            models = get_marker_models()
            con.info(f"Extracting Markdown text from {os.path.basename(source)} using Marker...")
            
            from marker.convert import convert_single_pdf
            
            # Request Russian to enable eng+rus OCR via Tesseract
            full_text, out_metadata = convert_single_pdf(
                source,
                models,
                metadata={"language": "Russian"},
                parallel_factor=1
            )
            
            full_text = full_text.strip()
            
            # Extract references from the Marker-generated Markdown text
            marker_references = self._extract_references_from_markdown(full_text)
            if marker_references:
                references = marker_references
                
            paper.properties["pdf_parser"] = "marker"
                
        except Exception as e:
            con.warning(f"Marker PDF parsing failed for {source}: {e}. Falling back to standard PDF parser.")
            logger.warning(f"Marker PDF parsing failed for {source}: {e}. Falling back to standard PDF parser.", exc_info=True)
            full_text = legacy_text
            paper.properties["pdf_parser"] = "fitz"
            paper.properties["pdf_parser_fallback"] = True
            paper.properties["pdf_parser_error"] = str(e)

        # Update paper file path and return
        paper.file_path = source
        return paper, references, full_text

    def _extract_references_from_markdown(self, markdown_text: str) -> List[str]:
        """Heuristic to extract clean references list from Marker's markdown output."""
        references = []
        ref_idx = -1
        
        # Look for references section heading
        for ref_keyword in ["references", "bibliography", "литература", "список литературы"]:
            idx = markdown_text.lower().rfind(ref_keyword)
            if idx > ref_idx:
                # Ensure it looks like a section start (e.g. preceded by newline or #)
                line_start = markdown_text.rfind("\n", 0, idx)
                line_end = markdown_text.find("\n", idx)
                line = markdown_text[line_start:line_end].strip()
                if len(line) < 30:
                    ref_idx = idx

        if ref_idx != -1:
            ref_section = markdown_text[ref_idx:]
            # Split by markdown list items or numbering patterns
            # e.g., \n1. or \n- or \n[1]
            split_patterns = re.compile(r'\n(?:[*\-+]|\d+\.|\s*\[\d+\])\s+')
            raw_refs = split_patterns.split(ref_section)
            for r in raw_refs:
                if not r:
                    continue
                r_clean = r.strip().replace("\n", " ")
                # Clean multiple spaces
                r_clean = re.sub(r'\s+', ' ', r_clean)
                if len(r_clean) > 20 and not any(kw in r_clean.lower() for kw in ["references", "bibliography", "page"]):
                    references.append(r_clean)

        return references
