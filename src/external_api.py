import urllib.request
import urllib.parse
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"

import time

def fetch_paper_metadata(
    doi: Optional[str] = None,
    title: Optional[str] = None,
    arxiv_id: Optional[str] = None,
    timeout: int = 15
) -> Optional[Dict[str, Any]]:
    """
    Fetches scientific paper metadata from the Semantic Scholar API.
    Can query by DOI, arXiv ID, or by Title.
    
    Returns a dictionary containing:
        - title (str)
        - authors (List[str])
        - year (int/None)
        - abstract (str/None)
        - doi (str/None)
        - references (List[Dict[str, Any]]) - papers cited by this paper
        - citations (List[Dict[str, Any]]) - papers citing this paper
    """
    fields = "paperId,title,authors,year,abstract,externalIds,citations.title,citations.externalIds,references.title,references.externalIds"
    
    # 1. Decide on query URL
    url = None
    if doi:
        # Standardize DOI query format
        doi_clean = doi.strip()
        if doi_clean.lower().startswith("doi:"):
            doi_clean = doi_clean[4:]
        url = f"{SEMANTIC_SCHOLAR_BASE}/paper/DOI:{urllib.parse.quote(doi_clean)}?fields={fields}"
    elif arxiv_id:
        # Standardize arXiv ID query format
        arxiv_clean = arxiv_id.strip()
        if arxiv_clean.lower().startswith("arxiv:"):
            arxiv_clean = arxiv_clean[6:]
        url = f"{SEMANTIC_SCHOLAR_BASE}/paper/arXiv:{urllib.parse.quote(arxiv_clean)}?fields={fields}"
    elif title:
        query_encoded = urllib.parse.quote(title.strip())
        url = f"{SEMANTIC_SCHOLAR_BASE}/paper/search?query={query_encoded}&limit=1&fields={fields}"
    else:
        return None

    logger.info(f"[*] Querying Semantic Scholar: {url}")
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PDF-Graph-Analyzer/1.0 (local; research)"}
    )
    
    max_retries = 3
    backoff = 2
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    
                    # If we queried search, we get a list in data['data']
                    if title and not doi and not arxiv_id:
                        results = data.get("data", [])
                        if not results:
                            logger.warning("[!] No search results found on Semantic Scholar.")
                            return None
                        paper_data = results[0]
                    else:
                        paper_data = data
                    
                    # Normalize response format
                    return _normalize_response(paper_data)
                else:
                    logger.warning(f"[!] Semantic Scholar returned status {response.status}")
        except urllib.error.HTTPError as he:
            if he.code == 404:
                logger.warning(f"[!] Paper not found on Semantic Scholar (404) for URL: {url}")
                return None
            elif he.code in (429, 500, 502, 503, 504):
                logger.warning(f"[!] Semantic Scholar returned HTTP error {he.code} (attempt {attempt + 1}/{max_retries})")
            else:
                logger.warning(f"[!] Semantic Scholar HTTP error {he.code}: {he.reason}")
                return None
        except Exception as e:
            logger.warning(f"[!] Semantic Scholar query failed: {e} (attempt {attempt + 1}/{max_retries})")
        
        if attempt < max_retries - 1:
            time.sleep(backoff ** attempt)
            
    return None



def _normalize_response(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to convert Semantic Scholar raw response to our standard format."""
    # Extract clean DOI
    doi = raw.get("externalIds", {}).get("DOI")
    
    # Extract authors list of names
    authors = [a.get("name") for a in raw.get("authors", []) if a.get("name")]
    
    # Extract references
    references = []
    for ref in raw.get("references", []):
        ref_title = ref.get("title")
        ref_doi = ref.get("externalIds", {}).get("DOI")
        if ref_title:
            references.append({"title": ref_title, "doi": ref_doi})
            
    # Extract citations
    citations = []
    for cit in raw.get("citations", []):
        cit_title = cit.get("title")
        cit_doi = cit.get("externalIds", {}).get("DOI")
        if cit_title:
            citations.append({"title": cit_title, "doi": cit_doi})
            
    return {
        "title": raw.get("title"),
        "authors": authors,
        "year": raw.get("year"),
        "abstract": raw.get("abstract"),
        "doi": doi,
        "references": references,
        "citations": citations
    }
