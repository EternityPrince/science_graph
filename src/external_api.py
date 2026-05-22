import urllib.request
import urllib.parse
import json
import logging
import time
import random
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"


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
    
    import os
    s2_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY")
    headers = {"User-Agent": "PDF-Graph-Analyzer/1.0 (local; research)"}
    if s2_api_key:
        headers["x-api-key"] = s2_api_key

    req = urllib.request.Request(
        url,
        headers=headers
    )
    
    max_retries = 3
    backoff = 2
    for attempt in range(max_retries):
        sleep_time = backoff ** attempt
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
                if he.code == 429:
                    # Handle rate limiting with Retry-After header or exponential backoff
                    retry_after = he.headers.get("Retry-After")
                    sleep_time = None
                    if retry_after:
                        try:
                            # Retry-After can be integer seconds or a HTTP-date; we only handle seconds
                            sleep_time = float(retry_after)
                            # Validate sleep time: cap between 1 and 60 seconds
                            sleep_time = max(1.0, min(sleep_time, 60.0))
                        except (ValueError, TypeError):
                            sleep_time = None
                    if sleep_time is None:
                        # No valid Retry-After: use exponential backoff with 10s base for rate limits
                        base_backoff = 10  # Minimum 10s for 429 errors
                        sleep_time = base_backoff * (backoff ** attempt)
                    # Add jitter to prevent thundering herd (±20% of sleep time)
                    jitter = sleep_time * 0.2 * (random.random() - 0.5)  # -10% to +10%
                    sleep_time = max(1.0, sleep_time + jitter)
                    logger.warning(f"[!] Rate limited (429). Retrying after {sleep_time:.1f} seconds...")
                else:
                    # For 5xx errors, use standard exponential backoff
                    sleep_time = backoff ** (attempt + 1)
            else:
                logger.warning(f"[!] Semantic Scholar HTTP error {he.code}: {he.reason}")
                return None
        except Exception as e:
            logger.warning(f"[!] Semantic Scholar query failed: {e} (attempt {attempt + 1}/{max_retries})")
        
        if attempt < max_retries - 1:
            time.sleep(sleep_time)
            
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
