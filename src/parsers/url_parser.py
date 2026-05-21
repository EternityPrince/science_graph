import re
from typing import Tuple
from bs4 import BeautifulSoup
import markdownify
from curl_cffi import requests
from src.models import Paper
from src import console as con

def parse_url(url: str) -> Tuple[Paper, str]:
    """
    Fetches a URL using curl_cffi to bypass bot protection, 
    parses HTML to markdown, and returns a Paper (note) and the markdown text.
    """
    con.dim(f"Fetching URL with anti-detect browser: {url}")
    try:
        # impersonate="chrome120" makes the request look like a real Chrome browser
        response = requests.get(url, impersonate="chrome120", timeout=30)
        response.raise_for_status()
        html_content = response.text
    except Exception as e:
        raise RuntimeError(f"Failed to fetch URL {url}: {e}")

    # Parse HTML
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract title
    title = soup.title.string if soup.title else url
    if title:
        title = title.strip()
    else:
        title = url

    # Clean up unnecessary tags before markdown conversion
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
        element.decompose()

    # Convert to markdown
    main_content = soup.find("main") or soup.find("article") or soup.body
    if not main_content:
        main_content = soup
    
    html_to_convert = str(main_content)
    
    md_content = markdownify.markdownify(
        html_to_convert,
        heading_style="ATX",
        bullets="-",
    ).strip()

    # Clean up excessive newlines
    md_content = re.sub(r'\n{3,}', '\n\n', md_content)

    paper_id = re.sub(r'[^a-z0-9]+', '_', url.lower().replace("https://", "").replace("http://", "").strip("_"))[:120]

    paper = Paper(
        id=paper_id,
        title=title,
        authors=[],
        year=None,
        abstract="",
        doi="",
        file_path=url,
        properties={"source_type": "webpage", "url": url}
    )

    return paper, md_content
