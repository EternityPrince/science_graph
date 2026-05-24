from typing import List, Dict, Any, Optional
from src.models import Paper, Chunk, Author, slugify

def create_paper(
    id: Optional[str] = None,
    title: str = "Test Paper Title",
    authors: Optional[List[str]] = None,
    year: Optional[int] = 2026,
    doi: Optional[str] = None,
    abstract: Optional[str] = "This is a test abstract.",
    file_path: Optional[str] = "/path/to/test.pdf",
    properties: Optional[Dict[str, Any]] = None,
) -> Paper:
    """Factory helper to create a Paper instance with sensible defaults."""
    if authors is None:
        authors = ["John Doe"]
    if id is None:
        id = doi or slugify(title)
    return Paper(
        id=id,
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        abstract=abstract,
        file_path=file_path,
        properties=properties or {},
    )

def create_author(
    id: Optional[str] = None,
    name: str = "John Doe",
    properties: Optional[Dict[str, Any]] = None,
) -> Author:
    """Factory helper to create an Author instance with sensible defaults."""
    if id is None:
        id = slugify(name)
    return Author(
        id=id,
        name=name,
        properties=properties or {},
    )

def create_chunk(
    id: Optional[str] = None,
    paper_id: str = "test_paper",
    text_content: str = "This is some test content of the chunk.",
    page_number: int = 1,
    embedding: Optional[List[float]] = None,
) -> Chunk:
    """Factory helper to create a Chunk instance with sensible defaults."""
    if id is None:
        id = f"{paper_id}#0"
    return Chunk(
        id=id,
        paper_id=paper_id,
        text_content=text_content,
        page_number=page_number,
        embedding=embedding,
    )
