from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Paper:
    id: str  # DOI or hash/slug of the title
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    file_path: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Author:
    id: str  # slug of name (e.g. "vladimir_kasterin")
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Concept:
    id: str  # slug of concept (e.g. "self_attention")
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Chunk:
    id: str  # paper_id#index
    paper_id: str
    text_content: str
    page_number: int
    embedding: Optional[List[float]] = None

@dataclass
class Edge:
    source_id: str
    target_id: str
    type: str  # CITES, AUTHORED, MENTIONS_CONCEPT, RELATED_TO
    properties: Dict[str, Any] = field(default_factory=dict)
