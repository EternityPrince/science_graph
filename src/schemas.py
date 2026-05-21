from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    question: str
    limit: Optional[int] = 5

class NoteCreate(BaseModel):
    title: str
    content: str
    authors: Optional[List[str]] = Field(default_factory=list)
    tags: Optional[List[str]] = Field(default_factory=list)

class NoteResponse(BaseModel):
    id: str
    title: str
    created_at: Optional[str] = None
    authors: List[str] = []
    summary: Optional[str] = None
    abstract: Optional[str] = None

class OpenFileRequest(BaseModel):
    file_path: str

class GraphNode(BaseModel):
    id: str
    label: str
    title: str
    color: str
    size: int
    group: str
    shape: str = "dot"
    created_at: Optional[str] = None
    source_type: str = "paper"
    full_title: str

class GraphEdgeFont(BaseModel):
    size: int = 8
    align: str = "top"

class GraphEdgeColor(BaseModel):
    color: str = "#adb5bd"
    highlight: str = "#74c0fc"

class GraphEdge(BaseModel):
    from_node: str = Field(..., alias="from")
    to: str
    label: str
    arrows: str = "to"
    font: GraphEdgeFont = GraphEdgeFont()
    color: GraphEdgeColor = GraphEdgeColor()

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True
    )

class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]

class SearchResultItem(BaseModel):
    id: str
    title: str
    year: Optional[int] = None
    source_type: str = "paper"

class SearchResponse(BaseModel):
    results: List[SearchResultItem]
