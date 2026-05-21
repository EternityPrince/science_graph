from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.models import Paper, Author, Concept, Chunk, Edge

class GraphRepository(ABC):
    @abstractmethod
    def save_paper(self, paper: Paper) -> None:
        """Saves a paper node to the database."""
        pass

    @abstractmethod
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Retrieves a paper by its ID."""
        pass

    @abstractmethod
    def get_papers_batch(self, paper_ids: List[str]) -> Dict[str, Paper]:
        """Retrieves a dictionary of paper_id -> Paper for the requested paper IDs."""
        pass

    @abstractmethod
    def find_paper_by_title(self, title: str) -> Optional[Paper]:
        """Finds a paper or note by its exact or case-insensitive title."""
        pass

    @abstractmethod
    def save_author(self, author: Author) -> None:
        """Saves an author node to the database."""
        pass

    @abstractmethod
    def get_author(self, author_id: str) -> Optional[Author]:
        """Retrieves an author by ID."""
        pass

    @abstractmethod
    def save_concept(self, concept: Concept) -> None:
        """Saves a concept node to the database."""
        pass

    @abstractmethod
    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Retrieves a concept by ID."""
        pass

    @abstractmethod
    def add_edge(self, source_id: str, target_id: str, edge_type: str, properties: Dict[str, Any] = None) -> None:
        """Adds a directed edge between two nodes."""
        pass

    @abstractmethod
    def get_neighbors(self, node_id: str, max_depth: int = 1) -> List[tuple[str, str, str, str, str, str]]:
        """
        Returns connections from node_id.
        Each connection is a tuple: (src_id, src_label, edge_type, target_id, target_label, edge_properties_json)
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, int]:
        """Returns statistical numbers of nodes and edges."""
        pass

    @abstractmethod
    def cleanup_orphaned_concepts(self) -> int:
        """Deletes Concept nodes with degree 0 and returns the number of deleted nodes."""
        pass

    @abstractmethod
    def get_all_nodes(self) -> List[tuple[str, str, str]]:
        """Retrieves all node rows as (id, label, properties)."""
        pass

    @abstractmethod
    def get_all_edges(self) -> List[tuple[str, str, str, str]]:
        """Retrieves all edge rows as (source_id, target_id, type, properties)."""
        pass

    @abstractmethod
    def get_node_by_id(self, node_id: str) -> Optional[tuple[str, str]]:
        """Retrieves label and properties for a node ID."""
        pass

    @abstractmethod
    def get_papers_by_author(self, author_id: str) -> List[Paper]:
        """Retrieves all papers written by a given author ID."""
        pass

    @abstractmethod
    def get_papers_by_entity(self, entity_id: str, edge_type: str) -> List[Paper]:
        """Retrieves all papers connected to a specific entity with a specific edge type."""
        pass

    @abstractmethod
    def get_distinct_targets(self, source_ids: List[str], edge_type: str) -> List[tuple[str, str]]:
        """Retrieves distinct target IDs and target properties for source IDs and an edge type."""
        pass

    @abstractmethod
    def search_papers_by_title(self, query: str, limit: int = 20) -> List[Paper]:
        """Searches papers by title (case-insensitive) using the title column."""
        pass

    @abstractmethod
    def get_notes(self) -> List[Paper]:
        """Retrieves all nodes that represent notes (source_type = 'note')."""
        pass


class VectorRepository(ABC):
    @abstractmethod
    def save_chunks(self, chunks: List[Chunk]) -> None:
        """Saves text chunks and their embeddings."""
        pass

    @abstractmethod
    def search_similar_chunks(self, query_embedding: List[float], limit: int = 5) -> List[tuple[Chunk, float]]:
        """
        Searches for similar chunks using cosine similarity.
        Returns a list of tuples (Chunk, score).
        """
        pass

    @abstractmethod
    def search_text_fts5(self, query: str, limit: int = 10) -> List[tuple[Chunk, float]]:
        """
        Searches for chunks using SQLite FTS5 BM25 search.
        Returns a list of tuples (Chunk, score).
        """
        pass

    @abstractmethod
    def get_chunks_for_paper(self, paper_id: str) -> List[Chunk]:
        """Retrieves all text chunks belonging to a paper."""
        pass

    @abstractmethod
    def get_all_chunks(self) -> List[Chunk]:
        """Retrieves all chunks from the database."""
        pass
