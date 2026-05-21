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
    def get_chunks_for_paper(self, paper_id: str) -> List[Chunk]:
        """Retrieves all text chunks belonging to a paper."""
        pass

    @abstractmethod
    def get_all_chunks(self) -> List[Chunk]:
        """Retrieves all chunks from the database."""
        pass
