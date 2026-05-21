import os
import tempfile
import textwrap
import unittest
from unittest.mock import MagicMock

from src.models import Paper
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
from src.indexer import Indexer

class TestIndexerWikilinks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()

        self.graph_repo = SQLiteGraphRepository(self.tmp.name)
        self.vector_repo = SQLiteVectorRepository(self.tmp.name)

        self.emb_engine = MagicMock()
        self.emb_engine.get_embeddings.side_effect = lambda texts: [[0.0] * 3 for _ in texts]

        self.indexer = Indexer(self.graph_repo, self.vector_repo, self.emb_engine)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def _write_md(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        )
        f.write(content)
        f.close()
        return f.name

    def test_wikilink_resolves_to_existing_paper(self):
        # 1. Create and index target paper
        target_content = textwrap.dedent("""\
            ---
            title: "Attention Mechanism"
            ---
            This note is about attention.
            """)
        target_path = self._write_md(target_content)

        # 2. Create source note linking to target and a non-existent concept
        source_content = textwrap.dedent("""\
            ---
            title: "Transformers Note"
            ---
            This note references [[Attention Mechanism]] and [[Quantum Computing]].
            """)
        source_path = self._write_md(source_content)

        try:
            target_id = self.indexer.index_markdown(target_path)
            source_id = self.indexer.index_markdown(source_path)

            # Check that both notes are indexed
            target_paper = self.graph_repo.get_paper(target_id)
            self.assertEqual(target_paper.title, "Attention Mechanism")

            source_paper = self.graph_repo.get_paper(source_id)
            self.assertEqual(source_paper.title, "Transformers Note")

            # Check neighbors of source note
            neighbors = self.graph_repo.get_neighbors(source_id, max_depth=1)
            
            # We expect a RELATED_TO edge directly from source_id to target_id (Attention Mechanism)
            # and a RELATED_TO edge to the concept "Quantum Computing"
            direct_paper_links = []
            concept_links = []
            for src_id, src_label, edge_type, tgt_id, tgt_label, edge_props in neighbors:
                if edge_type == "RELATED_TO":
                    # Determine neighbor node relative to source_id
                    if src_id == source_id:
                        neighbor_id = tgt_id
                        neighbor_label = tgt_label
                    else:
                        neighbor_id = src_id
                        neighbor_label = src_label
                        
                    if neighbor_label == "Paper":
                        direct_paper_links.append(neighbor_id)
                    elif neighbor_label == "Concept":
                        concept_links.append(neighbor_id)

            self.assertIn(target_id, direct_paper_links)
            self.assertEqual(len(concept_links), 1)
            # The concept id should be slugified "quantum_computing"
            self.assertEqual(concept_links[0], "quantum_computing")

        finally:
            os.unlink(target_path)
            os.unlink(source_path)
