import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os
import shutil
from pathlib import Path

from src.models import Chunk, Paper
from src.services.note_service import NoteService
from src.services.rag_service import RAGService


class TestNoteService(unittest.TestCase):
    def setUp(self):
        self.graph_repo = MagicMock()
        self.vector_repo = MagicMock()
        self.emb_engine = MagicMock()
        self.llm_engine = MagicMock()
        self.note_service = NoteService(
            graph_repo=self.graph_repo,
            vector_repo=self.vector_repo,
            embedding_engine=self.emb_engine,
            llm_engine=self.llm_engine
        )
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get_notes(self):
        # Setup dummy note papers
        p1 = Paper(
            id="note_1",
            title="Note 1",
            authors=["Author A"],
            year=2026,
            abstract="Note 1 Abstract",
            created_at="2026-05-21T12:00:00",
            properties={"summary": "Note 1 Summary"}
        )
        self.graph_repo.get_notes.return_value = [p1]

        notes = self.note_service.get_notes()
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["id"], "note_1")
        self.assertEqual(notes[0]["title"], "Note 1")
        self.assertEqual(notes[0]["summary"], "Note 1 Summary")
        self.assertEqual(notes[0]["abstract"], "Note 1 Abstract")

    @patch("src.services.note_service.Indexer")
    @patch("src.services.note_service.config")
    def test_create_note(self, mock_config, mock_indexer_class):
        # Override config.data_dir to temp_dir
        mock_config.data_dir = self.temp_dir

        mock_indexer = MagicMock()
        mock_indexer.index_markdown.return_value = "note_paper_id"
        mock_indexer_class.return_value = mock_indexer

        # Create note
        paper_id, note_path = self.note_service.create_note(
            title="Test Note Title",
            content="This is the content of the test note.",
            authors=["Author X", "Author Y"],
            tags=["AI", "RAG"]
        )

        self.assertEqual(paper_id, "note_paper_id")
        self.assertTrue(os.path.exists(note_path))

        # Check content
        with open(note_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("title: Test Note Title", content)
        self.assertIn("authors:", content)
        self.assertIn("- Author X", content)
        self.assertIn("- Author Y", content)
        self.assertIn("tags:", content)
        self.assertIn("- AI", content)
        self.assertIn("- RAG", content)
        self.assertIn("This is the content of the test note.", content)

        mock_indexer.index_markdown.assert_called_once_with(str(note_path))

    @patch("src.services.note_service.Indexer")
    @patch("src.services.note_service.config")
    def test_update_note_success(self, mock_config, mock_indexer_class):
        # Override config.data_dir to temp_dir
        mock_config.data_dir = self.temp_dir
        notes_dir = Path(self.temp_dir) / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)

        # Create original note file
        orig_file_path = notes_dir / "old_title.md"
        orig_file_path.write_text("---\ntitle: Old Title\nauthors:\n- Author Old\ntags:\n- tag-old\n---\n\nOld Content", encoding="utf-8")

        mock_paper = MagicMock()
        mock_paper.title = "Old Title"
        mock_paper.file_path = str(orig_file_path)
        self.graph_repo.get_paper.return_value = mock_paper

        mock_indexer = MagicMock()
        mock_indexer.index_markdown.return_value = "new_note_id"
        mock_indexer_class.return_value = mock_indexer

        # Update note
        result = self.note_service.update_note(
            note_id="old_note_id",
            title="New Title",
            content="New Content",
            authors=["Author New"],
            tags=["tag-new"]
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["id"], "new_note_id")
        
        # Verify old file was deleted because title changed
        self.assertFalse(os.path.exists(orig_file_path))
        
        # Verify new file exists and has updated content
        new_file_path = Path(result["file_path"])
        self.assertTrue(new_file_path.exists())
        self.assertEqual(new_file_path.name, "new_title.md")
        
        content = new_file_path.read_text(encoding="utf-8")
        self.assertIn("title: New Title", content)
        self.assertIn("Author New", content)
        self.assertIn("tag-new", content)
        self.assertIn("New Content", content)

        self.graph_repo.delete_node.assert_called_once_with("old_note_id")
        mock_indexer.index_markdown.assert_called_once_with(str(new_file_path))

    def test_update_note_not_found(self):
        self.graph_repo.get_paper.return_value = None
        with self.assertRaises(ValueError) as context:
            self.note_service.update_note("nonexistent_id", title="Title")
        self.assertIn("not found", str(context.exception))

    def test_update_note_file_not_found(self):
        mock_paper = MagicMock()
        mock_paper.file_path = "/nonexistent/path.md"
        self.graph_repo.get_paper.return_value = mock_paper
        with self.assertRaises(ValueError) as context:
            self.note_service.update_note("some_id", title="Title")
        self.assertIn("not found on disk", str(context.exception))


class TestRAGService(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.graph_repo = MagicMock()
        self.vector_repo = MagicMock()
        self.emb_engine = MagicMock()
        self.llm_engine = MagicMock()
        self.rag_service = RAGService(
            graph_repo=self.graph_repo,
            vector_repo=self.vector_repo,
            embedding_engine=self.emb_engine,
            llm_engine=self.llm_engine
        )

    def test_retrieve_relevant_chunks(self):
        c1 = Chunk(id="chunk_1", paper_id="paper_1", text_content="Attention mechanisms are widely used in transformer neural networks.", page_number=1)
        c2 = Chunk(id="chunk_2", paper_id="paper_1", text_content="Reinforcement learning relies on reward functions to train policy networks.", page_number=2)

        self.emb_engine.get_embedding.return_value = [0.1] * 384
        self.vector_repo.search_similar_chunks.return_value = [(c2, 0.9)]
        self.vector_repo.search_text_fts5.return_value = [(c1, 1.2)]

        mock_reranker = MagicMock()
        def mock_predict(pairs):
            return [1.5 if "Attention" in doc else 0.5 for query, doc in pairs]
        mock_reranker.predict.side_effect = mock_predict
        self.rag_service._reranker = mock_reranker

        results = self.rag_service.retrieve_relevant_chunks("query text", limit=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0].id, "chunk_1")  # Score 1.5 is greater than 0.5
        self.assertEqual(results[1][0].id, "chunk_2")

    async def test_stream_rag_response_success(self):
        c1 = Chunk(id="chunk_1", paper_id="paper_1", text_content="Attention mechanisms.", page_number=1)
        
        self.emb_engine.get_embedding.return_value = [0.1] * 384
        self.vector_repo.search_similar_chunks.return_value = [(c1, 0.9)]
        self.vector_repo.search_text_fts5.return_value = []
        
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [1.5]
        self.rag_service._reranker = mock_reranker
        
        self.graph_repo.get_paper.return_value = Paper(id="paper_1", title="Attention paper", authors=["Author A"], year=2021)
        self.graph_repo.get_neighbors.return_value = []
        
        self.rag_service.llm_engine.model = None
        self.rag_service.llm_engine.generate_response.return_value = "Response from LLM"
        
        events = []
        async for event in self.rag_service.generate_stream("question here", limit=1):
            events.append(event)
            
        self.assertTrue(len(events) > 1)
        self.assertEqual(events[-1], {"type": "done"})
        text_tokens = "".join([e["text"] for e in events if e["type"] == "token"])
        self.assertEqual(text_tokens.strip(), "Response from LLM")


if __name__ == "__main__":
    unittest.main()
