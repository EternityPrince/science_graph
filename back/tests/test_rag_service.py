import unittest
from unittest.mock import MagicMock, patch
import json
import asyncio
from typing import AsyncGenerator

from src.models import Chunk, Paper, Author, Concept
from src.services.rag_service import RAGService


class TestRAGService(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.graph_repo = MagicMock()
        self.vector_repo = MagicMock()
        self.emb_engine = MagicMock()
        self.llm_engine = MagicMock()
        self.expander = MagicMock()
        self.service = RAGService(
            self.graph_repo,
            self.vector_repo,
            self.emb_engine,
            self.llm_engine,
            self.expander
        )

    @patch("sentence_transformers.CrossEncoder")
    def test_get_reranker(self, mock_cross_encoder):
        mock_ce_instance = MagicMock()
        mock_cross_encoder.return_value = mock_ce_instance
        
        reranker = self.service._get_reranker()
        self.assertEqual(reranker, mock_ce_instance)
        mock_cross_encoder.assert_called_once()
        
        # Test cache
        reranker2 = self.service._get_reranker()
        self.assertEqual(reranker2, mock_ce_instance)
        self.assertEqual(mock_cross_encoder.call_count, 1)

    def test_resolve_node_name(self):
        # Paper
        mock_paper = MagicMock(spec=Paper)
        mock_paper.title = "Paper Title"
        self.graph_repo.get_paper.return_value = mock_paper
        res = self.service._resolve_node_name("p1", "Paper")
        self.assertEqual(res, "'Paper Title'")
        
        self.graph_repo.get_paper.return_value = None
        res = self.service._resolve_node_name("p1", "Paper")
        self.assertEqual(res, "'p1'")

        # Author
        mock_author = MagicMock(spec=Author)
        mock_author.name = "Author Name"
        self.graph_repo.get_author.return_value = mock_author
        res = self.service._resolve_node_name("a1", "Author")
        self.assertEqual(res, "Author Name")
        
        self.graph_repo.get_author.return_value = None
        res = self.service._resolve_node_name("a1", "Author")
        self.assertEqual(res, "a1")

        # Concept
        mock_concept = MagicMock(spec=Concept)
        mock_concept.name = "Concept Name"
        self.graph_repo.get_concept.return_value = mock_concept
        res = self.service._resolve_node_name("c1", "Concept")
        self.assertEqual(res, "Concept Name")

        self.graph_repo.get_concept.return_value = None
        res = self.service._resolve_node_name("c1", "Concept")
        self.assertEqual(res, "c1")
        
        # Default label
        res = self.service._resolve_node_name("x1", "Other")
        self.assertEqual(res, "x1")

    def test_build_context(self):
        chunk1 = MagicMock(spec=Chunk)
        chunk1.paper_id = "p1"
        chunk1.page_number = 2
        chunk1.text_content = "Chunk 1 content"

        paper1 = MagicMock(spec=Paper)
        paper1.title = "Title A"
        paper1.year = 2020
        paper1.authors = ["Author A"]

        self.graph_repo.get_papers_batch.return_value = {"p1": paper1}
        
        # Mock neighbors
        self.graph_repo.get_neighbors.return_value = [
            ("a1", "Author", "AUTHORED", "p1", "Paper", None),
            ("p1", "Paper", "MENTIONS_CONCEPT", "c1", "Concept", None),
            ("p1", "Paper", "CITES", "p2", "Paper", json.dumps({"raw_text": "cited B because of X"})),
            ("p1", "Paper", "CITES", "p3", "Paper", None),
            ("p1", "Paper", "OTHER_REL", "x1", "Other", None),
        ]
        
        # Mock name resolution
        def get_paper_mock(nid):
            if nid == "p1":
                return paper1
            elif nid == "p2":
                return MagicMock(title="Title B")
            else:
                return MagicMock(title="Title C")
        self.graph_repo.get_paper.side_effect = get_paper_mock
        
        mock_author = MagicMock()
        mock_author.name = "Author A"
        self.graph_repo.get_author.return_value = mock_author
        
        mock_concept = MagicMock()
        mock_concept.name = "Concept A"
        self.graph_repo.get_concept.return_value = mock_concept

        text_ctx, graph_ctx = self.service.build_context([(chunk1, 0.95)])
        
        # Check text block formatting
        self.assertIn("Block 1 (Score: 0.950) | Paper: Title A by Author A, 2020 (Page 2):", text_ctx)
        self.assertIn("\"\"\"\nChunk 1 content\n\"\"\"", text_ctx)
        
        # Check graph lines
        self.assertIn("- Author A (Author) authored paper 'Title A'", graph_ctx)
        self.assertIn("- Paper 'Title A' mentions concept/topic 'Concept A'", graph_ctx)
        self.assertIn("- Paper 'Title A' cites: cited B because of X", graph_ctx)
        self.assertIn("- Paper 'Title A' cites paper 'Title C'", graph_ctx)
        self.assertIn("- Node ''Title A'' is connected to 'x1' via OTHER_REL", graph_ctx)

    def test_retrieve_relevant_chunks_success(self):
        self.emb_engine.get_embedding.return_value = [0.1, 0.2]
        
        chunk1 = MagicMock(spec=Chunk)
        chunk1.id = "c1"
        chunk1.text_content = "content 1"
        
        chunk2 = MagicMock(spec=Chunk)
        chunk2.id = "c2"
        chunk2.text_content = "content 2"
        
        self.vector_repo.search_similar_chunks.return_value = [(chunk1, 0.9), (chunk2, 0.8)]
        self.vector_repo.search_text_fts5.return_value = [(chunk2, 0.7)]
        
        # Mock reranker: chunk2 (index 0) gets 0.85, chunk1 (index 1) gets 0.95
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.85, 0.95]
        self.service._reranker = mock_reranker
        
        res = self.service.retrieve_relevant_chunks("query", limit=2)
        
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0][0].id, "c1")
        self.assertAlmostEqual(res[0][1], 0.95)
        self.assertEqual(res[1][0].id, "c2")
        self.assertAlmostEqual(res[1][1], 0.85)

    def test_retrieve_relevant_chunks_no_results(self):
        self.vector_repo.search_similar_chunks.return_value = []
        self.vector_repo.search_text_fts5.return_value = []
        
        res = self.service.retrieve_relevant_chunks("query")
        self.assertEqual(res, [])

    def test_retrieve_relevant_chunks_reranker_fail_fallback_rrf(self):
        self.emb_engine.get_embedding.return_value = [0.1]
        
        chunk1 = MagicMock(spec=Chunk)
        chunk1.id = "c1"
        
        chunk2 = MagicMock(spec=Chunk)
        chunk2.id = "c2"
        
        self.vector_repo.search_similar_chunks.return_value = [(chunk1, 0.9)]
        self.vector_repo.search_text_fts5.return_value = [(chunk2, 0.8)]
        
        # Make reranker fail
        mock_reranker = MagicMock()
        mock_reranker.predict.side_effect = Exception("Reranker failed")
        self.service._reranker = mock_reranker
        
        res = self.service.retrieve_relevant_chunks("query", limit=2)
        self.assertEqual(len(res), 2)
        # RRF scores should decide order or presence
        self.assertEqual(res[0][0].id, "c1")
        self.assertEqual(res[1][0].id, "c2")

    def test_ask(self):
        # Empty retrieval
        with patch.object(self.service, "retrieve_relevant_chunks", return_value=[]):
            res = self.service.ask("query")
            self.assertTrue(res.startswith("Не найдено"))

        chunk = MagicMock(spec=Chunk)
        self.llm_engine.generate_response.return_value = "Answer Text"
        
        # With expander
        self.service.expander = MagicMock()
        self.service.expander.reranker = None
        self.service.expander.expand.return_value = "expansion block"
        
        with patch.object(self.service, "retrieve_relevant_chunks", return_value=[(chunk, 0.95)]):
            with patch.object(self.service, "build_context", return_value=("text context", "graph context")):
                with patch.object(self.service, "_get_reranker", return_value=MagicMock()):
                    res = self.service.ask("query")
                    self.assertEqual(res, "Answer Text")
                    self.service.expander.expand.assert_called_once()
                    self.llm_engine.generate_response.assert_called_once()

        # Without expander
        self.service.expander = None
        self.llm_engine.generate_response.reset_mock()
        with patch.object(self.service, "retrieve_relevant_chunks", return_value=[(chunk, 0.95)]):
            with patch.object(self.service, "build_context", return_value=("text context", "graph context")):
                res = self.service.ask("query")
                self.assertEqual(res, "Answer Text")
                self.llm_engine.generate_response.assert_called_once()

    async def test_generate_stream_success_no_mlx(self):
        chunk = MagicMock(spec=Chunk)
        self.service.expander = None
        self.llm_engine.model = None # triggers fallback
        self.llm_engine.generate_response.return_value = "Answer Text"
        
        with patch.object(self.service, "retrieve_relevant_chunks", return_value=[(chunk, 0.95)]):
            with patch.object(self.service, "build_context", return_value=("text context", "graph context")):
                stream = self.service.generate_stream("query")
                tokens = []
                async for chunk_dict in stream:
                    tokens.append(chunk_dict)
                
                # Verify we got tokens and done
                self.assertEqual(tokens[-1], {"type": "done"})
                token_texts = "".join([t["text"] for t in tokens if t["type"] == "token"])
                self.assertEqual(token_texts.strip(), "Answer Text")

    async def test_generate_stream_success_mlx(self):
        chunk = MagicMock(spec=Chunk)
        self.service.expander = MagicMock()
        self.service.expander.reranker = None
        self.service.expander.expand.return_value = "expanded context"
        
        self.llm_engine.model = MagicMock()
        self.llm_engine.tokenizer = MagicMock()
        
        # Mock stream_generate
        mock_response1 = MagicMock()
        mock_response1.text = "Answer"
        mock_response2 = MagicMock()
        mock_response2.text = " Text"
        
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.stream_generate.return_value = [mock_response1, mock_response2]
        
        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            with patch.object(self.service, "retrieve_relevant_chunks", return_value=[(chunk, 0.95)]):
                with patch.object(self.service, "_get_reranker", return_value=MagicMock()):
                    stream = self.service.generate_stream("query")
                    tokens = []
                    async for chunk_dict in stream:
                        tokens.append(chunk_dict)
                    
                    self.assertEqual(tokens[-1], {"type": "done"})
                    token_texts = "".join([t["text"] for t in tokens if t["type"] == "token"])
                    self.assertEqual(token_texts, "Answer Text")

    async def test_generate_stream_failures(self):
        # retrieval error
        with patch.object(self.service, "retrieve_relevant_chunks", side_effect=Exception("DB crash")):
            stream = self.service.generate_stream("query")
            results = [r async for r in stream]
            self.assertEqual(results[0]["type"], "error")
            self.assertIn("Retrieval failed", results[0]["text"])

        # no documents
        with patch.object(self.service, "retrieve_relevant_chunks", return_value=[]):
            stream = self.service.generate_stream("query")
            results = [r async for r in stream]
            self.assertEqual(results[0]["type"], "error")
            self.assertIn("No documents indexed yet", results[0]["text"])

        # context building error
        chunk = MagicMock(spec=Chunk)
        self.service.expander = None
        with patch.object(self.service, "retrieve_relevant_chunks", return_value=[(chunk, 0.95)]):
            with patch.object(self.service, "build_context", side_effect=Exception("Context building failed")):
                stream = self.service.generate_stream("query")
                results = [r async for r in stream]
                self.assertEqual(results[0]["type"], "error")
                self.assertIn("Context building failed", results[0]["text"])

        # generator thread exception
        self.llm_engine.model = None
        self.llm_engine.generate_response.side_effect = Exception("LLM connection timed out")
        with patch.object(self.service, "retrieve_relevant_chunks", return_value=[(chunk, 0.95)]):
            with patch.object(self.service, "build_context", return_value=("text context", "graph context")):
                stream = self.service.generate_stream("query")
                results = [r async for r in stream]
                self.assertEqual(results[0]["type"], "error")
                self.assertIn("Generation failed: LLM connection timed out", results[0]["text"])

    def test_trim_context_no_change(self):
        # Context < limit: nothing changes
        chunk1 = MagicMock(spec=Chunk)
        chunk1.paper_id = "p1"
        chunk1.page_number = 1
        chunk1.text_content = "text1"

        chunk2 = MagicMock(spec=Chunk)
        chunk2.paper_id = "p2"
        chunk2.page_number = 1
        chunk2.text_content = "text2"

        final_chunks = [(chunk1, 0.9), (chunk2, 0.8)]
        
        # Mock build_context
        def build_context_mock(chunks):
            txt = "\n\n".join([f"Block {i}: {c[0].text_content}" for i, c in enumerate(chunks, 1)])
            graph = "No direct graph relations found."
            return txt, graph
        
        with patch.object(self.service, "build_context", side_effect=build_context_mock):
            trimmed_text, trimmed_graph, trimmed_chunks = self.service.trim_context(
                context_text="Block 1: text1\n\nBlock 2: text2",
                context_graph="No direct graph relations found.",
                final_chunks=final_chunks,
                query="query",
                history_str="",
                system_prompt="system",
                model_max_context=1000,
                reserved_tokens=500
            )
            # Should keep both chunks
            self.assertEqual(len(trimmed_chunks), 2)
            self.assertEqual(trimmed_chunks[0][0].text_content, "text1")
            self.assertEqual(trimmed_chunks[1][0].text_content, "text2")

    def test_trim_context_trim_chunks(self):
        # Context > limit: chunks are cut from tail
        chunk1 = MagicMock(spec=Chunk)
        chunk1.paper_id = "p1"
        chunk1.page_number = 1
        chunk1.text_content = "text1"

        chunk2 = MagicMock(spec=Chunk)
        chunk2.paper_id = "p2"
        chunk2.page_number = 1
        chunk2.text_content = "text2"

        final_chunks = [(chunk1, 0.9), (chunk2, 0.8)]
        
        # Mock build_context
        def build_context_mock(chunks):
            txt = "\n\n".join([f"Block {i}: {c[0].text_content}" for i, c in enumerate(chunks, 1)])
            graph = "No direct graph relations found."
            return txt, graph

        with patch.object(self.service, "build_context", side_effect=build_context_mock):
            trimmed_text, trimmed_graph, trimmed_chunks = self.service.trim_context(
                context_text="Block 1: text1\n\nBlock 2: text2",
                context_graph="No direct graph relations found.",
                final_chunks=final_chunks,
                query="query",
                history_str="",
                system_prompt="system",
                model_max_context=40,
                reserved_tokens=25 # Limit is 15 tokens
            )
            # Should trim to 1 chunk
            self.assertEqual(len(trimmed_chunks), 1)
            self.assertEqual(trimmed_chunks[0][0].text_content, "text1")

    def test_trim_context_trim_graph(self):
        # Context > limit even with 1 chunk: graph is trimmed
        chunk1 = MagicMock(spec=Chunk)
        chunk1.paper_id = "p1"
        chunk1.page_number = 1
        chunk1.text_content = "text1"

        final_chunks = [(chunk1, 0.9)]
        
        # Mock build_context
        def build_context_mock(chunks):
            return "Block 1: text1", "- a1 (Author) authored paper p1\n- Paper p1 cites paper p2"
            
        # Mock get_neighbors
        self.graph_repo.get_neighbors.return_value = [
            ("a1", "Author", "AUTHORED", "p1", "Paper", json.dumps({"score": 0.5})),
            ("p1", "Paper", "CITES", "p2", "Paper", json.dumps({"score": 0.9})),
        ]
        # We need name resolution for nodes
        self.service._resolve_node_name = MagicMock(side_effect=lambda nid, lbl: nid)

        with patch.object(self.service, "build_context", side_effect=build_context_mock):
            trimmed_text, trimmed_graph, trimmed_chunks = self.service.trim_context(
                context_text="Block 1: text1",
                context_graph="- a1 (Author) authored paper p1\n- Paper p1 cites paper p2",
                final_chunks=final_chunks,
                query="query",
                history_str="",
                system_prompt="system",
                model_max_context=31,
                reserved_tokens=15 # limit is 16 tokens
            )
            
            self.assertIn("cites paper p2", trimmed_graph)
            self.assertNotIn("authored paper", trimmed_graph)

    @patch("src.services.rag_service.config")
    def test_expand_query_success(self, mock_config):
        mock_config.max_expanded_queries = 3
        mock_config.llm_max_tokens = 1000
        
        self.llm_engine.generate_response.return_value = '["cluster analysis", "grouping algorithm"]'
        self.llm_engine.extract_json.side_effect = lambda x: x
        
        res = self.service._expand_query("clustering")
        self.assertEqual(res, ["clustering", "cluster analysis", "grouping algorithm"])
        
        # Verify prompt format/intent
        self.llm_engine.generate_response.assert_called_once()
        prompt = self.llm_engine.generate_response.call_args[0][0]
        self.assertIn("You are a search query expansion assistant", prompt)

    @patch("src.services.rag_service.config")
    def test_expand_query_invalid_json_fallback(self, mock_config):
        mock_config.max_expanded_queries = 3
        mock_config.llm_max_tokens = 1000
        
        self.llm_engine.generate_response.return_value = 'invalid json'
        self.llm_engine.extract_json.side_effect = lambda x: x
        
        res = self.service._expand_query("clustering")
        self.assertEqual(res, ["clustering"])

    @patch("src.services.rag_service.config")
    def test_expand_query_exception_fallback(self, mock_config):
        mock_config.max_expanded_queries = 3
        mock_config.llm_max_tokens = 1000
        
        self.llm_engine.generate_response.side_effect = Exception("LLM unavailable")
        
        res = self.service._expand_query("clustering")
        self.assertEqual(res, ["clustering"])

    @patch("src.services.rag_service.config")
    def test_expand_query_length_limit(self, mock_config):
        mock_config.max_expanded_queries = 3
        
        long_query = "a" * 201
        res = self.service._expand_query(long_query)
        self.assertEqual(res, [long_query])
        self.llm_engine.generate_response.assert_not_called()

    @patch("src.services.rag_service.config")
    def test_expand_query_disabled(self, mock_config):
        mock_config.max_expanded_queries = 1
        
        res = self.service._expand_query("clustering")
        self.assertEqual(res, ["clustering"])
        self.llm_engine.generate_response.assert_not_called()

    @patch("sentence_transformers.CrossEncoder")
    def test_retrieve_relevant_chunks_with_expansion(self, mock_cross_encoder):
        mock_ce_instance = MagicMock()
        mock_ce_instance.predict.side_effect = lambda pairs: [0.95, 0.85, 0.75]
        self.service._reranker = mock_ce_instance

        # Mock query expansion
        self.service._expand_query = MagicMock(return_value=["clustering", "cluster analysis"])
        
        # We have 3 chunks
        chunk1 = MagicMock(spec=Chunk)
        chunk1.id = "c1"
        chunk1.text_content = "text 1"
        
        chunk2 = MagicMock(spec=Chunk)
        chunk2.id = "c2"
        chunk2.text_content = "text 2"
        
        chunk3 = MagicMock(spec=Chunk)
        chunk3.id = "c3"
        chunk3.text_content = "text 3"

        # Mock embedding engine
        self.emb_engine.get_embedding.side_effect = lambda q: [0.1] if q == "clustering" else [0.2]
        
        # Mock vector search
        def mock_search_similar(emb, limit):
            # If embedding is [0.1] (clustering)
            if emb == [0.1]:
                return [(chunk1, 0.9), (chunk2, 0.8)]
            # If embedding is [0.2] (cluster analysis)
            else:
                return [(chunk2, 0.85), (chunk3, 0.7)]
        self.vector_repo.search_similar_chunks.side_effect = mock_search_similar
        
        # Mock FTS5 to return chunk1 with lower score
        self.vector_repo.search_text_fts5.return_value = [(chunk1, 0.5)]
        
        res = self.service.retrieve_relevant_chunks("clustering", limit=3)
        
        # Let's check calls to search_similar_chunks
        self.assertEqual(self.vector_repo.search_similar_chunks.call_count, 2)
        
        # Let's check call to search_text_fts5 is run with original query only
        self.vector_repo.search_text_fts5.assert_called_once_with("clustering", limit=6)
        
        # Let's check the result contains the candidates sorted by reranker score
        self.assertEqual(len(res), 3)
        
        # Reranker pairs should have been called with the original query "clustering"
        mock_ce_instance.predict.assert_called_once()
        pairs = mock_ce_instance.predict.call_args[0][0]
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0][0], "clustering")
        self.assertEqual(pairs[1][0], "clustering")
        self.assertEqual(pairs[2][0], "clustering")
