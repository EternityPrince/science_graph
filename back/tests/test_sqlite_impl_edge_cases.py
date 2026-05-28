import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.models import Paper, Author, Concept, Chunk
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository, ConnectionProxy


class TestSQLiteImplEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.graph_repo = SQLiteGraphRepository(self.tmp.name)
        self.vector_repo = SQLiteVectorRepository(self.tmp.name)

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def test_connection_proxy_transaction_isolation(self):
        """Test ConnectionProxy commit/rollback behavior based on is_transaction value."""
        mock_conn = MagicMock()
        
        # 1. When is_transaction=False (normal connection proxy)
        proxy = ConnectionProxy(mock_conn, is_transaction=False)
        proxy.commit()
        mock_conn.commit.assert_called_once()
        
        proxy.rollback()
        mock_conn.rollback.assert_called_once()
        
        # Check __enter__ and __exit__ call connection
        proxy.__enter__()
        mock_conn.__enter__.assert_called_once()
        proxy.__exit__(None, None, None)
        mock_conn.__exit__.assert_called_once()
        
        # 2. When is_transaction=True (in-transaction connection proxy)
        mock_conn.reset_mock()
        tx_proxy = ConnectionProxy(mock_conn, is_transaction=True)
        tx_proxy.commit()
        # Commit should not be called on the raw connection because transaction manager handles it
        mock_conn.commit.assert_not_called()
        
        tx_proxy.rollback()
        mock_conn.rollback.assert_not_called()
        
        # Enter and exit should not be called
        tx_proxy.__enter__()
        mock_conn.__enter__.assert_not_called()
        tx_proxy.__exit__(None, None, None)
        mock_conn.__exit__.assert_not_called()

    def test_connection_proxy_close_on_del(self):
        """Test ConnectionProxy close logic in destructor."""
        mock_conn = MagicMock()
        proxy = ConnectionProxy(mock_conn, is_transaction=False)
        proxy.__del__()
        mock_conn.close.assert_called_once()

    def test_conflict_resolution_indexed_over_placeholder(self):
        """Verify inserting fully-indexed node over a placeholder node works, and vice-versa."""
        # 1. Save placeholder paper node
        placeholder_paper = Paper(
            id="p_test",
            title="Placeholder Title",
            authors=[],
            properties={"is_placeholder": True}
        )
        self.graph_repo.save_paper(placeholder_paper)
        
        # Check database stats and columns
        stats_before = self.graph_repo.get_stats()
        self.assertEqual(stats_before["mentioned_papers"], 1)
        self.assertEqual(stats_before["indexed_papers"], 0)
        
        # 2. Save fully-indexed paper node over the placeholder
        real_paper = Paper(
            id="p_test",
            title="Actual Brilliant Paper",
            authors=["Alice", "Bob"],
            year=2026,
            abstract="Summary here",
            properties={"source_type": "paper"}
        )
        self.graph_repo.save_paper(real_paper)
        
        stats_after = self.graph_repo.get_stats()
        self.assertEqual(stats_after["mentioned_papers"], 0)
        self.assertEqual(stats_after["indexed_papers"], 1)
        
        p = self.graph_repo.get_paper("p_test")
        self.assertEqual(p.title, "Actual Brilliant Paper")
        self.assertEqual(p.authors, ["Alice", "Bob"])
        self.assertEqual(p.abstract, "Summary here")
        
        # 3. Save placeholder again over fully-indexed node (should NOT overwrite actual paper properties)
        placeholder_paper_2 = Paper(
            id="p_test",
            title="Another Placeholder",
            authors=[],
            properties={"is_placeholder": True}
        )
        self.graph_repo.save_paper(placeholder_paper_2)
        
        # Properties should remain those of the real paper, not the placeholder
        p_final = self.graph_repo.get_paper("p_test")
        self.assertEqual(p_final.title, "Actual Brilliant Paper")
        self.assertEqual(p_final.authors, ["Alice", "Bob"])

    def test_save_edges_bulk_creates_placeholders(self):
        """Verify that save_edges_bulk automatically creates placeholders for unknown node IDs."""
        # Save edge between nonexistent source and nonexistent target
        edges = [
            ("p:unknown_1", "p:unknown_2", "CITES", {"intent": "support"})
        ]
        self.graph_repo.save_edges_bulk(edges)
        
        # Verification: Nodes should have been auto-created as placeholders
        p1 = self.graph_repo.get_paper("p:unknown_1")
        p2 = self.graph_repo.get_paper("p:unknown_2")
        self.assertIsNotNone(p1)
        self.assertIsNotNone(p2)
        
        # Verify node virtual properties
        with self.graph_repo._get_connection() as conn:
            row = conn.execute("SELECT label, is_placeholder FROM nodes WHERE id = ?", ("p:unknown_1",)).fetchone()
            self.assertEqual(row["label"], "Paper")
            self.assertEqual(row["is_placeholder"], 1)
            
            row2 = conn.execute("SELECT label, is_placeholder FROM nodes WHERE id = ?", ("p:unknown_2",)).fetchone()
            self.assertEqual(row2["label"], "Paper")
            self.assertEqual(row2["is_placeholder"], 1)

    def test_save_nodes_bulk_conflict_behavior(self):
        """Verify save_nodes_bulk updates placeholders correctly but preserves real nodes."""
        # 1. Add real node and placeholder node to database
        self.graph_repo.save_paper(Paper(id="real_node", title="Real Title", authors=[]))
        self.graph_repo.save_paper(Paper(id="placeholder_node", title="Placeholder Title", authors=[], properties={"is_placeholder": True}))
        
        # 2. Call save_nodes_bulk with conflicts
        nodes_to_save = [
            ("real_node", "Paper", {"title": "Attempt Overwrite Real", "is_placeholder": True}), # Should not overwrite
            ("placeholder_node", "Paper", {"title": "Updated Real Title", "is_placeholder": False}), # Should overwrite since target was a placeholder
            ("new_node", "Paper", {"title": "New Node Title"})
        ]
        self.graph_repo.save_nodes_bulk(nodes_to_save)
        
        # Real node should NOT be overwritten
        p_real = self.graph_repo.get_paper("real_node")
        self.assertEqual(p_real.title, "Real Title")
        
        # Placeholder node SHOULD be updated and become fully indexed
        p_placeholder = self.graph_repo.get_paper("placeholder_node")
        self.assertEqual(p_placeholder.title, "Updated Real Title")
        
        # New node should exist
        p_new = self.graph_repo.get_paper("new_node")
        self.assertEqual(p_new.title, "New Node Title")

    def test_transaction_recursive_concurrency(self):
        """Test transaction re-entry behavior (nested transaction yields without raising exceptions)."""
        # Top level transaction
        with self.graph_repo.transaction():
            self.graph_repo.save_paper(Paper(id="p1", title="Paper 1"))
            
            # Nested transaction (should reuse the active connection without re-beginning raw SQL transaction)
            with self.graph_repo.transaction():
                self.graph_repo.save_paper(Paper(id="p2", title="Paper 2"))
                
        # Verification: both papers should be saved successfully
        self.assertIsNotNone(self.graph_repo.get_paper("p1"))
        self.assertIsNotNone(self.graph_repo.get_paper("p2"))

    def test_sqlite_repository_methods_edge_cases(self):
        """Cover various edge cases and exception handling paths in SQLiteGraphRepository."""
        # 1. save_nodes_bulk and save_chunks_bulk with empty inputs
        self.graph_repo.save_nodes_bulk([])
        self.vector_repo.save_chunks_bulk([])
        self.vector_repo.save_chunks([])

        # 2. get_papers_batch with empty list
        self.assertEqual(self.graph_repo.get_papers_batch([]), {})

        # 3. get_distinct_targets with empty list
        self.assertEqual(self.graph_repo.get_distinct_targets([], "MENTIONS_CONCEPT"), [])

        # 4. find_paper_by_title matching exact ID
        paper_id_only = Paper(id="exact_id_title", title="Some Title", authors=[])
        self.graph_repo.save_paper(paper_id_only)
        found = self.graph_repo.find_paper_by_title("exact_id_title")
        self.assertIsNotNone(found)
        self.assertEqual(found.id, "exact_id_title")

        # 5. find_paper_by_doi and find_paper_by_content_hash with empty input
        self.assertIsNone(self.graph_repo.find_paper_by_doi(""))
        self.assertIsNone(self.graph_repo.find_paper_by_content_hash(""))

        # 6. get_neighbors with max_depth < 1 and max_depth > 1 linear traversal
        self.assertEqual(self.graph_repo.get_neighbors("exact_id_title", max_depth=0), [])
        
        # Linear chain for max_depth > 1
        self.graph_repo.save_paper(Paper(id="node_a", title="Node A", authors=[]))
        self.graph_repo.save_paper(Paper(id="node_b", title="Node B", authors=[]))
        self.graph_repo.save_paper(Paper(id="node_c", title="Node C", authors=[]))
        self.graph_repo.add_edge("node_a", "node_b", "CITES")
        self.graph_repo.add_edge("node_b", "node_c", "CITES")
        
        neighbors_depth2 = self.graph_repo.get_neighbors("node_a", max_depth=2)
        # Should find at least two edges (node_a -> node_b and node_b -> node_c)
        self.assertTrue(len(neighbors_depth2) >= 2)

        # 7. get_node_by_id and get_node_properties
        node_tuple = self.graph_repo.get_node_by_id("node_a")
        self.assertIsNotNone(node_tuple)
        self.assertEqual(node_tuple[0], "Paper")
        props = self.graph_repo.get_node_properties("node_a")
        self.assertIsNotNone(props)
        self.assertEqual(props.get("title"), "Node A")

        # Non-existent node properties
        self.assertIsNone(self.graph_repo.get_node_properties("nonexistent_node"))

        # 8. get_papers_by_entity
        self.graph_repo.save_concept(Concept(id="concept_x", name="Concept X"))
        self.graph_repo.add_edge("node_a", "concept_x", "MENTIONS_CONCEPT")
        papers_by_entity = self.graph_repo.get_papers_by_entity("concept_x", "MENTIONS_CONCEPT")
        self.assertEqual(len(papers_by_entity), 1)
        self.assertEqual(papers_by_entity[0].id, "node_a")

        # 9. search_papers_by_title
        papers_searched = self.graph_repo.search_papers_by_title("Node A")
        self.assertEqual(len(papers_searched), 1)
        self.assertEqual(papers_searched[0].id, "node_a")

        # 10. get_paper_ids and get_non_placeholder_paper_ids
        pids = self.graph_repo.get_paper_ids()
        self.assertIn("node_a", pids)
        non_placeholder_pids = self.graph_repo.get_non_placeholder_paper_ids()
        self.assertIn("node_a", non_placeholder_pids)

        # 11. cleanup_orphaned_concepts
        self.graph_repo.save_concept(Concept(id="orphan_concept", name="Orphan Concept"))
        removed_count = self.graph_repo.cleanup_orphaned_concepts()
        self.assertEqual(removed_count, 1)

        # 12. get_browse_rows and get_browse_count for documents, authors, concepts
        # documents count and rows
        self.assertTrue(self.graph_repo.get_browse_count("documents") > 0)
        self.assertTrue(len(self.graph_repo.get_browse_rows("documents", page=1, limit=5)) > 0)
        # documents with query search
        self.assertEqual(self.graph_repo.get_browse_count("documents", "NonexistentSearch"), 0)
        self.assertEqual(len(self.graph_repo.get_browse_rows("documents", page=1, limit=5, search_query="NonexistentSearch")), 0)
        
        # authors count and rows
        self.graph_repo.save_author(Author(id="auth_a", name="Author A"))
        self.assertTrue(self.graph_repo.get_browse_count("authors") > 0)
        self.assertTrue(len(self.graph_repo.get_browse_rows("authors", page=1, limit=5)) > 0)
        self.assertEqual(self.graph_repo.get_browse_count("authors", "Author A"), 1)
        self.assertEqual(len(self.graph_repo.get_browse_rows("authors", page=1, limit=5, search_query="Author A")), 1)

        # concepts count and rows
        self.assertTrue(self.graph_repo.get_browse_count("concepts") > 0)
        self.assertTrue(len(self.graph_repo.get_browse_rows("concepts", page=1, limit=5)) > 0)
        self.assertEqual(self.graph_repo.get_browse_count("concepts", "Concept X"), 1)
        self.assertEqual(len(self.graph_repo.get_browse_rows("concepts", page=1, limit=5, search_query="Concept X")), 1)

        # 13. get_distinct_targets normal flow
        targets = self.graph_repo.get_distinct_targets(["node_a"], "MENTIONS_CONCEPT")
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0][0], "concept_x")

        # 14. get_paper_source_types with invalid JSON properties
        with self.graph_repo._get_connection() as conn:
            conn.execute("INSERT OR REPLACE INTO nodes (id, label, properties) VALUES ('invalid_json_paper', 'Paper', '\"not_an_object\"')")
            conn.commit()
        # Should skip parsing error and not crash
        source_types = self.graph_repo.get_paper_source_types()
        self.assertNotIn("invalid_json_paper", source_types)

        # 15. get_concept_aliases with invalid JSON properties
        with self.graph_repo._get_connection() as conn:
            conn.execute("INSERT OR REPLACE INTO nodes (id, label, properties) VALUES ('invalid_json_concept', 'Concept', '\"not_an_object\"')")
            conn.commit()
        aliases = self.graph_repo.get_concept_aliases()
        # Should not crash
        self.assertIsNotNone(aliases)

    def test_fts5_and_vector_repo_edge_cases(self):
        """Cover search_text_fts5, USearch load failure, and empty search results."""
        # 1. FTS5 empty query
        self.assertEqual(self.vector_repo.search_text_fts5(""), [])
        self.assertEqual(self.vector_repo.search_text_fts5("   "), [])

        # 2. FTS5 query causing sqlite3.OperationalError (unclosed quote or syntax error)
        # MATCH query parser will throw on syntax errors
        results = self.vector_repo.search_text_fts5('"')
        self.assertEqual(results, [])

        # 3. similar chunks search with empty index
        self.assertEqual(self.vector_repo.search_similar_chunks([0.1]*384), [])

        # 4. Save chunk without embedding (should be skipped)
        chunk_no_emb = Chunk(id="c_no_emb", paper_id="node_a", text_content="No embedding", page_number=1, embedding=None)
        self.vector_repo.save_chunks([chunk_no_emb])
        self.assertEqual(len(self.vector_repo.get_chunks_for_paper("c_no_emb")), 0)

        # 5. USearch load failure/self-healing mock
        # Mock Index load to raise exception
        from usearch.index import Index
        with patch.object(Index, "load", side_effect=RuntimeError("Index corrupt")):
            # Trigger load inside _get_index
            self.vector_repo._usearch_index = None
            idx = self.vector_repo._get_index(384)
            self.assertIsNotNone(idx)
            self.assertEqual(len(idx), 0)

        # 6. similar chunks search where key is not in databases (h not in key_to_dist or results match len == 0)
        # Save a chunk to have it in db and usearch
        chunk = Chunk(id="chunk_1", paper_id="node_a", text_content="text content", page_number=1, embedding=[0.1]*384)
        self.vector_repo.save_chunks([chunk])
        
        # When querying similar chunks, it searches using the mock distances
        # If we return matches from search that aren't in sqlite chunks table, they should be filtered out.
        # Let's mock index.search to return a dummy key
        mock_matches = MagicMock()
        mock_matches.keys = [999999] # nonexistent key
        mock_matches.distances = [0.1]
        
        with patch.object(Index, "search", return_value=mock_matches):
            results = self.vector_repo.search_similar_chunks([0.1]*384, limit=5)
            self.assertEqual(results, [])

    def test_connection_proxy_del_exception(self):
        """Verify ConnectionProxy destructor ignores exceptions during connection close."""
        mock_conn = MagicMock()
        mock_conn.close.side_effect = Exception("Close error")
        proxy = ConnectionProxy(mock_conn, is_transaction=False)
        # Should not raise exception
        proxy.__del__()
        mock_conn.close.assert_called_once()

