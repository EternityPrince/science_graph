import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from src.models import Concept, Paper, Author, Chunk
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
from src.services.doctor_service import DoctorService, clean_text
from src.cli import app
from tests.output_utils import plain_output

runner = CliRunner()


class TestDoctor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.graph_repo = SQLiteGraphRepository(self.tmp.name)
        self.vector_repo = SQLiteVectorRepository(self.tmp.name)
        self.doctor_service = DoctorService(self.graph_repo, self.vector_repo)
        
        self.ner_patcher = None
        if self._testMethodName != "test_doctor_ner_author_enrichment":
            self.ner_patcher = patch("src.ner_engine.extract_persons_from_text", return_value=[])
            self.ner_patcher.start()

    def tearDown(self):
        if self.ner_patcher:
            self.ner_patcher.stop()
        os.unlink(self.tmp.name)

    def test_clean_text(self):
        self.assertEqual(clean_text("<think>reasoning</think>Clean Name"), "Clean Name")
        self.assertEqual(clean_text("<thought>reasoning</thought>Clean Name"), "Clean Name")
        self.assertEqual(clean_text("```json\n{\"name\": \"value\"}\n```"), "{\"name\": \"value\"}")
        self.assertEqual(clean_text("\"Wrapping Quotes\""), "Wrapping Quotes")
        self.assertEqual(clean_text("Line 1\n\n\n\nLine 2"), "Line 1\n\nLine 2")
        self.assertEqual(clean_text("   Many   Spaces   "), "Many Spaces")
        self.assertEqual(clean_text("<think>unclosed thought"), "")

    def test_doctor_diagnostics_check_only(self):
        # 1. Save an uncleaned paper
        paper = Paper(
            id="p1",
            title="```json\nTitle with codeblock\n```",
            authors=["\"Author One\"", "<think>x</think>Author Two"],
            abstract="<think>some reasoning</think>This is abstract."
        )
        self.graph_repo.save_paper(paper)

        # 2. Save an uncleaned concept
        concept = Concept(
            id="uncleaned_concept",
            name="\"Machine Learning\"",
            properties={"description": "<think>thought</think>Actual description."}
        )
        self.graph_repo.save_concept(concept)

        # 3. Save an uncleaned chunk
        chunk = Chunk(
            id="p1#0",
            paper_id="p1",
            text_content="```\nChunk content\n```",
            page_number=1,
            embedding=[0.1] * 384
        )
        self.vector_repo.save_chunks([chunk])

        # Run diagnostics in check-only mode
        report = self.doctor_service.run_diagnostics(fix=False)
        
        # Verify stats
        self.assertEqual(report["stats"]["papers_checked"], 1)
        self.assertEqual(report["stats"]["papers_fixed"], 1)
        self.assertEqual(report["stats"]["concepts_checked"], 1)
        self.assertEqual(report["stats"]["concepts_migrated"], 1)
        self.assertEqual(report["stats"]["chunks_checked"], 1)
        self.assertEqual(report["stats"]["chunks_fixed"], 1)

        # Verify database is UNCHANGED (fix=False)
        p = self.graph_repo.get_paper("p1")
        self.assertIn("```json", p.title)
        c = self.graph_repo.get_concept("uncleaned_concept")
        self.assertIsNotNone(c)
        self.assertIn("\"", c.name)

    def test_doctor_diagnostics_fix(self):
        # 1. Create a paper and an author
        paper = Paper(
            id="p1",
            title="```json\nTitle with codeblock\n```",
            authors=["\"Author One\""],
            abstract="<think>some reasoning</think>This is abstract."
        )
        self.graph_repo.save_paper(paper)

        author = Author(
            id="uncleaned_author",
            name="\"Author One\""
        )
        self.graph_repo.save_author(author)
        self.graph_repo.add_edge("uncleaned_author", "p1", "AUTHORED")

        # 2. Save concept and link it
        concept = Concept(
            id="uncleaned_concept",
            name="\"Machine Learning\"",
            properties={"description": "<think>thought</think>Actual description."}
        )
        self.graph_repo.save_concept(concept)
        self.graph_repo.add_edge("p1", "uncleaned_concept", "MENTIONS_CONCEPT")

        # 3. Save chunk
        chunk = Chunk(
            id="p1#0",
            paper_id="p1",
            text_content="```\nChunk content\n```",
            page_number=1,
            embedding=[0.1] * 384
        )
        self.vector_repo.save_chunks([chunk])

        # Run diagnostics in FIX mode
        report = self.doctor_service.run_diagnostics(fix=True)

        # 4. Verify report
        self.assertEqual(report["stats"]["papers_fixed"], 1)
        self.assertEqual(report["stats"]["authors_migrated"], 1)
        self.assertEqual(report["stats"]["concepts_migrated"], 1)
        self.assertEqual(report["stats"]["chunks_fixed"], 1)

        # 5. Verify database corrections
        # Paper
        p = self.graph_repo.get_paper("p1")
        self.assertEqual(p.title, "Title with codeblock")
        self.assertEqual(p.abstract, "This is abstract.")
        self.assertEqual(p.authors, ["Author One"])

        # Author ID should have migrated from 'uncleaned_author' to 'author_one'
        old_author = self.graph_repo.get_author("uncleaned_author")
        self.assertIsNone(old_author)
        new_author = self.graph_repo.get_author("author_one")
        self.assertIsNotNone(new_author)
        self.assertEqual(new_author.name, "Author One")

        # Concept ID should have migrated from 'uncleaned_concept' to 'machine_learning'
        old_concept = self.graph_repo.get_concept("uncleaned_concept")
        self.assertIsNone(old_concept)
        new_concept = self.graph_repo.get_concept("machine_learning")
        self.assertIsNotNone(new_concept)
        self.assertEqual(new_concept.name, "Machine Learning")
        self.assertEqual(new_concept.properties.get("description"), "Actual description.")

        # Edges should be migrated
        # Check author edge
        author_neighbors = self.graph_repo.get_neighbors("author_one")
        self.assertTrue(any(row[3] == "p1" for row in author_neighbors))

        # Check concept edge
        paper_neighbors = self.graph_repo.get_neighbors("p1")
        self.assertTrue(any(row[3] == "machine_learning" for row in paper_neighbors))

        # Chunk text should be corrected
        chunks = self.vector_repo.get_all_chunks()
        self.assertEqual(chunks[0].text_content, "Chunk content")

    @patch("src.services.metadata_enricher.MetadataEnricher.enrich")
    def test_doctor_missing_fields_check_and_fix(self, mock_enrich):
        # Mock Semantic Scholar metadata enrichment to return abstract
        mock_enrich.return_value = {
            "title": "Title",
            "abstract": "Enriched Abstract",
            "authors": ["Author One"],
            "year": 2026,
            "doi": "10.1000/xyz123"
        }

        # 1. Save a paper without abstract and summary
        paper = Paper(
            id="p_missing",
            title="Title",
            authors=["Author One"],
            abstract="", # Missing abstract
            properties={} # Missing summary (no "summary" key)
        )
        self.graph_repo.save_paper(paper)

        # 2. Add chunk for this paper
        chunk = Chunk(
            id="p_missing#0",
            paper_id="p_missing",
            text_content="Some chunk text content",
            page_number=1,
            embedding=[0.1] * 384
        )
        self.vector_repo.save_chunks([chunk])

        # Mock LLM engine for summary generation
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "Generated Summary via LLM"
        self.doctor_service.llm_engine = mock_llm

        # Run diagnostics in check-only mode
        report_check = self.doctor_service.run_diagnostics(fix=False)
        self.assertEqual(report_check["stats"]["papers_fixed"], 1)
        self.assertTrue(report_check["anomalies"]["papers"][0]["missing_abstract"])
        self.assertTrue(report_check["anomalies"]["papers"][0]["missing_summary"])

        # Check database: still unchanged
        p_check = self.graph_repo.get_paper("p_missing")
        self.assertEqual(p_check.abstract, "")
        self.assertNotIn("summary", p_check.properties)

        # Run diagnostics in FIX mode
        report_fix = self.doctor_service.run_diagnostics(fix=True)
        self.assertEqual(report_fix["stats"]["papers_fixed"], 1)

        # Verify updates in database
        p_fixed = self.graph_repo.get_paper("p_missing")
        self.assertEqual(p_fixed.abstract, "Enriched Abstract") # filled by MetadataEnricher mock
        self.assertEqual(p_fixed.properties.get("summary"), "Generated Summary via LLM") # filled by LLM summary mock

    @patch("src.services.metadata_enricher.MetadataEnricher.enrich")
    def test_doctor_missing_abstract_llm_fallback(self, mock_enrich):
        # Mock Semantic Scholar metadata enrichment to return nothing (e.g. offline or not found)
        mock_enrich.return_value = None

        # 1. Save a paper without abstract
        paper = Paper(
            id="p_missing_llm",
            title="Title",
            authors=["Author One"],
            abstract="", # Missing abstract
            properties={"summary": "Has summary already"}
        )
        self.graph_repo.save_paper(paper)

        # 2. Add chunk for this paper
        chunk = Chunk(
            id="p_missing_llm#0",
            paper_id="p_missing_llm",
            text_content="Some chunk text content representing the paper",
            page_number=1,
            embedding=[0.1] * 384
        )
        self.vector_repo.save_chunks([chunk])

        # Mock LLM engine for abstract generation
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "Generated Abstract via LLM Fallback"
        self.doctor_service.llm_engine = mock_llm

        # Run diagnostics in FIX mode
        report_fix = self.doctor_service.run_diagnostics(fix=True)
        self.assertEqual(report_fix["stats"]["papers_fixed"], 1)

        # Verify abstract is generated using LLM fallback
        p_fixed = self.graph_repo.get_paper("p_missing_llm")
        self.assertEqual(p_fixed.abstract, "Generated Abstract via LLM Fallback")

    @patch("src.cli.get_services")
    def test_cli_doctor_command(self, mock_get_services):
        mock_emb = MagicMock()
        mock_emb.get_embedding.return_value = [0.1] * 384
        mock_get_services.return_value = (self.graph_repo, self.vector_repo, mock_emb, MagicMock())

        # Setup uncleaned concept
        concept = Concept(
            id="uncleaned_concept",
            name="\"Machine Learning\"",
            properties={"description": "<think>thought</think>Actual description."}
        )
        self.graph_repo.save_concept(concept)

        # CLI run: check mode
        result = runner.invoke(app, ["doctor"])
        self.assertEqual(result.exit_code, 0)
        stdout = plain_output(result.stdout)
        self.assertIn("Starting Science Graph Database Doctor", stdout)
        self.assertIn("Found 1 anomalies.", stdout)

        # CLI run: fix mode
        result_fix = runner.invoke(app, ["doctor", "--fix"])
        self.assertEqual(result_fix.exit_code, 0)
        self.assertIn("Successfully corrected 1 anomalies across all", plain_output(result_fix.stdout))

    def test_doctor_concept_lemmatization_embedding_regeneration(self):
        # Create a concept
        concept = Concept(
            id="non_lemmatized_concept",
            name="Computers",
            properties={"description": "Some description"}
        )
        self.graph_repo.save_concept(concept)

        mock_emb = MagicMock()
        mock_emb.get_embedding.return_value = [0.9] * 384
        self.doctor_service.emb_engine = mock_emb

        # Mock the normalizer to return a different name to force a change
        with patch.object(self.doctor_service.normalizer, "normalize_concept_name", return_value="Computer"):
            report = self.doctor_service.run_diagnostics(fix=True)

        # Assertions
        self.assertEqual(report["stats"]["concepts_migrated"], 1)
        mock_emb.get_embedding.assert_called_with("Computer", is_query=False)
        
        # Verify the new concept in DB has the new name and regenerated embedding
        c_fixed = self.graph_repo.get_concept("computer")
        self.assertIsNotNone(c_fixed)
        self.assertEqual(c_fixed.name, "Computer")
        self.assertEqual(c_fixed.properties.get("embedding"), [0.9] * 384)

    def test_doctor_hallucination_retry_loop(self):
        # Create a paper with missing abstract
        paper = Paper(
            id="p_hallucinate",
            title="A Good Title",
            authors=["Some Author"],
            abstract="",  # Missing
            properties={"summary": "A Good Summary"}
        )
        self.graph_repo.save_paper(paper)
        
        # Save a chunk so there is full text to extract abstract from
        chunk = Chunk(
            id="p_hallucinate#0",
            paper_id="p_hallucinate",
            text_content="Some chunk text content of the paper",
            page_number=1,
            embedding=[0.1] * 384
        )
        self.vector_repo.save_chunks([chunk])
        
        # Configure LLM engine to hallucinate on first call, succeed on second call
        mock_llm = MagicMock()
        mock_llm.generate_response.side_effect = [
            "Abstract abstract abstract abstract abstract.", 
            "This is a clean valid abstract without repetitions."
        ]
        self.doctor_service.llm_engine = mock_llm
        
        # Mock metadata enricher to not return anything
        with patch("src.services.metadata_enricher.MetadataEnricher.enrich", return_value=None):
            report = self.doctor_service.run_diagnostics(fix=True)
            
        # Assertions
        self.assertEqual(report["stats"]["papers_fixed"], 1)
        self.assertEqual(mock_llm.generate_response.call_count, 2)
        
        # The first call should have been at temp=0.7, second at temp=0.3
        call_args_list = mock_llm.generate_response.call_args_list
        self.assertAlmostEqual(call_args_list[0].kwargs["temp"], 0.7)
        self.assertAlmostEqual(call_args_list[1].kwargs["temp"], 0.3)
        
        # Verify paper has the clean abstract saved
        p_fixed = self.graph_repo.get_paper("p_hallucinate")
        self.assertEqual(p_fixed.abstract, "This is a clean valid abstract without repetitions.")

    @patch("src.ner_engine.extract_persons_from_text")
    def test_doctor_ner_author_enrichment(self, mock_extract_persons):
        # Create a paper with one author and an abstract
        paper = Paper(
            id="p_ner",
            title="A Great Title",
            authors=["Jane Smith"],
            abstract="In this paper we discuss things by John Doe and Jane Smith.",
            properties={"summary": "A Good Summary"}
        )
        self.graph_repo.save_paper(paper)
        
        # Existing author node
        jane = Author(id="jane_smith", name="Jane Smith")
        self.graph_repo.save_author(jane)
        self.graph_repo.add_edge("jane_smith", "p_ner", "AUTHORED")
        
        # Mock extract_persons_from_text to return John Doe and Jane Smith
        mock_extract_persons.return_value = ["John Doe", "Jane Smith"]
        
        # Run diagnostics in FIX mode
        report = self.doctor_service.run_diagnostics(fix=True)
        
        # Assertions
        self.assertEqual(report["stats"]["papers_fixed"], 1)
        
        # Verify the paper's authors metadata has been enriched
        p_fixed = self.graph_repo.get_paper("p_ner")
        self.assertIn("John Doe", p_fixed.authors)
        self.assertIn("Jane Smith", p_fixed.authors)
        
        # Verify that John Doe author node and AUTHORED edge have been created
        john = self.graph_repo.get_node_by_id("john_doe")
        self.assertIsNotNone(john)
        
        edges = self.graph_repo.get_all_edges()
        authored_edge_exists = any(
            src == "john_doe" and tgt == "p_ner" and etype == "AUTHORED"
            for src, tgt, etype, _ in edges
        )
        self.assertTrue(authored_edge_exists)

    @patch("src.services.metadata_enricher.MetadataEnricher.enrich")
    def test_doctor_author_merge(self, mock_enrich):
        mock_enrich.return_value = None
        # Create destination author: Bob Smith -> ID 'bob_smith'
        bob = Author(id="bob_smith", name="Bob Smith")
        self.graph_repo.save_author(bob)

        # Create uncleaned source author: "Bob Smith" -> ID 'uncleaned_author'
        bob_uncleaned = Author(id="uncleaned_author", name="\"Bob Smith\"")
        self.graph_repo.save_author(bob_uncleaned)

        # Add a paper and link the uncleaned author to it
        paper = Paper(id="p_merge", title="Merge Paper", authors=["\"Bob Smith\""])
        self.graph_repo.save_paper(paper)
        self.graph_repo.add_edge("uncleaned_author", "p_merge", "AUTHORED")

        # Run diagnostics check only
        report = self.doctor_service.run_diagnostics(fix=False)
        self.assertEqual(report["stats"]["authors_merged"], 1)

        # Run diagnostics with fix=True
        report_fix = self.doctor_service.run_diagnostics(fix=True)
        self.assertEqual(report_fix["stats"]["authors_merged"], 1)

        # Verify old author is deleted and edges are migrated to the clean one
        self.assertIsNone(self.graph_repo.get_author("uncleaned_author"))
        self.assertIsNotNone(self.graph_repo.get_author("bob_smith"))
        
        edges = self.graph_repo.get_all_edges()
        self.assertTrue(any(
            src == "bob_smith" and tgt == "p_merge" and etype == "AUTHORED"
            for src, tgt, etype, _ in edges
        ))

    def test_doctor_enrich_exception_handling(self):
        # Create a paper with missing abstract
        paper = Paper(
            id="p_enrich_fail",
            title="Title",
            authors=["Author One"],
            abstract="",
            properties={}
        )
        self.graph_repo.save_paper(paper)

        # Mock MetadataEnricher to raise Exception
        with patch("src.services.metadata_enricher.MetadataEnricher.enrich", side_effect=Exception("Enrich API down")):
            # Run diagnostics, it should catch the exception and proceed
            report = self.doctor_service.run_diagnostics(fix=True)
            self.assertEqual(report["stats"]["papers_fixed"], 1)

    def test_doctor_get_all_chunks_exception_handling(self):
        # Mock vector_repo.get_all_chunks to raise Exception
        with patch.object(self.vector_repo, "get_all_chunks", side_effect=Exception("DB Corrupted")):
            # Run diagnostics, it should catch the exception and return empty chunks report
            report = self.doctor_service.run_diagnostics(fix=True)
            self.assertEqual(report["stats"]["chunks_checked"], 0)

    @patch("src.services.doctor_service.logger")
    def test_doctor_json_load_exception_handling(self, mock_logger):
        # Mock get_all_nodes to return a node with invalid JSON properties
        self.graph_repo.get_all_nodes = MagicMock(return_value=[
            ("p_bad_json", "Paper", "invalid { json }")
        ])

        # Run diagnostics, it should catch JSONDecodeError and log it
        report = self.doctor_service.run_diagnostics(fix=False)
        self.assertEqual(report["stats"]["papers_checked"], 1)
        mock_logger.error.assert_called()

    def test_doctor_concept_emb_generation_exception_handling(self):
        # Create a concept that needs renaming/lemmatization
        concept = Concept(
            id="concept_fail",
            name="Computers",
            properties={"description": "Desc"}
        )
        self.graph_repo.save_concept(concept)

        mock_emb = MagicMock()
        mock_emb.get_embedding.side_effect = Exception("Embeddings engine model failure")
        self.doctor_service.emb_engine = mock_emb

        # Mock the normalizer to rename it
        with patch.object(self.doctor_service.normalizer, "normalize_concept_name", return_value="Computer"):
            # Run diagnostics with fix=True, should catch embedding failure and complete successfully
            report = self.doctor_service.run_diagnostics(fix=True)
            self.assertEqual(report["stats"]["concepts_migrated"], 1)
            
        c = self.graph_repo.get_concept("computer")
        self.assertIsNotNone(c)
        self.assertNotIn("embedding", c.properties)

    def test_doctor_concept_merge_and_update_edge_cases(self):
        # 1. Concept Merge edge case
        c1 = Concept(
            id="machine_learning",
            name="Machine Learning",
            properties={"description": "Original description"}
        )
        self.graph_repo.save_concept(c1)
        
        # ID is different initially
        c2 = Concept(
            id="machine_learning_dot",
            name="Machine Learning.",
            properties={"description": "Description to merge"}
        )
        self.graph_repo.save_concept(c2)
        # Link source to a paper to test edge migration
        paper = Paper(id="p1", title="Paper Title", authors=[], abstract="Some abstract", properties={"summary": "Some summary"})
        self.graph_repo.save_paper(paper)
        self.graph_repo.add_edge("p1", "machine_learning_dot", "MENTIONS_CONCEPT")

        # 2. Concept Update (same ID) edge case
        c3 = Concept(
            id="deep_learning",
            name="Deep Learning",
            properties={"description": "Description with space "}
        )
        self.graph_repo.save_concept(c3)

        # Mock embedding engine to test embedding regeneration during update/merge
        mock_emb = MagicMock()
        mock_emb.get_embedding.return_value = [0.95] * 384
        self.doctor_service.emb_engine = mock_emb

        # Run diagnostics in FIX mode
        report = self.doctor_service.run_diagnostics(fix=True)

        # Verify merge
        self.assertEqual(report["stats"]["concepts_merged"], 1)
        self.assertIsNone(self.graph_repo.get_concept("machine_learning_dot"))
        c1_fixed = self.graph_repo.get_concept("machine_learning")
        self.assertIsNotNone(c1_fixed)
        self.assertEqual(c1_fixed.name, "Machine Learning")
        
        # Verify edge migrated
        neighbors = self.graph_repo.get_neighbors("p1")
        self.assertTrue(any(row[3] == "machine_learning" for row in neighbors))

        # Verify update
        self.assertEqual(report["stats"]["concepts_fixed"], 1)
        c3_fixed = self.graph_repo.get_concept("deep_learning")
        self.assertIsNotNone(c3_fixed)
        self.assertEqual(c3_fixed.properties.get("description"), "Description with space")
