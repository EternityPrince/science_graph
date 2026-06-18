import unittest
from unittest.mock import patch
import json
from typing import Optional, Type
from pydantic import BaseModel

from src.llm_schemas import (
    validate_extraction_response,
    validate_clustering_response
)
from src.llm_engine import BaseLLMEngine


class TestLLMValidation(unittest.TestCase):
    def test_extraction_happy_path(self):
        """Test extraction validation with clean, correct inputs."""
        raw_data = {
            "authors": ["Alice Smith", "Bob Jones"],
            "concepts": [
                {"name": "Self-Attention", "description": "Relates different positions of a single sequence."},
                {"name": "Transformer Model", "description": "Encoder-decoder model based on attention."}
            ],
            "tags": ["deep learning", "nlp"]
        }
        model, warnings = validate_extraction_response(raw_data)
        
        self.assertEqual(warnings, [])
        self.assertEqual(model.authors, ["Alice Smith", "Bob Jones"])
        self.assertEqual(len(model.concepts), 2)
        self.assertEqual(model.concepts[0].name, "Self-Attention")
        self.assertEqual(model.tags, ["deep learning", "nlp"])

    def test_extraction_author_filtering(self):
        """Test that institutional names, citations, emails, and malformed names are filtered from authors."""
        raw_data = {
            "authors": [
                "Jane Doe",
                "University of Toronto",  # institution
                "Department of CS",       # dept
                "John Smith et al.",      # citation/et al
                "bob@example.com",        # email
                "https://example.com",    # URL
                "Vol. 12",                # volume
                "12345",                  # digits only
                "A",                      # too short
                "  Alice Cooper, "        # trailing punctuation/whitespace
            ],
            "concepts": [],
            "tags": []
        }
        model, warnings = validate_extraction_response(raw_data)
        
        self.assertEqual(model.authors, ["Jane Doe", "Alice Cooper"])
        self.assertTrue(any("University of Toronto" in w for w in warnings))
        self.assertTrue(any("Department of CS" in w for w in warnings))
        self.assertTrue(any("John Smith et al." in w for w in warnings))
        self.assertTrue(any("bob@example.com" in w for w in warnings))

    def test_extraction_concept_filtering(self):
        """Test filtering of concepts with names that are too long, contain citations, or are empty."""
        raw_data = {
            "authors": [],
            "concepts": [
                {"name": "Self-Attention", "description": "Good description"},
                {"name": "Transformer Architecture [1]", "description": "Has citation"},
                {"name": "This is a very long concept name with many words", "description": "Too long name"},
                {"name": "", "description": "Empty name"},
                {"name": "Short Name", "description": "No"}  # too short description warning
            ],
            "tags": []
        }
        model, warnings = validate_extraction_response(raw_data)
        
        # Self-Attention is kept.
        # "Transformer Architecture [1]" is kept but cleaned to "Transformer Architecture".
        # "This is a very long concept name..." is filtered out.
        # "" is filtered out.
        # "Short Name" is kept but generates a description length warning.
        concept_names = [c.name for c in model.concepts]
        self.assertIn("Self-Attention", concept_names)
        self.assertIn("Transformer Architecture", concept_names)
        self.assertNotIn("This is a very long concept name with many words", concept_names)
        
        self.assertTrue(any("too long" in w for w in warnings))
        self.assertTrue(any("unusual length" in w for w in warnings))

    def test_extraction_tag_filtering(self):
        """Test tag validation including length limits and invalid characters."""
        raw_data = {
            "authors": [],
            "concepts": [],
            "tags": [
                "deep learning",
                "natural language processing models",  # 4 words - allowed
                "this tag is a whole sentence and is too long",  # 10 words - filtered
                "bad@tag",  # invalid characters - filtered
                "tag[1]"    # citation brackets - filtered
            ]
        }
        model, warnings = validate_extraction_response(raw_data)
        
        self.assertEqual(model.tags, ["deep learning", "natural language processing models"])
        self.assertTrue(any("too long" in w for w in warnings))
        self.assertTrue(any("invalid characters" in w for w in warnings))

    def test_extraction_type_mismatches(self):
        """Test that type mismatches (e.g. list fields not lists) are handled gracefully."""
        raw_data = {
            "authors": "Not A List",
            "concepts": "Not A List",
            "tags": "Not A List"
        }
        model, warnings = validate_extraction_response(raw_data)
        self.assertEqual(model.authors, [])
        self.assertEqual(model.concepts, [])
        self.assertEqual(model.tags, [])
        self.assertTrue(any("Expected 'authors' to be a list" in w for w in warnings))

    def test_clustering_happy_path(self):
        """Test clustering validation with clean correct inputs."""
        raw_data = {
            "Introduction": ["chunk_1", "chunk_2"],
            "Methodology": ["chunk_3"]
        }
        model, warnings = validate_clustering_response(raw_data)
        
        self.assertEqual(warnings, [])
        self.assertEqual(model.root, raw_data)

    def test_clustering_filtering(self):
        """Test filtering of empty sections, non-list chunk IDs, empty lists, etc."""
        raw_data = {
            "Introduction": ["chunk_1", ""],
            "EmptySection": [],
            "  ": ["chunk_2"],
            "NonListSection": "chunk_3",
            123: ["chunk_4"]
        }
        model, warnings = validate_clustering_response(raw_data)
        
        self.assertIn("Introduction", model.root)
        self.assertEqual(model.root["Introduction"], ["chunk_1"])
        self.assertNotIn("EmptySection", model.root)
        self.assertNotIn("  ", model.root)
        self.assertNotIn("NonListSection", model.root)
        self.assertTrue(any("is not a list" in w for w in warnings))
        self.assertTrue(any("has no valid chunk IDs" in w for w in warnings))


class DummyEngine(BaseLLMEngine):
    def __init__(self):
        self.response = ""

    def generate_response(self, prompt: str, max_tokens: int = None, temp: float = None, task: str = None) -> str:
        return self.response

    def generate_json(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        temp: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        return self.response


class TestLLMEngineIntegration(unittest.TestCase):
    def setUp(self):
        self.engine = DummyEngine()

    @patch("src.llm_engine.base.con")
    def test_engine_extract_success(self, mock_con):
        """Test extract_concepts_and_metadata with valid LLM output."""
        self.engine.response = json.dumps({
            "authors": ["John Doe"],
            "concepts": [{"name": "Attention", "description": "Focus on specific parts."}],
            "tags": ["deep learning"]
        })
        
        res = self.engine.extract_concepts_and_metadata("sample text")
        
        self.assertIsNotNone(res)
        self.assertEqual(res["authors"], ["John Doe"])
        self.assertEqual(res["tags"], ["deep learning"])
        mock_con.success.assert_called_with("LLM extraction output validated successfully.")
        mock_con.info.assert_called()

    @patch("src.llm_engine.base.con")
    def test_engine_extract_invalid_json(self, mock_con):
        """Test extract_concepts_and_metadata handles invalid JSON response gracefully."""
        self.engine.response = "invalid json {..."
        res = self.engine.extract_concepts_and_metadata("sample text")
        
        self.assertIsNone(res)
        mock_con.warning.assert_called()

    @patch("src.llm_engine.base.con")
    def test_engine_extract_validation_warnings(self, mock_con):
        """Test extraction triggers warning logs on low-quality output, but returns cleaned model."""
        self.engine.response = json.dumps({
            "authors": ["John Doe", "University of Nowhere"],
            "concepts": [
                {"name": "Attention", "description": "Focus on specific parts."},
                {"name": "Very long concept name that is actually a sentence", "description": "Too long"}
            ],
            "tags": ["deep learning"]
        })
        
        res = self.engine.extract_concepts_and_metadata("sample text")
        
        self.assertIsNotNone(res)
        self.assertEqual(res["authors"], ["John Doe"])
        self.assertEqual(len(res["concepts"]), 1)  # long concept is filtered out
        
        # Verify warnings were logged to console
        mock_con.warning.assert_any_call("LLM extraction output validated with warnings:")
        any_warning = any("noisy" in call.args[0] or "too long" in call.args[0] for call in mock_con.warning.call_args_list)
        self.assertTrue(any_warning)

    @patch("src.llm_engine.base.con")
    def test_engine_cluster_success(self, mock_con):
        """Test cluster_chunks_by_topic happy path validation."""
        self.engine.response = json.dumps({
            "Introduction": ["chunk_1", "chunk_2"]
        })
        
        res = self.engine.cluster_chunks_by_topic("summary", "topic")
        
        self.assertIsNotNone(res)
        self.assertEqual(res, {"Introduction": ["chunk_1", "chunk_2"]})
        mock_con.success.assert_called_with("LLM clustering output validated successfully.")

    def test_clean_json_response_with_trailing_conversational_text(self):
        """Test that _clean_json_response correctly ignores conversational text and extra braces."""
        raw_output = (
            "{\n"
            "  \"authors\": [],\n"
            "  \"concepts\": [],\n"
            "  \"tags\": []\n"
            "}\n"
            "Here is the analysis of the domain: Example Domain is a domain {reserved} for use in illustrative examples."
        )
        cleaned = self.engine._clean_json_response(raw_output)
        self.assertEqual(cleaned, "{\n  \"authors\": [],\n  \"concepts\": [],\n  \"tags\": []\n}")

    def test_clean_json_response_with_markdown_blocks(self):
        """Test that markdown code blocks are cleaned and correctly traced."""
        raw_output = (
            "```json\n"
            "{\n"
            "  \"authors\": [],\n"
            "  \"concepts\": [],\n"
            "  \"tags\": []\n"
            "}\n"
            "```\n"
            "This is some trailing text."
        )
        cleaned = self.engine._clean_json_response(raw_output)
        self.assertEqual(cleaned, "{\n  \"authors\": [],\n  \"concepts\": [],\n  \"tags\": []\n}")

    def test_clean_json_response_array(self):
        """Test that json arrays are also correctly extracted and traced."""
        raw_output = "[\"item1\", \"item2\"] with extra text [ignored]"
        cleaned = self.engine._clean_json_response(raw_output)
        self.assertEqual(cleaned, "[\"item1\", \"item2\"]")

    def test_strip_thinking_tokens(self):
        """Test strip_thinking_tokens removes both closed and unclosed thinking blocks."""
        from src.llm_engine import strip_thinking_tokens

        # 1. Normal text
        self.assertEqual(strip_thinking_tokens("hello world"), "hello world")
        self.assertEqual(strip_thinking_tokens(""), "")
        self.assertEqual(strip_thinking_tokens(None), None)

        # 2. Closed block
        self.assertEqual(strip_thinking_tokens("<think>thinking process</think>hello world"), "hello world")

        # 3. Multiline closed block
        multiline = (
            "<think>\n"
            "thinking\n"
            "process\n"
            "</think>\n"
            "hello world"
        )
        self.assertEqual(strip_thinking_tokens(multiline), "hello world")

        # 4. Unclosed trailing block
        unclosed = (
            "hello world\n"
            "<think>\n"
            "incomplete thinking"
        )
        self.assertEqual(strip_thinking_tokens(unclosed), "hello world")

        # 5. Only thinking
        only_thinking = "<think>only this</think>"
        self.assertEqual(strip_thinking_tokens(only_thinking), "")

        # 6. Unclosed only thinking
        unclosed_only = "<think>only this"
        self.assertEqual(strip_thinking_tokens(unclosed_only), "")

        # 7. Multiple thinking blocks
        multiple = "<think>first</think> hello <think>second</think> world"
        self.assertEqual(strip_thinking_tokens(multiple), "hello  world")

        # 8. Technical tokens
        self.assertEqual(strip_thinking_tokens("<|im_start|>hello <|im_end|>"), "hello")
        self.assertEqual(strip_thinking_tokens("<|im_end|> <|im_start|> <|im_end|> <|im_start|> hello"), "hello")
        self.assertEqual(strip_thinking_tokens("[INST] hello [/INST]"), "hello")
        self.assertEqual(strip_thinking_tokens("<s>hello</s>"), "hello")
        self.assertEqual(strip_thinking_tokens("hello <|eot_id|>"), "hello")

        # 9. Leaked role prefixes
        self.assertEqual(strip_thinking_tokens("assistant\nhello"), "hello")
        self.assertEqual(strip_thinking_tokens("assistant: hello"), "hello")
        self.assertEqual(strip_thinking_tokens("system\nhello"), "hello")
        self.assertEqual(strip_thinking_tokens("user: hello"), "hello")

        # 10. Legitimate words should be preserved
        self.assertEqual(strip_thinking_tokens("The assistant helped the user."), "The assistant helped the user.")

    def test_pydantic_field_validators_deduplication(self):
        """Verify that Pydantic field validators clean and deduplicate lists of strings and objects."""
        from src.llm_schemas import LLMExtractionResponse, LLMVideoSummaryResponse

        # 1. Deduplicate string lists (authors, tags, institutions, sponsored_by) case-insensitively and strip whitespace
        raw = {
            "authors": ["Alice Smith", " alice smith ", "Bob Jones", "bob jones"],
            "tags": ["deep learning", "Deep Learning", "nlp"],
            "institutions": ["MIT", "mit", "Google"],
            "sponsored_by": ["Google", "google", "IBM"],
            "code_repositories": ["https://github.com/test", "https://github.com/test", "invalid_url_skipped"]
        }
        model = LLMExtractionResponse.model_validate(raw)
        self.assertEqual(model.authors, ["Alice Smith", "Bob Jones"])
        self.assertEqual(model.tags, ["deep learning", "nlp"])
        self.assertEqual(model.institutions, ["MIT", "Google"])
        self.assertEqual(model.sponsored_by, ["Google", "IBM"])
        self.assertEqual(model.code_repositories, ["https://github.com/test"])

        # 2. Unique concepts validation
        raw_concepts = {
            "concepts": [
                {"name": "Self-Attention", "description": "Desc 1"},
                {"name": "self-attention", "description": "Desc 2"},
                {"name": "Transformer", "description": "Desc 3"}
            ]
        }
        model_c = LLMExtractionResponse.model_validate(raw_concepts)
        self.assertEqual(len(model_c.concepts), 2)
        self.assertEqual(model_c.concepts[0].name, "Self-Attention")
        self.assertEqual(model_c.concepts[1].name, "Transformer")

        # 3. Unique citation intents validation
        raw_citations = {
            "citation_intents": [
                {"target_title": "Paper A", "intent": "USES_METHOD"},
                {"target_title": "paper a", "intent": "uses_method"},
                {"target_title": "Paper A", "intent": "BACKGROUND"}
            ]
        }
        model_cit = LLMExtractionResponse.model_validate(raw_citations)
        self.assertEqual(len(model_cit.citation_intents), 2)
        self.assertEqual(model_cit.citation_intents[0].target_title, "Paper A")
        self.assertEqual(model_cit.citation_intents[0].intent, "USES_METHOD")
        self.assertEqual(model_cit.citation_intents[1].target_title, "Paper A")
        self.assertEqual(model_cit.citation_intents[1].intent, "BACKGROUND")

        # 4. Unique concept relations validation
        raw_relations = {
            "concept_relations": [
                {"source": "Concept A", "target": "Concept B", "relation_type": "IS_A"},
                {"source": "concept a", "target": "concept b", "relation_type": "is_a"},
                {"source": "Concept A", "target": "Concept C", "relation_type": "IS_A"}
            ]
        }
        model_rel = LLMExtractionResponse.model_validate(raw_relations)
        self.assertEqual(len(model_rel.concept_relations), 2)
        self.assertEqual(model_rel.concept_relations[0].source, "Concept A")
        self.assertEqual(model_rel.concept_relations[0].target, "Concept B")
        self.assertEqual(model_rel.concept_relations[0].relation_type, "IS_A")

        # 5. Video summary deduplication
        raw_video = {
            "overview": "Concise summary",
            "themes": ["Theme A", "theme a", "Theme B"],
            "outline": ["Outline A", "outline a", "Outline B"]
        }
        model_vid = LLMVideoSummaryResponse.model_validate(raw_video)
        self.assertEqual(model_vid.themes, ["Theme A", "Theme B"])
        self.assertEqual(model_vid.outline, ["Outline A", "Outline B"])

    def test_pydantic_field_validators_blank_values(self):
        """Verify that Pydantic field validators do not filter out blank values, leaving them for the normalization pipeline."""
        from src.llm_schemas import LLMExtractionResponse

        # 1. Blank concept name
        raw_concepts = {
            "concepts": [
                {"name": "  ", "description": "Desc 1"},
                {"name": "", "description": "Desc 2"}
            ]
        }
        model_c = LLMExtractionResponse.model_validate(raw_concepts)
        # The first blank concept is preserved, the second is deduplicated (both strip to "")
        self.assertEqual(len(model_c.concepts), 1)
        self.assertEqual(model_c.concepts[0].name, "  ")

        # 2. Blank citation intent target title
        raw_citations = {
            "citation_intents": [
                {"target_title": "  ", "intent": "BACKGROUND"},
                {"target_title": "", "intent": "BACKGROUND"}
            ]
        }
        model_cit = LLMExtractionResponse.model_validate(raw_citations)
        self.assertEqual(len(model_cit.citation_intents), 1)
        self.assertEqual(model_cit.citation_intents[0].target_title, "  ")

        # 3. Blank concept relation source/target
        raw_relations = {
            "concept_relations": [
                {"source": "  ", "target": "Concept B", "relation_type": "IS_A"},
                {"source": "Concept A", "target": "", "relation_type": "IS_A"}
            ]
        }
        model_rel = LLMExtractionResponse.model_validate(raw_relations)
        self.assertEqual(len(model_rel.concept_relations), 2)

    def test_extraction_invalid_types_and_fallbacks(self):
        """Test extraction validation with malformed field types and invalid characters."""
        raw_data = {
            # Resetting lists
            "authors": "Not A List",
            "concepts": None,
            "tags": 12345,
            "institutions": {},
            "author_institutions": "String",
            "sponsored_by": False,
            "datasets": True,
            "code_repositories": "No list",
            "citation_intents": "None",
            "concept_relations": 3.14
        }
        model, warnings = validate_extraction_response(raw_data)
        self.assertEqual(model.authors, [])
        self.assertEqual(model.concepts, [])
        self.assertEqual(model.tags, [])
        self.assertEqual(model.institutions, [])
        self.assertEqual(model.author_institutions, [])
        self.assertEqual(model.sponsored_by, [])
        self.assertEqual(model.datasets, [])
        self.assertEqual(model.code_repositories, [])
        self.assertEqual(model.citation_intents, [])
        self.assertEqual(model.concept_relations, [])
        self.assertTrue(any("Expected 'authors' to be a list" in w for w in warnings))

    def test_extraction_element_filtering_and_cleaning(self):
        """Test extraction validation with mixed valid/invalid element types and values."""
        raw_data = {
            "authors": ["Alice Smith", 123, "   ", "A", "University of Cambridge"],
            "concepts": [
                "Not a dict concept",
                {"name": 123, "description": "Desc"},
                {"name": "Concept A", "description": 123},
                {"name": "   ", "description": "Valid description"},
                {"name": "Too Long Concept Name Here Because It Has More Than Five Words", "description": "Valid desc"},
                {"name": "Concept B", "description": "Short desc", "aliases": ["alias1", 123, "  "]}
            ],
            "tags": ["valid tag", 123, "   ", "tag too long with too many words here", "bad_tag_@_char", "valid tag"],
            "institutions": ["Valid Inst", 123, "  "],
            "author_institutions": ["Not a dict", {"author": 123, "institution": "Inst"}, {"author": "Author A", "institution": "Inst"}],
            "sponsored_by": ["Valid Sponsor", 123, "  "],
            "datasets": [
                "Dataset String A",
                {"name": "Dataset Dict B", "relation": "INTRODUCED_DATASET"},
                {"name": "Dataset Dict C", "relation": "INVALID_RELATION"},
                {"name": 123, "relation": "USED_DATASET"},
                "  "
            ],
            "code_repositories": ["https://github.com/repo", 123, "not-a-url"],
            "journal_or_conference": "Conference Name",
            "citation_intents": [
                "Not a dict",
                {"target_title": "Paper A", "intent": "BACKGROUND"},
                {"target_title": 123, "intent": "BACKGROUND"},
                {"target_title": "  ", "intent": "BACKGROUND"}
            ],
            "concept_relations": [
                "Not a dict",
                {"source": "Concept A", "target": "Concept B", "relation_type": "IS_A"},
                {"source": "Concept A", "target": "Concept B", "relation_type": "INVALID_REL"},
                {"source": 123, "target": "Concept B", "relation_type": "IS_A"}
            ]
        }
        
        model, warnings = validate_extraction_response(raw_data)
        
        # Verify authors
        self.assertEqual(model.authors, ["Alice Smith"])
        # Verify concepts
        self.assertEqual(len(model.concepts), 1)
        self.assertEqual(model.concepts[0].name, "Concept B")
        self.assertEqual(model.concepts[0].aliases, ["alias1"])
        # Verify tags
        self.assertEqual(model.tags, ["valid tag"])
        # Verify institutions
        self.assertEqual(model.institutions, ["Valid Inst"])
        # Verify author_institutions
        self.assertEqual(model.author_institutions, [{"author": "Author A", "institution": "Inst"}])
        # Verify sponsored_by
        self.assertEqual(model.sponsored_by, ["Valid Sponsor"])
        # Verify datasets
        self.assertEqual(len(model.datasets), 3)
        self.assertEqual(model.datasets[0].name, "Dataset String A")
        self.assertEqual(model.datasets[0].relation, "USED_DATASET")
        self.assertEqual(model.datasets[1].name, "Dataset Dict B")
        self.assertEqual(model.datasets[1].relation, "INTRODUCED_DATASET")
        self.assertEqual(model.datasets[2].name, "Dataset Dict C")
        self.assertEqual(model.datasets[2].relation, "USED_DATASET") # Default fallback
        # Verify code_repositories
        self.assertEqual(model.code_repositories, ["https://github.com/repo"])
        # Verify journal_or_conference
        self.assertEqual(model.journal_or_conference, "Conference Name")
        # Verify citation_intents
        self.assertEqual(len(model.citation_intents), 1)
        self.assertEqual(model.citation_intents[0].target_title, "Paper A")
        # Verify concept_relations
        self.assertEqual(len(model.concept_relations), 1)
        self.assertEqual(model.concept_relations[0].source, "Concept A")
        self.assertEqual(model.concept_relations[0].relation_type, "IS_A")

    def test_clustering_response_invalid_inputs(self):
        """Test clustering validation with malformed raw clustering data."""
        # 1. Non-dict input
        model1, warnings1 = validate_clustering_response("Not a dict")
        self.assertEqual(model1.root, {})
        self.assertTrue(any("Expected clustering response to be a JSON dictionary" in w for w in warnings1))

        # 2. Dict input with non-list values or non-string list elements
        raw_data = {
            "Cluster A": ["Paper 1", "Paper 2"],
            "Cluster B": "Not a list",
            "Cluster C": [123, "Paper 3"],
            "Cluster D": []
        }
        model2, warnings2 = validate_clustering_response(raw_data)
        self.assertEqual(model2.root["Cluster A"], ["Paper 1", "Paper 2"])
        self.assertNotIn("Cluster B", model2.root)
        self.assertEqual(model2.root["Cluster C"], ["Paper 3"])
        self.assertNotIn("Cluster D", model2.root)
        self.assertTrue(any("Cluster B" in w and "is not a list" in w for w in warnings2))
        self.assertTrue(any("Cluster C" in w and "non-string chunk ID" in w for w in warnings2))



