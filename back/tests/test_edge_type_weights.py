"""
Tests for configurable edge-type heuristic weights (w(τ)).

Covers:
1. Config default values
2. Config overridden values
3. _get_scored_graph_lines reading weights from config
4. run_custom_retrieve.py merging weight overrides via CLI and file config
"""
import unittest
import copy
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from src.config import Config, DEFAULT_CONFIG
from src.services.rag_service import RAGService

# Ensure benchmarks module is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "rag"))
from run_custom_retrieve import (
    build_custom_config,
    CUSTOM_PRESET_HYPERPARAMS,
    get_custom_preset_weights,
    CUSTOM_PRESET_HYPERPARAMS_NT,
)


# ─── 1. Config class: default and overridden values ──────────────────────────

class TestConfigEdgeWeightDefaults(unittest.TestCase):
    """Verify default edge-type weight properties on the Config class."""

    def setUp(self):
        with patch.object(Config, "__init__", lambda self: None):
            self.cfg = Config()
            self.cfg.data = copy.deepcopy(DEFAULT_CONFIG)

    def test_default_weight_authored(self):
        self.assertAlmostEqual(self.cfg.graph_weight_authored, 0.8)

    def test_default_weight_cites(self):
        self.assertAlmostEqual(self.cfg.graph_weight_cites, 0.7)

    def test_default_weight_mentions_concept(self):
        self.assertAlmostEqual(self.cfg.graph_weight_mentions_concept, 0.6)

    def test_default_weight_default(self):
        self.assertAlmostEqual(self.cfg.graph_weight_default, 0.5)


class TestConfigEdgeWeightOverrides(unittest.TestCase):
    """Verify edge-type weight properties change when config data is modified."""

    def setUp(self):
        with patch.object(Config, "__init__", lambda self: None):
            self.cfg = Config()
            self.cfg.data = copy.deepcopy(DEFAULT_CONFIG)

    def test_override_weight_authored(self):
        self.cfg.data["hyperparameters"]["graph"]["weight_authored"] = 0.95
        self.assertAlmostEqual(self.cfg.graph_weight_authored, 0.95)

    def test_override_weight_cites(self):
        self.cfg.data["hyperparameters"]["graph"]["weight_cites"] = 0.3
        self.assertAlmostEqual(self.cfg.graph_weight_cites, 0.3)

    def test_override_weight_mentions_concept(self):
        self.cfg.data["hyperparameters"]["graph"]["weight_mentions_concept"] = 0.1
        self.assertAlmostEqual(self.cfg.graph_weight_mentions_concept, 0.1)

    def test_override_weight_default(self):
        self.cfg.data["hyperparameters"]["graph"]["weight_default"] = 0.99
        self.assertAlmostEqual(self.cfg.graph_weight_default, 0.99)

    def test_missing_graph_section_falls_back(self):
        """If 'graph' section is entirely missing, defaults should be returned."""
        del self.cfg.data["hyperparameters"]["graph"]
        self.assertAlmostEqual(self.cfg.graph_weight_authored, 0.8)
        self.assertAlmostEqual(self.cfg.graph_weight_cites, 0.7)
        self.assertAlmostEqual(self.cfg.graph_weight_mentions_concept, 0.6)
        self.assertAlmostEqual(self.cfg.graph_weight_default, 0.5)


# ─── 2. _get_scored_graph_lines: weights read from config ────────────────────

class TestGetScoredGraphLinesWeights(unittest.TestCase):
    """Verify that _get_scored_graph_lines uses config weights, not hardcoded."""

    def setUp(self):
        self.graph_repo = MagicMock()
        self.vector_repo = MagicMock()
        self.emb_engine = MagicMock()
        self.llm_engine = MagicMock()
        self.service = RAGService(
            self.graph_repo,
            self.vector_repo,
            self.emb_engine,
            self.llm_engine,
            None,
        )
        # Stub name resolution
        self.graph_repo.get_paper.return_value = MagicMock(title="Paper X")
        self.graph_repo.get_author.return_value = MagicMock(name="Author Y")
        self.graph_repo.get_concept.return_value = MagicMock(name="Concept Z")

    def _setup_single_neighbor(self, edge_type: str):
        """Helper: configure one neighbor of the given edge type (no props)."""
        self.graph_repo.get_neighbors.return_value = [
            ("src", "Paper", edge_type, "tgt", "Paper", None),
        ]

    @patch("src.services.rag_service.config")
    def test_authored_uses_config_weight(self, mock_config):
        mock_config.graph_weight_authored = 0.91
        mock_config.rag_components = {"graph_expansion": True}
        self._setup_single_neighbor("AUTHORED")

        scored = self.service._get_scored_graph_lines(["p1"])
        self.assertEqual(len(scored), 1)
        _, score = scored[0]
        self.assertAlmostEqual(score, 0.91)

    @patch("src.services.rag_service.config")
    def test_cites_uses_config_weight(self, mock_config):
        mock_config.graph_weight_cites = 0.33
        mock_config.rag_components = {"graph_expansion": True}
        self._setup_single_neighbor("CITES")

        scored = self.service._get_scored_graph_lines(["p1"])
        _, score = scored[0]
        self.assertAlmostEqual(score, 0.33)

    @patch("src.services.rag_service.config")
    def test_mentions_concept_uses_config_weight(self, mock_config):
        mock_config.graph_weight_mentions_concept = 0.42
        mock_config.rag_components = {"graph_expansion": True}
        self._setup_single_neighbor("MENTIONS_CONCEPT")

        scored = self.service._get_scored_graph_lines(["p1"])
        _, score = scored[0]
        self.assertAlmostEqual(score, 0.42)

    @patch("src.services.rag_service.config")
    def test_unknown_edge_uses_config_default_weight(self, mock_config):
        mock_config.graph_weight_default = 0.11
        mock_config.rag_components = {"graph_expansion": True}
        self._setup_single_neighbor("SOME_UNKNOWN_REL")

        scored = self.service._get_scored_graph_lines(["p1"])
        _, score = scored[0]
        self.assertAlmostEqual(score, 0.11)

    @patch("src.services.rag_service.config")
    def test_explicit_score_in_props_overrides_config(self, mock_config):
        """Properties 'score' in edge data should override the config weight."""
        mock_config.graph_weight_authored = 0.91
        mock_config.rag_components = {"graph_expansion": True}
        self.graph_repo.get_neighbors.return_value = [
            ("src", "Paper", "AUTHORED", "tgt", "Paper", json.dumps({"score": 0.123})),
        ]

        scored = self.service._get_scored_graph_lines(["p1"])
        _, score = scored[0]
        self.assertAlmostEqual(score, 0.123)

    @patch("src.services.rag_service.config")
    def test_ordering_after_weight_change(self, mock_config):
        """Changing weights should change the sort order of scored lines."""
        mock_config.graph_weight_authored = 0.2  # lower than CITES
        mock_config.graph_weight_cites = 0.9     # higher than AUTHORED
        mock_config.graph_weight_default = 0.1
        mock_config.rag_components = {"graph_expansion": True}

        self.graph_repo.get_neighbors.return_value = [
            ("a1", "Author", "AUTHORED", "p1", "Paper", None),
            ("p1", "Paper", "CITES", "p2", "Paper", None),
        ]

        scored = self.service._get_scored_graph_lines(["p1"], limit=5)
        # After sort (desc), CITES (0.9) should come first, AUTHORED (0.2) second
        self.assertAlmostEqual(scored[0][1], 0.9)
        self.assertAlmostEqual(scored[1][1], 0.2)


# ─── 3. run_custom_retrieve build_custom_config: weight merging ──────────────

class TestBuildCustomConfigWeights(unittest.TestCase):
    """Verify that weight params are correctly merged in build_custom_config."""

    def _make_args(self, **kwargs):
        """Create a namespace with all expected fields defaulting to None."""
        defaults = {
            "custom": False,
            # Component fields
            "intent_classifier": None, "graph_ontology_lookup": None,
            "llm_query_expansion": None, "hyde": None,
            "lexical_search": None, "dense_search": None,
            "dynamic_alpha_blending": None, "rrf": None,
            "graph_expansion": None, "reranker": None,
            "score_blending": None, "context_trimming": None,
            "citation_repair": None,
            # RAG hyper fields
            "score_blend_reranker_weight": None, "score_blend_rrf_weight": None,
            "rrf_k": None,
            "dynamic_alpha_threshold_low": None, "dynamic_alpha_val_low": None,
            "dynamic_alpha_threshold_mid": None, "dynamic_alpha_val_mid": None,
            "dynamic_alpha_val_high": None,
            # Graph hyper fields
            "graph_p_base": None, "graph_gamma": None,
            "graph_crawl_stop_threshold": None,
            "graph_semantic_score_threshold": None,
            "graph_semantic_score_top_p": None,
            "graph_sigmoid_score_threshold": None,
            "graph_sigmoid_score_top_p": None,
            "graph_essential_fact_threshold": None,
            "graph_sigmoid_slope": None, "graph_sigmoid_center": None,
            "graph_weight_authored": None, "graph_weight_cites": None,
            "graph_weight_mentions_concept": None, "graph_weight_default": None,
            # BM25
            "bm25_k1": None, "bm25_b": None,
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_custom_preset_includes_weights(self):
        """--custom flag should apply the hardcoded preset weights."""
        args = self._make_args(custom=True)
        _, custom_hype = build_custom_config(args)

        expected = get_custom_preset_weights(CUSTOM_PRESET_HYPERPARAMS_NT)
        self.assertEqual(custom_hype["graph"]["weight_authored"], expected["weight_authored"])
        self.assertEqual(custom_hype["graph"]["weight_cites"], expected["weight_cites"])
        self.assertEqual(custom_hype["graph"]["weight_mentions_concept"], expected["weight_mentions_concept"])
        self.assertEqual(custom_hype["graph"]["weight_default"], expected["weight_default"])

    def test_cli_weight_overrides(self):
        """CLI arguments should override the preset values."""
        args = self._make_args(
            custom=True,
            graph_weight_authored=0.99,
            graph_weight_cites=0.11,
            graph_weight_mentions_concept=0.22,
            graph_weight_default=0.33,
        )
        _, custom_hype = build_custom_config(args)

        self.assertAlmostEqual(custom_hype["graph"]["weight_authored"], 0.99)
        self.assertAlmostEqual(custom_hype["graph"]["weight_cites"], 0.11)
        self.assertAlmostEqual(custom_hype["graph"]["weight_mentions_concept"], 0.22)
        self.assertAlmostEqual(custom_hype["graph"]["weight_default"], 0.33)

    def test_file_config_weight_overrides(self):
        """YAML file config should be merged into weights."""
        args = self._make_args(custom=False)
        file_config = {
            "hyperparameters": {
                "graph": {
                    "weight_authored": 0.55,
                    "weight_cites": 0.44,
                }
            }
        }
        _, custom_hype = build_custom_config(args, file_config=file_config)

        self.assertAlmostEqual(custom_hype["graph"]["weight_authored"], 0.55)
        self.assertAlmostEqual(custom_hype["graph"]["weight_cites"], 0.44)

    def test_cli_overrides_file_config(self):
        """CLI args should take precedence over file config."""
        args = self._make_args(
            custom=False,
            graph_weight_authored=0.77,
        )
        file_config = {
            "hyperparameters": {
                "graph": {
                    "weight_authored": 0.22,
                }
            }
        }
        _, custom_hype = build_custom_config(args, file_config=file_config)
        self.assertAlmostEqual(custom_hype["graph"]["weight_authored"], 0.77)


# ─── 4. Preset constant validation ──────────────────────────────────────────

class TestCustomPresetContainsWeights(unittest.TestCase):
    """Ensure CUSTOM_PRESET_HYPERPARAMS has the weight keys."""

    def test_preset_has_weight_authored(self):
        self.assertIn("weight_authored", CUSTOM_PRESET_HYPERPARAMS["graph"])

    def test_preset_has_weight_cites(self):
        self.assertIn("weight_cites", CUSTOM_PRESET_HYPERPARAMS["graph"])

    def test_preset_has_weight_mentions_concept(self):
        self.assertIn("weight_mentions_concept", CUSTOM_PRESET_HYPERPARAMS["graph"])

    def test_preset_has_weight_default(self):
        self.assertIn("weight_default", CUSTOM_PRESET_HYPERPARAMS["graph"])


# ─── 5. DEFAULT_CONFIG contains weight keys ──────────────────────────────────

class TestDefaultConfigContainsWeights(unittest.TestCase):
    """Ensure DEFAULT_CONFIG structure includes the weight keys."""

    def test_default_config_has_weight_authored(self):
        val = DEFAULT_CONFIG["hyperparameters"]["graph"]["weight_authored"]
        self.assertAlmostEqual(val, 0.8)

    def test_default_config_has_weight_cites(self):
        val = DEFAULT_CONFIG["hyperparameters"]["graph"]["weight_cites"]
        self.assertAlmostEqual(val, 0.7)

    def test_default_config_has_weight_mentions_concept(self):
        val = DEFAULT_CONFIG["hyperparameters"]["graph"]["weight_mentions_concept"]
        self.assertAlmostEqual(val, 0.6)

    def test_default_config_has_weight_default(self):
        val = DEFAULT_CONFIG["hyperparameters"]["graph"]["weight_default"]
        self.assertAlmostEqual(val, 0.5)


if __name__ == "__main__":
    unittest.main()
