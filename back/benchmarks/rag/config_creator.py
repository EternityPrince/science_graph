#!/usr/bin/env python3
"""
Science Graph — Configuration Creator and Preset Manager.
Provides shared utilities, presets, and monkey-patching logic for custom runs.
"""

import copy
import yaml
import argparse
from pathlib import Path
from typing import NamedTuple, Tuple, Dict, Any

from src.config import config
import core.config

# Store default config settings for reference and rollback
DEFAULT_COMPONENTS = copy.deepcopy(config.data.get("rag_components", {}))
DEFAULT_HYPERPARAMS = copy.deepcopy(config.data.get("hyperparameters", {}))

# Hardcoded CUSTOM preset configuration (components and hyperparameters).
CUSTOM_PRESET_COMPONENTS = {
    "intent_classifier": False,
    "graph_ontology_lookup": True,
    "llm_query_expansion": False,
    "hyde": False,
    "lexical_search": True,
    "dense_search": True,
    "dynamic_alpha_blending": False,
    "rrf": True,
    "graph_expansion": False,
    "reranker": True,
    "score_blending": False,
    "context_trimming": True,
    "citation_repair": True,
}

class RAGPreset(NamedTuple):
    score_blend_reranker_weight: float
    score_blend_rrf_weight: float
    rrf_k: float
    dynamic_alpha_threshold_low: float
    dynamic_alpha_val_low: float
    dynamic_alpha_threshold_mid: float
    dynamic_alpha_val_mid: float
    dynamic_alpha_val_high: float

class GraphPreset(NamedTuple):
    p_base: float
    gamma: float
    crawl_stop_threshold: float
    semantic_score_threshold: float
    semantic_score_top_p: float
    sigmoid_score_threshold: float
    sigmoid_score_top_p: float
    essential_fact_threshold: float
    sigmoid_slope: float
    sigmoid_center: float
    weight_authored: float
    weight_cites: float
    weight_mentions_concept: float
    weight_default: float

class BM25Preset(NamedTuple):
    k1: float
    b: float

class CustomPresetHyperparams(NamedTuple):
    rag: RAGPreset
    graph: GraphPreset
    bm25: BM25Preset

CUSTOM_PRESET_HYPERPARAMS_NT = CustomPresetHyperparams(
    rag=RAGPreset(
        score_blend_reranker_weight=0.75,
        score_blend_rrf_weight=0.25,
        rrf_k=60.0,
        dynamic_alpha_threshold_low=1.2,
        dynamic_alpha_val_low=1.0,
        dynamic_alpha_threshold_mid=3.0,
        dynamic_alpha_val_mid=0.5,
        dynamic_alpha_val_high=1.0,
    ),
    graph=GraphPreset(
        p_base=0.0,
        gamma=0.0,
        crawl_stop_threshold=1.0,
        semantic_score_threshold=0.35,
        semantic_score_top_p=0.9,
        sigmoid_score_threshold=0.4,
        sigmoid_score_top_p=0.9,
        essential_fact_threshold=0.5,
        sigmoid_slope=0.0,
        sigmoid_center=0.5,
        weight_authored=1.0,
        weight_cites=1.0,
        weight_mentions_concept=1.,
        weight_default=1.0,
    ),
    bm25=BM25Preset(
        k1=1.5,
        b=0.75,
    )
)

CUSTOM_PRESET_HYPERPARAMS = {
    "rag": CUSTOM_PRESET_HYPERPARAMS_NT.rag._asdict(),
    "graph": CUSTOM_PRESET_HYPERPARAMS_NT.graph._asdict(),
    "bm25": CUSTOM_PRESET_HYPERPARAMS_NT.bm25._asdict(),
}

def get_custom_preset_weights(preset_hype: CustomPresetHyperparams) -> dict:
    """Returns the custom preset weights configured for edge-type heuristics from the provided NamedTuple."""
    return {
        "weight_authored": preset_hype.graph.weight_authored,
        "weight_cites": preset_hype.graph.weight_cites,
        "weight_mentions_concept": preset_hype.graph.weight_mentions_concept,
        "weight_default": preset_hype.graph.weight_default,
    }

def add_custom_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds custom config command line arguments to the parser."""
    parser.add_argument(
        "--config-file", type=str, default=None,
        help="Path to a custom YAML configuration file containing overrides."
    )
    parser.add_argument(
        "--custom", action="store_true",
        help="Apply the hardcoded custom preset components and hyperparameters."
    )

    # Component overrides (Boolean Optional actions)
    parser.add_argument("--intent-classifier", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--graph-ontology-lookup", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--llm-query-expansion", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--hyde", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lexical-search", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dense-search", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dynamic-alpha-blending", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rrf", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--graph-expansion", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reranker", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--score-blending", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--context-trimming", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--citation-repair", action=argparse.BooleanOptionalAction, default=None)

    # RAG Hyperparameters overrides
    parser.add_argument("--score-blend-reranker-weight", type=float, default=None)
    parser.add_argument("--score-blend-rrf-weight", type=float, default=None)
    parser.add_argument("--rrf-k", type=float, default=None)
    parser.add_argument("--dynamic-alpha-threshold-low", type=float, default=None)
    parser.add_argument("--dynamic-alpha-val-low", type=float, default=None)
    parser.add_argument("--dynamic-alpha-threshold-mid", type=float, default=None)
    parser.add_argument("--dynamic-alpha-val-mid", type=float, default=None)
    parser.add_argument("--dynamic-alpha-val-high", type=float, default=None)

    # Graph Hyperparameters overrides
    parser.add_argument("--graph-p-base", type=float, default=None)
    parser.add_argument("--graph-gamma", type=float, default=None)
    parser.add_argument("--graph-crawl-stop-threshold", type=float, default=None)
    parser.add_argument("--graph-semantic-score-threshold", type=float, default=None)
    parser.add_argument("--graph-semantic-score-top-p", type=float, default=None)
    parser.add_argument("--graph-sigmoid-score-threshold", type=float, default=None)
    parser.add_argument("--graph-sigmoid-score-top-p", type=float, default=None)
    parser.add_argument("--graph-essential-fact-threshold", type=float, default=None)
    parser.add_argument("--graph-sigmoid-slope", type=float, default=None)
    parser.add_argument("--graph-sigmoid-center", type=float, default=None)
    parser.add_argument("--graph-weight-authored", type=float, default=None)
    parser.add_argument("--graph-weight-cites", type=float, default=None)
    parser.add_argument("--graph-weight-mentions-concept", type=float, default=None)
    parser.add_argument("--graph-weight-default", type=float, default=None)

    # BM25 Hyperparameters overrides
    parser.add_argument("--bm25-k1", type=float, default=None)
    parser.add_argument("--bm25-b", type=float, default=None)


def build_custom_config(args: Any, file_config: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Builds custom components and hyperparameters dictionaries by merging
    defaults, hardcoded preset (if args.custom), file config (if provided), and CLI overrides.
    """
    custom_comp = copy.deepcopy(DEFAULT_COMPONENTS)
    custom_hype = copy.deepcopy(DEFAULT_HYPERPARAMS)

    # Apply hardcoded custom preset if --custom is specified
    if getattr(args, "custom", False):
        custom_comp.update(CUSTOM_PRESET_COMPONENTS)
        for section, params in CUSTOM_PRESET_HYPERPARAMS.items():
            if section not in custom_hype:
                custom_hype[section] = {}
            custom_hype[section].update(params)

    # If file config is provided, merge it
    if file_config:
        if "rag_components" in file_config:
            custom_comp.update(file_config["rag_components"])
        if "hyperparameters" in file_config:
            # Deep merge hyperparameters
            for section, params in file_config["hyperparameters"].items():
                if section not in custom_hype:
                    custom_hype[section] = {}
                if params:
                    custom_hype[section].update(params)

    # Merge CLI arguments for components
    comp_fields = [
        "intent_classifier", "graph_ontology_lookup", "llm_query_expansion", 
        "hyde", "lexical_search", "dense_search", "dynamic_alpha_blending", 
        "rrf", "graph_expansion", "reranker", "score_blending", 
        "context_trimming", "citation_repair"
    ]
    for field in comp_fields:
        val = getattr(args, field, None)
        if val is not None:
            custom_comp[field] = val

    # Merge CLI arguments for hyperparameters
    # RAG hyperparameters
    rag_hype_fields = [
        "score_blend_reranker_weight", "score_blend_rrf_weight", "rrf_k",
        "dynamic_alpha_threshold_low", "dynamic_alpha_val_low",
        "dynamic_alpha_threshold_mid", "dynamic_alpha_val_mid",
        "dynamic_alpha_val_high"
    ]
    for field in rag_hype_fields:
        val = getattr(args, field, None)
        if val is not None:
            if "rag" not in custom_hype:
                custom_hype["rag"] = {}
            custom_hype["rag"][field] = val

    # Graph hyperparameters
    graph_hype_fields = [
        ("graph_p_base", "p_base"),
        ("graph_gamma", "gamma"),
        ("graph_crawl_stop_threshold", "crawl_stop_threshold"),
        ("graph_semantic_score_threshold", "semantic_score_threshold"),
        ("graph_semantic_score_top_p", "semantic_score_top_p"),
        ("graph_sigmoid_score_threshold", "sigmoid_score_threshold"),
        ("graph_sigmoid_score_top_p", "sigmoid_score_top_p"),
        ("graph_essential_fact_threshold", "essential_fact_threshold"),
        ("graph_sigmoid_slope", "sigmoid_slope"),
        ("graph_sigmoid_center", "sigmoid_center"),
        ("graph_weight_authored", "weight_authored"),
        ("graph_weight_cites", "weight_cites"),
        ("graph_weight_mentions_concept", "weight_mentions_concept"),
        ("graph_weight_default", "weight_default")
    ]
    for arg_name, conf_name in graph_hype_fields:
        val = getattr(args, arg_name, None)
        if val is not None:
            if "graph" not in custom_hype:
                custom_hype["graph"] = {}
            custom_hype["graph"][conf_name] = val

    # BM25 hyperparameters
    bm25_hype_fields = [
        ("bm25_k1", "k1"),
        ("bm25_b", "b")
    ]
    for arg_name, conf_name in bm25_hype_fields:
        val = getattr(args, arg_name, None)
        if val is not None:
            if "bm25" not in custom_hype:
                custom_hype["bm25"] = {}
            custom_hype["bm25"][conf_name] = val

    return custom_comp, custom_hype


def patch_config_for_custom(custom_comp: dict, custom_hype: dict):
    """Dynamically patches core config and retrieval functions to support CUSTOM baseline."""
    import core.config
    orig_get_baseline_config = core.config.get_baseline_config

    def custom_get_baseline_config(baseline: str, config_rag_components: dict) -> dict:
        if baseline == "CUSTOM":
            # Apply custom hyperparameters to the active config instance
            config.data["hyperparameters"] = copy.deepcopy(custom_hype)
            return copy.deepcopy(custom_comp)
        else:
            # Restore default/original hyperparameters for other baselines
            config.data["hyperparameters"] = copy.deepcopy(DEFAULT_HYPERPARAMS)
            return orig_get_baseline_config(baseline, config_rag_components)

    # Monkeypatch core.config and core.retrieval
    core.config.get_baseline_config = custom_get_baseline_config
    
    try:
        import core.retrieval
        core.retrieval.get_baseline_config = custom_get_baseline_config
    except ImportError:
        pass

    try:
        import core.generation
        core.generation.get_baseline_config = custom_get_baseline_config
    except ImportError:
        pass
