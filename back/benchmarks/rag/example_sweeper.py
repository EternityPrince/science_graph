#!/usr/bin/env python3
"""
Science Graph — Example RAG Hyperparameter Sweeper.
An example script showcasing how to use BaseHyperparameterSweeper to run a sweep
over reranker blending weights and check its impact on retrieval recall/precision.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any

# Ensure parent directory is in path to resolve base_sweeper import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from base_sweeper import BaseHyperparameterSweeper


class RerankerWeightSweeper(BaseHyperparameterSweeper):
    """
    Sweeper that optimizes the reranker weight in the score blending component.
    Tests values from 0.0 (only RRF score) to 1.0 (only Reranker score).
    """

    def get_runs(self) -> List[Dict[str, Any]]:
        """
        Generate configurations for different reranker weights in score blending.
        """
        runs = []
        # We'll test 4 different blending weights
        weights_to_test = [0.0, 0.3, 0.7, 1.0]

        for w in weights_to_test:
            runs.append({
                "name": f"reranker_weight_{int(w * 100)}",
                # Ensure components required for blending are enabled
                "components": {
                    "lexical_search": True,
                    "dense_search": True,
                    "rrf": True,
                    "reranker": True,
                    "score_blending": True,
                },
                "hyperparameters": {
                    "rag": {
                        "score_blend_reranker_weight": w,
                        "score_blend_rrf_weight": round(1.0 - w, 2),
                    }
                }
            })

        return runs


if __name__ == "__main__":
    # Call the convenience class runner which parses command line arguments and runs the sweep
    RerankerWeightSweeper.main()
