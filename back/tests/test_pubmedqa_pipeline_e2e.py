import unittest
import sys
import subprocess
import yaml
from pathlib import Path

# Setup paths
back_root = Path(__file__).resolve().parents[1]
if str(back_root) not in sys.path:
    sys.path.insert(0, str(back_root))
rag_dir = back_root / "benchmarks" / "rag"
if str(rag_dir) not in sys.path:
    sys.path.insert(0, str(rag_dir))


class TestPubMedQAPipelineE2E(unittest.TestCase):
    def test_pubmedqa_pipeline_skip_eval_smoke(self):
        """E2E smoke test running run_pipeline.py on PubMedQA dataset with --limit 3 and --skip-eval."""
        dataset_path = rag_dir / "dataset" / "pubmedqa" / "pubmedqa_golden.yaml"
        self.assertTrue(dataset_path.exists(), f"Dataset file missing: {dataset_path}")

        run_script = rag_dir / "run_pipeline.py"
        self.assertTrue(run_script.exists(), f"Pipeline script missing: {run_script}")

        # Run pipeline with limit=3, baselines B1,B2, and --skip-eval
        cmd = [
            sys.executable,
            str(run_script),
            "--dataset", str(dataset_path),
            "--baselines", "B1,B2",
            "--limit", "3",
            "--skip-eval",
            "--no-unique-dir",
            "--output-dir", str(rag_dir / "graphs" / "smoke_test_output")
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(back_root))
        self.assertEqual(
            result.returncode, 0,
            f"Pipeline run failed with returncode {result.returncode}.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

        eval_results = rag_dir / "graphs" / "smoke_test_output" / "evaluation_results.yaml"
        self.assertTrue(eval_results.exists(), f"Expected output results missing: {eval_results}")

        with open(eval_results, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.assertIn("results", data)
        self.assertGreater(len(data["results"]), 0)
        first_result = data["results"][0]
        self.assertIn("baselines", first_result)
        self.assertIn("B1", first_result["baselines"])
        b1_res = first_result["baselines"]["B1"]
        self.assertEqual(b1_res.get("status"), "success")
        self.assertIsNotNone(b1_res.get("generated_answer"))


if __name__ == "__main__":
    unittest.main()
