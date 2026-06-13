#!/usr/bin/env python3
"""
Science Graph — RAG Quality Evaluator (LLM-as-a-Judge + Retrieval Metrics).
Iterates over RAG benchmark results, runs cloud LLM evaluation with robust
rate limiting & retries, computes traditional metrics, and outputs a consolidated report.
"""

import os
import sys
import time
import argparse
import asyncio
import json
import re
import random
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

# Set up python path to resolve src imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import config
from src import console as con

# Imports for Rich formatting
from rich.table import Table
from rich.panel import Panel

# Rate Limiter helper
class AsyncRateLimiter:
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.last_call_time = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        if self.interval <= 0:
            return
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_call_time
            if elapsed < self.interval:
                await asyncio.sleep(self.interval - elapsed)
            self.last_call_time = time.monotonic()


class CloudEvaluator:
    def __init__(self, api_key: str, base_url: str, model_name: str, concurrency: int, rpm: int):
        import openai
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(concurrency)
        self.rate_limiter = AsyncRateLimiter(rpm)

    async def call_llm(self, system_prompt: str, user_prompt: str, max_retries: int = 5) -> str:
        async with self.semaphore:
            for attempt in range(max_retries):
                # Enforce rate limits
                await self.rate_limiter.wait()

                try:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.0,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(0.1, 1.0)
                    con.warning(
                        f"Cloud API Call Error (Attempt {attempt+1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(wait_time)
            raise RuntimeError("Cloud LLM request failed after maximum retries.")

    async def evaluate_metric(self, evaluator_config: Dict[str, Any], metric_name: str, **kwargs) -> Dict[str, Any]:
        system_prompt = evaluator_config["system_prompt"]
        user_prompt = evaluator_config["user_prompt_template"].format(**kwargs)

        max_parse_retries = 3
        for parse_attempt in range(max_parse_retries):
            raw_response = await self.call_llm(system_prompt, user_prompt)
            try:
                parsed = self.clean_and_parse_json(raw_response)
                if "score" not in parsed:
                    raise ValueError("JSON response is missing the 'score' key.")
                return parsed
            except Exception as e:
                con.warning(
                    f"JSON Parse Error for '{metric_name}' (Attempt {parse_attempt+1}/{max_parse_retries}): {e}. "
                    f"Raw response: {raw_response[:200]}..."
                )
                if parse_attempt == max_parse_retries - 1:
                    # Return error state
                    return {"score": 0.0, "error": str(e), "raw_response": raw_response}
                # Brief sleep before retrying LLM query
                await asyncio.sleep(1.0)
        return {"score": 0.0, "error": "Evaluation parsing failed."}

    def clean_and_parse_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        # Find markdown code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)
        else:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                text = text[start:end+1]
        
        return json.loads(text)


# Traditional Metrics
def calculate_retrieval_recall(expected_papers: List[str], retrieved_papers: List[str]) -> float:
    if not expected_papers:
        return 1.0
    expected_set = {p.strip().lower() for p in expected_papers if p.strip()}
    retrieved_set = {p.strip().lower() for p in retrieved_papers if p.strip()}
    if not expected_set:
        return 1.0
    intersection = expected_set.intersection(retrieved_set)
    return round(len(intersection) / len(expected_set), 4)


def calculate_context_precision(expected_papers: List[str], retrieved_chunks: List[Dict[str, Any]]) -> float:
    if not expected_papers:
        return 1.0
    expected_set = {p.strip().lower() for p in expected_papers if p.strip()}
    if not expected_set:
        return 1.0
    if not retrieved_chunks:
        return 0.0

    precision_sum = 0.0
    relevant_hits = 0
    for idx, chunk in enumerate(retrieved_chunks):
        paper_id = chunk.get("paper_id", "")
        if paper_id and paper_id.strip().lower() in expected_set:
            relevant_hits += 1
            precision_sum += relevant_hits / (idx + 1)
            
    if relevant_hits == 0:
        return 0.0
    return round(precision_sum / relevant_hits, 4)


# Prompts/Context Helpers
def build_context_string(retrieved_chunks: List[Dict[str, Any]]) -> str:
    if not retrieved_chunks:
        return "Context is empty."
    blocks = []
    for idx, chunk in enumerate(retrieved_chunks):
        paper_id = chunk.get("paper_id", "Unknown")
        page = chunk.get("page_number", "Unknown")
        text = chunk.get("text_content", "").strip()
        blocks.append(f"Block {idx+1} (Paper: {paper_id}, Page: {page}):\n{text}")
    return "\n\n".join(blocks)


def load_prompts(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Checkpoint management
def load_checkpoint(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            con.warning(f"Could not load checkpoint: {e}. Starting fresh.")
    return {}


def save_checkpoint(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        temp_path.replace(path)
    except Exception as e:
        con.warning(f"Failed to save checkpoint: {e}")


async def evaluate_baseline_case(
    evaluator: CloudEvaluator,
    prompts: Dict[str, Any],
    case_id: str,
    query: str,
    golden_answer: str,
    expected_papers: List[str],
    baseline_name: str,
    baseline_data: Dict[str, Any],
    checkpoint_data: Dict[str, Any],
    checkpoint_path: Path
) -> Dict[str, Any]:
    checkpoint_key = f"{case_id}_{baseline_name}"
    
    # Return from checkpoint if already evaluated
    if checkpoint_key in checkpoint_data:
        return checkpoint_data[checkpoint_key]

    eval_metrics = {
        "retrieval_recall": 0.0,
        "context_precision": 0.0,
        "faithfulness": 0.0,
        "answer_relevance": 0.0,
        "citation_fidelity": 0.0,
        "semantic_accuracy": 0.0,
    }

    if baseline_data.get("status") == "error":
        # Keep metrics as 0.0 if baseline failed
        checkpoint_data[checkpoint_key] = eval_metrics
        save_checkpoint(checkpoint_path, checkpoint_data)
        return eval_metrics

    generated_answer = baseline_data.get("generated_answer", "")
    retrieved_papers = baseline_data.get("retrieved_papers", [])
    retrieved_chunks = baseline_data.get("retrieved_chunks", [])

    # 1. Calculate traditional retrieval metrics
    eval_metrics["retrieval_recall"] = calculate_retrieval_recall(expected_papers, retrieved_papers)
    eval_metrics["context_precision"] = calculate_context_precision(expected_papers, retrieved_chunks)

    # If answer is missing, skip LLM calls
    if not generated_answer.strip():
        checkpoint_data[checkpoint_key] = eval_metrics
        save_checkpoint(checkpoint_path, checkpoint_data)
        return eval_metrics

    # Prepare LLM evaluator inputs
    context_str = build_context_string(retrieved_chunks)
    
    llm_tasks = {}

    # Always evaluate relevance and semantic accuracy
    llm_tasks["answer_relevance"] = evaluator.evaluate_metric(
        prompts["answer_relevance_evaluator"],
        metric_name="answer_relevance",
        query=query,
        answer=generated_answer
    )
    llm_tasks["semantic_accuracy"] = evaluator.evaluate_metric(
        prompts["semantic_accuracy_evaluator"],
        metric_name="semantic_accuracy",
        golden_answer=golden_answer,
        answer=generated_answer
    )

    # Evaluate faithfulness and citation fidelity only if there is retrieval context
    if baseline_name != "B0" and retrieved_chunks:
        llm_tasks["faithfulness"] = evaluator.evaluate_metric(
            prompts["faithfulness_evaluator"],
            metric_name="faithfulness",
            context=context_str,
            answer=generated_answer
        )
        llm_tasks["citation_fidelity"] = evaluator.evaluate_metric(
            prompts["citation_fidelity_evaluator"],
            metric_name="citation_fidelity",
            sources=context_str,
            answer=generated_answer
        )

    # Run LLM calls concurrently for this baseline
    keys = list(llm_tasks.keys())
    eval_results = await asyncio.gather(*llm_tasks.values(), return_exceptions=True)

    for key, val in zip(keys, eval_results):
        if isinstance(val, Exception):
            con.error(f"Failed evaluating {key} for {checkpoint_key}: {val}")
            eval_metrics[key] = 0.0
        else:
            eval_metrics[key] = float(val.get("score", 0.0))

    # Save to checkpoint
    checkpoint_data[checkpoint_key] = eval_metrics
    save_checkpoint(checkpoint_path, checkpoint_data)
    
    con.info(
        f"  Evaluated {case_id} [{baseline_name}]: "
        f"Recall={eval_metrics['retrieval_recall']:.2f}, "
        f"Faithfulness={eval_metrics['faithfulness']:.2f}, "
        f"Relevance={eval_metrics['answer_relevance']:.2f}, "
        f"Semantic={eval_metrics['semantic_accuracy']:.2f}"
    )

    return eval_metrics


def get_cloud_credentials(force_cloud: bool) -> tuple[str, str, str]:
    api_key = config.llm_cloud_api_key
    base_url = config.llm_cloud_base_url
    model_name = config.llm_cloud_model_name

    # Fallbacks to environment variables
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    
    if not base_url:
        if api_key and "openrouter" in api_key.lower():
            base_url = "https://openrouter.ai/api/v1"
        else:
            base_url = "https://api.openai.com/v1"

    if not model_name:
        model_name = "google/gemini-2.5-flash"

    if not api_key:
        con.error("Error: Cloud LLM API Key is not configured.")
        con.info("Please set it in ~/.config/pdf-graph-analyzer/config.yaml under llm.cloud.api_key")
        con.info("Or set the OPENAI_API_KEY / OPENROUTER_API_KEY environment variable.")
        sys.exit(1)

    return api_key, base_url, model_name


async def main_async():
    parser = argparse.ArgumentParser(description="Science Graph RAG Quality Evaluator (LLM-as-a-Judge)")
    parser.add_argument(
        "--input", "-i", type=str, default="reports/evaluation_results.yaml",
        help="Path to input evaluation results YAML. Defaults to reports/evaluation_results.yaml"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="reports/result_metrics.yaml",
        help="Path to save output result metrics. Defaults to reports/result_metrics.yaml"
    )
    parser.add_argument(
        "--baselines", "-b", type=str, default="all",
        help="Comma-separated baselines to evaluate (e.g. B0,B2,B6) or 'all'."
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=None,
        help="Limit the number of questions to evaluate (for testing)."
    )
    parser.add_argument(
        "--concurrency", "-c", type=int, default=3,
        help="Max concurrent API calls to the cloud provider."
    )
    parser.add_argument(
        "--rpm", "-r", type=int, default=60,
        help="Rate limit in requests per minute (RPM)."
    )
    parser.add_argument(
        "--cloud", action="store_true", default=True,
        help="Use cloud LLM engine for evaluation (always required, enabled by default)."
    )
    parser.add_argument(
        "--clear-checkpoint", action="store_true",
        help="Ignore existing evaluation checkpoints and restart from scratch."
    )
    args = parser.parse_args()

    # Determine input and output paths
    script_dir = Path(__file__).resolve().parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = script_dir / input_path
        
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir / output_path

    if not input_path.exists():
        con.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Load input data
    con.info(f"Loading benchmark results from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = yaml.safe_load(f)

    if not input_data or "results" not in input_data:
        con.error("Invalid benchmark results file structure. Must contain a 'results' key.")
        sys.exit(1)

    # Load judge prompts
    prompts_path = script_dir / "prompts" / "judge_prompts.yaml"
    if not prompts_path.exists():
        con.error(f"Judge prompts file not found: {prompts_path}")
        sys.exit(1)
    
    prompts = load_prompts(prompts_path)
    con.success("Loaded judge prompts.")

    # Initialize evaluator
    api_key, base_url, model_name = get_cloud_credentials(args.cloud)
    con.info(f"Initializing Cloud LLM Evaluator ({model_name}) ...")
    evaluator = CloudEvaluator(api_key, base_url, model_name, args.concurrency, args.rpm)

    # Handle checkpoint
    checkpoint_path = output_path.parent / ".eval_checkpoint.json"
    if args.clear_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
        con.info("Cleared existing checkpoint.")
    
    checkpoint_data = load_checkpoint(checkpoint_path)
    if checkpoint_data:
        con.info(f"Resuming evaluation using checkpoint. ({len(checkpoint_data)} items already evaluated)")

    # Resolve baselines to evaluate
    if args.baselines.lower() == "all":
        baselines_to_run = "all"
    else:
        baselines_to_run = [b.strip().upper() for b in args.baselines.split(",") if b.strip()]

    # Limit cases if specified
    cases = input_data["results"]
    if args.limit:
        cases = cases[:args.limit]
        con.info(f"Limiting evaluation to first {args.limit} questions.")

    con.info("Starting evaluation task generation...")

    # We will build a list of tasks and execute them
    evaluation_futures = []
    
    for case in cases:
        case_id = case.get("id")
        query = case.get("query")
        golden_answer = case.get("golden_answer")
        expected_papers = case.get("expected_papers", [])

        for baseline_name, baseline_data in case.get("baselines", {}).items():
            if baselines_to_run != "all" and baseline_name not in baselines_to_run:
                continue

            # Append the coroutine task
            evaluation_futures.append((
                case_id,
                baseline_name,
                baseline_data,
                evaluate_baseline_case(
                    evaluator,
                    prompts,
                    case_id,
                    query,
                    golden_answer,
                    expected_papers,
                    baseline_name,
                    baseline_data,
                    checkpoint_data,
                    checkpoint_path
                )
            ))

    con.info(f"Total baseline evaluations to verify: {len(evaluation_futures)}")
    
    # Run all evaluations
    results_map = {}
    for case_id, baseline_name, _, coro in evaluation_futures:
        # Run it sequentially/concurrently via asyncio task wrapper
        results_map[f"{case_id}_{baseline_name}"] = asyncio.create_task(coro)

    # Wait for completion of all tasks
    await asyncio.gather(*results_map.values())

    # Build the final output structure
    con.info("All evaluations complete. Aggregating results...")
    
    final_results = []
    summary_stats = {}

    for case in cases:
        case_id = case.get("id")
        query = case.get("query")
        golden_answer = case.get("golden_answer")
        expected_papers = case.get("expected_papers", [])

        case_output = {
            "id": case_id,
            "category": case.get("category", "general"),
            "query": query,
            "golden_answer": golden_answer,
            "expected_papers": expected_papers,
            "baselines": {}
        }

        for baseline_name, baseline_data in case.get("baselines", {}).items():
            if baselines_to_run != "all" and baseline_name not in baselines_to_run:
                continue

            key = f"{case_id}_{baseline_name}"
            # Extract metrics from completed tasks
            eval_metrics = results_map[key].result()

            # Initialize summary stats tracker
            if baseline_name not in summary_stats:
                summary_stats[baseline_name] = {
                    "latency_sec": [],
                    "retrieval_recall": [],
                    "context_precision": [],
                    "faithfulness": [],
                    "answer_relevance": [],
                    "citation_fidelity": [],
                    "semantic_accuracy": []
                }

            latency = baseline_data.get("latency_sec")
            if latency is not None:
                summary_stats[baseline_name]["latency_sec"].append(latency)

            # Record stats
            for k, val in eval_metrics.items():
                # For B0, skip faithfulness and citation metrics from average if they are always zero/NA
                if baseline_name == "B0" and k in ("faithfulness", "citation_fidelity", "context_precision"):
                    continue
                # For non-B0, if retrieved_chunks was empty, we skip context metrics to avoid pulling down averages of runs that actually searched
                if baseline_name != "B0" and not baseline_data.get("retrieved_chunks") and k in ("faithfulness", "citation_fidelity", "context_precision"):
                    continue
                summary_stats[baseline_name][k].append(val)

            case_output["baselines"][baseline_name] = {
                "status": baseline_data.get("status", "success"),
                "latency_sec": latency,
                "retrieved_papers": baseline_data.get("retrieved_papers", []),
                "eval_metrics": eval_metrics,
                "generated_answer": baseline_data.get("generated_answer", "")
            }

        final_results.append(case_output)

    # Compute averages for summary
    final_summary = {}
    for baseline, metrics in summary_stats.items():
        final_summary[baseline] = {}
        for m_name, values in metrics.items():
            if values:
                final_summary[baseline][f"avg_{m_name}"] = round(sum(values) / len(values), 4)
            else:
                final_summary[baseline][f"avg_{m_name}"] = 0.0

    # Write output report
    output_report = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "original_metadata": input_data.get("metadata", {}),
            "evaluation_llm": {
                "model_name": model_name,
                "provider": base_url
            }
        },
        "summary": final_summary,
        "results": final_results
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(output_report, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    con.success(f"Evaluation finished successfully! Report saved to: {output_path}")

    # Print a beautiful Rich summary table
    table = Table(title="RAG baselines - Aggregated Evaluation Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Baseline", style="cyan")
    table.add_column("Recall", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Faithfulness", justify="right")
    table.add_column("Relevance", justify="right")
    table.add_column("Citations", justify="right")
    table.add_column("Semantic", justify="right")
    table.add_column("Latency (s)", justify="right")

    for baseline in sorted(final_summary.keys()):
        stats = final_summary[baseline]
        # B0 values for citation and faithfulness are marked as N/A to indicate they were excluded
        recall = f"{stats.get('avg_retrieval_recall', 0.0):.2%}" if baseline != "B0" else "N/A"
        precision = f"{stats.get('avg_context_precision', 0.0):.2%}" if baseline != "B0" else "N/A"
        faithfulness = f"{stats.get('avg_faithfulness', 0.0):.2%}" if baseline != "B0" else "N/A"
        relevance = f"{stats.get('avg_answer_relevance', 0.0):.2%}"
        citations = f"{stats.get('avg_citation_fidelity', 0.0):.2%}" if baseline != "B0" else "N/A"
        semantic = f"{stats.get('avg_semantic_accuracy', 0.0):.2%}"
        latency = f"{stats.get('avg_latency_sec', 0.0):.2f}s"

        table.add_row(
            baseline,
            recall,
            precision,
            faithfulness,
            relevance,
            citations,
            semantic,
            latency
        )

    con.console.print(table)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
