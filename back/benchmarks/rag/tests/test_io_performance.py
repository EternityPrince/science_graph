import time
import json
import yaml
from pathlib import Path
from core.pipelined import (
    BufferedYAMLWriter,
    safe_read_modify_write_yaml,
    flush_yaml_buffer,
    save_generation_baseline_result,
    save_evaluation_baseline_result,
)
from core.evaluator import save_checkpoint, load_checkpoint


def test_buffered_yaml_writer_buffering_and_flush(tmp_path):
    file_path = tmp_path / "test_buffered.yaml"
    writer = BufferedYAMLWriter(flush_interval_sec=10.0, max_unflushed=100)

    def modify_fn(data):
        if not data:
            data = {"count": 0}
        data["count"] += 1
        return data

    # First modification with force_flush=False
    writer.modify(file_path, modify_fn, force_flush=False)

    # File should NOT exist yet on disk because flush interval hasn't passed and unflushed count < 100
    assert not file_path.exists()

    # Document in memory should be updated
    path_str = str(file_path.resolve())
    assert writer.documents[path_str]["count"] == 1

    # Multiple modifications
    for _ in range(5):
        writer.modify(file_path, modify_fn, force_flush=False)

    assert not file_path.exists()
    assert writer.documents[path_str]["count"] == 6

    # Flush explicitly
    writer.flush(file_path)
    assert file_path.exists()

    with open(file_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert loaded == {"count": 6}


def test_buffered_yaml_writer_max_unflushed_trigger(tmp_path):
    file_path = tmp_path / "test_trigger.yaml"
    writer = BufferedYAMLWriter(flush_interval_sec=100.0, max_unflushed=3)

    def modify_fn(data):
        if not data:
            data = {"items": []}
        data["items"].append("x")
        return data

    writer.modify(file_path, modify_fn, force_flush=False)
    assert not file_path.exists()

    writer.modify(file_path, modify_fn, force_flush=False)
    assert not file_path.exists()

    # 3rd edit reaches max_unflushed=3 -> auto flushes
    writer.modify(file_path, modify_fn, force_flush=False)
    assert file_path.exists()

    with open(file_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert len(loaded["items"]) == 3


def test_save_generation_and_evaluation_result_buffered(tmp_path):
    gen_file = tmp_path / "eval_results.yaml"
    eval_file = tmp_path / "metrics_results.yaml"

    case_info = {
        "category": "science",
        "query": "What is photosyntesis?",
        "golden_answer": "Process by plants",
        "expected_papers": ["P1"]
    }
    b_data = {
        "status": "success",
        "latency_sec": 0.5,
        "generated_answer": "Process by plants",
        "retrieved_chunks": [{"id": "c1", "paper_id": "P1"}]
    }

    # Save generation result with force_flush=False
    save_generation_baseline_result(gen_file, "Q1", case_info, "B1", b_data, {}, force_flush=False)
    assert not gen_file.exists()

    # Flush buffer
    flush_yaml_buffer(gen_file)
    assert gen_file.exists()

    with open(gen_file, "r", encoding="utf-8") as f:
        gen_loaded = yaml.safe_load(f)
    assert len(gen_loaded["results"]) == 1
    assert gen_loaded["results"][0]["baselines"]["B1"]["generated_answer"] == "Process by plants"

    # Save evaluation result with force_flush=False
    eval_metrics = {"faithfulness": 0.95, "answer_relevance": 0.9}
    save_evaluation_baseline_result(eval_file, "Q1", case_info, "B1", b_data, eval_metrics, {}, force_flush=False)
    assert not eval_file.exists()

    flush_yaml_buffer(eval_file)
    assert eval_file.exists()

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_loaded = yaml.safe_load(f)
    assert eval_loaded["summary"]["B1"]["avg_faithfulness"] == 0.95


def test_save_checkpoint_buffering(tmp_path):
    ckpt_path = tmp_path / ".eval_checkpoint.json"

    data = {"key_1": "val_1"}
    save_checkpoint(ckpt_path, data, force=False)

    data["key_2"] = "val_2"
    save_checkpoint(ckpt_path, data, force=False)

    # Force write to verify persistence
    save_checkpoint(ckpt_path, data, force=True)
    assert ckpt_path.exists()

    loaded = load_checkpoint(ckpt_path)
    assert loaded == {"key_1": "val_1", "key_2": "val_2"}
