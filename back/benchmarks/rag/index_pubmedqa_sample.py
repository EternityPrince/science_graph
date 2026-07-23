#!/usr/bin/env python3
"""
Index PubMedQA sample notes into science_graph.db and Vector DB.
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure back root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACK_ROOT = SCRIPT_DIR.parent.parent
if str(BACK_ROOT) not in sys.path:
    sys.path.insert(0, str(BACK_ROOT))

from src.config import config
from src.services.container import container
from src.services.indexing_orchestrator import run_batch_index
from src import console as con

PARQUET_PATH = BACK_ROOT / "benchmarks" / "rag" / "dataset" / "pubmedqa" / "pubmedqf.parquet"
MANIFEST_PATH = BACK_ROOT / "benchmarks" / "rag" / "dataset" / "pubmedqa" / "sampled_300.json"
DEFAULT_NOTES_DIR = BACK_ROOT / "benchmarks" / "rag" / "dataset" / "pubmedqa" / "notes"
DEFAULT_DB_PATH = BACK_ROOT / "science_graph.db"

def ensure_sample_and_notes(notes_dir: Path, manifest_path: Path):
    notes_dir.mkdir(parents=True, exist_ok=True)
    existing_notes = list(notes_dir.glob("*.md"))
    
    if len(existing_notes) == 300 and manifest_path.exists():
        con.info(f"300 notes already exist in {notes_dir} and manifest at {manifest_path}.")
        return

    con.info(f"Loading dataset from {PARQUET_PATH}...")
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet file not found at {PARQUET_PATH}")
        
    df = pd.read_parquet(PARQUET_PATH)
    con.info(f"Dataset loaded with {len(df)} total rows. Sampling 300 rows (random_state=42)...")
    sampled_df = df.sample(n=300, random_state=42)

    manifest = []
    for idx, row in sampled_df.iterrows():
        pubid = int(row["pubid"])
        paper_id = f"pmid-{pubid}"
        question = str(row["question"]).strip()
        
        c = row["context"]
        if isinstance(c, dict):
            c = c.get("contexts", [])
        if isinstance(c, (list, tuple, np.ndarray)):
            context_str = "\n\n".join([str(x).strip() for x in c])
        else:
            context_str = str(c).strip()

        long_answer = str(row["long_answer"]).strip()
        final_decision = str(row["final_decision"]).strip()

        item = {
            "pubid": pubid,
            "paper_id": paper_id,
            "question": question,
            "context": context_str,
            "long_answer": long_answer,
            "final_decision": final_decision
        }
        manifest.append(item)

        title_yaml = json.dumps(f"PubMed {pubid}: {question}")
        note_content = f"""---
title: {title_yaml}
pmid: {pubid}
tags: ["pubmedqa", "medical"]
---
# PubMed {pubid}: {question}

## Context
{context_str}

## Answer
{long_answer}
"""
        note_file = notes_dir / f"pmid_{pubid}.md"
        note_file.write_text(note_content, encoding="utf-8")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    con.success(f"Saved manifest to {manifest_path} and created 300 notes in {notes_dir}.")

def parse_args():
    parser = argparse.ArgumentParser(description="Index PubMedQA 300 sample notes into science_graph.db")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        dest="use_llm",
        default=False,
        help="Use LLM for concept extraction (slower)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_false",
        dest="use_llm",
        help="Disable LLM for concept extraction (fast regex mode)"
    )
    parser.add_argument(
        "--notes-dir",
        type=str,
        default=str(DEFAULT_NOTES_DIR),
        help="Directory containing PubMedQA markdown notes"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to target SQLite database (defaults to active production database)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    notes_dir = Path(args.notes_dir).resolve()
    from core.config import resolve_project_db_path
    db_path = resolve_project_db_path(args.db_path)
    manifest_path = MANIFEST_PATH.resolve()

    ensure_sample_and_notes(notes_dir, manifest_path)
        
    md_files = list(notes_dir.glob("*.md"))
    if not md_files:
        con.error(f"No markdown files found in {notes_dir}")
        sys.exit(1)

    con.info(f"Target database: {db_path}")
    con.info(f"Target notes directory: {notes_dir} ({len(md_files)} files)")
    con.info(f"LLM extraction enabled: {args.use_llm}")
    
    # Configure database path in config and reset container instances
    config.data["db_path"] = str(db_path)
    container._graph_repo = None
    container._vector_repo = None

    t0 = time.time()
    results = run_batch_index(
        target=str(notes_dir),
        use_llm=args.use_llm,
        trace=False,
        cloud=False
    )
    t1 = time.time()
    
    con.success(f"Batch indexing finished in {t1 - t0:.2f} seconds.")
    
    # Verify saved papers count in science_graph.db
    graph_repo = container.get_graph_repo()
    stats = graph_repo.get_stats()
    indexed_papers_count = stats.get("papers", 0)
    
    con.info(f"Database Stats: {stats}")
    con.success(f"Verified {indexed_papers_count} papers saved in {db_path}.")
    
    if indexed_papers_count < len(md_files):
        con.warning(f"Expected {len(md_files)} papers, but found {indexed_papers_count}.")
        sys.exit(1)
    else:
        con.success(f"All {len(md_files)} PubMedQA sample notes successfully indexed into {db_path} and verified!")

if __name__ == "__main__":
    main()
