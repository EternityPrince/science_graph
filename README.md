# 🔬 Science Graph

**A local-first, privacy-preserving AI knowledge base for scientific papers, books, and research notes.**

Build a queryable knowledge graph from PDFs, Markdown notes, and EPUB books — powered entirely by on-device AI (Apple Silicon MLX). No cloud APIs, no data leaving your machine.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![Platform: macOS](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-lightgrey)](https://developer.apple.com/silicon/)

---

## ✨ What it does

You drop PDFs, notes, or EPUB books into Science Graph. It extracts text, generates semantic embeddings, and builds a rich knowledge graph linking papers → authors → concepts → citations. Then you ask questions in plain language and get cited, context-aware answers from your **own** local AI.

**No OpenAI. No internet required. All data stays on your Mac.**

---

## 🗺️ Architecture

```
PDF / Markdown / EPUB
         │
         ▼
   ┌─────────────┐    ┌──────────────────────────────────┐
   │   Parser    │───▶│  Knowledge Graph (SQLite)         │
   │ (PyMuPDF /  │    │  nodes: Paper, Author, Concept    │
   │  ebooklib / │    │  edges: CITES, AUTHORED,          │
   │  frontmatter│    │         MENTIONS_CONCEPT,         │
   └─────────────┘    │         RELATED_TO                │
                      └──────────────────────────────────┘
         │
         ▼
   ┌─────────────────────────┐
   │  Embedding Engine       │
   │  (sentence-transformers)│
   └────────────┬────────────┘
                │
                ▼
   ┌────────────────────────────┐
   │  Vector Index (USearch)    │
   │  stored inside SQLite      │
   └──────────┬─────────────────┘
              │
              ▼ at query time
   ┌──────────────────────────────────────────────┐
   │  Hybrid RAG Pipeline                          │
   │  Dense retrieval + BM25 + RRF fusion          │
   │  + Cross-Encoder reranking                    │
   │  + Graph context (citations, co-authors)      │
   └────────────────┬─────────────────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │  Local LLM      │
          │  (MLX — Gemma / │
          │   Qwen / LLaMA) │
          └─────────────────┘
                    │
                    ▼
          Cited, context-aware answer
```

---

## 🚀 Quick Start

### Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip
- A local **MLX model** (Gemma-3, Qwen-3, LLaMA — see [Configuration](#configuration))

### Installation

```bash
git clone https://github.com/your-org/science-graph.git
cd science-graph

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### First run

```bash
# Check your configuration
python3 main.py config

# Index a paper
python3 main.py index --file ~/Downloads/attention_is_all_you_need.pdf

# Ask a question
python3 main.py query "What attention mechanism does the Transformer use?"

# Or start the interactive chat
python3 main.py chat
```

---

## 📦 Commands

| Command | Description |
|---------|-------------|
| `index` | Index PDF, Markdown, or EPUB files |
| `query` | One-shot RAG question answering |
| `chat` | Interactive TUI chat with memory |
| `review` | Generate a full Markdown literature review |
| `serve` | Launch the Web UI (FastAPI + vis-network) |
| `stats` | Show knowledge base statistics |
| `config` | Show all configuration and model paths |
| `visualize` | Export an interactive HTML graph |

### `index` — Indexing documents

```bash
# Single PDF
python3 main.py index --file paper.pdf

# Single Markdown note (Obsidian-compatible)
python3 main.py index --file notes/my_note.md

# Single EPUB book
python3 main.py index --file books/deep_learning.epub

# Entire directory (auto-detects format)
python3 main.py index --dir ~/research/

# With LLM-assisted concept extraction (richer graph, slower)
python3 main.py index --dir ~/research/ --use-llm

# Force a specific type
python3 main.py index --dir ~/obsidian-vault/ --type md
```

### `query` — Ask questions

```bash
python3 main.py query "What are the main differences between BERT and GPT?"
python3 main.py query "Which papers propose LoRA for fine-tuning?" --limit 10
```

### `review` — Auto-generate literature reviews

Runs a full agentic pipeline:
1. Hybrid retrieval of relevant chunks
2. LLM-based thematic clustering
3. Sequential section synthesis
4. Comparison table + APA bibliography

```bash
python3 main.py review "attention mechanisms in transformers"
python3 main.py review "diffusion models for image generation" --output report.md
python3 main.py review "RAG methods" --fast  # skip LLM clustering
```

### `serve` — Web UI

```bash
python3 main.py serve                    # opens http://127.0.0.1:8000
python3 main.py serve --port 8080        # custom port
python3 main.py serve --no-open          # don't auto-open browser
```

The Web UI features:
- **Interactive knowledge graph** with vis-network (filter by Papers / Notes / Books / Authors / Concepts)
- **Streaming RAG chat** with Markdown rendering (SSE)
- **Node details panel** — abstract, concepts, citations, DOI link
- **Drag & drop upload** — index files directly from the browser
- **Live search** — fuzzy title search with graph focus

### `config` — Inspect configuration

```bash
python3 main.py config
```

Shows 4 tables:
- **Paths** — database, archive directory, config file (with existence check ✓/✗)
- **LLM Model** — path, detected model family, max tokens, temperature
- **Embedding Model** — model name, reranker, chunk size/overlap
- **Environment** — HF_TOKEN status, verbosity settings

---

## ⚙️ Configuration

Configuration lives at `~/.config/pdf-graph-analyzer/config.yaml`.

```yaml
# Database & storage
db_path: ~/.local/share/pdf-graph-analyzer/graph.db
archive_dir: ~/.local/share/pdf-graph-analyzer/archive

# Local LLM (MLX format — must be on disk)
llm:
  model_path: /path/to/your/model      # e.g. gemma-3-text-12b-it-4bit
  max_tokens: 1000
  temp: 0.1

# Embeddings (sentence-transformers — auto-downloaded on first use)
embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  chunk_size: 1000
  chunk_overlap: 200
```

### Recommended MLX models

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `mlx-community/gemma-3-text-12b-it-4bit` | ~7 GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| `mlx-community/Qwen3-8B-4bit` | ~5 GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | ~2 GB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

Download with:
```bash
# Install mlx_lm if not already
uv add mlx-lm

# Download a model
python3 -c "from mlx_lm import load; load('mlx-community/gemma-3-text-12b-it-4bit')"
```

### HuggingFace Token (optional)

Without a token, sentence-transformers downloads are rate-limited. To fix:
```bash
export HF_TOKEN=hf_your_token_here
```
Or add it to your shell config (`~/.zshrc`).

---

## 🧪 Testing

```bash
uv run pytest            # run all tests
uv run pytest -v         # verbose output
uv run pytest tests/test_repository.py   # specific module
```

All 13 tests pass. Test coverage includes:
- SQLite graph and vector repositories
- Markdown parser (front-matter, wiki-links, inline tags)
- Hybrid search (BM25 + dense + reranking)
- TUI chat session logic
- External API (Semantic Scholar)

---

## 🗂️ Project Structure

```
science-graph/
├── main.py                    # Entry point
├── pyproject.toml             # Dependencies
├── src/
│   ├── cli.py                 # Typer CLI commands
│   ├── console.py             # Rich-based styled output
│   ├── config.py              # Configuration loading
│   ├── indexer.py             # PDF/MD/EPUB indexing pipeline
│   ├── rag.py                 # Hybrid RAG pipeline
│   ├── llm_engine.py          # Local MLX LLM wrapper
│   ├── vector_search.py       # Embeddings + BM25
│   ├── review_agent.py        # Auto-review agentic pipeline
│   ├── web_app.py             # FastAPI Web UI backend
│   ├── web/
│   │   └── index.html         # SPA Web UI (vis-network + SSE chat)
│   ├── parsers/
│   │   ├── md_parser.py       # Obsidian Markdown parser
│   │   └── epub_parser.py     # EPUB parser (ebooklib)
│   ├── parser.py              # PDF parser (PyMuPDF)
│   ├── models.py              # Core data models
│   ├── repository/
│   │   ├── base.py            # Abstract repository interfaces
│   │   └── sqlite_impl.py     # SQLite + USearch implementation
│   ├── external_api.py        # Semantic Scholar API client
│   └── tui.py                 # Rich TUI chat
└── tests/
    ├── test_repository.py
    ├── test_hybrid_search.py
    ├── test_md_epub_indexer.py
    ├── test_external_api.py
    └── test_tui.py
```

---

## 🧠 Key Design Decisions

### Local-first & privacy

All inference runs on-device via [MLX](https://github.com/ml-explore/mlx). No API keys needed (except optionally HF_TOKEN for faster model downloads). Your research data never leaves your machine.

### Hybrid retrieval

Science Graph combines three retrieval signals:
1. **Dense search** — semantic similarity via sentence-transformers embeddings and USearch HNSW index
2. **BM25** — keyword overlap, great for exact term matching
3. **Cross-Encoder reranking** — `mixedbread-ai/mxbai-rerank-xsmall-v1` rescores top candidates with a pairwise model for maximum precision

Scores are fused using **Reciprocal Rank Fusion (RRF)** before reranking.

### Graph-augmented context

Beyond text chunks, the RAG pipeline pulls in graph context: co-authorship links, citation chains, and concept co-occurrence. This gives the LLM richer signal for complex multi-hop questions.

### SQLite as the unified backend

Both the graph (nodes/edges) and the vector index live inside a single `.db` file. The USearch index is stored as a companion `.usearch` file. This makes the whole knowledge base a single portable directory.

---

## 🤝 Contributing

Contributions are welcome! Areas where help is most useful:

- **More file formats** — arXiv HTML, DOCX, HTML pages
- **Smarter chunking** — section-aware splitting for scientific papers
- **Graph analytics** — PageRank, community detection, centrality
- **Export** — BibTeX/RIS export from the graph
- **Cross-platform** — support for Linux / Windows (requires non-MLX LLM backend)

Please open an issue before large changes to discuss the approach.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

Built on top of excellent open-source libraries:

- [MLX](https://github.com/ml-explore/mlx) & [mlx-lm](https://github.com/ml-explore/mlx-examples) — Apple Silicon LLM inference
- [sentence-transformers](https://www.sbert.net/) — semantic embeddings & cross-encoder reranking
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF parsing
- [ebooklib](https://github.com/aerkalov/ebooklib) — EPUB parsing
- [USearch](https://github.com/unum-cloud/usearch) — fast HNSW vector index
- [FastAPI](https://fastapi.tiangolo.com/) — Web UI backend
- [vis-network](https://visjs.github.io/vis-network/) — interactive graph visualization
- [Rich](https://github.com/Textualize/rich) — terminal output
- [Typer](https://typer.tiangolo.com/) — CLI framework
