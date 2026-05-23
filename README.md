# 🔬 Science Graph

**A local-first, privacy-preserving AI knowledge base for scientific papers, books, and research notes.**

Build a queryable knowledge graph from PDFs, Markdown notes, and EPUB books — powered entirely by on-device AI (Apple Silicon MLX). No cloud APIs, no data leaving your machine.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![Platform: macOS](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-lightgrey)](https://developer.apple.com/silicon/)

---

## ✨ What it does

You drop PDFs, notes (supporting Obsidian-style `[[wiki-links]]`), or EPUB books into Science Graph. It extracts text, generates semantic embeddings, and builds a rich knowledge graph linking papers → authors → concepts → citations → related notes. Then you ask questions in plain language and get cited, context-aware answers from your **own** local AI, keep notes chronologically, view your research timeline, and explore a contribution calendar.

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
| `index` | Index PDF, Markdown, EPUB files, or Web/arXiv URLs |
| `reindex meta` | Partially re-index paper metadata without regenerating embeddings |
| `reindex full` | Fully re-index papers (re-chunk, recreate embeddings) by re-ingesting original files/URLs |
| `query` | One-shot RAG question answering |
| `chat` | Interactive TUI chat with memory |
| `storage` | Interactive TUI database manager (with search, abstract preview, file opening, and LLM summary generation) |
| `review` | Generate a full Markdown literature review |
| `serve` | Launch the Web UI (FastAPI, chat, timeline, contribution heatmap, and interactive notetaker) |
| `stats` | Show knowledge base statistics |
| `config` | Show all configuration and model paths |
| `visualize` | Export an interactive HTML graph with dynamic year/date filtering |
| `extract-file` | Extract authors, concepts, and tags from a text document and output as JSON |
| `cleanup` | Remove orphaned Concept nodes with degree 0 |
| `reset` | Completely reset the database, vector index, and local archives |


### `index` — Indexing documents

```bash
# Single PDF
python3 main.py index --file paper.pdf

# Single Markdown note (Obsidian-compatible)
python3 main.py index --file notes/my_note.md

# Single EPUB book
python3 main.py index --file books/deep_learning.epub

# Direct URL / arXiv / DOI link indexing (auto-enriches authors, abstract, and tags)
python3 main.py index --file https://arxiv.org/abs/1706.03762

# Entire directory (auto-detects format)
python3 main.py index --dir ~/research/

# With LLM-assisted concept/tag extraction (richer graph, slower)
python3 main.py index --dir ~/research/ --use-llm

# Force a specific type
python3 main.py index --dir ~/obsidian-vault/ --type md
```

### `reindex` — Metadata or Full Re-indexing

Updates paper metadata or fully re-indexes existing documents.

#### `reindex meta` — Partially re-index paper metadata (without regenerating embeddings)

Updates metadata (authors, publication year, topic tags, and citations) from Semantic Scholar or fallbacks:

```bash
# Update papers that have no authors in the database
python3 main.py reindex meta --missing-authors

# Update papers that have no topic tags
python3 main.py reindex meta --missing-tags

# Re-index metadata for all papers
python3 main.py reindex meta --all-metadata

# Re-index using LLM for tag extraction (slower)
python3 main.py reindex meta --missing-tags --use-llm

# Limit the number of updated documents
python3 main.py reindex meta --all-metadata --limit 10
```

#### `reindex full` — Fully re-index papers (re-chunk and recreate embeddings)

Deletes the existing paper node and its chunks, then fully re-ingests/re-indexes the original local file or URL:

```bash
# Re-index all papers in the database
python3 main.py reindex full --all

# Re-index a single paper by ID
python3 main.py reindex full --id <paper_id>

# Limit the number of papers to re-index
python3 main.py reindex full --all --limit 10

# Re-index using LLM for tag and concept extraction
python3 main.py reindex full --all --use-llm
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
- **Premium Obsidian-like dark layout** for sleek visual excellence.
- **Top Tab-based Navigation** — switches between Graph, Chat, Notes, Chronology, and Upload views.
- **Dedicated Details Sidebar** — displayed strictly inside the Graph view, rendering polymorphic detail cards tailored for Papers, Authors, Concepts, and Tags.
- **Interactive knowledge graph** with local `vis-network` (filter by node type, dynamic zoom, and date filters).
- **Tags as Meta-relationships** — topic tags are represented as distinct pink nodes (`#e64980`, group `tag`) in the graph, with a "Теги" filter chip, and click-to-focus interactivity.
- **Streaming RAG chat** with Markdown rendering (SSE).
- **Заметки (Notes)** — a Simple Notetaker form to write and save Markdown notes directly, which are auto-indexed with wikilink resolution.
- **Хронология (Chronology)** — visualizes a 53-week contribution calendar heatmap (CSS-grid layout) and a vertical scrollable timeline of papers sorted by creation date.
- **Drag & drop upload** — index files directly from the browser.
- **Live search** — fuzzy title/concept/author search with graph focus.

### `config` — Inspect configuration

```bash
python3 main.py config
```

Shows 4 tables:
- **Paths** — database, archive directory, config file (with existence check ✓/✗)
- **LLM Model** — active provider, local model settings, cloud API settings, max tokens, temperature
- **Embedding Model** — model name, reranker, chunk size/overlap
- **Environment** — HF_TOKEN status, verbosity settings

---

## ⚙️ Configuration

Configuration lives at `~/.config/pdf-graph-analyzer/config.yaml`.

```yaml
# Path to the SQLite database file
db_path: "~/.local/share/pdf-graph-analyzer/graph.db"

# Directory where local archives of websites/PDFs are stored
archive_dir: "~/.local/share/pdf-graph-analyzer/archive"

# HuggingFace token for downloading gated models/embeddings (optional)
hf_token: ""

# Large Language Model (LLM) configuration
llm:
  # Provider: 'mlx' (for local Apple Silicon) or 'openai' (for OpenAI / OpenRouter / compatible APIs)
  provider: "mlx"

  # Global default maximum output tokens for LLM response
  max_tokens: 1000

  # Default temperature (0.0 = deterministic, 1.0 = creative)
  temp: 0.1

  # Local model settings (used if provider is 'mlx')
  local:
    model_path: "~/models/llm/gemma-3-text-12b-it-4bit"

  # Cloud model settings (used if provider is 'openai')
  cloud:
    provider: "openai"
    model_name: "google/gemini-2.5-flash"
    api_key: ""
    base_url: "https://openrouter.ai/api/v1"

  # Task-specific input token limits (used to dynamically truncate inputs to fit context)
  extraction_input_limit: 5000
  clustering_input_limit: 6000
  synthesis_input_limit: 5000

  # Task-specific output token limits (passed to model during generation)
  extraction_output_limit: 2048
  clustering_output_limit: 1500
  synthesis_output_limit: 1500

# Embedding model configuration (used for vector search and indexing)
embedding:
  # HuggingFace model name for sentence embeddings
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 1000
  chunk_overlap: 200

# spaCy model configuration (used for lemmatization)
spacy:
  # spaCy model name (e.g. "en_core_web_sm") or path
  model_name: "en_core_web_sm"

# NER model configuration (used for name extraction)
ner:
  # NER model name or HuggingFace repo ID or local path
  model_name: "dslim/bert-base-NER"

# PDF compression settings (used to downsample high-DPI scanned PDFs)
pdf_compression:
  enabled: true
  dpi_threshold: 151
  dpi_target: 150
  quality: 75
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

All 156 tests pass. Test coverage includes:
- SQLite graph and vector repositories (creation, deletion, updates)
- Markdown parser (front-matter parsing, Obsidian-style `[[wikilinks]]` node resolution, fallback filesystem creation dates)
- URL parser (arXiv ID, DOI, fallback meta tags extraction, local archive copies)
- Hybrid search (BM25 + dense + reranking logic)
- TUI chat session logic and CLI commands
- TUI `storage` interactions (instant unambiguous digit selection & wait-for-second-digit selection)
- External API (Semantic Scholar query with exponential retries and arXiv queries)
- Metadata-only and Full re-indexing pipelines (`reindex meta` and `reindex full`)
- Concept description resolution (predefined dictionary & LLM fallbacks)
- Entity extraction service and LLM schema validation
- LLM limit constraints and input truncation logic
- Normalization pipeline and concept lemmatization via spaCy
- PDF compression and resolution downsampling
- Database cleanup (removal of degree-0 orphaned concepts)

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
│   ├── indexer.py             # Ingestion orchestrator
│   ├── rag.py                 # RAG orchestration helpers
│   ├── llm_engine.py          # Local MLX LLM & OpenAI API wrapper
│   ├── llm_schemas.py         # Structured outputs Pydantic models for LLM
│   ├── ner_engine.py          # NER helper for author names (dslim/bert-base-NER)
│   ├── schemas.py             # API request/response schemas
│   ├── vector_search.py       # Embeddings + BM25 & reranking
│   ├── review_agent.py        # Auto-review agentic pipeline
│   ├── web_app.py             # FastAPI Web UI backend
│   ├── frontend/              # SPA Web UI assets
│   │   ├── css/               # Tailwind or Custom CSS files
│   │   ├── js/                # Javascript for SPA routing & interaction
│   │   ├── index.html         # Main SPA view (vis-network + SSE chat)
│   │   └── favicon.png
│   ├── parsers/
│   │   ├── base.py            # Abstract parser interface
│   │   ├── factory.py         # Parser factory
│   │   ├── md_parser.py       # Obsidian Markdown parser
│   │   ├── epub_parser.py     # EPUB parser (ebooklib)
│   │   └── url_parser.py      # Academic URL/arXiv/DOI parser
│   ├── parser.py              # PDF parser (PyMuPDF)
│   ├── models.py              # Core data models & slugification
│   ├── repository/
│   │   ├── base.py            # Abstract repository interfaces
│   │   └── sqlite_impl.py     # SQLite + USearch implementation
│   ├── services/              # Decoupled business logic
│   │   ├── extraction_service.py # Entity/relation extraction
│   │   ├── metadata_enricher.py  # External metadata lookup
│   │   ├── normalization_pipeline.py # spaCy-based lemmatization & alias resolution
│   │   ├── note_service.py       # Note ingestion & wikilinks
│   │   └── rag_service.py        # Dense/sparse/graph hybrid retriever
│   ├── external_api.py        # Semantic Scholar API client
│   └── tui.py                 # Rich TUI chat
└── tests/                     # 156 automated pytest tests
    ├── test_cleanup.py
    ├── test_cli.py
    ├── test_external_api.py
    ├── test_extraction_service.py
    ├── test_hybrid_search.py
    ├── test_indexer_pipeline.py
    ├── test_indexer_wikilinks.py
    ├── test_llm_limits.py
    ├── test_llm_validation.py
    ├── test_md_epub_indexer.py
    ├── test_md_parser.py
    ├── test_metadata_enricher.py
    ├── test_pdf_compression.py
    ├── test_pipeline_refactoring.py
    ├── test_refinements.py
    ├── test_repository.py
    ├── test_repository_delete.py
    ├── test_services.py
    ├── test_split_llm_config.py
    ├── test_taxonomy.py
    ├── test_tui.py
    └── test_url_parser.py
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
