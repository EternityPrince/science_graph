# 🔬 Science Graph

**A local-first, privacy-preserving AI knowledge base for scientific papers, books, YouTube videos, and research notes.**

Build a queryable knowledge graph from PDFs, Markdown notes, EPUB books, and YouTube videos — powered entirely by on-device AI (Apple Silicon MLX & local Whisper transcription). No cloud APIs, no data leaving your machine.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![Platform: macOS](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-lightgrey)](https://developer.apple.com/silicon/)

---

## ✨ What it does

You drop PDFs, notes (supporting Obsidian-style `[[wiki-links]]`), EPUB books, or YouTube video links into Science Graph. It extracts text (transcribing video audio tracks locally using Whisper), generates semantic embeddings, and builds a rich knowledge graph linking papers → authors → concepts → citations → related notes. Then you ask questions in plain language and get cited, context-aware answers from your **own** local AI, keep notes chronologically, view your research timeline, and explore a contribution calendar.

**No OpenAI. No internet required. All data stays on your Mac.**

---

## 🗺️ Architecture

```
PDF / Markdown / EPUB / YouTube
                 │
                 ▼
          ┌─────────────┐    ┌──────────────────────────────────┐
          │   Parser    │───▶│  Knowledge Graph (SQLite)         │
          │ (PyMuPDF /  │    │  nodes: Paper, Author, Concept    │
          │  ebooklib / │    │  edges: CITES, AUTHORED,          │
          │  frontmatter│    │         MENTIONS_CONCEPT,         │
          │  whisper)   │    │         RELATED_TO, HAS_TAG       │
          └─────────────┘    └──────────────────────────────────┘
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
- **Node.js 18+** (for local frontend development)
- **[uv](https://docs.astral.sh/uv/)** (highly recommended) or pip
- A local **MLX model** (Gemma-3, Qwen-3, LLaMA — see [Configuration](#configuration))
- A local **Whisper model** (e.g. `faster-whisper-large-v3-turbo` for YouTube video transcription)

### Installation

```bash
git clone https://github.com/your-org/science-graph.git
cd science-graph

# Sync dependencies with uv (runs at root workspace level)
uv sync
```

### Running Locally (Developer Mode)

To run the full development stack (Next.js frontend + FastAPI backend) concurrently:

```bash
# Start Next.js dev server (on port 3000) and FastAPI server (on port 8000)
./dev.sh
```

### Running via Docker Compose

To launch the containerized production build of the backend and frontend:

```bash
docker-compose up --build
```

### First run CLI

You can interact with Science Graph directly using the `uv run graph` command from the project root:

```bash
# Check your configuration and active models
uv run graph config

# Index a paper
uv run graph index ~/Downloads/attention_is_all_you_need.pdf

# Ask a question
uv run graph query "What attention mechanism does the Transformer use?"

# Or start the interactive TUI chat
uv run graph chat
```

---

## 📦 Commands

| Command | Description |
|---------|-------------|
| `index` | Index PDF, Markdown, EPUB files, or Web/arXiv/YouTube URLs (supports single/multiple targets) |
| `reindex meta` | Partially re-index paper metadata without regenerating embeddings |
| `reindex full` | Fully re-index papers (re-chunk, recreate embeddings) by re-ingesting original files/URLs |
| `query` | One-shot RAG question answering |
| `chat` | Interactive TUI chat with memory |
| `storage` | Interactive TUI database manager (with search, abstract preview, file opening, and LLM summary generation) |
| `review` | Generate a full Markdown literature review |
| `serve` | Launch the FastAPI Web UI backend (ports Next.js frontend assets from `frontend/out`) |
| `stats` | Show knowledge base statistics and disk storage details |
| `config` | Show all configuration and model paths |
| `visualize` | Export an interactive HTML graph with dynamic year/date filtering |
| `extract-file` | Extract authors, concepts, and tags from a text document and output as JSON |
| `cleanup` | Remove orphaned Concept nodes with degree 0 |
| `reset` | Completely reset the database, vector index, and local archives |
| `doctor` | Scan database, detect, and fix LLM output artifacts, unapplied formatting, and incorrect identifiers |
| `init` | Bring the configuration file up-to-date with current settings (adding new / removing obsolete fields) |
| `export-db` | Export the database contents (nodes, edges, chunks) as JSON or YAML to stdout |


### `index` — Ingesting documents and links

The `index` command accepts a single target, directory, or multiple comma- or semicolon-separated targets. It automatically determines the appropriate parser based on file extensions or URL structures.

```bash
# Single PDF
uv run graph index paper.pdf

# Single Markdown note (Obsidian-compatible with wikilinks [[link]])
uv run graph index notes/my_note.md

# Single EPUB book
uv run graph index books/deep_learning.epub

# Direct URL / arXiv / DOI link indexing (auto-enriches metadata)
uv run graph index https://arxiv.org/abs/1706.03762

# YouTube video URL (downloads audio and transcribes locally using Whisper)
uv run graph index "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Multiple targets separated by commas or semicolons
uv run graph index "paper.pdf; https://arxiv.org/abs/1706.03762; notes/my_note.md"

# Entire directory (recursively indexes supporting file types)
uv run graph index ~/research/

# Control parallel LLM processing chunk pool size (prevent local GPU/CPU overload)
uv run graph index ~/research/ --chunk-pool 2

# Run without LLM-assisted concept extraction (faster indexing, fallback extraction)
uv run graph index ~/research/ --no-llm
```

### `reindex` — Metadata or Full Re-indexing

Updates paper metadata or fully re-indexes existing documents.

#### `reindex meta` — Partially re-index paper metadata (without regenerating embeddings)

Updates metadata (authors, publication year, topic tags, and citations) from Semantic Scholar or fallbacks:

```bash
# Update papers that have no authors in the database
uv run graph reindex meta --missing-authors

# Update papers that have no topic tags
uv run graph reindex meta --missing-tags

# Re-index metadata for all papers
uv run graph reindex meta --all-metadata

# Re-index using LLM for tag extraction (slower)
uv run graph reindex meta --missing-tags --use-llm

# Limit the number of updated documents
uv run graph reindex meta --all-metadata --limit 10
```

#### `reindex full` — Fully re-index papers (re-chunk and recreate embeddings)

Deletes the existing paper node and its chunks, then fully re-ingests/re-indexes the original local file or URL:

```bash
# Re-index all papers in the database
uv run graph reindex full --all

# Re-index a single paper by ID
uv run graph reindex full --id <paper_id>

# Limit the number of papers to re-index
uv run graph reindex full --all --limit 10

# Re-index using LLM for tag and concept extraction
uv run graph reindex full --all --use-llm
```

### `query` — Ask questions

```bash
uv run graph query "What are the main differences between BERT and GPT?"
uv run graph query "Which papers propose LoRA for fine-tuning?" --limit 10
```

### `review` — Auto-generate literature reviews

Runs an agentic literature review pipeline over the indexed knowledge base.

```bash
uv run graph review "attention mechanisms in transformers"
uv run graph review "diffusion models for image generation" --output report.md
uv run graph review "RAG methods" --fast  # skips LLM clustering stage
```

### `doctor` — Sanitize Database Text

Scans SQLite tables through the repository layer to detect and fix LLM output anomalies (artifacts, unapplied Markdown tags, formatting issues, spacing anomalies).

```bash
# Scan and report anomalies without writing changes (check-only mode)
uv run graph doctor

# Scan and correct anomalies in place (fix mode)
uv run graph doctor --fix
```

### `init` — Configuration Schema Update

Synchronizes your `config.yaml` to ensure it is in lockstep with the latest codebase configurations.

```bash
uv run graph init
```

### `export-db` — Export Database Graphs

Dump nodes, edges, and chunks in JSON or YAML formats (excluding dense embeddings) directly to stdout.

```bash
# Export complete graph database to a YAML file
uv run graph export-db > export.yaml

# Export complete graph database to JSON, omitting the text chunks
uv run graph export-db --format json --no-chunks > export.json
```

### `serve` — Web UI and API Host

Starts the FastAPI server hosting backend API endpoints and serving static Next.js frontend pages.

```bash
uv run graph serve                    # opens http://127.0.0.1:8000
uv run graph serve --port 8080        # custom port
uv run graph serve --no-open          # do not auto-open browser
```

The Web UI features:
- **Premium Obsidian-like dark layout** for sleek visual excellence.
- **Top Tab-based Navigation** — switches between Graph, Chat, Library, Notes, Chronology, and Upload views.
- **Dedicated Details Sidebar** — rendered strictly inside the Graph view, providing detail cards for Papers, Authors, Concepts, and Tags.
- **Interactive knowledge graph** via `vis-network` (featuring node filters, dynamic zooming, physics toggles, and year/date filtering).
- **Tags as Meta-relationships** — topic tags are represented as distinct pink nodes (`#e64980`, group `tag`) in the graph, with a "Теги" filter chip, and click-to-focus interactivity.
- **Streaming RAG chat** with Markdown rendering (SSE-backed).
- **Заметки (Notes)** — A simple Markdown editor to capture research notes and link them to papers/authors/concepts via wikilinks.
- **Хронология (Chronology)** — visualizes a 53-week contribution calendar heatmap (CSS-grid layout) and a vertical scrollable timeline of papers sorted by creation date.
- **Drag & drop upload** — index files directly from the browser.
- **Live search** — fuzzy title/concept/author search with graph focus.

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
uv run pytest back/tests/test_repository.py   # specific module
```

All 198 tests pass. Test coverage includes:
- SQLite graph and vector repositories (creation, deletion, updates)
- Markdown parser (front-matter parsing, Obsidian-style `[[wikilinks]]` node resolution, fallback filesystem creation dates)
- URL parser (arXiv ID, DOI, fallback meta tags extraction, local archive copies)
- YouTube video parser (metadata, local Whisper audio transcription, timestamps)
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
- Database Doctor sanitization (fixing LLM outputs and spacing)

---

## 🗂️ Project Structure

```
science-graph/
├── dev.sh                     # Dev stack command launcher
├── docker-compose.yml         # Dev/Production container setup
├── pyproject.toml             # Root project workspace settings
├── uv.lock                    # Dependency lockfile
├── back/                      # Python Backend Service
│   ├── main.py                # Main backend entrypoint
│   ├── pyproject.toml         # Backend dependencies & package scripts
│   ├── Dockerfile.backend     # Backend Docker configuration
│   ├── src/                   # Python Backend Core
│   │   ├── cli.py             # CLI interface commands (doctor, init, etc)
│   │   ├── config.py          # Config loader and writer
│   │   ├── indexer.py         # Ingestion Orchestrator
│   │   ├── llm_engine.py      # LLM handler (local MLX / cloud OpenAI)
│   │   ├── vector_search.py   # Hybrid vector/BM25 retrieval engine
│   │   ├── web_app.py         # FastAPI REST & SSE backend
│   │   ├── parsers/           # Parser sub-module (PDF, EPUB, MD, YouTube)
│   │   │   ├── base.py
│   │   │   ├── epub_parser.py
│   │   │   ├── md_parser.py
│   │   │   ├── pdf_parser.py
│   │   │   ├── url_parser.py
│   │   │   └── youtube_parser.py
│   │   └── services/          # Supporting services (doctor, note, extraction)
│   │       ├── doctor_service.py
│   │       └── ...
│   └── tests/                 # 198 automated unit & integration tests
└── frontend/                  # Next.js SPA Web UI (React, Lucide, vis-network)
    ├── package.json           # Frontend scripts & dependencies
    ├── next.config.ts         # Proxy API routes to backend
    ├── Dockerfile             # Frontend Docker configuration
    ├── app/                   # Next.js pages and layouts
    └── components/            # UI components (Graph visualizer, Chat UI)
```

---

## 🧠 Key Design Decisions & Optimizations

### Local-first & Privacy

All inference runs on-device via [MLX](https://github.com/ml-explore/mlx) and local `faster-whisper` models. No API keys needed (except optionally HF_TOKEN for faster model downloads). Your research data never leaves your machine.

### Hybrid Retrieval

Science Graph combines three retrieval signals:
1. **Dense search** — semantic similarity via sentence-transformers embeddings and USearch HNSW index
2. **BM25** — keyword overlap, great for exact term matching
3. **Cross-Encoder reranking** — `mixedbread-ai/mxbai-rerank-xsmall-v1` rescores top candidates with a pairwise model for maximum precision

Scores are fused using **Reciprocal Rank Fusion (RRF)** before reranking.

### Staged Batch Ingestion Pipeline

To prevent local resources from being overwhelmed, batch indexing follows a structured staged pipeline:
1. **Batch Parsing**: Concurrently parses all documents (PDFs, Markdown, EPUBs, YouTube videos) to extract text and initial metadata.
2. **Duplicate Checking & Filtering**: Runs shingles and title/author checking to skip already-indexed documents.
3. **Batch Embedding**: Optimizes sentence embeddings generation by passing all text chunks across the batch to the sentence-transformer model in a single vectorized batch execution.
4. **Rate-limited LLM Extraction**: Throttles concurrent LLM concept extraction and summary tasks using a configurable semaphore (`--chunk-pool`), ensuring Apple Silicon neural engines or CPU cores are not overloaded.
5. **Atomic Persistence**: Persists all nodes, edges, and chunks atomically into SQLite and the vector index.

### Graph-augmented Context

Beyond text chunks, the RAG pipeline pulls in graph context: co-authorship links, citation chains, and concept co-occurrence. This gives the LLM richer signal for complex multi-hop questions.

### SQLite as the Unified Backend

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
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — fast local Whisper transcriber
- [sentence-transformers](https://www.sbert.net/) — semantic embeddings & cross-encoder reranking
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF parsing
- [ebooklib](https://github.com/aerkalov/ebooklib) — EPUB parsing
- [USearch](https://github.com/unum-cloud/usearch) — fast HNSW vector index
- [FastAPI](https://fastapi.tiangolo.com/) — Web UI backend
- [Next.js](https://nextjs.org/) & [vis-network](https://visjs.github.io/vis-network/) — SPA Frontend and Interactive Graph
- [Rich](https://github.com/Textualize/rich) — terminal output
- [Typer](https://typer.tiangolo.com/) — CLI framework
