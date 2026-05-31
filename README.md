# 🔬 Science Graph

**A local-first, privacy-preserving AI knowledge base for scientific papers, books, YouTube videos, and research notes.**

Build a queryable knowledge graph from PDFs, Markdown notes, EPUB books, and YouTube videos — powered entirely by on-device AI (Apple Silicon MLX & local Whisper transcription). No cloud APIs, no data leaving your machine.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![Platform: macOS](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-lightgrey)](https://developer.apple.com/silicon/)

---

## ⚡ What Makes it Unique?

Compared to generic vector search databases or typical RAG frameworks, **Science Graph** introduces several structural innovations designed specifically for researchers:

1. **Graph-Augmented RAG (not just flat vectors)**: Instead of treating your library as disconnected text chunks, Science Graph structures them into a queryable semantic graph connecting `Paper` ↔ `Author` ↔ `Concept` ↔ `Tag` nodes. The LLM receives both textual context and relational context (citations, co-authors, and tag chains).
2. **Concept Ontology & SpaCy Lemmatization**: Extracted concepts are processed through spaCy for multilingual lemmatization and alias matching. Synonyms (e.g., *"convolutional network"*, *"сверточная сеть"*, and *"CNN"*) are automatically resolved to a single canonical `Concept` node, preventing graph fragmentation.
3. **Local Apple Silicon Optimizations**: Built with native support for [MLX](https://github.com/ml-explore/mlx) to run Gemma, Qwen, and LLaMA models directly on macOS GPUs with high speed and zero inference latency or cloud subscription fees.
4. **Multimodal local parsers**: Ingests Markdown (Obsidian-style `[[wikilinks]]`), PDFs, EPUBs, and YouTube videos. For PDFs, it uses the high-performance **Marker** parser by default with local OCR and layout analysis to convert PDFs into structured Markdown with LaTeX math and tables (Russian and English languages supported). For videos, it extracts the audio track, runs local **Whisper** transcription, and filters transcript chunks to discard conversational fluff before indexing.
5. **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**: Combines dense embeddings search (`USearch` HNSW index) and sparse keyword matches (SQLite `FTS5`) with adaptive BM25 weighting based on match strength.
6. **Smart Context Trimming**: To ensure inputs fit local LLM context limits (e.g. 4k/8k tokens), the system dynamically prunes context. It groups chunks by paper, soft-trims sentences from the *middle* of paragraphs (retaining critical intro/conclusion sentences), and drops low-importance graph edges last.
7. **Model Context Protocol (MCP) Server**: Ships with a built-in MCP server. Agents like Claude Desktop, Cursor, or peer AIs can connect to it to query, explore, index files, or manage research notes directly.

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

## 🚀 Getting Started with `uv`

Science Graph is configured as a Python workspace using [uv](https://docs.astral.sh/uv/), the ultra-fast Python package installer and resolver.

### Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.12+**
- **Node.js 18+** (for local frontend development)
- **[uv](https://docs.astral.sh/uv/)** (recommended: `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh`)

### Installation & Sync

Clone the repository and run `uv sync` from the workspace root. This automatically sets up a virtual environment (`.venv`) and installs both core package dependencies and developer testing tools:

```bash
git clone https://github.com/your-org/science-graph.git
cd science-graph

# Sync dependencies across the workspace
uv sync
```

### Running Locally (Developer Mode)

To run the full development stack (Next.js frontend on port 3000 + FastAPI backend on port 8000) concurrently with live reload:

```bash
# Starts both frontend and backend concurrently
./dev.sh
```

### Running via Docker Compose

To launch containerized production builds of the backend and frontend:

```bash
docker-compose up --build
```

---

## 📖 Ingestion & Indexing Process (Deep Dive)

The `index` command processes local files or URLs. Rather than running a simple sequential parser, the ingestion pipeline utilizes a structured **Staged DAG Execution Flow** to prevent local CPU/GPU resource starvation:

```
                  ┌─────────────── Ingest Target ──────────────┐
                  │                                            │
                  ▼                                            ▼
         ┌─────────────────┐                          ┌──────────────────┐
         │ Duplicate Check │ (Shingle & title match)  │  Parse Document  │
         └────────┬────────┘                          └────────┬─────────┘
                  │                                            │
                  │ (If unique)                                ▼
                  └───────────────────────────────────► [ Staged Ingestion DAG ]
                                                               │
                              ┌────────────────────────────────┼────────────────────────────────┐
                              ▼                                ▼                                ▼
                   ┌───────────────────────┐       ┌───────────────────────┐       ┌──────────────────────┐
                   │ Path A: Meta Enrich   │       │ Path B: Concept Extr. │       │ Path C: Compress/Arch│
                   │ (Semantic Scholar API)│       │ (NER & spaCy Lemmat.) │       │ (DPI Downsampling)   │
                   └──────────┬────────────┘       └───────────┬───────────┘       └────────────┬─────────┘
                              │                                │                                │
                              └────────────────────────────────┼────────────────────────────────┘
                                                               │
                                                               ▼
                                                   ┌───────────────────────┐
                                                   │ Path D: LLM Summary   │
                                                   └───────────┬───────────┘
                                                               │
                                                               ▼
                                                   ┌───────────────────────┐
                                                   │ Vector Embed Batch    │
                                                   └───────────┬───────────┘
                                                               │
                                                               ▼
                                                   ┌───────────────────────┐
                                                   │   Atomic Commit to    │
                                                   │   SQLite & USearch    │
                                                   └───────────────────────┘
```

1. **Duplicate Detection & Shingle Matching**: Extracted text signatures are generated via shingles. Science Graph computes Jaccard similarity and checks existing database titles/IDs. If a duplicate is detected, it terminates the process early, saving computation and LLM tokens.
2. **Parser Selection**:
   - **PDFs**: Parsed via **Marker** (default, layout-aware Markdown and LaTeX formulas converter) or **PyMuPDF** (legacy fast parser).
   - **Markdown notes**: Loaded with front-matter extraction (supporting Obsidian-style `[[wiki-links]]` for creating related Concept/Paper nodes).
   - **EPUBs**: Parsed using `ebooklib`.
   - **YouTube videos**: Ingested via `yt-dlp`. The audio track is downloaded, transcribed locally using `faster-whisper`, and the resulting text chunks are filtered for database relevance.
3. **Four Concurrent Async Paths (DAG Ingestion)**:
   - **Path A (Metadata Enrichment)**: Interrogates the Semantic Scholar and arXiv APIs to fetch publication years, DOIs, formal citation list schemas, and author names.
   - **Path B (Concept & Tag Extraction)**: Utilizes spaCy and a local NER model (`bert-base-NER`) to parse core entities, filtering them through a lemmatization pipeline to resolve plurals or multilingual synonyms into unique Concept nodes.
   - **Path C (Archiving & Compression)**: Copies the file to the local directory. If a PDF is too large, it downsamples high-DPI scans based on settings (e.g., 150 DPI threshold, 75% quality) to preserve disk space.
   - **Path D (LLM Summary Generation)**: Prompts the LLM to generate a concise, structured research summary.
4. **Vectorized Embedding Batching**: Instead of generating embeddings one-by-one, the parsed chunks are batched and sent to `sentence-transformers` in a single vectorized call.
5. **Atomic Persistence**: Writes nodes, edges, and vector chunks inside a single SQLite database transaction to maintain graph integrity.

---

## 🧠 Advanced Hybrid RAG Architecture (Deep Dive)

When you ask a question via `query` or in the TUI/Web Chat, the pipeline executes a multi-stage retrieval and synthesis flow:

```
                            [ User Query ]
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │ Intent & Filter Extract  │ (LLM extracts Author, Year, etc.)
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │ Query Expansion & HyDE   │ (Concept ontology + LLM variants)
                     └────────────┬─────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
    ┌───────────────────┐                   ┌───────────────────┐
    │ Dense Vector Srch │                   │ Sparse FTS5 Srch  │
    │  (USearch HNSW)   │                   │ (Keyword Match)   │
    └─────────┬─────────┘                   └─────────┬─────────┘
              │                                       │
              └───────────────────┬───────────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │  Adaptive RRF Blending   │ (Reciprocal Rank Fusion)
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │  Cross-Encoder Reranker  │ (mixedbread-ai rescoring)
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │  Graph Context & Trim    │ (Neighbor paths + Token-aware prune)
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │  Local LLM Generation    │ (MLX response generation)
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │  Citation Repair Engine  │ (Cross-validates brackets [1], [2])
                     └──────────────────────────┘
```

1. **Intent Classification & Metadata Extraction**: The user's query is analyzed by an LLM classifier to extract structured metadata filters. A natural query like *"BERT papers from 2021 by Devlin"* extracts filters: `{ "year_start": 2021, "year_end": 2021, "author": "Devlin" }`, which are converted directly into SQLite queries.
2. **Dual Query Expansion & HyDE**:
   - **Ontological synonyms**: Maps the query term to alternative concept aliases in the graph.
   - **LLM variants**: Expands the search phrase to multilingual synonyms.
   - **HyDE (Hypothetical Document Embeddings)**: Generates a temporary hypothetical answer to embed and search alongside the real query.
3. **Dense & Sparse Search Fusion**:
   - Executes dense retrieval using the HNSW index (`USearch`) for all query variants.
   - Runs sparse retrieval via SQLite `FTS5` for keyword overlap.
   - Blends results using Reciprocal Rank Fusion (RRF), dynamically tuning the keyword weight depending on the strength of the FTS5 matches (downscaling FTS weight if keyword matches are weak).
4. **Cross-Encoder Reranking**: The top candidates from RRF are rescored using a local `mixedbread-ai/mxbai-rerank-xsmall-v1` cross-encoder model.
5. **Score Blending**: Ranks candidates using a blended metric: `0.7 * Reranker_Score + 0.3 * RRF_Score` to prevent dense-only vector bias.
6. **Graph Context Enrichment**: Gathers neighboring relations of the retrieved papers (e.g., citation chains, author links, concept tags) and appends them to the prompt context. Edges are weighted by strength (e.g., `AUTHORED`: 0.8, `CITES`: 0.7, `MENTIONS_CONCEPT`: 0.6).
7. **Smart Token Trimming**: Measures total context token length against the LLM's limit. If too large, it groups chunks by paper, and iteratively prunes sentences from the middle of the paragraph (retaining the first/last sentences which contain the key thesis statements) rather than throwing away entire documents. If still too large, it prunes low-priority graph links.
8. **LLM Generation**: Generates the final cited response using the local MLX engine.
9. **Citation Validation & Repair**: A regex engine verifies the bracketed citations (e.g. `[1]`, `[2]`) in the output, maps them to the actual document metadata indexes, and filters out hallucinated citations.

---

## 📦 Command Reference

Run any command using `uv run graph <command>`:

| Command | Description |
|---------|-------------|
| `index` | Ingests PDF, Markdown notes, EPUB books, or Web/arXiv/YouTube URLs |
| `reindex meta` | Updates metadata (authors, year, tags, citations) from APIs |
| `reindex full` | Fully re-ingests and re-chunks documents, regenerating vector embeddings |
| `query` | One-shot RAG query answering |
| `chat` | Interactive TUI chat session with history memory |
| `storage` | Interactive TUI SQLite database browser and editor |
| `review` | Auto-generates a detailed Markdown literature review report |
| `serve` | Starts the FastAPI Web UI backend (serving Next.js assets from `frontend/out`) |
| `serve-mcp` | Starts the Science Graph MCP Server (Stdio/SSE modes supported) |
| `stats` | Displays database node counts and storage details |
| `config` | Shows configuration paths and active model details |
| `visualize` | Exports an interactive HTML graph network file with timeline filters |
| `extract-file` | Extracts authors, concepts, and tags from text and returns JSON |
| `cleanup` | Safely purges orphaned Concept nodes with degree 0 |
| `reset` | Resets the SQLite database, USearch vectors, and local file archives |
| `doctor` | Scans the database, reporting and fixing LLM formatting/whitespace artifacts |
| `init` | Syncs `config.yaml` schema to include new parameters |
| `export-db` | Exports complete graph contents (excluding embeddings) as JSON/YAML |

---

### Command Examples

#### Indexing Documents

```bash
# Index a single local PDF (using the default Marker parser)
uv run graph index paper.pdf

# Index a single local PDF using the legacy Fitz (PyMuPDF) parser
uv run graph index paper.pdf --legacy-pdf

# Ingest an arXiv paper link directly (downloads PDF, queries metadata)
uv run graph index https://arxiv.org/abs/1706.03762

# Transcribe a YouTube video locally and index key concepts
uv run graph index "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Index a directory recursively, limiting parallel GPU load
uv run graph index ~/research/ --chunk-pool 2
```

#### Running Queries & Literature Reviews

```bash
# One-shot query
uv run graph query "Explain the core mechanics of LoRA."

# Run a structured literature review and write report to markdown
uv run graph review "RAG optimization techniques" --output review.md
```

#### Reindexing Database

```bash
# Fetch missing authors for all indexed papers
uv run graph reindex meta --missing-authors

# Fully re-chunk and re-embed all documents
uv run graph reindex full --all
```

#### Running the Database Doctor

```bash
# Scan and fix formatting inconsistencies in concept titles and descriptions
uv run graph doctor --fix
```

---

## 🔌 Model Context Protocol (MCP) Server Setup

The built-in MCP server allows developer tools (like Claude Desktop or Cursor) to communicate directly with your Science Graph library.

### 1. Starting the MCP Server

By default, the server runs over standard I/O (stdio), which is ideal for desktop agents:

```bash
uv run graph serve-mcp
```

To run the MCP server over HTTP Server-Sent Events (SSE):

```bash
uv run graph serve-mcp --sse --host 127.0.0.1 --port 8010
```

### 2. Configuring Claude Desktop

Add Science Graph to your `claude_desktop_config.json` (typically located at `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "science-graph": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/vladimirkasterin/python/graph",
        "run",
        "graph",
        "serve-mcp"
      ]
    }
  }
}
```

Now Claude Desktop will have access to tools like `search_papers`, `query_rag`, `get_paper_details`, `create_note`, and `index_file`.

---

## ⚙️ Configuration File

Configuration parameters are loaded from `~/.config/pdf-graph-analyzer/config.yaml`.

```yaml
# Path to the SQLite database file
db_path: "~/.local/share/pdf-graph-analyzer/graph.db"

# Directory where local archives of websites/PDFs are stored
archive_dir: "~/.local/share/pdf-graph-analyzer/archive"

# PDF parser to use: 'marker' (default, layout-aware OCR/Markdown) or 'fitz' (legacy fast PyMuPDF)
pdf_parser: "marker"

# HuggingFace token for downloading gated models/embeddings (optional)
hf_token: ""

# Large Language Model (LLM) configuration
llm:
  # Provider: 'mlx' (local Apple Silicon) or 'openai' (OpenAI / OpenRouter / compatible APIs)
  provider: "mlx"
  max_tokens: 1000
  temp: 0.1

  local:
    model_path: "~/models/llm/gemma-3-text-12b-it-4bit"

  cloud:
    provider: "openai"
    model_name: "google/gemini-2.5-flash"
    api_key: ""
    base_url: "https://openrouter.ai/api/v1"

  # Dynamic input token limits to fit context
  extraction_input_limit: 5000
  clustering_input_limit: 6000
  synthesis_input_limit: 5000

# Embedding model configuration
embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 1000
  chunk_overlap: 200

# spaCy configuration for lemmatization
spacy:
  model_name: "en_core_web_sm"

# Named Entity Recognition model
ner:
  model_name: "dslim/bert-base-NER"

# PDF compression settings to downsample scanned pages
pdf_compression:
  enabled: true
  dpi_threshold: 151
  dpi_target: 150
  quality: 75
```

---

## 🧪 Testing & Current Project State

Science Graph is a robust, local-first research tool. The codebase is backed by **495 automated unit and integration tests** checking:

- Core SQLite repository transactions and neighbor query patterns.
- EPUB, PDF, and Markdown parsing (wikilink parsing, fallback metadata heuristics).
- YouTube audio downloads and Whisper local transcription.
- Hybrid BM25 & USearch vector indexing, query expansions, and RRF blending.
- LLM response limits, input context truncation, and Spacy lemmatization pipelines.
- TUI CLI components and Doctor database sanitizers.

To run the complete test suite:

```bash
uv run pytest
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
