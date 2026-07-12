# RAG CLI

A local, open-source Retrieval-Augmented Generation CLI tool that lets you ask natural-language questions about a codebase and get grounded answers with source citations.

Everything runs **locally** — no paid APIs, no cloud services. Powered by [Ollama](https://ollama.com/) for both embeddings and generation, and [FAISS](https://github.com/facebookresearch/faiss) for vector search.

## Features

- **Code-aware chunking** — splits files on function/class boundaries where possible, not just arbitrary token counts.
- **Persistent index** — FAISS index + metadata stored on disk; survives process restarts.
- **Idempotent ingestion** — re-running `rag index` skips unchanged files.
- **Streaming answers** — LLM output streams to your terminal in real-time.
- **Source citations** — every answer includes exact file paths and line ranges.
- **Automatic model selection** — detects available VRAM/RAM and picks the largest Qwen 2.5 Coder variant your system can handle.

## Setup

### Prerequisites

- **Python 3.11+**
- **Ollama** — install from [ollama.com](https://ollama.com/)

### 1. Install Ollama and pull models

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Start the Ollama server (if not already running)
ollama serve &

# Pull the required models
ollama pull nomic-embed-text        # embedding model (~274 MB)
ollama pull qwen2.5-coder:7b       # generation model (~4.7 GB)

# If your machine has < 8 GB RAM/VRAM, use a smaller variant:
# ollama pull qwen2.5-coder:3b     # ~2.0 GB
# ollama pull qwen2.5-coder:1.5b   # ~1.0 GB
```

### 2. Install RAG CLI

```bash
# Clone the repo
git clone <repo-url> && cd RAGcli

# Install in a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

## Usage

### Index a repository

```bash
rag index /path/to/your/codebase
```

This will:
1. Walk the directory, skipping binary files, `.git`, `node_modules`, build artifacts, etc.
2. Split each text/code file into overlapping chunks (~300-500 tokens).
3. Embed all chunks via the local Ollama embedding model.
4. Store the FAISS index and metadata in `~/.rag_index/`.

Progress is logged as files are processed. Re-running the command on the same repo skips unchanged files.

### Ask a question

```bash
rag ask "How does the authentication middleware work?"
```

Options:
- `--k N` — number of source chunks to retrieve (default: 5)

```bash
rag ask "What does the parse_config function do?" --k 10
```

### Other options

```bash
rag --help           # show top-level help
rag index --help     # help for the index command
rag ask --help       # help for the ask command
rag -v index ./src   # verbose/debug logging
```

### Running without installing

```bash
# Alternative: run directly as a Python module
python -m rag index /path/to/repo
python -m rag ask "What does main() do?"
```

## Example Session

```
$ rag index ~/projects/my-api

Embedding model : nomic-embed-text
Index directory : /home/user/.rag_index
Repo            : /home/user/projects/my-api

Scanning repository...
Found 142 files to process.

  [10/142] src/routes/auth.ts  (3 chunks)
  [20/142] src/middleware/cors.ts  (1 chunks)
  ...
  [142/142] tests/unit/auth.test.ts  (4 chunks)

Done in 38.2s. 142 files indexed, 0 unchanged, 487 chunks stored (487 total vectors).

$ rag ask "How does JWT validation work in the auth middleware?"

✓ Generation model: qwen2.5-coder:7b  (selected for 16384 MB RAM)

────────────────────────────────────────────────────────
Answer:
────────────────────────────────────────────────────────
The JWT validation is handled in `src/middleware/auth.ts`. The `validateToken`
function (lines 24-41) extracts the Bearer token from the Authorization header,
then calls `jwt.verify()` with the secret from `config.JWT_SECRET`. If
verification fails, it returns a 401 response...

────────────────────────────────────────────────────────
Sources:
  [1] src/middleware/auth.ts  (lines 18-52, similarity 0.847)
  [2] src/config/index.ts  (lines 1-28, similarity 0.721)
  [3] src/routes/auth.ts  (lines 55-89, similarity 0.698)
  [4] tests/unit/auth.test.ts  (lines 1-34, similarity 0.654)
  [5] src/types/auth.ts  (lines 1-19, similarity 0.612)
────────────────────────────────────────────────────────
```

## Architecture

```
rag/
├── cli.py          # argparse CLI entry point
├── config.py       # constants, paths, ignore patterns
├── ingest.py       # repo walker — yields text files with metadata
├── chunker.py      # code-aware chunking with overlap
├── embeddings.py   # Ollama embedding client
├── store.py        # FAISS index + JSON metadata persistence
├── models.py       # VRAM/RAM detection + model selection
└── query.py        # retrieve → prompt → generate pipeline
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Cannot connect to Ollama` | Run `ollama serve` in another terminal |
| `No embedding model found` | Run `ollama pull nomic-embed-text` |
| `Model '...' isn't pulled yet` | Run the `ollama pull <model>` command shown in the error |
| `No index found` | Run `rag index <repo_path>` before `rag ask` |
| Index seems stale | Delete `~/.rag_index/` and re-index |

## Next Steps (v1 — out of scope for v0)

The following improvements are planned but **deliberately not implemented** in this version:

- **Hybrid search** — combine vector similarity with BM25 keyword matching for better recall.
- **Reranking** — use a cross-encoder reranker on the top-k results before passing to the LLM.
- **Incremental re-indexing** — detect file changes at the chunk level instead of re-processing entire files.
- **AST-aware chunking** — use tree-sitter for language-aware parsing instead of regex heuristics.
- **Knowledge graph** — extract and link entities (functions, classes, imports) across files.
- **Multi-repo support** — index and query across multiple repositories.
- **Cloud deployment** — serve the index and LLM behind an API for team usage.
- **Conversation memory** — support follow-up questions within a session.

## License

MIT
