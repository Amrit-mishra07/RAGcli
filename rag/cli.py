"""Command-line interface for the RAG tool."""

import argparse
import logging
import sys
import time
from pathlib import Path

from rag.chunker import chunk_file
from rag.config import DEFAULT_INDEX_DIR, DEFAULT_TOP_K
from rag.embeddings import embed_texts, get_embedding_model_name
from rag.ingest import walk_repo
from rag.query import ask
from rag.store import ChunkMeta, VectorStore

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Index command ─────────────────────────────────────────────────────

def cmd_index(args: argparse.Namespace) -> None:
    repo_path = Path(args.repo_path).resolve()
    if not repo_path.is_dir():
        print(f"Error: '{repo_path}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    store = VectorStore()
    embedding_model = get_embedding_model_name()
    print(f"Embedding model : {embedding_model}")
    print(f"Index directory : {store.index_dir}")
    print(f"Repo            : {repo_path}")
    print()

    # Collect files first for progress reporting
    print("Scanning repository...")
    files = list(walk_repo(repo_path))
    total = len(files)

    if total == 0:
        print("No indexable files found.")
        return

    print(f"Found {total} files to process.\n")

    start_time = time.time()
    chunks_added = 0
    files_skipped = 0
    files_updated = 0

    BATCH_SIZE = 32  # embed in small batches for progress feedback

    for file_idx, fc in enumerate(files, 1):
        content_hash = store.file_hash(fc.path, fc.content)

        if not store.needs_update(fc.path, content_hash):
            files_skipped += 1
            if file_idx % 50 == 0 or file_idx == total:
                print(f"  [{file_idx}/{total}] (skipped unchanged) {fc.path}")
            continue

        # File changed or new — re-chunk
        store.remove_file_chunks(fc.path)
        chunks = chunk_file(fc.path, fc.content)

        if not chunks:
            store.mark_ingested(fc.path, content_hash)
            continue

        # Embed in batches
        texts = [c.text for c in chunks]
        all_vectors: list[list[float]] = []
        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch = texts[batch_start : batch_start + BATCH_SIZE]
            all_vectors.extend(embed_texts(batch))

        metas = [
            ChunkMeta(
                file_path=c.file_path,
                start_line=c.start_line,
                end_line=c.end_line,
                text=c.text,
            )
            for c in chunks
        ]
        store.add(all_vectors, metas)
        store.mark_ingested(fc.path, content_hash)
        chunks_added += len(chunks)
        files_updated += 1

        if file_idx % 10 == 0 or file_idx == total:
            print(f"  [{file_idx}/{total}] {fc.path}  ({len(chunks)} chunks)")

    store.save()
    elapsed = time.time() - start_time
    print(
        f"\nDone in {elapsed:.1f}s. "
        f"{files_updated} files indexed, {files_skipped} unchanged, "
        f"{chunks_added} chunks stored ({store.total_vectors} total vectors)."
    )


# ── Ask command ───────────────────────────────────────────────────────

def cmd_ask(args: argparse.Namespace) -> None:
    store = VectorStore()
    if store.is_empty():
        print(
            "Error: No index found. Run 'rag index <repo_path>' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        ask(args.question, store, k=args.k)
    except ConnectionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="RAG CLI — ask natural-language questions about a codebase.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    sub = parser.add_subparsers(dest="command")

    # rag index <repo_path>
    p_index = sub.add_parser("index", help="Index a local code repository.")
    p_index.add_argument("repo_path", help="Path to the repository to index.")

    # rag ask "<question>" [--k N]
    p_ask = sub.add_parser("ask", help="Ask a question about the indexed repo.")
    p_ask.add_argument("question", help="Natural-language question.")
    p_ask.add_argument(
        "--k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K}).",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "index":
            cmd_index(args)
        elif args.command == "ask":
            cmd_ask(args)
    except ConnectionError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
