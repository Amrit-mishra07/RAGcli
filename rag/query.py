"""Query pipeline: embed question → retrieve chunks → generate answer."""

import logging

import ollama

from rag.config import DEFAULT_TOP_K, OLLAMA_BASE_URL
from rag.embeddings import embed_single
from rag.models import select_generation_model
from rag.store import ChunkMeta, VectorStore

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a helpful code assistant. Answer the user's question based ONLY on the provided source code context. If the context doesn't contain enough information, say so clearly. Always reference the file paths and line numbers when discussing code."""


def _build_context(chunks: list[tuple[ChunkMeta, float]]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    parts: list[str] = []
    for i, (meta, score) in enumerate(chunks, 1):
        header = f"[Source {i}] {meta.file_path} (lines {meta.start_line}-{meta.end_line})"
        parts.append(f"{header}\n{meta.text}")
    return "\n\n---\n\n".join(parts)


def _format_sources(chunks: list[tuple[ChunkMeta, float]]) -> str:
    lines: list[str] = []
    for i, (meta, score) in enumerate(chunks, 1):
        lines.append(
            f"  [{i}] {meta.file_path}  (lines {meta.start_line}-{meta.end_line}, "
            f"similarity {score:.3f})"
        )
    return "\n".join(lines)


def ask(
    question: str,
    store: VectorStore,
    k: int = DEFAULT_TOP_K,
) -> None:
    """Run the full RAG query pipeline and print the answer."""
    if store.is_empty():
        raise RuntimeError(
            "No index found. Run 'rag index <repo_path>' first to build the index."
        )

    # 1. Embed the question
    query_vec = embed_single(question)

    # 2. Retrieve top-k chunks
    results = store.search(query_vec, k=k)
    if not results:
        print("No relevant chunks found.")
        return

    # 3. Build prompt
    context = _build_context(results)
    user_message = (
        f"Context from the codebase:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}"
    )

    # 4. Generate answer
    model = select_generation_model()
    client = ollama.Client(host=OLLAMA_BASE_URL)

    print()  # blank line before answer
    print("─" * 60)
    print("Answer:")
    print("─" * 60)

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        stream=True,
    )

    for chunk in response:
        token = chunk.message.content
        if token:
            print(token, end="", flush=True)

    print()  # newline after streamed response
    print()
    print("─" * 60)
    print("Sources:")
    print(_format_sources(results))
    print("─" * 60)
