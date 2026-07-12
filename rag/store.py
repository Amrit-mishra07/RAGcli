"""FAISS vector store with persistent JSON metadata sidecar."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from rag.config import (
    DEFAULT_INDEX_DIR,
    FAISS_INDEX_FILE,
    INGEST_HASH_FILE,
    METADATA_FILE,
)

logger = logging.getLogger(__name__)


# ── Chunk metadata entry ──────────────────────────────────────────────

class ChunkMeta:
    """Metadata for a single stored chunk."""
    __slots__ = ("file_path", "start_line", "end_line", "text")

    def __init__(self, file_path: str, start_line: int, end_line: int, text: str):
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.text = text

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChunkMeta":
        return cls(
            file_path=d["file_path"],
            start_line=d["start_line"],
            end_line=d["end_line"],
            text=d["text"],
        )


# ── VectorStore ───────────────────────────────────────────────────────

class VectorStore:
    """Thin wrapper around a FAISS IndexFlatIP and a JSON metadata file."""

    def __init__(self, index_dir: Path | None = None):
        self.index_dir = (index_dir or DEFAULT_INDEX_DIR).resolve()
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._faiss_path = self.index_dir / FAISS_INDEX_FILE
        self._meta_path = self.index_dir / METADATA_FILE
        self._hash_path = self.index_dir / INGEST_HASH_FILE

        self._index: faiss.IndexFlatIP | None = None
        self._metadata: list[ChunkMeta] = []
        self._hashes: dict[str, str] = {}  # file_path -> content_hash

        self._load()

    # ── persistence ────────────────────────────────────────────────

    def _load(self) -> None:
        if self._faiss_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._faiss_path))
            raw = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._metadata = [ChunkMeta.from_dict(d) for d in raw]
            logger.info(
                "Loaded existing index: %d vectors.", self._index.ntotal
            )
        else:
            self._index = None
            self._metadata = []

        if self._hash_path.exists():
            self._hashes = json.loads(
                self._hash_path.read_text(encoding="utf-8")
            )

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        if self._index is None:
            return
        faiss.write_index(self._index, str(self._faiss_path))
        self._meta_path.write_text(
            json.dumps([m.to_dict() for m in self._metadata], ensure_ascii=False),
            encoding="utf-8",
        )
        self._hash_path.write_text(
            json.dumps(self._hashes, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Index saved to %s (%d vectors).", self.index_dir, self._index.ntotal)

    # ── idempotent ingestion helpers ───────────────────────────────

    def file_hash(self, file_path: str, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def needs_update(self, file_path: str, content_hash: str) -> bool:
        """Return True if this file has changed since last ingestion."""
        return self._hashes.get(file_path) != content_hash

    def remove_file_chunks(self, file_path: str) -> None:
        """Remove all chunks for a given file (before re-adding).

        Because FAISS IndexFlatIP doesn't support deletion, we rebuild
        the index without the removed entries.  This is fine at v0 scale.
        """
        if self._index is None or self._index.ntotal == 0:
            return

        keep_indices = [
            i for i, m in enumerate(self._metadata) if m.file_path != file_path
        ]
        if len(keep_indices) == len(self._metadata):
            return  # nothing to remove

        if keep_indices:
            all_vectors = faiss.rev_swig_ptr(
                self._index.get_xb(), self._index.ntotal * self._index.d
            )
            all_vectors = np.array(all_vectors, dtype=np.float32).reshape(
                self._index.ntotal, self._index.d
            )
            kept_vectors = all_vectors[keep_indices]
            self._index = faiss.IndexFlatIP(self._index.d)
            self._index.add(kept_vectors)
        else:
            dim = self._index.d
            self._index = faiss.IndexFlatIP(dim)

        self._metadata = [self._metadata[i] for i in keep_indices]

    def mark_ingested(self, file_path: str, content_hash: str) -> None:
        self._hashes[file_path] = content_hash

    # ── add / search ──────────────────────────────────────────────

    def add(self, vectors: list[list[float]], metas: list[ChunkMeta]) -> None:
        """Add vectors and their metadata to the store."""
        arr = np.array(vectors, dtype=np.float32)
        # L2-normalise so inner-product == cosine similarity
        faiss.normalize_L2(arr)

        if self._index is None:
            dim = arr.shape[1]
            self._index = faiss.IndexFlatIP(dim)

        self._index.add(arr)
        self._metadata.extend(metas)

    def search(self, query_vector: list[float], k: int = 5) -> list[tuple[ChunkMeta, float]]:
        """Return top-k (meta, score) pairs."""
        if self._index is None or self._index.ntotal == 0:
            return []

        arr = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(arr)
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(arr, k)

        results: list[tuple[ChunkMeta, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self._metadata[idx], float(score)))
        return results

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0

    def is_empty(self) -> bool:
        return self.total_vectors == 0
