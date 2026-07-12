"""Ollama embedding client."""

import logging
from typing import Sequence

import ollama

from rag.config import EMBEDDING_MODEL_PREFERENCES, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

_client: ollama.Client | None = None
_selected_model: str | None = None


def _get_client() -> ollama.Client:
    global _client
    if _client is None:
        _client = ollama.Client(host=OLLAMA_BASE_URL)
    return _client


def _check_ollama_available() -> None:
    """Raise a clear error if Ollama is unreachable."""
    try:
        _get_client().list()
    except Exception as exc:
        raise ConnectionError(
            "Cannot connect to Ollama. Is it running?\n"
            "  Start it with: ollama serve\n"
            f"  (tried {OLLAMA_BASE_URL})"
        ) from exc


def _resolve_embedding_model() -> str:
    """Pick the first available embedding model from the preference list."""
    global _selected_model
    if _selected_model is not None:
        return _selected_model

    _check_ollama_available()
    client = _get_client()
    available = {m.model.split(":")[0] for m in client.list().models}
    # Also check with full tag
    available_full = {m.model for m in client.list().models}

    for pref in EMBEDDING_MODEL_PREFERENCES:
        if pref in available or pref in available_full:
            _selected_model = pref
            logger.info("Using embedding model: %s", pref)
            return pref

    # None available — give actionable error
    cmds = "\n".join(f"  ollama pull {m}" for m in EMBEDDING_MODEL_PREFERENCES)
    raise RuntimeError(
        f"No embedding model found. Pull one with:\n{cmds}"
    )


def get_embedding_model_name() -> str:
    """Return the name of the selected embedding model."""
    return _resolve_embedding_model()


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch of texts via Ollama. Returns a list of float vectors."""
    model = _resolve_embedding_model()
    client = _get_client()
    vectors: list[list[float]] = []
    for text in texts:
        resp = client.embed(model=model, input=text)
        vectors.append(resp.embeddings[0])
    return vectors


def embed_single(text: str) -> list[float]:
    """Embed a single text string."""
    return embed_texts([text])[0]
