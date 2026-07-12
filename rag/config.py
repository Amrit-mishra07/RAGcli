"""Configuration constants and default settings."""

from pathlib import Path

# --- Index storage ---
DEFAULT_INDEX_DIR = Path.home() / ".rag_index"
FAISS_INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"
INGEST_HASH_FILE = "ingest_hashes.json"

# --- Chunking ---
CHUNK_TARGET_TOKENS = 400
CHUNK_MIN_TOKENS = 100
CHUNK_OVERLAP_TOKENS = 50
APPROX_CHARS_PER_TOKEN = 4  # rough estimate for code

# --- Retrieval ---
DEFAULT_TOP_K = 5

# --- File limits ---
MAX_FILE_SIZE_BYTES = 1_000_000  # 1 MB

# --- Ollama ---
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL_PREFERENCES = ["nomic-embed-text", "bge-small-en"]
GENERATION_MODEL_PREFERENCES = [
    ("qwen2.5-coder:7b", 8_000),   # (model_tag, min_MB_ram)
    ("qwen2.5-coder:3b", 4_000),
    ("qwen2.5-coder:1.5b", 2_000),
    ("qwen2.5-coder:0.5b", 1_000),
]

# --- Ignore patterns ---
IGNORE_DIRS: set[str] = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "vendor",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "target",       # Rust / Java
    "out",
    ".venv",
    "venv",
    "env",
    ".tox",
    ".eggs",
    "*.egg-info",
    ".gradle",
    ".idea",
    ".vscode",
    ".rag_index",
}

IGNORE_EXTENSIONS: set[str] = {
    # Binary / compiled
    ".pyc", ".pyo", ".so", ".o", ".a", ".dylib", ".dll", ".exe",
    ".class", ".jar", ".war",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    # Images / media
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac",
    ".webp", ".webm",
    # Fonts
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    # Data blobs
    ".bin", ".dat", ".db", ".sqlite", ".sqlite3",
    ".pkl", ".pickle", ".npy", ".npz",
    # PDFs / docs
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    # Lock files
    ".lock",
    # FAISS index
    ".faiss",
}
