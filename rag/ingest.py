"""Walk a repository and yield text file contents with metadata."""

import logging
from pathlib import Path
from typing import Iterator, NamedTuple

from rag.config import IGNORE_DIRS, IGNORE_EXTENSIONS, MAX_FILE_SIZE_BYTES

logger = logging.getLogger(__name__)


class FileContent(NamedTuple):
    """A single file's text content with its repo-relative path."""
    path: str        # repo-relative, forward-slash
    content: str


def _should_ignore_dir(name: str) -> bool:
    return name in IGNORE_DIRS


def _should_ignore_file(path: Path) -> bool:
    return path.suffix.lower() in IGNORE_EXTENSIONS


def _is_text_file(path: Path) -> bool:
    """Quick heuristic: read first 8 KB looking for null bytes."""
    try:
        chunk = path.read_bytes()[:8192]
        return b"\x00" not in chunk
    except OSError:
        return False


def walk_repo(repo_path: Path) -> Iterator[FileContent]:
    """Yield FileContent for every indexable text file in *repo_path*."""
    repo_path = repo_path.resolve()
    if not repo_path.is_dir():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    file_count = 0
    skipped = 0

    for item in sorted(repo_path.rglob("*")):
        # Skip ignored directories — check all parent components
        parts = item.relative_to(repo_path).parts
        if any(_should_ignore_dir(p) for p in parts):
            continue

        if not item.is_file():
            continue

        if _should_ignore_file(item):
            skipped += 1
            continue

        # Size guard
        try:
            size = item.stat().st_size
        except OSError:
            skipped += 1
            continue

        if size > MAX_FILE_SIZE_BYTES:
            logger.warning("Skipping large file (%d bytes): %s", size, item)
            skipped += 1
            continue

        if size == 0:
            continue

        # Binary guard
        if not _is_text_file(item):
            skipped += 1
            continue

        # Read content
        try:
            content = item.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", item, exc)
            skipped += 1
            continue

        rel = item.relative_to(repo_path).as_posix()
        file_count += 1
        yield FileContent(path=rel, content=content)

    logger.info(
        "File walk complete: %d files indexed, %d skipped.", file_count, skipped
    )
