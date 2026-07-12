"""Split file content into overlapping chunks with code-aware heuristics."""

import re
from typing import NamedTuple

from rag.config import (
    APPROX_CHARS_PER_TOKEN,
    CHUNK_MIN_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
)

# Patterns that indicate a logical boundary in source code.
_BOUNDARY_RE = re.compile(
    r"^"
    r"(?:"
    r"(?:export\s+)?(?:async\s+)?function\s"
    r"|def\s"
    r"|class\s"
    r"|public\s|private\s|protected\s"
    r"|impl\s"
    r"|fn\s"
    r"|func\s"
    r"|package\s"
    r"|module\s"
    r")",
    re.MULTILINE,
)


class Chunk(NamedTuple):
    """A text chunk with provenance metadata."""
    file_path: str
    start_line: int  # 1-indexed
    end_line: int    # 1-indexed, inclusive
    text: str


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // APPROX_CHARS_PER_TOKEN)


def _find_split_point(lines: list[str], target_idx: int) -> int:
    """Search around *target_idx* for a code boundary or blank line.

    Returns the best line index to split *before*.
    """
    window = max(10, len(lines) // 20)  # search ±window lines
    best: int | None = None
    best_dist = window + 1

    lo = max(0, target_idx - window)
    hi = min(len(lines), target_idx + window)

    for i in range(lo, hi):
        line = lines[i]
        dist = abs(i - target_idx)
        # Prefer code boundaries first, then blank lines
        is_boundary = bool(_BOUNDARY_RE.match(line))
        is_blank = line.strip() == ""
        if is_boundary and dist < best_dist:
            best = i
            best_dist = dist
        elif is_blank and best is None and dist < best_dist:
            best = i
            best_dist = dist

    return best if best is not None else target_idx


def chunk_file(file_path: str, content: str) -> list[Chunk]:
    """Split *content* into chunks respecting code boundaries."""
    lines = content.splitlines(keepends=True)
    if not lines:
        return []

    target_chars = CHUNK_TARGET_TOKENS * APPROX_CHARS_PER_TOKEN
    overlap_chars = CHUNK_OVERLAP_TOKENS * APPROX_CHARS_PER_TOKEN
    min_chars = CHUNK_MIN_TOKENS * APPROX_CHARS_PER_TOKEN

    chunks: list[Chunk] = []
    start = 0  # current chunk start (line index)

    while start < len(lines):
        # Accumulate lines until we hit the target size
        char_count = 0
        end = start
        while end < len(lines) and char_count < target_chars:
            char_count += len(lines[end])
            end += 1

        # If we haven't consumed all lines, try to find a nice split point
        if end < len(lines):
            split_at = _find_split_point(lines, end)
            # Don't make tiny chunks
            if split_at > start and (split_at - start) * APPROX_CHARS_PER_TOKEN >= min_chars:
                end = split_at

        chunk_text = "".join(lines[start:end])
        if chunk_text.strip():  # skip empty chunks
            chunks.append(Chunk(
                file_path=file_path,
                start_line=start + 1,
                end_line=end,  # end is exclusive idx, so end == last line 1-indexed
                text=chunk_text,
            ))

        # Advance with overlap
        overlap_lines = 0
        overlap_counted = 0
        for i in range(end - 1, start - 1, -1):
            overlap_counted += len(lines[i])
            overlap_lines += 1
            if overlap_counted >= overlap_chars:
                break

        next_start = max(start + 1, end - overlap_lines)
        if next_start <= start:
            next_start = start + 1  # guarantee forward progress
        start = next_start

    return chunks
