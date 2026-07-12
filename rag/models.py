"""Detect available resources and select the best generation model."""

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

import ollama

from rag.config import GENERATION_MODEL_PREFERENCES, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


def _get_total_ram_mb() -> int:
    """Return total system RAM in MB (Linux / macOS)."""
    try:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return mem // (1024 * 1024)
    except (ValueError, OSError):
        pass
    # Fallback: read /proc/meminfo
    try:
        text = Path("/proc/meminfo").read_text()
        match = re.search(r"MemTotal:\s+(\d+)\s+kB", text)
        if match:
            return int(match.group(1)) // 1024
    except OSError:
        pass
    return 0


def _get_vram_mb() -> int:
    """Attempt to detect NVIDIA VRAM via nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return sum(int(line.strip()) for line in out.strip().splitlines() if line.strip())
    except (subprocess.SubprocessError, ValueError):
        return 0


def select_generation_model() -> str:
    """Choose the best generation model the system can handle.

    Strategy:
    1. Compute usable memory = max(VRAM, total_RAM).
    2. Walk GENERATION_MODEL_PREFERENCES (largest first).
    3. Pick the biggest model whose min_MB fits AND is already pulled.
    4. If none are pulled, pick the biggest that fits and tell the user to pull it.
    """
    vram = _get_vram_mb()
    ram = _get_total_ram_mb()
    usable = max(vram, ram)
    mem_source = "VRAM" if vram >= ram else "RAM"

    logger.info("Detected %d MB VRAM, %d MB RAM → using %d MB (%s).", vram, ram, usable, mem_source)

    client = ollama.Client(host=OLLAMA_BASE_URL)
    try:
        available = {m.model for m in client.list().models}
    except Exception as exc:
        raise ConnectionError(
            "Cannot connect to Ollama. Is it running?\n"
            f"  Start it with: ollama serve\n  (tried {OLLAMA_BASE_URL})"
        ) from exc

    # Normalise available names — Ollama may list "qwen2.5-coder:7b" or with a hash
    available_short = set()
    for name in available:
        available_short.add(name)
        available_short.add(name.split(":")[0])  # base name without tag

    best_fits: str | None = None  # biggest that fits memory
    best_available: str | None = None  # biggest that fits AND is pulled

    for model_tag, min_mb in GENERATION_MODEL_PREFERENCES:
        if usable < min_mb:
            continue
        if best_fits is None:
            best_fits = model_tag
        if model_tag in available_short:
            if best_available is None:
                best_available = model_tag

    if best_available:
        print(
            f"✓ Generation model: {best_available}"
            f"  (selected for {usable} MB {mem_source})"
        )
        return best_available

    if best_fits:
        raise RuntimeError(
            f"Model '{best_fits}' would fit your {usable} MB {mem_source}, "
            f"but it isn't pulled yet.\n"
            f"  Run: ollama pull {best_fits}"
        )

    # Nothing fits at all
    smallest = GENERATION_MODEL_PREFERENCES[-1]
    raise RuntimeError(
        f"Not enough memory ({usable} MB) for even the smallest model "
        f"({smallest[0]}, needs {smallest[1]} MB).\n"
        f"  You can still try: ollama pull {smallest[0]}"
    )
