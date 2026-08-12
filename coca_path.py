"""Resolve the sibling CoCa source tree and put it on ``sys.path``.

Kept deliberately dependency-free and at the project root so that any module —
including ``intent.classifier``, which is imported *by* ``coca_adapter`` — can
use it without creating an import cycle.

Set ``COCA_ROOT`` if the two repositories are not siblings.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_COCA_ROOT = PROJECT_ROOT.parent / "sdn-vanet-optimizer"
COCA_ROOT = Path(os.environ.get("COCA_ROOT", DEFAULT_COCA_ROOT)).expanduser().resolve()


def bootstrap_coca() -> Path:
    """Add the CoCa source tree to ``sys.path`` and return its path."""
    if not (COCA_ROOT / "cache" / "distributed_manager.py").exists():
        raise FileNotFoundError(
            f"CoCa implementation not found at {COCA_ROOT}. "
            "Set COCA_ROOT to the sdn-vanet-optimizer directory."
        )
    root = str(COCA_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return COCA_ROOT


bootstrap_coca()

__all__ = ["COCA_ROOT", "PROJECT_ROOT", "bootstrap_coca"]
