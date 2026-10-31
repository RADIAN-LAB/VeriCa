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
# The public repository is RADIAN-LAB/CoCa; older checkouts use its former name.
_SIBLINGS = [PROJECT_ROOT.parent / name for name in ("CoCa", "sdn-vanet-optimizer")]
DEFAULT_COCA_ROOT = next((p for p in _SIBLINGS if p.exists()), _SIBLINGS[0])
COCA_ROOT = Path(os.environ.get("COCA_ROOT", DEFAULT_COCA_ROOT)).expanduser().resolve()


def bootstrap_coca() -> Path:
    """Add the CoCa source tree to ``sys.path`` and return its path."""
    if not (COCA_ROOT / "cache" / "distributed_manager.py").exists():
        raise FileNotFoundError(
            f"CoCa implementation not found at {COCA_ROOT}. "
            "Clone RADIAN-LAB/CoCa next to this repository or set COCA_ROOT."
        )
    root = str(COCA_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return COCA_ROOT


bootstrap_coca()

__all__ = ["COCA_ROOT", "PROJECT_ROOT", "bootstrap_coca"]
