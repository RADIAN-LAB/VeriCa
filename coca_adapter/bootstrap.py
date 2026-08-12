"""Locate the sibling CoCa implementation and expose it for imports.

Thin re-export of :mod:`coca_path`, which holds the actual resolution logic at
the project root so that modules imported *by* this package (notably
``intent.classifier``) can bootstrap without an import cycle.
"""

from __future__ import annotations

from coca_path import COCA_ROOT, PROJECT_ROOT, bootstrap_coca

bootstrap_coca()

__all__ = ["COCA_ROOT", "PROJECT_ROOT", "bootstrap_coca"]
