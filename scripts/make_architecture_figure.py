#!/usr/bin/env python3
"""Architecture figure for the paper (analogue of the CoCa draft's Fig. 1).

One RSU cell, drawn around the serve path of Algorithm 1: vehicles feed the
RSU's cell-state observer with cooperative awareness messages; a request is
mapped to a context-scoped key; the verifier compares the entry's stored
signature against the observed state before anything is served; a failed check
falls through to compute, whose fresh result re-enters the cache. The point the
figure must make visually: the verifier sits *between* the cache and the reply,
and it reads only state the RSU already has.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_SOFT, SURFACE = "#0b0b0b", "#52514e", "#ffffff"

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"


def box(ax, x, y, w, h, label, color, sublabel=None, lw=1.4):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012",
        facecolor=SURFACE, edgecolor=color, linewidth=lw, zorder=3))
    lines = label.count("\n")
    cy = y + h / 2 + ((0.045 if lines else 0.032) if sublabel else 0)
    ax.text(x + w / 2, cy, label, ha="center", va="center",
            fontsize=8.4, color=INK, zorder=4, fontweight="bold")
    if sublabel:
        ax.text(x + w / 2, y + 0.045, sublabel, ha="center",
                va="center", fontsize=7.0, color=INK_SOFT, zorder=4)


def arrow(ax, xy_from, xy_to, color=INK_SOFT, style="-|>", lw=1.3,
          dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, mutation_scale=11,
        linestyle=(0, (4, 3)) if dashed else "solid",
        color=color, linewidth=lw, zorder=2,
        connectionstyle=f"arc3,rad={rad}"))


def label(ax, x, y, text, color=INK_SOFT, size=7.2, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color,
            zorder=5)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- vehicles ----------------------------------------------------------
    for i, ytop in enumerate((0.72, 0.46, 0.20)):
        box(ax, 0.02, ytop, 0.115, 0.14,
            f"Vehicle $v_{i + 1}$", BLUE)
    label(ax, 0.078, 0.945, "vehicles in cell $\\kappa_r$", INK_SOFT, 7.6)

    # --- RSU enclosure -----------------------------------------------------
    ax.add_patch(FancyBboxPatch(
        (0.235, 0.05), 0.72, 0.90, boxstyle="round,pad=0.012",
        facecolor="#f4f3f0", edgecolor=INK_SOFT, linewidth=1.1, zorder=1))
    label(ax, 0.595, 0.905, "RSU $r$  (fog server)", INK, 8.8)

    # components inside the RSU
    box(ax, 0.26, 0.54, 0.155, 0.26, "Cell-state\nobserver",
        AQUA, "$S_r$ from CAMs")
    box(ax, 0.26, 0.14, 0.155, 0.22, "Key\nconstruction",
        BLUE, "$k=H(i,\\,g_i(x,y))$")
    box(ax, 0.47, 0.14, 0.145, 0.22, "Cache",
        YELLOW, "entries $\\langle e,\\sigma_e\\rangle$")
    box(ax, 0.665, 0.14, 0.135, 0.22, "Verifier",
        ORANGE, "$\\sigma_e \\approx S_r$?", lw=1.8)
    box(ax, 0.665, 0.56, 0.135, 0.22, "Compute",
        INK_SOFT, "$D_\\mathrm{comp}$")

    # --- flows -------------------------------------------------------------
    # CAM beacons feed the observer
    for ytop in (0.79, 0.53, 0.27):
        arrow(ax, (0.135, ytop), (0.258, 0.66), color=AQUA, dashed=True, lw=1.0)
    label(ax, 0.176, 0.585, "CAM\nbeacons", AQUA, 6.8)

    # request path
    arrow(ax, (0.135, 0.25), (0.258, 0.25), color=BLUE)
    label(ax, 0.196, 0.315, "request\n$\\langle i,x,y,t,v\\rangle$", BLUE, 6.8)
    arrow(ax, (0.415, 0.25), (0.468, 0.25), color=BLUE)
    arrow(ax, (0.615, 0.25), (0.663, 0.25), color=YELLOW)
    label(ax, 0.639, 0.205, "entry", INK_SOFT, 6.8)

    # observer feeds the verifier
    arrow(ax, (0.415, 0.63), (0.663, 0.315), color=AQUA, rad=-0.22)
    label(ax, 0.512, 0.455, "observed state $S_r$", AQUA, 7.0)

    # verified answer back to the vehicle
    arrow(ax, (0.732, 0.135), (0.078, 0.185), color=ORANGE, rad=0.10)
    label(ax, 0.435, 0.075, "verified answer (only correct results served)",
          ORANGE, 7.2)

    # failed verification -> compute -> cache
    arrow(ax, (0.732, 0.36), (0.732, 0.555), color=INK_SOFT)
    label(ax, 0.815, 0.46, "fail:\nrecompute", INK_SOFT, 6.8)
    arrow(ax, (0.663, 0.70), (0.543, 0.368), color=INK_SOFT, rad=0.25)
    label(ax, 0.578, 0.71, "store $\\langle res,\\sigma_e{=}S_r\\rangle$",
          INK_SOFT, 6.8)

    fig.tight_layout(pad=0.25)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_architecture.pdf")
    fig.savefig(OUT / "fig_architecture.png", dpi=200)
    print(f"wrote {OUT / 'fig_architecture.pdf'}")


if __name__ == "__main__":
    main()
