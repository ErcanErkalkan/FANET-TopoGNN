from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "plantuml"


def add_box(ax, xy, width, height, label, face, edge="#24323f"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=10.5,
        color="#101820",
        linespacing=1.2,
    )
    return box


def add_arrow(ax, start, end, color="#3b4652"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.35,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.28, 5.31))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        ((0.035, 0.56), 0.16, 0.28, "UAV motion\npositions + velocities\nphysical graph", "#e7f0fb"),
        ((0.235, 0.72), 0.16, 0.20, "Projected\ntopology", "#f4eddf"),
        ((0.235, 0.36), 0.16, 0.20, "H0 persistence\nimage features", "#e8f4ea"),
        ((0.445, 0.54), 0.16, 0.28, "Motion-conditioned\ntopological AI\nresidual forecast", "#efe8fb"),
        ((0.655, 0.72), 0.14, 0.20, "Future beta0\nfragmentation\nstate", "#f9e4e2"),
        ((0.655, 0.36), 0.14, 0.20, "Risk score\nwarning threshold", "#dff3f2"),
        ((0.835, 0.54), 0.13, 0.28, "Engineering\nwarning/action\nrelay + routing", "#edf6df"),
    ]
    for xy, width, height, label, face in boxes:
        add_box(ax, xy, width, height, label, face)

    add_arrow(ax, (0.195, 0.70), (0.235, 0.82))
    add_arrow(ax, (0.195, 0.66), (0.235, 0.46))
    add_arrow(ax, (0.395, 0.82), (0.445, 0.69))
    add_arrow(ax, (0.395, 0.46), (0.445, 0.62))
    add_arrow(ax, (0.605, 0.68), (0.655, 0.82))
    add_arrow(ax, (0.605, 0.61), (0.655, 0.46))
    add_arrow(ax, (0.795, 0.82), (0.835, 0.70))
    add_arrow(ax, (0.795, 0.46), (0.835, 0.62))

    ax.text(
        0.5,
        0.18,
        "Goal: short-horizon, interpretable fragmentation-risk forecasting for proactive UAV-network connectivity management",
        ha="center",
        va="center",
        fontsize=12,
        color="#17202a",
        weight="bold",
    )
    ax.text(
        0.5,
        0.10,
        "Validation outputs: implemented baselines, MAE/R2, risk-F1, lead-time, statistical tests, and network-level metrics",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#334155",
    )

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT_DIR / "graphical_abstract.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "graphical_abstract.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
