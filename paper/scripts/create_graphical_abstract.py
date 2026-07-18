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
        ((0.025, 0.54), 0.14, 0.30, "Observations\nflight motion\nradio graph", "#e7f0fb"),
        ((0.19, 0.54), 0.17, 0.30, "Candidate predictors\ncurrent state | topology\nmotion | source gated", "#f4eddf"),
        ((0.385, 0.54), 0.15, 0.30, "Leakage controls\ntrain/validation selection\nlocked seeds", "#e8f4ea"),
        ((0.56, 0.54), 0.13, 0.30, "Warning output\nevent risk\nvalidated threshold", "#efe8fb"),
        ((0.715, 0.54), 0.13, 0.30, "Engineering tests\ndomain | packet\ncontroller | runtime", "#dff3f2"),
        ((0.87, 0.54), 0.105, 0.30, "Evidence\ndecision\nclaim boundary", "#f9e4e2"),
    ]
    for xy, width, height, label, face in boxes:
        add_box(ax, xy, width, height, label, face)

    add_arrow(ax, (0.165, 0.69), (0.19, 0.69))
    add_arrow(ax, (0.36, 0.69), (0.385, 0.69))
    add_arrow(ax, (0.535, 0.69), (0.56, 0.69))
    add_arrow(ax, (0.69, 0.69), (0.715, 0.69))
    add_arrow(ax, (0.845, 0.69), (0.87, 0.69))

    ax.text(
        0.5,
        0.18,
        "Reproducible benchmark: richer sources do not consistently beat the current-state comparator",
        ha="center",
        va="center",
        fontsize=12,
        color="#17202a",
        weight="bold",
    )
    ax.text(
        0.5,
        0.10,
        "Locked paired tests | source factorial | domain transfer | packet/controller cost | measured host-side runtime",
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
