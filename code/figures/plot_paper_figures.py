#!/usr/bin/env python3
"""Rebuild result figures that appear in the paper.

Generated figures:
- category_distribution_pie.{pdf,png}
- category_bacc_boxplot.{pdf,png}
- batch_size_trend.{pdf,png}
- self_voting_gain.{pdf,png}

The non-result schematic figures in the paper (pipeline/construction diagrams) are
not generated here.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ruverbench-matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ruverbench-xdg-cache"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

RESULTS = ROOT / "results"
OUT = RESULTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    ("kimi-k2.6", "Kimi K2.6"),
    ("deepseek-v4-flash", "DeepSeek V4 Flash"),
    ("glm-5.1", "GLM-5.1"),
    ("doubao-seed-2.0-pro", "Seed 2.0 Pro"),
    ("gpt-oss-120b", "GPT-OSS-120B"),
    ("qwen3.5-27b", "Qwen3.5-27B"),
]
COLORS = {
    "kimi-k2.6": "#3A8B5A",
    "deepseek-v4-flash": "#B84E3F",
    "glm-5.1": "#3E79B8",
    "doubao-seed-2.0-pro": "#C9952E",
    "gpt-oss-120b": "#7557A6",
    "qwen3.5-27b": "#7B6B5C",
}
MARKERS = {
    "kimi-k2.6": "o",
    "deepseek-v4-flash": "s",
    "glm-5.1": "D",
    "doubao-seed-2.0-pro": "^",
    "gpt-oss-120b": "P",
    "qwen3.5-27b": "v",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def configure() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
    })


def save(fig, stem: str, dpi: int = 300) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def plot_category_distribution() -> None:
    candidates = [
        RESULTS / "dataset" / "ruverbench_category_distribution.csv",
        RESULTS / "dataset" / "category_distribution.csv",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError("Expected ruverbench_category_distribution.csv under results/dataset/.")
    rows = read_csv(source)
    by_domain: dict[str, list[tuple[str, int]]] = {"DeepResearch": [], "AgenticCoding": []}
    for row in rows:
        domain = row.get("domain", row.get("Domain", ""))
        if domain not in by_domain:
            continue
        category = row.get("category", row.get("Category", ""))
        category = {
            "format": "Format",
            "numbers": "Numbers",
            "logic": "Logic",
            "facts": "Facts",
            "task": "Task",
            "planning": "Planning",
            "tools": "Tools",
            "rules": "Rules",
        }.get(category, category)
        count = int(float(row.get("count", row.get("n", row.get("rubrics", row.get("Total", "0"))))))
        by_domain[domain].append((category, count))

    # Stable paper order in case CSV order changes.
    order = {
        "DeepResearch": ["Format", "Numbers", "Logic", "Facts"],
        "AgenticCoding": ["Task", "Planning", "Tools", "Rules"],
    }
    colors = {
        "Format": "#4F7CAC", "Numbers": "#D9922E", "Logic": "#62AFA7", "Facts": "#C95158",
        "Task": "#4F7CAC", "Planning": "#D9922E", "Tools": "#62AFA7", "Rules": "#C95158",
    }
    text = "#2F3745"
    gray = "#6B7280"
    grid = "#E7EAF0"

    fig, ax = plt.subplots(figsize=(3.45, 1.72), dpi=300)
    ys = {"DeepResearch": 0.95, "AgenticCoding": 0.25}
    for domain in ("DeepResearch", "AgenticCoding"):
        values = dict(by_domain[domain])
        data = [(cat, values[cat]) for cat in order[domain]]
        total = sum(v for _, v in data)
        y = ys[domain]
        left = 0.0
        height = 0.18
        ax.text(0, y + 0.24, domain, ha="left", va="bottom", fontsize=8.5, color=text)
        ax.text(100, y + 0.24, f"{total:,} rubrics", ha="right", va="bottom", fontsize=6.9, color=gray)
        for label, value in data:
            pct = value / total * 100
            ax.add_patch(Rectangle((left, y), pct, height, facecolor=colors[label], edgecolor="white", linewidth=1.15))
            ax.text(left + pct / 2, y + height / 2, f"{pct:.0f}%", ha="center", va="center", fontsize=6.2, fontweight="semibold", color="white")
            left += pct
        legend_y = y - 0.12
        for x, (label, _) in zip([0, 27, 54, 78], data):
            ax.scatter([x + 1.2], [legend_y], s=18, marker="s", color=colors[label], edgecolor="none")
            ax.text(x + 3.1, legend_y, label, ha="left", va="center", fontsize=6.8, color=text)
        ax.hlines(y - 0.24, 0, 100, color=grid, linewidth=0.7)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.08, 1.38)
    ax.axis("off")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.98, bottom=0.06)
    save(fig, "category_distribution_pie", dpi=300)


def plot_category_bacc_boxplot() -> None:
    data = json.loads((RESULTS / "main_leaderboard" / "main_leaderboard.json").read_text(encoding="utf-8"))
    dr = {"Format": [], "Numbers": [], "Logic": [], "Facts": []}
    ac = {"Task": [], "Planning": [], "Tools": [], "Rules": []}
    dr_map = {"format": "Format", "numbers": "Numbers", "logic": "Logic", "facts": "Facts"}
    ac_map = {"task": "Task", "planning": "Planning", "tools": "Tools", "rules": "Rules"}
    for row in data["DeepResearch"]:
        for key, label in dr_map.items():
            dr[label].append(row["categories"][key]["balanced_accuracy"] * 100)
    for row in data["AgenticCoding"]:
        for key, label in ac_map.items():
            ac[label].append(row["categories"][key]["balanced_accuracy"] * 100)

    def draw_panel(ax, panel_data, title, categories, ylims, yticks):
        style = {"DeepResearch": {"fill": "#B9D3DD", "edge": "#4A5564"}, "AgenticCoding": {"fill": "#E7BFAF", "edge": "#4A5564"}}[title]
        values = [np.asarray(panel_data[category]) for category in categories]
        positions = np.arange(1, len(categories) + 1)
        ax.boxplot(values, positions=positions, widths=0.5, patch_artist=True, showmeans=False, showfliers=False,
                   medianprops={"color": "#0F172A", "linewidth": 2.05},
                   boxprops={"facecolor": style["fill"], "edgecolor": "#8897A7", "linewidth": 1.25, "alpha": 0.68},
                   whiskerprops={"color": style["edge"], "linewidth": 1.2},
                   capprops={"color": style["edge"], "linewidth": 1.2})
        ax.set_title(title, loc="left", fontsize=12.0, fontweight="normal", fontfamily="DejaVu Sans", color="#111111", pad=3)
        ax.set_xticks(positions)
        ax.set_xticklabels(categories, fontsize=11.5)
        ax.tick_params(axis="x", length=4, width=0.8, color="#1F2933", pad=4)
        ax.tick_params(axis="y", labelsize=10.8, pad=2)
        ax.set_ylim(*ylims)
        ax.set_yticks(yticks)
        ax.grid(axis="y", color="#E1E5EA", linewidth=0.72)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#A3ABB6")
        ax.spines["bottom"].set_color("#A3ABB6")

    fig, axes = plt.subplots(2, 1, figsize=(3.65, 4.35), sharey=False)
    draw_panel(axes[0], dr, "DeepResearch", ["Logic", "Facts", "Numbers", "Format"], (80, 100), [80, 85, 90, 95, 100])
    draw_panel(axes[1], ac, "AgenticCoding", ["Tools", "Rules", "Planning", "Task"], (50, 100), [50, 60, 70, 80, 90, 100])
    fig.supylabel("Category BAcc", x=0.01, fontsize=11.0)
    fig.subplots_adjust(left=0.16, right=0.992, top=0.965, bottom=0.075, hspace=0.30)
    save(fig, "category_bacc_boxplot", dpi=300)


def style_strategy_panel(ax, title, ylims, yticks):
    ax.set_title(title, loc="left", fontsize=9.4, fontweight="normal", pad=5)
    ax.axhline(0, color="#3f3f46", linewidth=1.0, zorder=1)
    ax.grid(axis="y", color="#d9dde3", linewidth=0.65)
    ax.grid(axis="x", color="#edf0f3", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.set_ylim(*ylims)
    ax.set_yticks(yticks)
    ax.tick_params(axis="x", labelsize=8.6, pad=1.5)
    ax.tick_params(axis="y", labelsize=8.1, pad=1.2)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#9aa3af")
    ax.spines["bottom"].set_color("#9aa3af")


def plot_batch_size_trend() -> None:
    rows = read_csv(RESULTS / "strategies" / "batch" / "batch_size_trend.csv")
    points: dict[tuple[str, str], dict[int, float]] = {}
    for row in rows:
        domain = "DR" if row["domain"] == "DeepResearch" else "AC"
        model = next(k for k, v in MODEL_ORDER if v == row["model"])
        points.setdefault((domain, model), {})[int(row["actual_rubrics_per_call"])] = float(row["delta_bacc_vs_single_rubric"])
    config = [("DR", "DeepResearch", range(1, 5), (-8, 4), [-8, -4, 0, 4]), ("AC", "AgenticCoding", range(1, 6), (-40, 10), [-40, -30, -20, -10, 0, 10])]
    fig, axes = plt.subplots(2, 1, figsize=(3.65, 4.15), sharey=False)
    for ax, (domain_key, title, xs, ylims, yticks) in zip(axes, config):
        style_strategy_panel(ax, title, ylims, yticks)
        for model, label in MODEL_ORDER:
            ys = [points.get((domain_key, model), {}).get(x) for x in xs]
            ax.plot(list(xs), ys, color=COLORS[model], marker=MARKERS[model], markersize=2.85, linewidth=1.0, label=label, solid_capstyle="round")
        ax.set_xlim(min(xs) - 0.12, max(xs) + 0.12)
        ax.set_xticks(list(xs))
    axes[1].set_xlabel("Actual rubrics per call", fontsize=8.8, labelpad=2)
    fig.text(0.01, 0.53, "Change from batching", rotation=90, va="center", fontsize=8.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7.8, loc="lower center", bbox_to_anchor=(0.52, 0.005), ncol=2, handlelength=1.25, columnspacing=0.7, labelspacing=0.28)
    fig.subplots_adjust(left=0.132, right=0.992, top=0.965, bottom=0.235, hspace=0.27)
    save(fig, "batch_size_trend", dpi=280)


def plot_self_voting_gain() -> None:
    rows = read_csv(RESULTS / "strategies" / "voting" / "self_voting_gain.csv")
    points: dict[tuple[str, str], dict[int, float]] = {}
    for row in rows:
        domain = "deepresearch" if row["domain"] == "DeepResearch" else "agenticcoding"
        model = next(k for k, v in MODEL_ORDER if v == row["model"])
        points.setdefault((domain, model), {})[int(row["votes"])] = float(row["delta_bacc_vs_one_vote"])
    fig, axes = plt.subplots(2, 1, figsize=(3.65, 4.15), sharex=True, sharey=False)
    for ax, domain, title, ylims, yticks in [
        (axes[0], "deepresearch", "DeepResearch", (-1, 2), [-1, 0, 1, 2]),
        (axes[1], "agenticcoding", "AgenticCoding", (-4, 6), [-4, -2, 0, 2, 4, 6]),
    ]:
        style_strategy_panel(ax, title, ylims, yticks)
        ax.set_xticks([1, 3, 5, 7, 9])
        for model, label in MODEL_ORDER:
            xs = [1, 3, 5, 7, 9]
            ys = [points.get((domain, model), {}).get(x) for x in xs]
            ax.plot(xs, ys, color=COLORS[model], marker=MARKERS[model], markersize=2.85, linewidth=1.0, label=label, solid_capstyle="round")
        ax.set_xlim(1 - 0.15, 9 + 0.15)
    axes[1].set_xlabel("Number of votes", fontsize=8.8, labelpad=2)
    fig.text(0.01, 0.53, "Gain from voting", rotation=90, va="center", fontsize=8.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7.8, loc="lower center", bbox_to_anchor=(0.52, 0.005), ncol=2, handlelength=1.25, columnspacing=0.7, labelspacing=0.28)
    fig.subplots_adjust(left=0.118, right=0.992, top=0.965, bottom=0.235, hspace=0.27)
    save(fig, "self_voting_gain", dpi=280)


def main() -> None:
    configure()
    plot_category_distribution()
    plot_category_bacc_boxplot()
    plot_batch_size_trend()
    plot_self_voting_gain()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
