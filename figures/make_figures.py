#!/usr/bin/env python3
"""Generate compact window-sweep figures from results.json."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

import style

HERE = Path(__file__).resolve().parent


def plot_window(ax, data, title):
    windows = data["windows"]
    ax.errorbar(
        windows,
        data["closed"],
        yerr=data["closed_std"],
        marker="o",
        lw=1.2,
        capsize=2,
        color=style.BLUE,
        label="closed-set acc.",
    )
    ax.errorbar(
        windows,
        data["auroc"],
        yerr=data["auroc_std"],
        marker="s",
        lw=1.2,
        capsize=2,
        color=style.ORANGE,
        label="AUROC",
    )
    ax.set_title(title)
    ax.set_xlabel("prototype window W")
    ax.set_xticks(windows)
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    style.despine(ax)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(HERE / "out"))
    parser.add_argument("--results", default=str(HERE / "results.json"))
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.results).read_text())

    fig, axes = plt.subplots(1, 2, figsize=(style.TEXT_W, 2.0), sharey=True)
    plot_window(axes[0], data["wisig_window"], "WiSig")
    plot_window(axes[1], data["pycom_window"], "Pycom")
    axes[0].set_ylabel("score")
    axes[1].legend(loc="lower left", fontsize=7, frameon=False)
    style.save(fig, out / "fig_window_sweep.pdf")
    style.save(fig, out / "fig_window_sweep.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
