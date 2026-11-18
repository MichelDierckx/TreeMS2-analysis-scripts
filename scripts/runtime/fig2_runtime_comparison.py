"""
Plot the runtime of TreeMS2 and compareMS2 against the number of runs (Figure 2).

Run from the repository root:
    python scripts/runtime/fig2_runtime_comparison.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path.cwd()

# Input
TREEMS2_TIMINGS_PATH = PROJECT_DIR / "data" / "runtime" / "fig2-timings_TreeMS2.csv"
COMPAREMS2_TIMINGS_PATH = PROJECT_DIR / "data" / "runtime" / "fig2-timings_compareMS2.csv"

# Output
TIMINGS_FIGURE_PATH = PROJECT_DIR / "figures" / "timings_comparison_graph_runs.png"

# ============================================================
# Plot settings
# ============================================================

TREEMS2_COLOR = "#597e00"
COMPAREMS2_COLOR = "#397edc"


def plot_timing_comparison(df_treems2, df_comparems2, output_path):
    """Plot runtime (seconds) against number of runs for both tools."""
    plt.figure(figsize=(7, 3.5))

    for df, label, color in [
        (df_treems2, "TreeMS2", TREEMS2_COLOR),
        (df_comparems2, "CompareMS2", COMPAREMS2_COLOR),
    ]:
        plt.plot(
            df["num_rows"],
            df["time_seconds"],
            marker="o",
            linestyle="-",
            label=label,
            color=color,
        )

    plt.xlabel("Number of runs")
    plt.ylabel("Time (seconds)")
    plt.legend(frameon=False)
    plt.xticks([5, 10, 15, 20])
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()


def main():
    df_treems2 = pd.read_csv(TREEMS2_TIMINGS_PATH, delimiter="\t")
    df_comparems2 = pd.read_csv(COMPAREMS2_TIMINGS_PATH, delimiter="\t")

    TIMINGS_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_timing_comparison(df_treems2, df_comparems2, TIMINGS_FIGURE_PATH)


if __name__ == "__main__":
    main()
