"""
MDS embedding of the iPSC/EB single-cell TreeMS2 distance matrix, coloured by
cluster (Figure 5).

Run from the repository root:
    python scripts/single-cell/fig5_single_cell_mds.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import MDS

# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data" / "single-cell"

# Input
DISTANCE_MATRIX_PATH = DATA_DIR / "fig5-iPSC_EBs-distance_matrix.npy"
DISTANCE_MATRIX_LABELS_PATH = DATA_DIR / "fig5-iPSC_EBs-distance_matrix_labels.txt"
METADATA_PATH = DATA_DIR / "fig5-iPSC_EBs_metadata_with_clusters.tsv"

# Output
MDS_FIGURE_PATH = PROJECT_DIR / "figures" / "mds.png"

# ============================================================
# Settings
# ============================================================

CLUSTER_COLORS = {
    "0": "#0e1c36",
    "1": "#8ac6d0",
    "2": "#397edc",
    "3": "#ffc7c2",
    "4": "#f33b16",
    "5": "#597e00",
}
UNKNOWN_CLUSTER = "Unknown"
RANDOM_STATE = 42


def load_data(distance_matrix_path, labels_path, metadata_path):
    """
    Load the distance matrix and the metadata, with the metadata rows
    ordered like the distance matrix.
    """
    distance_matrix = np.load(distance_matrix_path)

    with open(labels_path) as f:
        labels = [line.strip() for line in f if line.strip()]

    if not distance_matrix.shape[0] == distance_matrix.shape[1] == len(labels):
        raise ValueError("Distance matrix and labels length mismatch")

    metadata = pd.read_csv(metadata_path, sep="\t").set_index("sample_name").loc[labels]

    return distance_matrix, metadata


def drop_unknown_clusters(distance_matrix, metadata):
    """Remove samples whose cluster is 'Unknown'."""
    mask = (metadata["cluster"] != UNKNOWN_CLUSTER).to_numpy()
    return distance_matrix[np.ix_(mask, mask)], metadata[mask]


def run_mds(distance_matrix, n_components=2):
    """Embed a precomputed distance matrix with metric MDS."""
    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        n_init=4,  # explicit: scikit-learn 1.9 changes the default to 1
        random_state=RANDOM_STATE,
    )
    return mds.fit_transform(distance_matrix)


def plot_mds(embedding, colors, color_map, output_path, legend_title=None):
    """Scatter plot of a 2D embedding with a legend built from color_map."""
    plt.figure(figsize=(7, 4))
    sns.set_context("paper")
    ax = plt.gca()

    ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=40, linewidths=0)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    sns.despine()

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8)
        for c in color_map.values()
    ]
    ax.legend(
        handles,
        color_map.keys(),
        loc="upper left",
        frameon=False,
        ncol=2,
        title=legend_title,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()
    print(f"Saved {output_path}")


def main():
    distance_matrix, metadata = load_data(
        DISTANCE_MATRIX_PATH, DISTANCE_MATRIX_LABELS_PATH, METADATA_PATH
    )
    distance_matrix, metadata = drop_unknown_clusters(distance_matrix, metadata)

    colors = metadata["cluster"].astype(str).map(CLUSTER_COLORS).fillna("gray")
    embedding = run_mds(distance_matrix)

    MDS_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_mds(embedding, colors, CLUSTER_COLORS, MDS_FIGURE_PATH, legend_title="Cluster")


if __name__ == "__main__":
    main()
