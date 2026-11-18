"""
UMAP embeddings of the Global FoodOmics TreeMS2 distance matrix (Figure 6 and
Supplementary Figure 3), coloured by food category, fermentation and alcohol
content.

Run from the repository root:
    python scripts/FoodOmics/fig6_foodomics_umap.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data" / "FoodOmics"
FIGURES_DIR = PROJECT_DIR / "figures"

# Input
DISTANCE_MATRIX_PATH = DATA_DIR / "fig6-foodomics-distance_matrix.npy"
LABELS_PATH = DATA_DIR / "fig6-foodomics-labels.txt"
METADATA_PATH = DATA_DIR / "fig6-foodomics_metadata_with_umap_cat.tsv"

# Output
FOOD_CATEGORY_FIGURE_PATH = FIGURES_DIR / "foodomics_rerun_umap50-annotated.png"
FERMENTED_FIGURE_PATH = FIGURES_DIR / "foodomics_rerun_umap50-fermented_fixed.png"
ALCOHOLIC_FIGURE_PATH = FIGURES_DIR / "foodomics_rerun_umap50-alcoholic_fixed.png"

# ============================================================
# Settings
# ============================================================

ID_COLUMN = "mega11_compatible_sample_name"
N_NEIGHBORS = 50
RANDOM_STATE = 42
MISSING_COLOR = "#B2BEB5"

# Each callout maps a sample to (text, x offset, y offset). Offsets are
# fractions of the axis range.

FOOD_CATEGORY_COLORS = {
    "dairy": "#efb1f9",  # soft pink
    "beverages": "#016bdf",  # blue
    "fruit": "#FFA500",  # orange
    "nuts_seeds_legumes": "#DAA06D",  # brown
    "vegetable": "#597e00",  # green
    "grains": "#F5DEB3",  # gold/yellow
    "meat": "#f33b16",  # red
    "seafood": "#8AC6D0",  # light blue
    "other": "#CDCBCB",  # gray
}
FOOD_CATEGORY_LEGEND = [
    "dairy",
    "beverages",
    "fruits",
    "nuts/seeds/legumes",
    "vegetables",
    "grains",
    "meat",
    "seafood",
    "other",
]
FOOD_CATEGORY_CALLOUTS = {
    "G96444": ("Ground coffee", -0.04, -0.06),
    "G96093": ("Cherry tomato", 0.04, 0.06),
    "11442.G75258": ("Home brewed beer", 0.02, 0.04),
    "G96168": ("Yogurt", 0.02, 0.04),
    "G96221": ("Ground beef", 0.02, 0.06),
    "11442.G79557": ("Trout", -0.04, -0.04),
    "11442.G72441": ("Lemon", 0.02, -0.05),
    "11442.G78938": ("Lamb", 0.02, 0.04),
    "11442.G72685": ("White rice", -0.02, 0.11),
    "G96687": ("Black tea", 0.02, 0.04),
    "11442.G73489": ("Whole milk", -0.04, 0.06),
    "11442.G83398": ("Peanut", 0.05, 0.09),
    "G96285": ("Whole beans", 0.04, 0.04),
    "11442.G75192": ("Lettuce", 0.07, -0.04),
    "G96196": ("Milk with yogurt culture", 0.04, -0.06),
    "CW70_G1_T12": ("Granny Smith cider", 0.04, 0.12),
}

FERMENTED_COLORS = {
    "yes": "#f33b16",
    "no": "#8ac6d0",
}
FERMENTED_LEGEND = ["fermented", "not fermented"]
FERMENTED_CALLOUTS = {
    "11442.G72827": ("Home brewed beer", 0.14, 0.08),
    "G96617": ("Oolong tea", 0.14, 0.04),
    "11442.G86846": ("Milk", -0.08, 0.06),
    "G96170": ("Yogurt", -0.05, -0.1),
    "11442.G73654": ("Cheddar cheese", 0.14, 0.1),
    "G96129": ("Milk with yogurt culture", 0.04, -0.12),
}

ALCOHOLIC_COLORS = {
    "alcoholic": "#f33b16",
    "non alcoholic": "#8ac6d0",
    "not applicable": "#0e1c36",
}
ALCOHOLIC_CALLOUTS = {
    "11442.G72835": ("Red wine", 0.12, 0.15),
    "CW70_G1_T12": ("Granny Smith cider", 0.12, 0.1),
    "11442.G75258": ("Home brewed beer", 0.25, 0.04),
    "F33_black.wine": ("Black wine", 0.08, 0.04),
    "11442.G72464": ("Pacifico beer", 0.12, 0.08),
}


def load_data(distance_matrix_path, labels_path, metadata_path, id_column):
    """
    Load the distance matrix and the metadata, with the metadata rows
    ordered like the distance matrix.
    """
    distance_matrix = np.load(distance_matrix_path)

    with open(labels_path) as f:
        labels = [line.strip() for line in f if line.strip()]

    if not distance_matrix.shape[0] == distance_matrix.shape[1] == len(labels):
        raise ValueError("Distance matrix and labels length mismatch")

    metadata = pd.read_csv(metadata_path, sep="\t").set_index(id_column).loc[labels]

    return distance_matrix, metadata


def compute_umap(distance_matrix, n_neighbors=N_NEIGHBORS):
    """Embed a precomputed distance matrix in 2D with UMAP."""
    reducer = umap.UMAP(
        metric="precomputed",
        random_state=RANDOM_STATE,
        n_jobs=1,
        n_neighbors=n_neighbors,
    )
    return reducer.fit_transform(distance_matrix)


def plot_umap(
    embedding,
    metadata,
    color_column,
    color_map,
    output_path,
    fig_size,
    legend_labels=None,
    callouts=None,
):
    """
    Scatter plot of a UMAP embedding coloured by color_column, with optional
    labelled callouts for selected samples.
    """
    colors = metadata[color_column].map(color_map).fillna(MISSING_COLOR)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        s=15,
        linewidths=0,
        alpha=0.4,
        marker=".",
    )
    ax.margins(x=0.01, y=0.01)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Callout offsets are relative to the axis range.
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    for sample_name, (text, offset_x, offset_y) in (callouts or {}).items():
        x, y = embedding[metadata.index.get_loc(sample_name)]
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(x + (x_max - x_min) * offset_x, y + (y_max - y_min) * offset_y),
            textcoords="data",
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            arrowprops=dict(arrowstyle="-", color="black", lw=0.2, shrinkA=0, shrinkB=0),
        )

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8)
        for color in color_map.values()
    ]
    ax.legend(
        handles,
        legend_labels or list(color_map.keys()),
        loc="upper right",
        frameon=True,
        ncol=1,
        title=None,
        fontsize=10,
    )

    plt.tight_layout(pad=0.1)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    print(f"Using umap-learn version: {umap.__version__}")

    distance_matrix, metadata = load_data(
        DISTANCE_MATRIX_PATH, LABELS_PATH, METADATA_PATH, ID_COLUMN
    )
    embedding = compute_umap(distance_matrix)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_umap(
        embedding,
        metadata,
        color_column="umap_category_fixed",
        color_map=FOOD_CATEGORY_COLORS,
        output_path=FOOD_CATEGORY_FIGURE_PATH,
        fig_size=(6, 4),
        legend_labels=FOOD_CATEGORY_LEGEND,
        callouts=FOOD_CATEGORY_CALLOUTS,
    )
    plot_umap(
        embedding,
        metadata,
        color_column="fermented_fixed",
        color_map=FERMENTED_COLORS,
        output_path=FERMENTED_FIGURE_PATH,
        fig_size=(4, 4),
        legend_labels=FERMENTED_LEGEND,
        callouts=FERMENTED_CALLOUTS,
    )
    plot_umap(
        embedding,
        metadata,
        color_column="sample_type_groupB1_fixed",
        color_map=ALCOHOLIC_COLORS,
        output_path=ALCOHOLIC_FIGURE_PATH,
        fig_size=(4, 4),
        callouts=ALCOHOLIC_CALLOUTS,
    )


if __name__ == "__main__":
    main()
