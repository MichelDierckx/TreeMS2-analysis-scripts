#!/usr/bin/env python3
"""
Compare the TreeMS2 distance matrix of the kingdom-of-life dataset with the
NCBI taxonomy tree from the tanglegram (Figure 4).

- Nearest-neighbour overlap per taxon, written to tables/ (Supplementary
  Table 3 is the summary table).
- Mantel tests (Pearson and Spearman)

Run fig4_2_tanglegram_mapping.py first; it creates the
tanglegram mapping used here.

Run from the repository root:
    python scripts/kingdom-of-life/fig4_3_compare_with_ncbi_tree.py
"""

from pathlib import Path

import dendropy
import numpy as np
import pandas as pd
from skbio.stats.distance import mantel

# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path.cwd()

# Input
TREEMS2_DISTANCE_MATRIX_PATH = (
    PROJECT_DIR / "data" / "kingdom-of-life" / "fig4-kingdom-of-life-distance_matrix.npy"
)
TREEMS2_DISTANCE_MATRIX_LABELS_PATH = (
    PROJECT_DIR / "data" / "kingdom-of-life" / "fig4-kingdom-of-life-distance_matrix_labels.txt"
)
TANGLEGRAM_PATH = PROJECT_DIR / "iTOL" / "kingdom-of-life" / "fig4-kingdom-of-life-tree-taxonids.nwk"
TANGLEGRAM_MAPPING = PROJECT_DIR / "temp" / "kingdom-of-life-tanglegram-mapping.tsv"

# Output
NEIGHBOUR_COMP_OUTPUT_PATH = (
    PROJECT_DIR / "tables" / "kingdom_of_life_tanglegram_neighbour_comparison.tsv"
)
NEIGHBOUR_COMP_SUMMARY_OUTPUT_PATH = (
    PROJECT_DIR / "tables" / "kingdom_of_life_tanglegram_neighbour_comparison_summary.tsv"
)

# ============================================================
# Settings
# ============================================================

N_NEIGHBOURS_LIST = [1, 2, 3, 5, 10]
MANTEL_PERMUTATIONS = 10000

# Species names to relabel in the summary table.
SUMMARY_SPECIES_RENAMES = {"Sus scrofa": "Sus scrofa; LLC-PK1"}


# ============================================================
# Loading
# ============================================================


def load_trees(tree_file):
    """Read the left and right tanglegram trees (first two non-empty lines)."""
    with open(tree_file) as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        raise ValueError("Tree file must contain at least two non-empty lines.")

    left_tree = dendropy.Tree.get(data=lines[0], schema="newick")
    right_tree = dendropy.Tree.get(data=lines[1], schema="newick")
    return left_tree, right_tree


def load_treems2_distance_matrix(matrix_path, labels_path, labels_relevant_taxa):
    """
    Load the TreeMS2 distance matrix and reduce it to labels_relevant_taxa,
    in that order.

    Returns the reduced matrix and a dict mapping label -> row/column index.
    """
    full_dm = np.load(matrix_path)
    if full_dm.shape[0] != full_dm.shape[1]:
        raise ValueError("TreeMS2 distance matrix is not square.")

    with open(labels_path) as f:
        full_labels = [line.strip() for line in f if line.strip()]

    if len(full_labels) != full_dm.shape[0]:
        raise ValueError("Number of labels does not match matrix dimensions.")

    full_label_to_index = {label: i for i, label in enumerate(full_labels)}

    missing = [label for label in labels_relevant_taxa if label not in full_label_to_index]
    if missing:
        raise ValueError(
            f"{len(missing)} taxa missing from TreeMS2 matrix. First few: {missing[:10]}"
        )

    indices = [full_label_to_index[label] for label in labels_relevant_taxa]
    subset_dm = full_dm[np.ix_(indices, indices)]

    if not np.allclose(subset_dm, subset_dm.T):
        raise ValueError("Subset matrix is not symmetric.")
    np.fill_diagonal(subset_dm, 0.0)

    label_to_index = {label: i for i, label in enumerate(labels_relevant_taxa)}
    return subset_dm, label_to_index


def tree_distance_matrix(tree, labels_relevant_taxa):
    """
    Patristic distance matrix of `tree` for labels_relevant_taxa, in that
    order. Prunes `tree` to those taxa.

    Returns the matrix and a dict mapping label -> row/column index.
    """
    tree_labels = {
        node.taxon.label for node in tree.leaf_node_iter() if node.taxon is not None
    }
    missing = [label for label in labels_relevant_taxa if label not in tree_labels]
    if missing:
        raise ValueError(
            f"{len(missing)} requested taxa are missing from the right-hand tree. "
            f"First few missing labels: {missing[:10]}\n"
            f"First few available labels: {sorted(tree_labels)[:10]}"
        )

    tree.retain_taxa_with_labels(labels=labels_relevant_taxa)
    pdc = tree.phylogenetic_distance_matrix()

    label_to_index = {label: i for i, label in enumerate(labels_relevant_taxa)}
    n = len(labels_relevant_taxa)
    matrix = np.zeros((n, n), dtype=float)

    n_pairs = 0
    for taxon_1, taxon_2 in pdc.distinct_taxon_pair_iter():
        i = label_to_index[taxon_1.label]
        j = label_to_index[taxon_2.label]
        matrix[i, j] = matrix[j, i] = float(pdc(taxon_1, taxon_2))
        n_pairs += 1

    np.fill_diagonal(matrix, 0.0)
    assert n_pairs == n * (n - 1) // 2, "Failed to completely fill distance matrix!"

    return matrix, label_to_index


# ============================================================
# Nearest-neighbour overlap
# ============================================================


def get_nearest_neighbours(distance_matrix, label_to_index, focal_label, n_neighbours):
    """
    Return the labels of the n_neighbours nearest taxa to focal_label,
    including ties (every taxon at most as far as the n-th neighbour).
    """
    if focal_label not in label_to_index:
        raise ValueError(f"{focal_label} not found in label_to_index.")

    index_to_label = {i: label for label, i in label_to_index.items()}
    focal_index = label_to_index[focal_label]

    distances = distance_matrix[focal_index]
    other_indices = np.arange(len(distances))
    other_indices = other_indices[other_indices != focal_index]
    other_distances = distances[other_indices]

    if n_neighbours >= len(other_distances):
        return {index_to_label[i] for i in other_indices}

    order = np.argsort(other_distances)
    sorted_indices = other_indices[order]
    sorted_distances = other_distances[order]

    threshold = sorted_distances[n_neighbours - 1]
    return {index_to_label[i] for i in sorted_indices[sorted_distances <= threshold]}


def compute_neighbour_overlap_table(
    df_mapping,
    left_dm,
    left_label_to_index,
    right_dm,
    right_label_to_index,
    n_neighbours_list,
    left_id_to_name,
    right_id_to_name,
):
    """
    For every taxon in both trees and every n in n_neighbours_list, compare
    its n nearest neighbours in the left and right tree.
    """
    rows = []

    for _, row in df_mapping.iterrows():
        left_label = str(row["TaxonID_left"]).strip()
        right_label = str(row["TaxonID_right"]).strip()

        if left_label not in left_label_to_index or right_label not in right_label_to_index:
            continue

        for n in n_neighbours_list:
            left_neighbours = get_nearest_neighbours(
                left_dm, left_label_to_index, left_label, n
            )
            right_neighbours = get_nearest_neighbours(
                right_dm, right_label_to_index, right_label, n
            )
            overlap = left_neighbours & right_neighbours
            union = left_neighbours | right_neighbours

            left_sorted = sorted(left_neighbours)
            right_sorted = sorted(right_neighbours)

            rows.append({
                "TaxonID_left": left_label,
                "Taxon_left_name": left_id_to_name.get(left_label, left_label),
                "TaxonID_right": right_label,
                "Taxon_right_name": right_id_to_name.get(right_label, right_label),
                "n_neighbours": n,
                "left_neighbours": ",".join(left_sorted),
                "right_neighbours": ",".join(right_sorted),
                "left_neighbours_names": ",".join(left_id_to_name.get(t, t) for t in left_sorted),
                "right_neighbours_names": ",".join(right_id_to_name.get(t, t) for t in right_sorted),
                "overlap_count": len(overlap),
                "left_set_size": len(left_neighbours),
                "right_set_size": len(right_neighbours),
                "jaccard_index": len(overlap) / len(union) if union else 0.0,
            })

    df_result = pd.DataFrame(rows)
    if not df_result.empty:
        df_result = df_result.sort_values(["TaxonID_left", "n_neighbours"])
    return df_result


def build_overlap_summary_table(df_neighbour_comparison):
    """
    Pivot the overlap table to one row per species with one
    overlap_count_n{n} column per neighbourhood size.
    """
    df_summary = df_neighbour_comparison.pivot_table(
        index="Taxon_left_name",
        columns="n_neighbours",
        values="overlap_count",
        aggfunc="first",
    )
    df_summary.columns = [f"overlap_count_n{int(n)}" for n in df_summary.columns]

    df_summary = df_summary.reset_index().rename(columns={"Taxon_left_name": "Species"})

    overlap_columns = sorted(
        [c for c in df_summary.columns if c.startswith("overlap_count_n")],
        key=lambda c: int(c.split("n")[-1]),
    )
    return df_summary[["Species"] + overlap_columns]


# ============================================================
# Mantel tests
# ============================================================


def run_mantel_tests(matrix_1, matrix_2):
    """Print Pearson and Spearman Mantel test results."""
    for method in ["pearson", "spearman"]:
        r, p_value, n = mantel(
            matrix_1, matrix_2, method=method, permutations=MANTEL_PERMUTATIONS
        )
        print(f"Mantel {method.capitalize()} R = {r:.3f}, p = {p_value:.4g}, n = {n}")


# ============================================================
# Main
# ============================================================


def main():
    df_mapping = pd.read_csv(TANGLEGRAM_MAPPING, sep="\t")
    labels_left = df_mapping["TaxonID_left"].astype(str).tolist()
    labels_right = df_mapping["TaxonID_right"].astype(str).tolist()

    # Left: TreeMS2 distances. Right: NCBI tree from the tanglegram.
    left_dm, left_label_to_index = load_treems2_distance_matrix(
        TREEMS2_DISTANCE_MATRIX_PATH,
        TREEMS2_DISTANCE_MATRIX_LABELS_PATH,
        labels_left,
    )
    _, right_tree = load_trees(TANGLEGRAM_PATH)
    right_dm, right_label_to_index = tree_distance_matrix(right_tree, labels_right)

    assert left_dm.shape == right_dm.shape

    unique_mapping = df_mapping.drop_duplicates()
    left_id_to_name = dict(zip(
        unique_mapping["TaxonID_left"].astype(str).str.strip(),
        unique_mapping["Dataset_name"].astype(str).str.strip(),
    ))
    right_id_to_name = dict(zip(
        unique_mapping["TaxonID_right"].astype(str).str.strip(),
        unique_mapping["NCBI_name"].astype(str).str.strip(),
    ))

    df_neighbour_comparison = compute_neighbour_overlap_table(
        df_mapping,
        left_dm,
        left_label_to_index,
        right_dm,
        right_label_to_index,
        N_NEIGHBOURS_LIST,
        left_id_to_name,
        right_id_to_name,
    )

    df_overlap_summary = build_overlap_summary_table(df_neighbour_comparison)
    df_overlap_summary["Species"] = df_overlap_summary["Species"].replace(
        SUMMARY_SPECIES_RENAMES
    )

    NEIGHBOUR_COMP_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_neighbour_comparison.to_csv(NEIGHBOUR_COMP_OUTPUT_PATH, sep="\t", index=False)
    df_overlap_summary.to_csv(NEIGHBOUR_COMP_SUMMARY_OUTPUT_PATH, sep="\t", index=False)

    run_mantel_tests(left_dm, right_dm)


if __name__ == "__main__":
    main()
