"""
Compare the TreeMS2 distance matrix of the 38SPD dataset with GTDB phylogenetic
distances (Mantel test; Figure 3 results, species mapping in Supplementary
Table 5).

Sample-level TreeMS2 distances are averaged to species level and compared with
patristic distances between the matching GTDB reference species.

Run from the repository root:
    python scripts/38SPD/fig3_38SPD_gtdb_mantel_test.py
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
DATA_DIR = PROJECT_DIR / "data" / "38SPD"
SUPPLEMENTARY_DIR = PROJECT_DIR / "supplementary"

# Input
GTDB_TREE_PATH = DATA_DIR / "bac120_r226.sp_labels.tree"
SAMPLE_TO_SPECIES_PATH = DATA_DIR / "sample_description_38SPD.tsv"
SPECIES_MAPPING_PATH = SUPPLEMENTARY_DIR / "Supplementary_Table_5_species_38SPD_to_GTDB.tsv"
TREEMS2_DISTANCE_MATRIX_PATH = DATA_DIR / "fig3-38SPD-distance_matrix.npy"

# ============================================================
# Settings
# ============================================================

SPECIES_COLUMN = "38SPD species"
GTDB_SPECIES_COLUMN = "GTDB reference species"

MANTEL_PERMUTATIONS = 10000
MANTEL_SEED = 42


def load_data():
    """Load the sample table, species mapping, GTDB tree and TreeMS2 matrix."""
    df_samples = pd.read_csv(SAMPLE_TO_SPECIES_PATH, sep="\t")
    df_mapping = pd.read_csv(SPECIES_MAPPING_PATH, sep="\t")

    tree = dendropy.Tree.get_from_path(
        GTDB_TREE_PATH,
        schema="newick",
        rooting="force-rooted",
        preserve_underscores=True,
    )

    distance_matrix = np.load(TREEMS2_DISTANCE_MATRIX_PATH)

    return df_samples, df_mapping, tree, distance_matrix


def build_species_to_gtdb_mapping(df_mapping):
    """
    Return a dict mapping each 38SPD species to its GTDB reference species.

    Raises an error if a 38SPD species maps to more than one GTDB species.
    """
    df_mapping = (
        df_mapping[[SPECIES_COLUMN, GTDB_SPECIES_COLUMN]]
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
    )

    duplicated = df_mapping[SPECIES_COLUMN].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            "Multiple GTDB mappings found for these 38SPD species: "
            f"{df_mapping.loc[duplicated, SPECIES_COLUMN].unique().tolist()}"
        )

    return df_mapping.set_index(SPECIES_COLUMN)[GTDB_SPECIES_COLUMN].to_dict()


def aggregate_sample_distances(distance_matrix, species_per_sample, species_order):
    """Mean distance between the samples of each species pair."""
    species_per_sample = np.asarray(species_per_sample)
    species_to_indices = {
        species: np.flatnonzero(species_per_sample == species)
        for species in species_order
    }

    n_species = len(species_order)
    species_matrix = np.zeros((n_species, n_species))

    for i in range(n_species):
        indices_1 = species_to_indices[species_order[i]]
        for j in range(i + 1, n_species):
            indices_2 = species_to_indices[species_order[j]]
            mean_distance = np.mean(distance_matrix[np.ix_(indices_1, indices_2)])
            species_matrix[i, j] = species_matrix[j, i] = mean_distance

    return species_matrix


def construct_gtdb_distance_matrix(tree, species_order, species_to_gtdb):
    """Patristic distances between the GTDB reference species of species_order."""
    gtdb_labels = [species_to_gtdb[species] for species in species_order]

    tree.retain_taxa_with_labels(labels=gtdb_labels)
    pdc = tree.phylogenetic_distance_matrix()
    gtdb_taxa = {taxon.label: taxon for taxon in pdc.taxon_namespace}

    n_species = len(species_order)
    species_matrix = np.zeros((n_species, n_species))

    for i in range(n_species):
        taxon_1 = gtdb_taxa[gtdb_labels[i]]
        for j in range(i + 1, n_species):
            taxon_2 = gtdb_taxa[gtdb_labels[j]]
            species_matrix[i, j] = species_matrix[j, i] = pdc(taxon_1, taxon_2)

    return species_matrix


def run_mantel_tests(matrix_1, matrix_2):
    """Print Pearson and Spearman Mantel test results."""
    for method in ["pearson", "spearman"]:
        r, p_value, n = mantel(
            matrix_1,
            matrix_2,
            method=method,
            permutations=MANTEL_PERMUTATIONS,
            seed=MANTEL_SEED,
        )
        print(f"Mantel {method.capitalize()} R = {r:.3f}, p = {p_value:.4g}, n = {n}")


def main():
    df_samples, df_mapping, gtdb_tree, treems2_distance_matrix = load_data()

    n_samples = len(df_samples)
    if treems2_distance_matrix.shape != (n_samples, n_samples):
        raise ValueError(
            "The dimensions of the TreeMS2 distance matrix do not match "
            "the number of samples in the sample-to-species TSV. "
            "Make sure both use the same samples and ordering."
        )

    species_to_gtdb = build_species_to_gtdb_mapping(df_mapping)

    # Keep only samples whose species has a GTDB reference species, in both
    # the sample table and the distance matrix.
    df_samples[SPECIES_COLUMN] = df_samples[SPECIES_COLUMN].replace("", pd.NA)
    has_gtdb_species = df_samples[SPECIES_COLUMN].map(species_to_gtdb).notna()

    valid_indices = np.flatnonzero(has_gtdb_species)
    treems2_distance_matrix = treems2_distance_matrix[np.ix_(valid_indices, valid_indices)]
    df_samples = df_samples.loc[has_gtdb_species].reset_index(drop=True)

    species_per_sample = df_samples[SPECIES_COLUMN].tolist()
    species_order = df_samples[SPECIES_COLUMN].unique().tolist()

    treems2_species_matrix = aggregate_sample_distances(
        treems2_distance_matrix, species_per_sample, species_order
    )
    gtdb_species_matrix = construct_gtdb_distance_matrix(
        gtdb_tree, species_order, species_to_gtdb
    )

    assert treems2_species_matrix.shape == gtdb_species_matrix.shape
    assert np.allclose(treems2_species_matrix, treems2_species_matrix.T)
    assert np.allclose(gtdb_species_matrix, gtdb_species_matrix.T)

    print(f"Number of retained samples: {len(df_samples)}")
    print(f"Number of species: {len(species_order)}")

    print("\nMantel tests")
    run_mantel_tests(treems2_species_matrix, gtdb_species_matrix)


if __name__ == "__main__":
    main()
