"""
Identify bacterial isolates from dairy products using the TreeMS2 distance
matrix of the 38SPD reference bacteria plus the dairy isolates (Results and
Methods: "Food-derived bacterial isolates classification").

The genus of each isolate is predicted with a distance-weighted k-nearest
neighbours classifier trained on the 38SPD reference samples. The species is
the closest of the five nearest reference species that belongs to the predicted
genus. Predictions are compared with the 16S rRNA identifications.

Run from the repository root:
    python scripts/38SPD-dairy-isolates/dairy_isolates_classification.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
TABLE_DIR = PROJECT_DIR / "tables"
CONFIGURATIONS_DIR = PROJECT_DIR / "TreeMS2-inputs"

# Input
DIST_MATRIX_PATH = DATA_DIR / "38SPD-dairy-isolates" / "38SPD-dairy-isolates-distance_matrix.npy"
SAMPLE_TO_GROUP_PATH = CONFIGURATIONS_DIR / "38SPD-dairy-isolates" / "38SPD-dairy-sample_to_group.tsv"
SAMPLE_DESCRIPTION_PATH = DATA_DIR / "38SPD" / "sample_description_38SPD.tsv"
DAIRY_PRODUCT_PATH = DATA_DIR / "38SPD-dairy-isolates" / "Dairy_Product.tsv"

# Output
OUTPUT_CLASSIFICATION_RESULTS_PATH = TABLE_DIR / "Dairy_Product_with_TreeMS2_predictions.tsv"

# ============================================================
# Settings
# ============================================================

N_NEIGHBORS = 5
SPECIES_COLUMN = "38SPD species"
GENUS_16S_COLUMN = "Genus according to 16S rRNA"
SPECIES_16S_COLUMN = "Species according to 16S rRNA"


def normalise_filenames(series):
    """Strip whitespace and a trailing '.mgf' from filenames."""
    return series.str.strip().str.replace(r"\.mgf$", "", regex=True)


def require_columns(df, columns, description):
    if not set(columns).issubset(df.columns):
        raise ValueError(f"{description} must contain columns: {set(columns)}")


def load_sample_files(mapping_path, distance_matrix):
    """
    Return the sample filenames in distance-matrix order, as listed in the
    TreeMS2 sample-to-group file.
    """
    df_mapping = pd.read_csv(mapping_path, sep="\t")
    require_columns(df_mapping, {"file", "group"}, "Mapping file")

    sample_files = normalise_filenames(df_mapping["file"].astype(str)).tolist()

    if len(sample_files) != len(set(sample_files)):
        raise ValueError("Duplicate sample filenames in mapping file")

    if distance_matrix.shape != (len(sample_files), len(sample_files)):
        raise ValueError(
            f"Distance matrix shape {distance_matrix.shape} does not match "
            f"the {len(sample_files)} samples in the mapping file"
        )

    return sample_files


def load_sample_to_species(sample_description_path):
    """Return a dict mapping each 38SPD sample filename to its species."""
    df_species = pd.read_csv(sample_description_path, sep="\t")
    require_columns(df_species, {"file", SPECIES_COLUMN}, "Sample description file")

    df_species["file"] = normalise_filenames(df_species["file"].astype(str))

    duplicated = df_species["file"].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            "Duplicate samples in sample description file: "
            f"{df_species.loc[duplicated, 'file'].unique().tolist()}"
        )

    return df_species.set_index("file")[SPECIES_COLUMN].to_dict()


def get_reference_species(sample_files, train_idx, sample_to_species):
    """
    Return the species of every reference sample (indexed like sample_files;
    None for food isolates). Raises an error for reference samples without a
    known species.
    """
    species_labels = [None] * len(sample_files)
    for i in train_idx:
        species = sample_to_species.get(sample_files[i])
        if species is not None and not pd.isna(species):
            species_labels[i] = str(species).strip()

    unknown_reference = [sample_files[i] for i in train_idx if species_labels[i] is None]
    if unknown_reference:
        raise ValueError(
            "Could not find species for the following reference samples: "
            f"{unknown_reference}"
        )

    return species_labels


def genus_of(species):
    return species.split()[0]


def predict_isolates(dist_matrix, food_indices, train_idx, species_labels, sample_files):
    """
    Predict genus and species for each food isolate.

    Returns three dicts keyed by isolate filename: predicted genus, predicted
    species, and a formatted list of the five closest reference species.
    """
    y_train = [genus_of(species_labels[i]) for i in train_idx]

    clf = KNeighborsClassifier(
        n_neighbors=N_NEIGHBORS,
        metric="precomputed",
        weights="distance",
    )
    clf.fit(dist_matrix[np.ix_(train_idx, train_idx)], y_train)

    genus_predictions = {}
    species_predictions = {}
    closest_species = {}

    for i in food_indices:
        isolate = sample_files[i]
        distances = dist_matrix[i, train_idx]

        predicted_genus = clf.predict(distances.reshape(1, -1))[0]

        nearest = np.argsort(distances)[:N_NEIGHBORS]
        closest_pairs = [(species_labels[train_idx[j]], distances[j]) for j in nearest]

        # Closest species within the predicted genus; fall back to the
        # closest species overall if none of the neighbours match.
        same_genus = [sp for sp, _ in closest_pairs if genus_of(sp) == predicted_genus]

        genus_predictions[isolate] = predicted_genus
        species_predictions[isolate] = same_genus[0] if same_genus else closest_pairs[0][0]
        closest_species[isolate] = "; ".join(f"{sp} ({dist:.4f})" for sp, dist in closest_pairs)

    return genus_predictions, species_predictions, closest_species


def add_evaluation_columns(df_food, training_genera, training_species):
    """Compare TreeMS2 predictions with the 16S rRNA identifications."""
    species_16s = (
        df_food[GENUS_16S_COLUMN].astype(str).str.strip()
        + " "
        + df_food[SPECIES_16S_COLUMN].astype(str).str.strip()
    )

    df_food["Genus_in_38SPD"] = df_food[GENUS_16S_COLUMN].isin(training_genera)
    df_food["Species_in_38SPD"] = species_16s.isin(training_species)
    df_food["Correct_Genus"] = df_food["Genus_in_38SPD"] & (
        df_food["Genus according to TreeMS2"] == df_food[GENUS_16S_COLUMN]
    )
    df_food["Correct_Species"] = df_food["Species_in_38SPD"] & (
        df_food["Species according to TreeMS2"] == species_16s
    )

    return df_food


def print_summary(df_food):
    print("\nSummary of evaluation:")

    for level, plural in [("Genus", "genera"), ("Species", "species")]:
        total = df_food[f"{level}_in_38SPD"].sum()
        correct = df_food[f"Correct_{level}"].sum()
        if total:
            print(f"{level} evaluation: {correct}/{total} matches ({correct / total:.1%})")
        else:
            print(f"No food isolates with represented {plural}.")


def main():
    dist_matrix = np.load(DIST_MATRIX_PATH)
    sample_files = load_sample_files(SAMPLE_TO_GROUP_PATH, dist_matrix)
    file_to_index = {filename: i for i, filename in enumerate(sample_files)}

    # Food isolates
    df_food = pd.read_csv(DAIRY_PRODUCT_PATH, sep="\t")
    require_columns(df_food, {"Raw file name"}, "Input TSV")

    food_filenames = normalise_filenames(df_food["Raw file name"].astype("string"))
    food_isolates = set(food_filenames.dropna())

    missing_food = food_isolates - set(file_to_index)
    if missing_food:
        raise ValueError(f"Food isolates missing from sample_to_group.tsv: {missing_food}")

    # Split samples into food isolates and reference bacteria (all others),
    # both in distance-matrix order.
    food_indices = [i for i, filename in enumerate(sample_files) if filename in food_isolates]
    train_idx = [i for i, filename in enumerate(sample_files) if filename not in food_isolates]

    if len(train_idx) < N_NEIGHBORS:
        raise ValueError(f"At least {N_NEIGHBORS} reference samples are required")

    sample_to_species = load_sample_to_species(SAMPLE_DESCRIPTION_PATH)
    species_labels = get_reference_species(sample_files, train_idx, sample_to_species)

    genus_predictions, species_predictions, closest_species = predict_isolates(
        dist_matrix, food_indices, train_idx, species_labels, sample_files
    )

    df_food["Genus according to TreeMS2"] = food_filenames.map(genus_predictions)
    df_food["Species according to TreeMS2"] = food_filenames.map(species_predictions)
    df_food["5 closest species (TreeMS2)"] = food_filenames.map(closest_species)

    training_species = {species_labels[i] for i in train_idx}
    training_genera = {genus_of(species) for species in training_species}
    df_food = add_evaluation_columns(df_food, training_genera, training_species)

    print_summary(df_food)

    OUTPUT_CLASSIFICATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_food.to_csv(OUTPUT_CLASSIFICATION_RESULTS_PATH, sep="\t", index=False)
    print(f"\nFinal results saved to {OUTPUT_CLASSIFICATION_RESULTS_PATH}")


if __name__ == "__main__":
    main()
