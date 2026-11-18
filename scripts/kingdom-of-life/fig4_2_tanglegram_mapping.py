"""
Create the TaxonID mapping between the left (TreeMS2) and right (NCBI) trees of
the kingdom-of-life tanglegram (Figure 4).

The output is used by fig4_3_compare_with_ncbi_tree.py.

Run from the repository root:
    python scripts/kingdom-of-life/fig4_2_tanglegram_mapping.py
"""

from pathlib import Path

import pandas as pd

# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path.cwd()

# Input
INPUT_FILE = PROJECT_DIR / "data" / "kingdom-of-life" / "fig4-kingdom-of-life-ncbi-taxonomy.tsv"

# Output
OUTPUT_FILE = PROJECT_DIR / "temp" / "kingdom-of-life-tanglegram-mapping.tsv"

# ============================================================
# Settings
# ============================================================

# Taxa whose TaxonID in the NCBI tree differs from the one in the dataset.
RIGHT_TAXON_ID_OVERRIDES = {
    "Phlebovirus riftense": 3052676,
    "Human Coronavirus": 2664420,
}


def create_tanglegram_mapping(df):
    """
    One row per included TaxonID, sorted by TaxonID, with the dataset name,
    NCBI name, superkingdom and the TaxonIDs used in the left and right tree.
    """
    df = df.loc[~df["Excluded"]].copy()
    df["TaxonID"] = pd.to_numeric(df["TaxonID"], errors="coerce").astype("Int64")

    # Use the NCBI species name, falling back to the dataset species name.
    df["NCBI_name"] = (
        df["NCBI_species"]
        .fillna(df["Species"])
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
    )

    mapping = (
        df[["TaxonID", "Species", "NCBI_name", "Superkingdom"]]
        .rename(columns={"Species": "Dataset_name"})
        .drop_duplicates(subset=["TaxonID"])
        .sort_values(by="TaxonID")
    )

    mapping["TaxonID_left"] = mapping["TaxonID"]
    mapping["TaxonID_right"] = mapping["TaxonID"]
    for ncbi_name, taxon_id in RIGHT_TAXON_ID_OVERRIDES.items():
        mapping.loc[mapping["NCBI_name"] == ncbi_name, "TaxonID_right"] = taxon_id

    mapping["TaxonID_left"] = mapping["TaxonID_left"].astype("Int64")
    mapping["TaxonID_right"] = mapping["TaxonID_right"].astype("Int64")

    return mapping


def main():
    df = pd.read_csv(INPUT_FILE, sep="\t")
    mapping = create_tanglegram_mapping(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"✔ Created {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
