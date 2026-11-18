"""
Build an NCBI taxonomy tree for the kingdom-of-life taxa, with branch lengths
derived from taxonomic rank (Figure 4).

Each rank gets an integer depth (species = 1, increasing towards the root).
Depths are scaled nonlinearly, so steps between higher ranks are longer, and a
branch length is the difference between the scaled depths of parent and child.

Requires the local ete3 NCBI taxonomy database (~/.etetoolkit/taxa.sqlite);
ete3 downloads it on first use.

Run from the repository root:
    python scripts/kingdom-of-life/fig4_1_ncbi_taxonomy_tree.py
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd
from ete3 import NCBITaxa

# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path.cwd()

# Input
INPUT_FILE = PROJECT_DIR / "data" / "kingdom-of-life" / "fig4-kingdom-of-life-ncbi-taxonomy.tsv"

# Output
OUTPUT_FILE = PROJECT_DIR / "temp" / "kingdom-of-life-ncbi-tree.nwk"

# ============================================================
# Settings
# ============================================================

BASE_STEP = 0.3
GROWTH_FACTOR = 1.2

# Taxa that should explicitly be treated as species.
SPECIAL_SPECIES_TAXIDS = {3052676, 2664420}

NCBI_ROOT_TAXID = 1

# Ranks from most specific to most general. Viruses (acellular) and cellular
# organisms share the ranks up to kingdom and then diverge.
SHARED_RANKS = [
    "species",
    "subgenus",
    "genus",
    "subfamily",
    "family",
    "suborder",
    "order",
    "superorder",
    "subclass",
    "class",
    "superclass",
    "subphylum",
    "phylum",
    "kingdom",
]
ACELLULAR_RANKS = ["realm", "acellular root"]
CELLULAR_RANKS = ["domain", "cellular root"]


def ranks_to_depths(ranks):
    """Map ranks to consecutive integer depths, starting at 1."""
    return {rank: depth for depth, rank in enumerate(ranks, start=1)}


RANK_DEPTH_ACELLULAR = ranks_to_depths(SHARED_RANKS + ACELLULAR_RANKS)
RANK_DEPTH_CELLULAR = ranks_to_depths(SHARED_RANKS + CELLULAR_RANKS)

SPECIES_DEPTH = RANK_DEPTH_CELLULAR["species"]
ROOT_DEPTH = max(RANK_DEPTH_CELLULAR.values()) + 1


def scale_depth(depth, base_step=BASE_STEP, growth=GROWTH_FACTOR):
    """
    Convert an integer depth into a scaled depth. Each successive step is
    `growth` times longer than the previous one.
    """
    scaled_depth = 0.0
    step = base_step
    for _ in range(depth):
        scaled_depth += step
        step *= growth
    return scaled_depth


def make_depth_function(ncbi, leaf_taxids):
    """
    Return a cached function giving the integer taxonomic depth of a taxid.

    Leaf taxa are treated as species. Internal nodes get the depth of their
    NCBI rank; nodes with an unrecognised rank (e.g. "clade") are placed one
    level below the most specific recognised rank in their lineage.
    """

    @lru_cache(maxsize=None)
    def get_depth(taxid):
        if taxid in SPECIAL_SPECIES_TAXIDS or taxid in leaf_taxids:
            return SPECIES_DEPTH

        if taxid == NCBI_ROOT_TAXID:
            return ROOT_DEPTH

        lineage = ncbi.get_lineage(taxid)
        ranks = ncbi.get_rank(lineage)
        lineage_ranks = [ranks.get(ancestor, "no rank") for ancestor in lineage]

        if "acellular root" in lineage_ranks:
            rank_depth = RANK_DEPTH_ACELLULAR
        elif "cellular root" in lineage_ranks:
            rank_depth = RANK_DEPTH_CELLULAR
        else:
            raise ValueError(
                f"Cannot determine whether taxid {taxid} belongs to "
                "a cellular or acellular lineage."
            )

        rank = ranks.get(taxid, "no rank")
        if rank in rank_depth:
            return rank_depth[rank]

        recognised_depths = [
            rank_depth[ranks[ancestor]]
            for ancestor in lineage
            if ranks.get(ancestor) in rank_depth
        ]
        if not recognised_depths:
            raise ValueError(f"No recognized taxonomic rank found in lineage for taxid {taxid}.")

        approximated_depth = max(min(recognised_depths) - 1, 0)
        print(
            f"[WARNING] Unrecognized rank for taxid {taxid}: "
            f"{rank!r}. Approximated depth = {approximated_depth}."
        )
        return approximated_depth

    return get_depth


def assign_branch_lengths(tree, get_depth):
    """Set each branch length to the scaled depth difference (never negative)."""
    for node in tree.traverse("postorder"):
        if node.is_root():
            node.dist = 0.0
            continue
        parent_depth = scale_depth(get_depth(node.up.taxid))
        node_depth = scale_depth(get_depth(node.taxid))
        node.dist = max(parent_depth - node_depth, 0.0)


def main():
    df = pd.read_csv(INPUT_FILE, sep="\t")
    df = df.loc[~df["Excluded"]]

    tax_ids = df["TaxonID"].dropna().drop_duplicates().astype(int).tolist()
    print(f"Loaded {len(tax_ids)} unique taxa.")

    ncbi = NCBITaxa()
    tree = ncbi.get_topology(tax_ids)

    assign_branch_lengths(tree, make_depth_function(ncbi, set(tax_ids)))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tree.write(format=1, outfile=str(OUTPUT_FILE))
    print(f"Tree written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
