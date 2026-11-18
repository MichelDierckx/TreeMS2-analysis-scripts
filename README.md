# TreeMS2 manuscript: analysis scripts

Scripts that reproduce the figures, tables and statistics of the TreeMS2 manuscript.

## Data

The input data are available on Zenodo: [10.5281/zenodo.24347634](https://doi.org/10.5281/zenodo.24347634).
The upload contains `data`, `iTOL`, `supplementary` and `TreeMS2-inputs` as zip files. Extract them into the repository root.

## Software

TreeMS2 is open source under the Apache 2.0 license: https://github.com/bittremieuxlab/TreeMS2.

## Usage

Run the scripts from the repository root:

```bash
python scripts/runtime/fig2_runtime_comparison.py
```

Outputs are written to `figures/`, `tables/` and `temp/`.

Tested with Python 3.12, numpy 2.0.1, pandas 2.3.1, matplotlib 3.10.0, seaborn 0.13.2, scikit-learn 1.7.1, umap-learn 0.5.11, dendropy 5.0.8, scikit-bio 0.7.0 and ete3 3.1.3.

## Scripts

| Script | Manuscript | Output |
|---|---|---|
| [`scripts/runtime/fig2_runtime_comparison.py`](scripts/runtime/fig2_runtime_comparison.py) | Figure 2 | `figures/timings_comparison_graph_runs.png` |
| [`scripts/38SPD/fig3_38SPD_gtdb_mantel_test.py`](scripts/38SPD/fig3_38SPD_gtdb_mantel_test.py) | Figure 3, Mantel test vs. GTDB | printed |
| [`scripts/38SPD-dairy-isolates/dairy_isolates_classification.py`](scripts/38SPD-dairy-isolates/dairy_isolates_classification.py) | Food-derived isolates classification | `tables/Dairy_Product_with_TreeMS2_predictions.tsv` |
| [`scripts/kingdom-of-life/fig4_1_ncbi_taxonomy_tree.py`](scripts/kingdom-of-life/fig4_1_ncbi_taxonomy_tree.py) | Figure 4, NCBI tree | `temp/kingdom-of-life-ncbi-tree.nwk` |
| [`scripts/kingdom-of-life/fig4_2_tanglegram_mapping.py`](scripts/kingdom-of-life/fig4_2_tanglegram_mapping.py) | Figure 4, tanglegram mapping | `temp/kingdom-of-life-tanglegram-mapping.tsv` |
| [`scripts/kingdom-of-life/fig4_3_compare_with_ncbi_tree.py`](scripts/kingdom-of-life/fig4_3_compare_with_ncbi_tree.py) | Figure 4 Mantel test, Supplementary Table 3 | `tables/kingdom_of_life_tanglegram_neighbour_comparison*.tsv`, printed |
| [`scripts/single-cell/fig5_single_cell_mds.py`](scripts/single-cell/fig5_single_cell_mds.py) | Figure 5 | `figures/mds.png` |
| [`scripts/FoodOmics/fig6_foodomics_umap.py`](scripts/FoodOmics/fig6_foodomics_umap.py) | Figure 6, Supplementary Figure 3 | `figures/foodomics_rerun_umap50-*.png` |

Notes:

- Run `fig4_2_tanglegram_mapping.py` before `fig4_3_compare_with_ncbi_tree.py`.
- `fig4_1_ncbi_taxonomy_tree.py` uses the ete3 NCBI taxonomy database, which ete3 downloads on first use.
- The trees in Figures 3 and 4 were drawn in [iTOL](https://itol.embl.de/) from the files in `iTOL/`.
