# November 2 Gerber Follow-up Bundle

This bundle adds a second analysis layer on top of the existing
`Analysis/post_gerber_pc3_extension.py`.

## What it addresses

Dr. Gerber's Nov. 2 requests are translated into concrete outputs:

- k=3 / k=4 / k=5 cluster assignments and profiles
- k=5 interactive 3D PCA triplot
- broader KMeans model-selection grid
- scaling sensitivity
- refined PC3 geographic color range
- ACS broadband-subscription map
- ZIP / coordinate / town-label provenance audit
- Ayer / Devens / Shirley comparison
- explicit mismatch-map methodology documentation
- model-based PCA density/outlier scoring using cross-validated KDE
- row counts / unique ZIP counts
- candidate cluster labels/ZIPs for deeper investigation

## Important terminology correction

`Broadband_Rate` is an ACS household broadband-subscription/adoption measure.
It is **not** infrastructure coverage or service availability.

The new map therefore says **ACS broadband subscription rate**. If the research
question is actual coverage/availability, add FCC Broadband Data Collection
(National Broadband Map) or Massachusetts Broadband Institute data separately.

## Geography finding

The current latitude/longitude fields come from the Census ZCTA Gazetteer
representative point.

The current `Town` label is created from the MassGIS school shapefile's
`TOWN` / `TOWN_MAIL` field grouped by ZIPCODE. That means it is a
**school-location town label**, not necessarily the canonical ZIP/postal place.

This is why 01434 currently appears as AYER even though 01434 is the Devens
ZIP/ZCTA. The new script keeps:

- `School_Town_Label`
- `Postal_Place`
- `Display_Place`
- `Postal_Place_Source`
- `Label_Audit_Flag`

as separate fields.

## Run

Put `post_nov2_gerber_followup.py` in the existing `Analysis/` folder next to
`post_gerber_pc3_extension.py`.

From `Analysis/`:

```bash
python post_nov2_gerber_followup.py \
  --primary final_results_2024/data/primary_analysis_C_math_ela_attainment_411.csv \
  --master final_results_2024/data/master_acs_mcas_2024_413.csv \
  --schools-shp Schools/SCHOOLS_PT.shp \
  --output final_results_2024/post_nov2_followup
```

If the local MassGIS school shapefile is not available, omit `--schools-shp`.
If the primary CSV already contains the current `Town` field, the script will
retain it as `School_Town_Label`.

## Main outputs

- `00_dataset_row_counts.csv`
- `01_k2_to_k6_3pc_metrics.csv`
- `02_k2_to_k6_initialization_stability.csv`
- `03_k3_k4_k5_cluster_profiles.csv`
- `04_k3_k4_k5_zip_assignments.csv`
- `05_cluster_labels_to_investigate.csv`
- `06_kmeans_model_selection_grid.csv`
- `06b_pca_kde_bandwidth_cv.csv`
- `06c_pca_kde_outlier_review.csv`
- `07_zip_coordinate_town_label_audit.csv`
- `08_ayer_devens_shirley_comparison.csv`
- `09_MISMATCH_MAP_METHODOLOGY.md`
- `10_k2_to_k6_silhouette.png`
- `11_k3_pc1_vs_pc3.png`
- `11_k4_pc1_vs_pc3.png`
- `11_k5_pc1_vs_pc3.png`
- `12_k5_participation_3d_triplot.html`
- `13_pc3_geographic_map_refined.png`
- `14_acs_broadband_subscription_map.png`
- `15_acs_broadband_subscription_map.html`
- `15b_pca_kde_density_outlier_map.png`
- `16_nov2_followup_enriched_411.csv`
- `NOV2_FOLLOWUP_FINDINGS.md`

## How to interpret the "grid search"

This is not supervised GridSearchCV because there is no target label or
ground-truth cluster assignment.

Instead it is a structured unsupervised sensitivity grid across:

- k = 2..8
- PCA dimensionality
- direct 10-feature space
- StandardScaler
- RobustScaler
- MinMaxScaler
- unscaled features (diagnostic only)

Use multiple criteria together:

- silhouette (higher better)
- Calinski-Harabasz (higher better)
- Davies-Bouldin (lower better)
- cluster-size balance
- initialization stability
- substantive interpretability

Do not choose k from one curve alone.

## Scaling

Keep scaling for the main KMeans analysis.

The features use very different units:
- income is in dollars
- rates are proportions
- MCAS measures are percentages

Without scaling, Euclidean distance is dominated by high-magnitude variables,
especially income. The unscaled row in the grid is included only to demonstrate
that sensitivity, not as a recommended specification.

## Recommended next external data step

For actual broadband **availability/coverage**, integrate one of:

1. FCC Broadband Data Collection / National Broadband Map fixed availability
   data, ideally summarized to ZCTA/ZIP; or
2. Massachusetts Broadband Institute infrastructure / deployment data.

That is a different concept from the current ACS household subscription measure.
