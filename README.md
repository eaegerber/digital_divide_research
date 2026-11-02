# Digital Divide Research

## 2024 Statewide Massachusetts Analysis

This branch documents the current statewide extension of the Massachusetts
digital-divide research project.

### Updated data pipeline

The analysis now combines:

- 2024 Massachusetts DESE MCAS Math, ELA, and Science outcomes
- MassGIS school locations for school-to-physical-ZIP assignment
- 2024 ACS 5-year socioeconomic and household digital-access estimates
- Census ZCTA geography for matching and spatial visualization

The merged dataset contains 413 school-location ZIPs. The primary complete-case
analysis contains 411 ZIP/ZCTA observations.

### Primary analysis

The preferred unsupervised specification uses:

- Broadband rate
- No-computer rate
- Computer-without-internet rate
- Median family income
- Family share below $25k
- Family share at/above $150k
- Math Meeting/Exceeding %
- ELA Meeting/Exceeding %

Main results:

- PC1 explains 66.6% of standardized variance
- PC1 + PC2 explain 80.9%
- PC1 represents a joint socioeconomic / digital-access /
  educational-attainment gradient
- KMeans selects k=2 with silhouette = 0.356
- The primary k=2 solution is stable across 100 KMeans initializations
- Adding education preserves the corresponding baseline grouping for
  approximately 92% of observations
- 32 of 411 ZIPs are reclassified after Math/ELA attainment is added
- Participation largely defines a separate PCA dimension
- Science participation produces a small unstable five-ZIP outlier group and
  is treated as a sensitivity result rather than a primary statewide cluster

### Current primary dataset

Master merged data:

`Analysis/final_results_2024/data/master_acs_mcas_2024_413.csv`

Primary analysis output:

`Analysis/final_results_2024/data/primary_analysis_C_math_ela_attainment_411.csv`

### Results

A detailed current results summary is available at:

`Analysis/final_results_2024/FINAL_RESULTS_SUMMARY.md`

Publication-ready figures are in:

`Analysis/final_results_2024/final_figures/`

Supporting tables are in:

`Analysis/final_results_2024/tables/`

### Important limitation

MCAS outcomes are assigned using each school's physical ZIP rather than
students' residential ZIPs. Results are ecological, cross-sectional, and
descriptive and should not be interpreted causally.

### Next research decision

The next phase will be determined with Professor Gerber and Massachusetts
Broadband Institute input, particularly whether to:

1. incorporate broadband availability/infrastructure data to distinguish
   availability from household adoption; or
2. move toward supervised modeling of educational outcomes.

## Post-Gerber PC3 and geographic extension

Following additional research feedback, the analysis was extended to examine
the third principal component, retain Math/ELA participation, test alternative
cluster granularities, and investigate geographic/outlier patterns.

Key additions include:

- PC3 analysis for both the original 8-feature model and a 10-feature model
  retaining Math/ELA participation
- Interactive Plotly PC1/PC2/PC3 triplots with feature loading vectors
- KMeans comparisons using the first 2 PCs, first 3 PCs, and full standardized
  feature space
- Explicit k=2, k=4, and k=6 cluster-profile comparisons
- Interactive Massachusetts and Greater Boston maps with ZIP-level hover data
- Geographic mapping of PC3
- ZCTA polygon choropleths for cluster assignment and PC3
- Domain-mismatch analysis to identify ZIPs where SES, digital access, and
  educational attainment do not follow the dominant statewide relationship
- Residual analysis identifying places with unusually high or low digital
  access relative to their SES and educational profile

The extension preserves the original 411-observation Math/ELA model as the
benchmark rather than replacing it.

### Initial PC3 findings

For the original 8-feature model:

- PC1: 66.63%
- PC2: 14.30%
- PC3: 7.14%
- PC1-PC3 cumulative explained variance: approximately 88.1%

When Math/ELA participation is retained:

- PC1: 53.40%
- PC2: 15.39%
- PC3: 11.31%
- PC1-PC3 cumulative explained variance: approximately 80.1%

Participation remains concentrated primarily on PC2, while PC3 captures a
different mixture dominated by digital-access variation.

Adding PC3 does not materially alter the primary k=2 segmentation. More granular
k=4 and k=6 solutions were also examined, but have weaker cluster separation.

Detailed outputs are available under:

`Analysis/final_results_2024/post_gerber_pc3/`

The full extension code is:

`Analysis/post_gerber_pc3_extension.py`
