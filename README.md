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
