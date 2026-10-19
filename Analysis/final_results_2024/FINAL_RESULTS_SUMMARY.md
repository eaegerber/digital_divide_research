# Massachusetts Digital Divide — 2024 Unsupervised Analysis

## Research objective

Assess whether socioeconomic status, household digital access, and educational
attainment reveal meaningful statewide patterns across Massachusetts ZIP/ZCTA
areas, and identify places whose education profile differs from their broader
socioeconomic/digital-access profile.

## Data

Primary merged dataset:
- 413 MCAS school-location ZIPs
- 412 matched to a 2024 ACS ZCTA
- 411 complete cases for the primary analysis

Sources:
- 2024 ACS 5-year estimates
- 2024 Massachusetts DESE MCAS
- MassGIS school locations for school-to-ZIP assignment
- Census ZCTA Gazetteer for geographic coordinates

Important geographic limitation:
MCAS outcomes are assigned using the physical ZIP of the school, not the
residential ZIP of each student.

## Primary model

Primary specification: C_math_ela_attainment

Features:
- Broadband rate
- No-computer rate
- Computer-without-internet rate
- Median family income
- Family share below $25k
- Family share at/above $150k
- Math Meeting/Exceeding %
- ELA Meeting/Exceeding %

N = 411

### PCA

PC1 explained variance: 66.6%
PC1 + PC2 cumulative variance: 80.9%

PC1 represents a joint socioeconomic, digital-access, and educational-attainment
gradient.

Large positive PC1 loadings:
- High-income share
- Median family income
- Math attainment
- ELA attainment
- Broadband access

Large negative PC1 loadings:
- No-computer rate
- Low-income share
- Computer-without-internet rate

Approximate PC1 domain contribution:
- Income: 40.4%
- Digital access: 32.5%
- Education attainment: 27.1%

## Clustering

KMeans selected k = 2 for the primary model.

Silhouette: 0.356

Cluster sizes:
- Cluster 0: 237
- Cluster 1: 174

KMeans initialization stability:
- Mean ARI across 100 starts: 1.000
- Minimum ARI: 1.000
- 100% of starts recovered the same partition

Bootstrap stability:
- Mean ARI: 0.860
- Median ARI: 0.919

KMeans/Ward agreement:
- 348 of 411 ZIPs agree
- 84.7% agreement
- ARI = 0.479

Interpretation:
The two groups are best treated as a useful segmentation of a largely continuous
statewide gradient, rather than evidence that Massachusetts contains exactly two
naturally discrete community types.

## Cluster profiles

Higher-profile cluster averages:
- Broadband rate: 94.9%
- Median family income: approximately $172,779
- Low-income family share: 3.4%
- High-income family share: 55.5%
- Math Meeting/Exceeding: 53.7%
- ELA Meeting/Exceeding: 50.8%

Lower-profile cluster averages:
- Broadband rate: 88.8%
- Median family income: approximately $102,809
- Low-income family share: 9.6%
- High-income family share: 29.7%
- Math Meeting/Exceeding: 29.3%
- ELA Meeting/Exceeding: 28.6%

These are descriptive ecological associations and should not be interpreted
causally.

## Comparison with baseline model

Baseline A used only compact socioeconomic and digital-access variables.

A vs C adjusted Rand index: 0.712.

After aligning arbitrary cluster labels:
- approximately 92.2% of ZIPs remain in the corresponding group
- 32 of 411 ZIPs are reclassified after education is added

Direction of reclassification:
- 28: higher SES/access group -> lower combined profile
- 4: lower SES/access group -> higher combined profile

Interpretation:
Educational attainment is strongly aligned with the existing SES/digital-access
gradient, but adds meaningful information for a smaller subset of ZIPs.

## Reclassified ZIP sample-size audit

Of the 32 reclassified ZIPs:
- 30 have at least 50 Math/ELA students
- 23 have at least 200 Math/ELA students
- only 2 have fewer than 50

Therefore, the reclassification pattern is not generally explained by very small
MCAS sample sizes.

Two very-small-sample cases should be flagged:
- 02116
- 01223

Do not describe reclassified ZIPs as "overperforming" or "underperforming."
Preferred terminology:
- reclassified ZIPs
- profile-mismatch ZIPs
- ZIPs whose education profile differs from their SES/digital-access grouping

## Participation sensitivity

Adding Math/ELA participation produces a distinct second PCA dimension.

Participation contributes approximately 95% of PC2 in the Math/ELA participation
specification.

This suggests participation captures information largely orthogonal to the main
SES/access/attainment gradient.

## Science sensitivity

Adding Science attainment alone preserves the k=2 solution.

Adding Science participation changes KMeans to k=3, but the third cluster contains
only 5 ZIPs.

That solution is substantially less stable:
- Mean initialization ARI: approximately 0.480
- Mean bootstrap ARI: approximately 0.595
- KMeans/Ward ARI: approximately 0.258

Conclusion:
The Science-participation third cluster should be treated as an outlier/sensitivity
phenomenon rather than evidence for a third statewide community type.

## Detailed-variable sensitivity

Adding all 16 income shares and many overlapping digital-access measures weakens
PCA concentration and cluster separation.

These expanded models are useful sensitivity checks but should not replace the
more interpretable primary model.

## Geography

For the primary model:
- PC1 vs latitude: r approximately 0.09
- PC1 vs longitude: r approximately 0.28

The dominant PCA gradient therefore is not simply a north/south or east/west
geographic gradient.

## Main conclusion

Massachusetts ZIP-level socioeconomic status, household digital access, and
educational attainment largely co-vary along a common statewide gradient.

Educational attainment mostly reinforces the original SES/digital-access structure,
but identifies a smaller subset of ZIPs whose education profile changes their
joint segmentation.

Participation behaves differently and contributes a largely separate source of
variation.

## Major limitations

1. School physical ZIP is not student residential ZIP.
2. USPS ZIP and Census ZCTA are related but not identical geographic concepts.
3. ACS 2024 5-year estimates pool multiple years of survey observations.
4. Analysis is ecological and cross-sectional.
5. Associations must not be interpreted causally.
6. MCAS grades 3-8 are pooled across grades.
7. Suppressed MCAS percentages remain missing rather than being reconstructed.
8. Cluster membership is a descriptive segmentation of a continuum.
