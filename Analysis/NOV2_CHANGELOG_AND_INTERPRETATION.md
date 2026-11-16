# What changed after the November 2 meeting

## 1. k=5 is now explicitly evaluated

The previous extension emphasized k=2, k=4, and k=6. The new follow-up script
adds k=3 and k=5, exports every ZIP's k=3/k=4/k=5 assignment, cluster profiles,
representative ZIPs, within-cluster extremes, and a k=5 interactive 3D triplot.

Important: do not call a bend in the silhouette curve an "elbow" without
qualification. The classical elbow heuristic is usually based on KMeans inertia
(within-cluster sum of squares). Silhouette should be interpreted as separation
and cohesion. k should be chosen using several criteria and interpretability.

## 2. "Grid search" is reframed as unsupervised model-selection sensitivity

A grid search can help, but there is no supervised target to score against.
The new grid evaluates:
- k = 2 through 8
- 2-PC, 3-PC, and 4-PC spaces
- the full 10-feature representation
- standard, robust, min-max, and unscaled sensitivity

The primary solution should stay scaled.

## 3. Broadband map terminology is corrected

The existing `Broadband_Rate` is an ACS household subscription/adoption
measure. A map of it is useful, but it is not true infrastructure coverage.

The new output is explicitly titled:
`ACS household broadband subscription rate`

Actual coverage/availability should be added from FCC BDC or MBI data.

## 4. PC3 map color range is made more informative

ZIP 01434 is such a large PC3 outlier that a raw min-max color scale compresses
most other observations into similar shades.

The refined map clips the *display scale only* at the 98th percentile of
absolute PC3 and keeps the raw PC3 values unchanged. It also labels the most
extreme PC3 ZIPs so they can be audited.

## 5. Town labels are no longer treated as canonical ZIP labels

Current provenance:
- ZIP/ZCTA: merged analysis key
- Latitude/Longitude: Census ZCTA Gazetteer representative point
- `Town`: MassGIS school shapefile `TOWN`/`TOWN_MAIL` grouped by ZIP

So `Town` is fundamentally a school-location label.

For 01434:
- ZCTA/ZIP is 01434
- canonical/display place should be Devens
- school-derived label currently says Ayer

The new audit output keeps both fields instead of silently overwriting one with
the other.

## 6. Mismatch map methodology is documented

The output now explains:
- exact standardized features used in SES/digital/education/participation scores
- 25th/75th percentile threshold logic
- category precedence
- digital-access residual regression
- education residual regression
- interpretation limitations

## 7. Dataset size is explicit

The existing project files show:
- merged master: 413 rows / ZIP-ZCTAs
- primary complete-case model: 411 rows / ZIP-ZCTAs

The new script writes these counts and checks duplicate ZIP keys.

## 8. A model-based PCA density/outlier diagnostic is added

To address the suggestion to make the unusual-ZIP analysis more model-based,
the follow-up now fits a Gaussian kernel density estimator in the first three
participation-model PCs.

The KDE bandwidth is chosen by 5-fold cross-validated held-out log density.

Each ZIP gets:
- PCA log density
- KDE outlier score = negative log density
- outlier percentile
- dominant PC axis (largest absolute PC coordinate)
- top loading variables for that dominant PC

This is complementary to the mismatch labels:
- mismatch labels are interpretable, rule-based domain contrasts
- residuals are conditional linear-model deviations
- KDE is a multivariate "how unusual is this point in PCA space?" score

If all three methods flag the same ZIP, that is a stronger reason to audit it.
