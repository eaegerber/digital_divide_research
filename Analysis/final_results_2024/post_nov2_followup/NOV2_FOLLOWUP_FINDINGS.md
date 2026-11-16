# November 2 Gerber follow-up outputs

## Dataset size

| Dataset                        |   Rows |   Unique_ZIP_ZCTA |   Duplicate_ZIP_rows |
|:-------------------------------|-------:|------------------:|---------------------:|
| Primary complete-case analysis |    411 |               411 |                    0 |
| Merged master dataset          |    413 |               413 |                    0 |

## k=3 / k=4 / k=5

The first-three-PC participation-model silhouette scores are:

- k=3: 0.358
- k=4: 0.375
- k=5: 0.327

Do not describe the silhouette curve as a classical "elbow" without qualification.
Silhouette is a separation/cohesion metric, whereas the classic elbow heuristic
normally refers to within-cluster sum of squares (inertia). Use the new metric
table plus cluster sizes, stability, and substantive interpretability together.

## Scaling / grid search

KMeans uses Euclidean distance, so the primary analysis should remain scaled.
The grid includes StandardScaler, RobustScaler, MinMaxScaler, and an explicitly
diagnostic unscaled version. Do not promote the unscaled solution because the
income variable is measured in dollars and would dominate distances.

A "grid search" is useful here as a structured sensitivity/model-selection
exercise, but it is not supervised hyperparameter tuning: there is no ground-truth
cluster label to optimize against.

## Broadband map

The generated broadband map uses the ACS `Broadband_Rate`. It should be called
a **household broadband subscription/adoption map**, not a coverage map.
True infrastructure availability/coverage should be added separately from an
FCC Broadband Data Collection / National Broadband Map or Massachusetts
Broadband Institute source.

## Geography / label audit

Latitude and longitude are Census ZCTA representative points.
The legacy `Town` label is derived from the MassGIS school shapefile and is
therefore a school-location town label, not a canonical ZIP/place name.

ZIP 01434 is explicitly preserved as:
- postal/display place: DEVENS
- school-town label: AYER

The new output keeps these as separate fields so the provenance is visible.

## Ayer / Devens / Shirley case study

|   ZIPCODE | Display_Place   | School_Town_Label   | Postal_Place   |   Latitude |   Longitude |   ACS_Total_Households |   ACS_Total_Families |   Broadband_Rate |   No_Computer_Rate |   Computer_No_Internet_Rate |   ACS_Median_Family_Income |   Low_Income_Share_lt25k |   High_Income_Share_150k_plus |   MCAS_Math_M_PLUS_E_PCT_ZIP |   MCAS_ELA_M_PLUS_E_PCT_ZIP |   MCAS_Math_STU_PART_PCT_WT |   MCAS_ELA_STU_PART_PCT_WT |   SES_Score |   Digital_Access_Score |   Education_Score |   Participation_Score |   PC1_Participation |   PC2_Participation |   PC3_Participation |   Digital_Access_Residual | Mismatch_Type                                  |   K3_3PC |   K4_3PC |   K5_3PC |
|----------:|:----------------|:--------------------|:---------------|-----------:|------------:|-----------------------:|---------------------:|-----------------:|-------------------:|----------------------------:|---------------------------:|-------------------------:|------------------------------:|-----------------------------:|----------------------------:|----------------------------:|---------------------------:|------------:|-----------------------:|------------------:|----------------------:|--------------------:|--------------------:|--------------------:|--------------------------:|:-----------------------------------------------|---------:|---------:|---------:|
|     01432 | AYER            | AYER                |                |    42.5666 |    -71.5751 |                   3567 |                 1885 |         0.905523 |          0.0518643 |                   0.0426128 |                     147309 |                0.061008  |                      0.490186 |                      40.5797 |                     44.4444 |                     99.6377 |                    99.6377 |    0.105439 |              -0.369026 |         0.0177283 |            0.363115   |           -0.2026   |           0.595752  |            0.482579 |               -0.435457   | No extreme cross-domain mismatch               |        1 |        1 |        0 |
|     01434 | DEVENS          | AYER                | DEVENS         |    42.5321 |    -71.6137 |                    260 |                  140 |         0.715385 |          0.119231  |                   0.165385  |                     222727 |                0.0285714 |                      0.9      |                      54.4715 |                     59.6774 |                     95.5691 |                    96      |    1.5573   |              -4.27723  |         0.904063  |           -2.04348    |           -1.86449  |          -1.58193   |            7.92031  |               -5.29109    | High SES + high education + low digital access |        2 |        2 |        4 |
|     01464 | SHIRLEY         | SHIRLEY             |                |    42.5737 |    -71.641  |                   2277 |                 1522 |         0.912604 |          0.0557751 |                   0.0316206 |                     138601 |                0.113009  |                      0.453351 |                      38.3764 |                     40.4806 |                     99.1402 |                    98.8965 |   -0.337097 |              -0.213017 |        -0.172084  |            0.00228545 |           -0.676626 |           0.0538209 |            0.21406  |                0.00524322 | No extreme cross-domain mismatch               |        1 |        1 |        1 |

The Devens record should be treated as an audit priority because it combines a
very large PC3 score with an unusually low ACS broadband-subscription rate and
a small ACS household denominator relative to neighboring ZIPs.
