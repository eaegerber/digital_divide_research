# Digital Divide Research — Next-Step Guide

## What Professor Gerber's latest direction implies

Your current notes indicate that **unsupervised learning should remain the primary focus for now**, with supervised/predictive work potentially coming later. The inherited sequence was EDA → correlations → PCA/factor analysis → later predictive/inferential modeling.

The new analysis file therefore does **not** jump straight into random forest or make regression the centerpiece. It first repairs the measurement issues, then deepens the unsupervised analysis.

## What the new code changes

1. Preserves an untouched raw dataset.
2. Audits missingness, constant variables, and suspicious education fields.
3. Converts household internet counts into rates:
   - Broadband_Rate
   - Fixed_Broadband_Rate
   - No_Computer_Rate
   - Computer_No_Internet_Rate
4. Treats the inherited income-bin variables as proportions (they already sum to about 1).
5. Creates:
   - Low_Income_Share_lt25k
   - High_Income_Share_150k_plus
6. Redoes a small number of interpretable EDA plots in original units.
7. Runs a **core PCA** using socioeconomic + digital-access measures only.
8. Saves explained variance, cumulative variance, PCA loadings, and scores.
9. Uses both elbow and silhouette diagnostics for K-means instead of assuming k=3.
10. Profiles clusters in original units so the clusters have substantive meaning.
11. Runs two-factor Factor Analysis as a second latent-structure check.
12. Runs a separate education-inclusive PCA only as a sensitivity analysis.
13. Keeps regression in an `optional_supervised` folder for later discussion rather than treating it as the main result.

## Verified findings from the current 32-ZIP dataset

### Data quality
- Rows: **32**
- Columns: **41**
- Non-missing Average_Rating observations: **24**
- Constant numeric columns: **Private_Awards, Private_Rating, PERCENT ALLOCATED_Families**
- Income-bin shares sum to approximately **0.9998** on average.
- `Private_Rating` has only the value **0**.
- Maximum inherited `Public_Rating`: **19**.

### Broadband count vs. broadband rate
The inherited raw-count result and normalized result are very different:

- Raw broadband household count vs Average_Rating:
  **r = -0.310, p = 0.140, n = 24**
- Broadband penetration rate vs Average_Rating:
  **r = -0.049, p = 0.821, n = 24**

This supports treating the inherited negative broadband-school-rating pattern cautiously: much of it appears to be related to using household counts rather than penetration rates.

### Core PCA
The core PCA uses only:
- Broadband_Rate
- No_Computer_Rate
- Computer_No_Internet_Rate
- Median family income
- Low-income share (<$25k)
- High-income share ($150k+)

The first component explains **79.9%** of variance.
The first two components together explain **89.0%**.

PC1 loadings are:

|                                  |    PC1 |    PC2 |
|:---------------------------------|-------:|-------:|
| Broadband_Rate                   |  0.433 | -0.404 |
| No_Computer_Rate                 | -0.385 | -0.226 |
| Computer_No_Internet_Rate        | -0.384 |  0.698 |
| Median income (dollars)_Families |  0.418 |  0.406 |
| Low_Income_Share_lt25k           | -0.418 | -0.1   |
| High_Income_Share_150k_plus      |  0.409 |  0.352 |

A reasonable *working interpretation* is that PC1 is a broad **socioeconomic/digital-access advantage-disadvantage axis**:
- broadband, median income, and high-income share load in one direction;
- no-computer, no-internet, and low-income share load in the opposite direction.

Do not lock in that label until you discuss the interpretation with Professor Gerber.

### K-means
Silhouette diagnostics:

|   k |   inertia |   silhouette_score |
|----:|----------:|-------------------:|
|   2 |    85.028 |              0.437 |
|   3 |    59.056 |              0.337 |
|   4 |    45.678 |              0.299 |
|   5 |    36.559 |              0.302 |
|   6 |    29.635 |              0.291 |

For this feature set, **k=2** has the highest silhouette score among k=2–6.

Cluster profiles:

|   Cluster |   n |   Broadband_Rate |   No_Computer_Rate |   Computer_No_Internet_Rate |   Median income (dollars)_Families |   Low_Income_Share_lt25k |   High_Income_Share_150k_plus |   Average_Rating |   Student_Teacher_Ratio |   Num_Public |   Num_Private |   Total: |
|----------:|----:|-----------------:|-------------------:|----------------------------:|-----------------------------------:|-------------------------:|------------------------------:|-----------------:|------------------------:|-------------:|--------------:|---------:|
|         0 |  19 |           0.9395 |             0.0265 |                      0.0331 |                           150607   |                   0.0755 |                        0.4804 |           3.6346 |                 22.2143 |       1.1579 |        1.7895 |  26377.2 |
|         1 |  13 |           0.8891 |             0.0473 |                      0.0629 |                            70328.3 |                   0.1961 |                        0.1973 |           3.9848 |                 19.9167 |       2.5385 |        1.3077 |  32682.8 |

The two clusters appear to separate:
- a higher-income / higher-broadband / lower-digital-exclusion group, and
- a lower-income / lower-broadband / higher-digital-exclusion group.

Average school rating is fairly similar between these two clusters in the current tiny sample, which is another reason not to overclaim an education relationship yet.

## What to write in your research notes

You can write the current result approximately as:

> The inherited exploratory analysis suggested a socioeconomic digital divide, but many internet-access measures were raw household counts and several school measures had unresolved aggregation problems. I therefore rebuilt the analysis around normalized household access rates and a smaller set of conceptually distinct socioeconomic features. In the current 32-ZIP sample, the first principal component explains roughly 80% of variance and contrasts higher broadband penetration, higher median income, and a larger high-income share against greater device/internet exclusion and a larger low-income share. K-means diagnostics favor a two-cluster solution in this small sample, broadly separating higher-resource/lower-exclusion and lower-resource/higher-exclusion ZIP codes. These results support a strong income–digital-access structure, while the relationship with education remains unresolved because the education measures require validation and Average_Rating is available for only 24 ZIP codes.

Use **associated with / patterned with / clusters suggest**, not causal language.

## What to show Professor Gerber next

Bring these outputs in this order:

1. `data_quality_audit.json`
2. `raw_count_vs_rate_broadband_rating.csv`
3. `eda/focused_correlation_matrix.png`
4. `pca/core_ses_digital_scree_and_cumulative.png`
5. `pca/core_ses_digital_loadings.csv`
6. `clustering/kmeans_silhouette.png`
7. `clustering/clusters_on_pca.png`
8. `clustering/cluster_profiles_original_units.csv`
9. `factor_analysis/factor_loadings.csv`

## What to ask Professor Gerber

1. **Is this the right unsupervised framing?**  
   “I kept the clustering/PCA centered on socioeconomic and digital-access measures and treated school rating as an external variable because the education fields have quality issues. Is that the way you want me to structure the unsupervised stage?”

2. **How should we interpret the main latent dimension?**  
   “PC1 explains about 80% of the variance and seems to contrast higher income/broadband with lower income/digital exclusion. Does it make sense to interpret this as a socioeconomic–digital access axis?”

3. **Should k=2 replace the inherited hard-coded k=3?**  
   “The old notebook manually chose k=3, but silhouette analysis currently favors k=2. Should I use the data-driven two-cluster solution, or do you want me to explore stability/other clustering methods before deciding?”

4. **Should Factor Analysis remain part of the project?**  
   “The original proposal mentioned PCA and/or Factor Analysis. I added a two-factor FA as a secondary check. Do you want both, or should I focus on PCA + clustering?”

5. **What should happen with education?**  
   “Private_Rating is constant zero, Public_Rating appears to be aggregated oddly, and Average_Rating is missing for 8 of 32 ZIPs. Should I reconstruct the GreatSchools aggregation, or switch to an official outcome such as MCAS/graduation data?”

6. **Can we expand statewide now?**  
   “The current merged sample is only 32 ZIPs. Should my next data task be obtaining statewide education data and merging it with the broader Massachusetts ACS/internet data?”

7. **What is the correct geographic unit?**  
   “Do you want the final analysis at ZIP/ZCTA, municipality, or school-district level? I want to decide that before expanding the data.”

8. **Do we know the years of each dataset?**  
   “Can we verify the ACS income year, ACS internet year, and GreatSchools/education year so the data are temporally aligned?”

9. **When should supervised learning start?**  
   “Once the unsupervised structure and dataset are cleaned/expanded, would you like the first supervised question to be school performance from internet + income, internet access from socioeconomic measures, or something else?”

10. **Is MBI still an intended stakeholder?**  
    “Should I design the eventual outputs around identifying underserved Massachusetts communities for an MBI-facing deliverable?”

## Recommended immediate next steps

Do these next, in order:

1. Run the new analysis file on the inherited CSV and review the outputs.
2. Discuss the PCA interpretation and k=2 clustering with Gerber.
3. Locate the raw school/GreatSchools aggregation data and the larger Massachusetts ACS files.
4. Resolve the geographic unit and years.
5. Rebuild/replace the education feature(s).
6. Expand the dataset statewide if approved.
7. Rerun the exact same pipeline on the expanded dataset.
8. Only after that, decide whether the supervised phase should be regression, tree-based prediction, or another question.

