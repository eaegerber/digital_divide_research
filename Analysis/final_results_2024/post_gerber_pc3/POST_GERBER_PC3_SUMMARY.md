# Post-Gerber PC3 / geography extension

## Key PCA findings

- Original 8-feature model: PC1=66.63%, PC2=14.30%, PC3=7.14%; cumulative PC1-PC3=88.06%.
- Participation-retained 10-feature model: PC1=53.40%, PC2=15.39%, PC3=11.31%; cumulative PC1-PC3=80.10%.
- In the participation model, PC2 is dominated by Math/ELA participation; PC3 is driven primarily by digital-access contrasts, with additional education/income contribution.

## Why low-income and high-income vectors are not opposite

- Correlation between low-income share and high-income share: -0.611.
- Angle between the loading vectors in PC1-PC2: 125.7 degrees.
- Angle in PC1-PC2-PC3: 127.7 degrees.
- Mean middle-income share (neither <25k nor >=150k): 49.4%.
- Therefore these two shares are not complements and should not be expected to point exactly 180 degrees apart.

## KMeans after retaining PC3 (participation model)

- k=2 silhouette: 0.401.
- k=4 silhouette: 0.375.
- k=6 silhouette: 0.331.
- k=2 remains the strongest separation. k=4 and k=6 can be inspected as exploratory subgroups, but they should not be promoted as better-supported statewide cluster counts solely because they are more granular.
- In the current 3-PC participation result, k=4 creates one very small participation-outlier cluster, which is a warning that the extra granularity may be isolating unusual observations rather than broad population types.

## Cross-domain mismatch analysis

- No extreme cross-domain mismatch: 404
- Low SES + high digital access: 3
- High SES + low digital access: 3
- High SES + high education + low digital access: 1

These mismatch categories are screening flags, not causal or policy conclusions. Use them to choose ZIPs for deeper data-quality, infrastructure, and policy investigation.

## Geography

- Interactive Massachusetts and Boston-area point maps include ZIP/town hover information, PC1-PC3, raw domain measures, and mismatch flags.
- Optional ZCTA polygon choropleths use the Census 2020 cartographic ZCTA boundary file for visualization and can be generated with `--build-polygons`.
- Optional population density combines 2024 ACS B01003 population with 2024 Census Gazetteer ZCTA land area and can be generated with `--build-density`.

## Methodological caution

MCAS outcomes are aggregated to the physical ZIP of each school, not student residence. This remains a school-location ecological analysis.
