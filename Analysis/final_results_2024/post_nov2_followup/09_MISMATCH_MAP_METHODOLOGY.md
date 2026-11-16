# Mismatch map methodology

## What data are used?

The mismatch analysis uses the same 411 complete-case school-location ZIP/ZCTA observations as the participation-retained PCA extension.

Four standardized domain scores are constructed:

- **SES score** = mean(z median family income, -z low-income share, z high-income share).
- **Digital-access score** = mean(z broadband subscription rate, -z no-computer rate, -z computer-without-internet rate).
- **Education score** = mean(z Math attainment, z ELA attainment).
- **Participation score** = mean(z Math participation, z ELA participation).

## Threshold-based mismatch labels

High/low is defined using the 75th/25th percentile of the relevant domain score within the 411-row analysis sample.

- SES 25th percentile: -0.499
- SES 75th percentile: 0.591
- Digital access 25th percentile: -0.450
- Digital access 75th percentile: 0.633
- Education 25th percentile: -0.787
- Education 75th percentile: 0.733

The code first assigns narrower three-domain combinations (for example high SES + high education + low digital access), then assigns broader two-domain mismatches to observations that remain unassigned.

These categories are screening flags, not learned classes and not causal findings.

## Residual-based mismatch diagnostic

A linear regression predicts the digital-access score from SES and education. The digital-access residual is:

`observed digital access - predicted digital access`

Large negative residuals indicate places with lower digital access than would be expected from their SES and education profile; large positive residuals indicate the reverse.

A second regression predicts education from SES and digital access and produces an analogous education residual.

## Important interpretation caveats

- MCAS is aggregated to the physical ZIP of schools, not student home ZIP.
- ACS broadband variables measure household subscription/adoption, not infrastructure availability.
- Small ZCTA denominators can make ACS estimates unusually volatile; flagged ZIPs should be audited before substantive interpretation.
- The threshold labels depend on this sample's quartiles; they are exploratory, not externally validated cutoffs.
