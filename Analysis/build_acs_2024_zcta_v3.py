#!/usr/bin/env python3
"""
Build an expanded 2024 ACS 5-year ZCTA dataset for the digital-divide project.

Adds:
- detailed B28002 internet-subscription measures
- all 16 B19101 family-income shares
- compact legacy SES/digital variables for comparison

Tables:
- B28002: Presence and Types of Internet Subscriptions in Household
- B28003: Presence of a Computer and Type of Internet Subscription in Household
- B19101: Family Income in the Past 12 Months
- B19113: Median Family Income in the Past 12 Months

Example:
python build_acs_2024_zcta_v3.py \
  --mcas mcas_zip_outputs_2024_expanded/mcas_zip_math_ela_science.csv \
  --output acs_2024_outputs_expanded
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests


YEAR = 2024
BASE = f"https://api.census.gov/data/{YEAR}/acs/acs5"

VARS = {
    # B28002: Presence and Types of Internet Subscriptions in Household
    "B28002_001E": "ACS_B28002_Total_Households",
    "B28002_002E": "ACS_Internet_Subscription",
    "B28002_003E": "ACS_Dialup_Only_B28002",
    "B28002_004E": "ACS_Broadband_Any",
    "B28002_005E": "ACS_Cellular_Data",
    "B28002_006E": "ACS_Cellular_Only",
    "B28002_007E": "ACS_Cable_Fiber_DSL",
    "B28002_008E": "ACS_Cable_Fiber_DSL_Only",
    "B28002_009E": "ACS_Satellite",
    "B28002_010E": "ACS_Satellite_Only",
    "B28002_011E": "ACS_Other_Service_Only",
    "B28002_012E": "ACS_Internet_Without_Subscription",
    "B28002_013E": "ACS_No_Internet_Access",

    # B28003: computer / internet
    "B28003_001E": "ACS_Total_Households",
    "B28003_002E": "ACS_Has_Computer",
    "B28003_003E": "ACS_Dialup_Only",
    "B28003_004E": "ACS_Broadband_Subscription",
    "B28003_005E": "ACS_Computer_No_Internet_Subscription",
    "B28003_006E": "ACS_No_Computer",

    # B19101: family income bins
    "B19101_001E": "ACS_Total_Families",
    "B19101_002E": "ACS_Income_lt10k",
    "B19101_003E": "ACS_Income_10k_14999",
    "B19101_004E": "ACS_Income_15k_19999",
    "B19101_005E": "ACS_Income_20k_24999",
    "B19101_006E": "ACS_Income_25k_29999",
    "B19101_007E": "ACS_Income_30k_34999",
    "B19101_008E": "ACS_Income_35k_39999",
    "B19101_009E": "ACS_Income_40k_44999",
    "B19101_010E": "ACS_Income_45k_49999",
    "B19101_011E": "ACS_Income_50k_59999",
    "B19101_012E": "ACS_Income_60k_74999",
    "B19101_013E": "ACS_Income_75k_99999",
    "B19101_014E": "ACS_Income_100k_124999",
    "B19101_015E": "ACS_Income_125k_149999",
    "B19101_016E": "ACS_Income_150k_199999",
    "B19101_017E": "ACS_Income_200k_plus",

    # B19113: median family income
    "B19113_001E": "ACS_Median_Family_Income",
}


INCOME_COUNT_COLS = {
    "Income_Share_lt10k": "ACS_Income_lt10k",
    "Income_Share_10k_14999": "ACS_Income_10k_14999",
    "Income_Share_15k_19999": "ACS_Income_15k_19999",
    "Income_Share_20k_24999": "ACS_Income_20k_24999",
    "Income_Share_25k_29999": "ACS_Income_25k_29999",
    "Income_Share_30k_34999": "ACS_Income_30k_34999",
    "Income_Share_35k_39999": "ACS_Income_35k_39999",
    "Income_Share_40k_44999": "ACS_Income_40k_44999",
    "Income_Share_45k_49999": "ACS_Income_45k_49999",
    "Income_Share_50k_59999": "ACS_Income_50k_59999",
    "Income_Share_60k_74999": "ACS_Income_60k_74999",
    "Income_Share_75k_99999": "ACS_Income_75k_99999",
    "Income_Share_100k_124999": "ACS_Income_100k_124999",
    "Income_Share_125k_149999": "ACS_Income_125k_149999",
    "Income_Share_150k_199999": "ACS_Income_150k_199999",
    "Income_Share_200k_plus": "ACS_Income_200k_plus",
}


def zfill5(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.extract(r"(\d+)", expand=False)
        .str.zfill(5)
    )


def download_acs(api_key: str) -> pd.DataFrame:
    params = {
        "get": "NAME," + ",".join(VARS.keys()),
        "for": "zip code tabulation area:*",
        "key": api_key,
    }
    print("Downloading 2024 ACS 5-year ZCTA estimates from Census API...")
    r = requests.get(
        BASE,
        params=params,
        timeout=120,
        headers={"User-Agent": "digital-divide-research/2.0"},
    )

    if not r.ok:
        body = r.text[:800].replace("\n", " ")
        raise RuntimeError(
            f"Census API request failed with HTTP {r.status_code}. "
            f"Response: {body}"
        )

    try:
        payload = r.json()
    except requests.exceptions.JSONDecodeError as exc:
        body = r.text[:800].replace("\n", " ")
        raise RuntimeError(
            "Census API returned a non-JSON response. "
            f"Response starts with: {body}"
        ) from exc

    df = pd.DataFrame(payload[1:], columns=payload[0])
    df = df.rename(columns=VARS)
    df = df.rename(columns={"zip code tabulation area": "ZIPCODE"})
    df["ZIPCODE"] = zfill5(df["ZIPCODE"])

    for c in VARS.values():
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] < 0, c] = np.nan

    return df


def safe_rate(num: pd.Series, den: pd.Series) -> pd.Series:
    return np.where(den > 0, num / den, np.nan)


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Baseline digital-access measures retained for comparability.
    out["Broadband_Rate"] = safe_rate(
        out["ACS_Broadband_Subscription"], out["ACS_Total_Households"]
    )
    out["No_Computer_Rate"] = safe_rate(
        out["ACS_No_Computer"], out["ACS_Total_Households"]
    )
    out["Computer_No_Internet_Rate"] = safe_rate(
        out["ACS_Computer_No_Internet_Subscription"], out["ACS_Total_Households"]
    )

    # Detailed B28002 internet measures.
    den = out["ACS_B28002_Total_Households"]
    b28002_rates = {
        "Internet_Subscription_Rate": "ACS_Internet_Subscription",
        "Dialup_Only_Rate": "ACS_Dialup_Only_B28002",
        "Broadband_Any_Rate": "ACS_Broadband_Any",
        "Cellular_Data_Rate": "ACS_Cellular_Data",
        "Cellular_Only_Rate": "ACS_Cellular_Only",
        "Cable_Fiber_DSL_Rate": "ACS_Cable_Fiber_DSL",
        "Cable_Fiber_DSL_Only_Rate": "ACS_Cable_Fiber_DSL_Only",
        "Satellite_Rate": "ACS_Satellite",
        "Satellite_Only_Rate": "ACS_Satellite_Only",
        "Other_Service_Only_Rate": "ACS_Other_Service_Only",
        "Internet_Without_Subscription_Rate": "ACS_Internet_Without_Subscription",
        "No_Internet_Access_Rate": "ACS_No_Internet_Access",
    }
    for rate_name, count_name in b28002_rates.items():
        out[rate_name] = safe_rate(out[count_name], den)

    # Compact SES variables retained as the baseline.
    low_income = (
        out["ACS_Income_lt10k"]
        + out["ACS_Income_10k_14999"]
        + out["ACS_Income_15k_19999"]
        + out["ACS_Income_20k_24999"]
    )
    high_income = out["ACS_Income_150k_199999"] + out["ACS_Income_200k_plus"]

    out["Low_Income_Share_lt25k"] = safe_rate(low_income, out["ACS_Total_Families"])
    out["High_Income_Share_150k_plus"] = safe_rate(
        high_income, out["ACS_Total_Families"]
    )

    # All 16 original B19101 categories as shares.
    for share_name, count_name in INCOME_COUNT_COLS.items():
        out[share_name] = safe_rate(
            out[count_name], out["ACS_Total_Families"]
        )

    # Retain the older grouped shares as named sensitivity features.
    grouped_bins = {
        "Income_Grouped_Share_15k_24999":
            out["ACS_Income_15k_19999"] + out["ACS_Income_20k_24999"],
        "Income_Grouped_Share_25k_34999":
            out["ACS_Income_25k_29999"] + out["ACS_Income_30k_34999"],
        "Income_Grouped_Share_35k_49999": (
            out["ACS_Income_35k_39999"]
            + out["ACS_Income_40k_44999"]
            + out["ACS_Income_45k_49999"]
        ),
        "Income_Grouped_Share_50k_74999":
            out["ACS_Income_50k_59999"] + out["ACS_Income_60k_74999"],
        "Income_Grouped_Share_100k_149999":
            out["ACS_Income_100k_124999"] + out["ACS_Income_125k_149999"],
    }
    for name, numerator in grouped_bins.items():
        out[name] = safe_rate(numerator, out["ACS_Total_Families"])

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mcas",
        type=Path,
        default=None,
        help="Optional ZIP-level MCAS CSV, ideally mcas_zip_math_ela_science.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("acs_2024_outputs_expanded"),
    )
    p.add_argument(
        "--api-key",
        default=None,
        help=(
            "Census API key. Prefer CENSUS_API_KEY environment variable "
            "so the key is not stored in shell history."
        ),
    )
    args = p.parse_args()

    api_key = args.api_key or os.environ.get("CENSUS_API_KEY")
    if not api_key:
        print(
            "ERROR: Census API key required.\n"
            "Set it with: export CENSUS_API_KEY='YOUR_KEY'"
        )
        sys.exit(2)

    args.output.mkdir(parents=True, exist_ok=True)

    acs = derive_features(download_acs(api_key)).sort_values("ZIPCODE")
    all_path = args.output / "acs_2024_all_zcta_expanded.csv"
    acs.to_csv(all_path, index=False)

    summary = {
        "acs_year": YEAR,
        "acs_product": "ACS 5-year estimates",
        "all_zctas_downloaded": int(acs["ZIPCODE"].nunique()),
        "tables": ["B28002", "B28003", "B19101", "B19113"],
        "income_share_variables": list(INCOME_COUNT_COLS.keys()),
    }

    print(f"Saved all ZCTAs: {all_path}")
    print(f"All ZCTAs downloaded: {summary['all_zctas_downloaded']:,}")

    if args.mcas is not None:
        mcas = pd.read_csv(args.mcas, dtype={"ZIPCODE": str})
        mcas["ZIPCODE"] = zfill5(mcas["ZIPCODE"])

        mcas_zips = set(mcas["ZIPCODE"].dropna())
        acs_zips = set(acs["ZIPCODE"].dropna())
        matched = mcas_zips & acs_zips
        unmatched = sorted(mcas_zips - acs_zips)

        subset = acs[acs["ZIPCODE"].isin(mcas_zips)].copy()
        subset.to_csv(
            args.output / "acs_2024_mcas_zcta_expanded.csv",
            index=False,
        )

        merged = mcas.merge(
            acs, on="ZIPCODE", how="left", validate="one_to_one"
        )
        merged_path = args.output / "mcas_acs_2024_expanded.csv"
        merged.to_csv(merged_path, index=False)

        pd.DataFrame({"ZIPCODE": unmatched}).to_csv(
            args.output / "mcas_zips_without_acs_zcta.csv",
            index=False,
        )

        summary.update(
            {
                "mcas_zip_count": len(mcas_zips),
                "mcas_zips_matched_to_acs": len(matched),
                "mcas_zips_without_acs_zcta": len(unmatched),
                "mcas_to_acs_match_rate": (
                    len(matched) / len(mcas_zips) if mcas_zips else None
                ),
                "merged_rows": int(len(merged)),
            }
        )

        print("\nMCAS -> ACS ZCTA audit")
        print(json.dumps(summary, indent=2))
        print(f"\nSaved merged file: {merged_path}")

    (args.output / "acs_build_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
