#!/usr/bin/env python3
"""
Download 2024 Census Gazetteer ZCTA coordinates and optionally subset them to
the ZIP/ZCTAs in the digital-divide analysis.

The Census Gazetteer is small (~1 MB compressed) and includes a representative
latitude/longitude for every ZCTA, which is sufficient for:
- plotting cluster assignments geographically as points
- coloring PCA observations by east/west (longitude)
- coloring PCA observations by north/south (latitude)

Example:
python build_zcta_geography_2024.py \
  --analysis acs_2024_outputs_expanded/mcas_acs_2024_expanded.csv \
  --output zcta_geography_2024
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests


URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2024_Gazetteer/2024_Gaz_zcta_national.zip"
)


def zfill5(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.extract(r"(\d+)", expand=False)
        .str.zfill(5)
    )


def download_gazetteer() -> pd.DataFrame:
    print("Downloading 2024 Census ZCTA Gazetteer...")
    r = requests.get(
        URL,
        timeout=120,
        headers={"User-Agent": "digital-divide-research/2.0"},
    )
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        txt_names = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        if not txt_names:
            raise RuntimeError("No .txt Gazetteer file found inside ZIP.")
        with zf.open(txt_names[0]) as fh:
            df = pd.read_csv(fh, sep="\t", dtype="string")

    df.columns = [str(c).strip() for c in df.columns]

    required = ["GEOID", "ALAND", "AWATER", "ALAND_SQMI", "AWATER_SQMI",
                "INTPTLAT", "INTPTLONG"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Gazetteer is missing expected columns: {missing}. "
            f"Available: {df.columns.tolist()}"
        )

    out = df[required].copy()
    out = out.rename(
        columns={
            "GEOID": "ZIPCODE",
            "INTPTLAT": "Latitude",
            "INTPTLONG": "Longitude",
            "ALAND": "Land_Area_m2",
            "AWATER": "Water_Area_m2",
            "ALAND_SQMI": "Land_Area_sqmi",
            "AWATER_SQMI": "Water_Area_sqmi",
        }
    )
    out["ZIPCODE"] = zfill5(out["ZIPCODE"])

    for c in [
        "Latitude", "Longitude", "Land_Area_m2", "Water_Area_m2",
        "Land_Area_sqmi", "Water_Area_sqmi",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--analysis",
        type=Path,
        default=None,
        help="Optional analysis CSV containing ZIPCODE.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("zcta_geography_2024"),
    )
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    gaz = download_gazetteer().sort_values("ZIPCODE")
    gaz.to_csv(args.output / "zcta_2024_gazetteer_all.csv", index=False)
    print(f"Gazetteer ZCTAs: {len(gaz):,}")

    if args.analysis is not None:
        analysis = pd.read_csv(args.analysis, dtype={"ZIPCODE": str})
        analysis["ZIPCODE"] = zfill5(analysis["ZIPCODE"])

        zips = set(analysis["ZIPCODE"].dropna())
        subset = gaz[gaz["ZIPCODE"].isin(zips)].copy()
        missing = sorted(zips - set(subset["ZIPCODE"]))

        subset.to_csv(
            args.output / "analysis_zcta_coordinates.csv",
            index=False,
        )
        pd.DataFrame({"ZIPCODE": missing}).to_csv(
            args.output / "analysis_zips_without_gazetteer.csv",
            index=False,
        )

        merged = analysis.merge(
            subset,
            on="ZIPCODE",
            how="left",
            validate="one_to_one",
        )
        merged.to_csv(
            args.output / "analysis_with_zcta_coordinates.csv",
            index=False,
        )

        print(f"Analysis ZIPs: {len(zips):,}")
        print(f"Matched coordinates: {len(subset):,}")
        print(f"Unmatched coordinates: {len(missing):,}")
        print(
            f"Saved: {args.output / 'analysis_zcta_coordinates.csv'}"
        )


if __name__ == "__main__":
    main()
