#!/usr/bin/env python3
"""
Build a ZIP-level MCAS education dataset by joining:
  1) Massachusetts DESE MCAS Achievement Results CSV
  2) MassGIS SCHOOLS_PT.shp (school ID -> ZIP crosswalk)

Designed to handle a very large MCAS CSV by reading it in chunks.

USAGE
-----
Step 1: Inspect the large MCAS file to discover exact codes/labels:
python build_mcas_zip_dataset.py inspect \
  --mcas MCAS_Achievement_Results_20260927.csv

Step 2: Build the ZIP-level dataset after you know the exact values:
python build_mcas_zip_dataset.py build \
  --mcas MCAS_Achievement_Results_20260927.csv \
  --schools SCHOOLS_PT.shp \
  --year 2025 \
  --all-students "All Students" \
  --math-code MATH \
  --ela-code ELA \
  --output mcas_zip_outputs

Optional: restrict to one or more test grades:
  --grades 8
or
  --grades 5 8

Notes:
- IDs are read as strings so leading zeros are preserved.
- Multi-campus school IDs that map to more than one ZIP are EXCLUDED by default
  and written to multi_zip_school_ids.csv for manual review.
- Rows with suppressed/non-numeric MCAS values are retained in the filtered
  school-level file, but only valid numeric values contribute to aggregates.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError:
    gpd = None


INSPECT_COLS = ["SY", "ORG_TYPE", "STU_GRP", "SUBJECT_CODE", "TEST_GRADE"]

MCAS_COLS = [
    "SY",
    "DIST_CODE",
    "DIST_NAME",
    "ORG_CODE",
    "ORG_NAME",
    "ORG_TYPE",
    "TEST_GRADE",
    "SUBJECT_CODE",
    "STU_GRP",
    "M_PLUS_E_CNT",
    "M_PLUS_E_PCT",
    "STU_CNT",
    "STU_PART_PCT",
    "AVG_SCALED_SCORE",
    "AVG_SGP",
    "AVG_SGP_INCL",
    "ACH_PERCENTILE",
]


def normalize_id(series, width=8):
    """Normalize IDs while preserving leading zeros and removing Excel-style .0."""
    s = series.astype("string").str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    # Only zero-pad values that are purely numeric.
    is_numeric = s.str.fullmatch(r"\d+", na=False)
    s.loc[is_numeric] = s.loc[is_numeric].str.zfill(width)
    return s


def normalize_zip(series):
    """Return first 5-digit ZIP where possible."""
    s = series.astype("string").str.strip()
    extracted = s.str.extract(r"(\d{5})", expand=False)
    return extracted


def inspect_mcas(path, chunksize=250_000):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MCAS file not found: {path}")

    values = {c: set() for c in INSPECT_COLS}
    row_count = 0

    for chunk in pd.read_csv(
        path,
        usecols=lambda c: c in INSPECT_COLS,
        dtype="string",
        chunksize=chunksize,
        low_memory=False,
    ):
        row_count += len(chunk)
        for col in INSPECT_COLS:
            if col in chunk.columns:
                vals = chunk[col].dropna().astype(str).str.strip()
                values[col].update(vals.unique().tolist())

    print(f"\nTotal MCAS rows scanned: {row_count:,}\n")
    for col in INSPECT_COLS:
        print("=" * 80)
        print(col)
        print("=" * 80)
        vals = sorted(values[col])
        for v in vals:
            print(repr(v))
        print()

    print("\nNEXT:")
    print("Use the exact labels/codes above for --year, --all-students,")
    print("--math-code, --ela-code, and optionally --grades.")


def read_school_crosswalk(shp_path, output_dir):
    if gpd is None:
        raise ImportError(
            "geopandas is required. Install it with:\n"
            "  python -m pip install geopandas pyogrio"
        )

    shp_path = Path(shp_path)
    if not shp_path.exists():
        raise FileNotFoundError(f"MassGIS shapefile not found: {shp_path}")

    schools = gpd.read_file(shp_path)

    required = [
        "SCHID", "NAME", "ZIPCODE", "TOWN", "DIST_CODE",
        "DIST_NAME", "TYPE", "TYPE_DESC", "GRADES"
    ]
    missing = [c for c in required if c not in schools.columns]
    if missing:
        raise ValueError(
            f"MassGIS file is missing expected columns: {missing}\n"
            f"Available columns: {schools.columns.tolist()}"
        )

    schools = pd.DataFrame(schools[required].copy())
    schools["SCHID"] = normalize_id(schools["SCHID"], width=8)
    schools["ZIPCODE"] = normalize_zip(schools["ZIPCODE"])

    # Save raw cleaned crosswalk for inspection.
    schools.to_csv(output_dir / "massgis_school_crosswalk_cleaned.csv", index=False)

    # Identify IDs duplicated across multiple rows.
    dup_all = schools[schools.duplicated("SCHID", keep=False)].copy()
    if not dup_all.empty:
        dup_all.to_csv(output_dir / "duplicate_school_ids_all_rows.csv", index=False)

    zip_counts = (
        schools.dropna(subset=["SCHID", "ZIPCODE"])
        .groupby("SCHID")["ZIPCODE"]
        .nunique()
    )
    multi_zip_ids = set(zip_counts[zip_counts > 1].index)

    multi_zip = schools[schools["SCHID"].isin(multi_zip_ids)].sort_values(
        ["SCHID", "ZIPCODE"]
    )
    multi_zip.to_csv(output_dir / "multi_zip_school_ids.csv", index=False)

    # Safe mapping: one school ID -> exactly one ZIP.
    safe = schools[
        schools["SCHID"].notna()
        & schools["ZIPCODE"].notna()
        & ~schools["SCHID"].isin(multi_zip_ids)
    ].copy()

    # Some IDs may have duplicate points in the SAME ZIP. Collapse those safely.
    safe = (
        safe.sort_values(["SCHID", "ZIPCODE"])
        .drop_duplicates(subset=["SCHID"], keep="first")
    )

    safe.to_csv(output_dir / "massgis_safe_school_to_zip_crosswalk.csv", index=False)

    return schools, safe, multi_zip_ids


def filter_mcas(
    path,
    year,
    all_students,
    math_code,
    ela_code,
    grades=None,
    chunksize=250_000,
):
    path = Path(path)

    kept = []
    scanned = 0

    target_year = str(year).strip()
    target_group = str(all_students).strip()
    math_code = str(math_code).strip()
    ela_code = str(ela_code).strip()
    grade_set = None if not grades else {str(g).strip() for g in grades}

    for chunk in pd.read_csv(
        path,
        usecols=lambda c: c in MCAS_COLS,
        dtype="string",
        chunksize=chunksize,
        low_memory=False,
    ):
        scanned += len(chunk)

        # Normalize fields used in filters.
        for c in ["SY", "ORG_TYPE", "STU_GRP", "SUBJECT_CODE", "TEST_GRADE"]:
            if c in chunk.columns:
                chunk[c] = chunk[c].astype("string").str.strip()

        mask = (
            (chunk["SY"] == target_year)
            & (chunk["ORG_TYPE"].isin(["Public School", "Charter School"]))
            & (chunk["STU_GRP"] == target_group)
            & (chunk["SUBJECT_CODE"].isin([math_code, ela_code]))
        )

        if grade_set is not None:
            mask &= chunk["TEST_GRADE"].isin(grade_set)

        sub = chunk.loc[mask].copy()
        if len(sub):
            kept.append(sub)

    if not kept:
        raise ValueError(
            "No MCAS rows matched your filters.\n"
            f"year={year!r}, all_students={all_students!r}, "
            f"math_code={math_code!r}, ela_code={ela_code!r}, grades={grades!r}\n"
            "Run the 'inspect' command first and use the exact values it prints."
        )

    mcas = pd.concat(kept, ignore_index=True)
    mcas["ORG_CODE"] = normalize_id(mcas["ORG_CODE"], width=8)
    mcas["DIST_CODE"] = normalize_id(mcas["DIST_CODE"], width=8)

    # Friendly subject names.
    mcas["SUBJECT"] = np.where(
        mcas["SUBJECT_CODE"] == math_code,
        "Math",
        np.where(mcas["SUBJECT_CODE"] == ela_code, "ELA", pd.NA),
    )

    # DESE percentage fields are published as strings such as "99%".
    # Strip percent signs while preserving missing/suppressed values.
    for pct_col in ["STU_PART_PCT", "M_PLUS_E_PCT"]:
        mcas[pct_col] = (
            mcas[pct_col]
            .astype("string")
            .str.replace("%", "", regex=False)
            .str.strip()
        )

    numeric_cols = [
        "M_PLUS_E_CNT",
        "M_PLUS_E_PCT",
        "STU_CNT",
        "STU_PART_PCT",
        "AVG_SCALED_SCORE",
        "AVG_SGP",
        "AVG_SGP_INCL",
        "ACH_PERCENTILE",
    ]
    for c in numeric_cols:
        mcas[c] = pd.to_numeric(mcas[c], errors="coerce")

    print(f"Scanned MCAS rows: {scanned:,}")
    print(f"Rows after filters: {len(mcas):,}")
    print(f"Unique filtered ORG_CODE values: {mcas['ORG_CODE'].nunique():,}")

    return mcas


def weighted_mean(group, value_col, weight_col):
    tmp = group[[value_col, weight_col]].dropna()
    tmp = tmp[tmp[weight_col] > 0]
    if tmp.empty:
        return np.nan
    return np.average(tmp[value_col], weights=tmp[weight_col])


def aggregate_zip_subject(group):
    # For M+E %, aggregate counts only for rows where DESE publishes M_PLUS_E_PCT.
    # This avoids reconstructing an unpublished/suppressed percentage from counts.
    valid_me = group[["M_PLUS_E_CNT", "M_PLUS_E_PCT", "STU_CNT"]].dropna()
    valid_me = valid_me[valid_me["STU_CNT"] > 0]

    me_count = valid_me["M_PLUS_E_CNT"].sum() if not valid_me.empty else np.nan
    stu_count_me = valid_me["STU_CNT"].sum() if not valid_me.empty else np.nan
    me_pct = (
        100.0 * me_count / stu_count_me
        if pd.notna(stu_count_me) and stu_count_me > 0
        else np.nan
    )

    return pd.Series(
        {
            "N_SCHOOLS": group["ORG_CODE"].nunique(),
            "N_SCHOOL_GRADE_ROWS": len(group),
            "STU_CNT_SUM_ALL_VALID": group["STU_CNT"].sum(min_count=1),
            "STU_CNT_SUM_FOR_ME_PCT": stu_count_me,
            "M_PLUS_E_CNT_SUM": me_count,
            "M_PLUS_E_PCT_ZIP": me_pct,
            "AVG_SCALED_SCORE_WT": weighted_mean(
                group, "AVG_SCALED_SCORE", "STU_CNT"
            ),
            "AVG_SGP_WT": weighted_mean(
                group, "AVG_SGP", "AVG_SGP_INCL"
            ),
            "STU_PART_PCT_WT": weighted_mean(
                group, "STU_PART_PCT", "STU_CNT"
            ),
        }
    )


def build(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # 1) MassGIS crosswalk and multi-campus audit.
    schools_all, schools_safe, multi_zip_ids = read_school_crosswalk(
        args.schools, out
    )

    # 2) Filter giant MCAS CSV in chunks.
    mcas = filter_mcas(
        args.mcas,
        year=args.year,
        all_students=args.all_students,
        math_code=args.math_code,
        ela_code=args.ela_code,
        grades=args.grades,
        chunksize=args.chunksize,
    )
    mcas.to_csv(out / "mcas_filtered_before_zip_merge.csv", index=False)

    # 3) Match audit BEFORE merge.
    mcas_ids = set(mcas["ORG_CODE"].dropna().unique())
    safe_ids = set(schools_safe["SCHID"].dropna().unique())
    all_school_ids = set(schools_all["SCHID"].dropna().unique())

    matched_safe = mcas_ids & safe_ids
    matched_any_massgis = mcas_ids & all_school_ids
    unmatched_ids = mcas_ids - all_school_ids
    multi_zip_mcas_ids = mcas_ids & multi_zip_ids

    unmatched = (
        mcas[mcas["ORG_CODE"].isin(unmatched_ids)]
        [["ORG_CODE", "ORG_NAME", "DIST_CODE", "DIST_NAME", "ORG_TYPE"]]
        .drop_duplicates()
        .sort_values(["DIST_NAME", "ORG_NAME"])
    )
    unmatched.to_csv(out / "unmatched_mcas_school_ids.csv", index=False)

    multi_zip_mcas = (
        mcas[mcas["ORG_CODE"].isin(multi_zip_mcas_ids)]
        [["ORG_CODE", "ORG_NAME", "DIST_NAME"]]
        .drop_duplicates()
        .merge(
            schools_all[
                schools_all["SCHID"].isin(multi_zip_mcas_ids)
            ][["SCHID", "NAME", "ZIPCODE", "TOWN", "TYPE", "GRADES"]],
            left_on="ORG_CODE",
            right_on="SCHID",
            how="left",
        )
        .sort_values(["ORG_CODE", "ZIPCODE"])
    )
    multi_zip_mcas.to_csv(
        out / "mcas_multi_zip_school_exceptions.csv", index=False
    )

    match_summary = {
        "filtered_mcas_rows": int(len(mcas)),
        "filtered_mcas_unique_org_codes": int(len(mcas_ids)),
        "massgis_unique_school_ids": int(len(all_school_ids)),
        "safe_one_zip_massgis_ids": int(len(safe_ids)),
        "mcas_ids_found_anywhere_in_massgis": int(len(matched_any_massgis)),
        "mcas_ids_safe_for_one_zip_merge": int(len(matched_safe)),
        "mcas_ids_not_in_massgis": int(len(unmatched_ids)),
        "mcas_ids_with_multiple_massgis_zips": int(len(multi_zip_mcas_ids)),
        "match_rate_any_massgis": (
            len(matched_any_massgis) / len(mcas_ids) if mcas_ids else np.nan
        ),
        "safe_one_zip_match_rate": (
            len(matched_safe) / len(mcas_ids) if mcas_ids else np.nan
        ),
    }

    with open(out / "mcas_massgis_match_summary.json", "w") as f:
        json.dump(match_summary, f, indent=2)

    print("\nMATCH AUDIT")
    print(json.dumps(match_summary, indent=2))

    # 4) Merge only safe 1-school-ID -> 1-ZIP mappings.
    merged = mcas.merge(
        schools_safe[
            [
                "SCHID",
                "ZIPCODE",
                "NAME",
                "TOWN",
                "TYPE",
                "TYPE_DESC",
                "GRADES",
                "DIST_CODE",
                "DIST_NAME",
            ]
        ],
        left_on="ORG_CODE",
        right_on="SCHID",
        how="inner",
        suffixes=("_MCAS", "_MASSGIS"),
        validate="many_to_one",
    )

    merged.to_csv(out / "mcas_school_rows_with_zip.csv", index=False)

    # 5) Aggregate to ZIP x subject.
    zip_long = (
        merged.groupby(["ZIPCODE", "SUBJECT"], dropna=False)
        .apply(aggregate_zip_subject, include_groups=False)
        .reset_index()
    )
    zip_long.to_csv(out / "mcas_zip_subject_long.csv", index=False)

    # 6) Pivot Math and ELA into one row per ZIP.
    metric_cols = [
        "N_SCHOOLS",
        "N_SCHOOL_GRADE_ROWS",
        "STU_CNT_SUM_ALL_VALID",
        "STU_CNT_SUM_FOR_ME_PCT",
        "M_PLUS_E_CNT_SUM",
        "M_PLUS_E_PCT_ZIP",
        "AVG_SCALED_SCORE_WT",
        "AVG_SGP_WT",
        "STU_PART_PCT_WT",
    ]

    wide_parts = []
    for subject in ["Math", "ELA"]:
        sub = zip_long[zip_long["SUBJECT"] == subject][
            ["ZIPCODE"] + metric_cols
        ].copy()
        sub = sub.rename(
            columns={c: f"MCAS_{subject}_{c}" for c in metric_cols}
        )
        wide_parts.append(sub)

    if wide_parts:
        zip_wide = wide_parts[0]
        for part in wide_parts[1:]:
            zip_wide = zip_wide.merge(part, on="ZIPCODE", how="outer")
    else:
        zip_wide = pd.DataFrame(columns=["ZIPCODE"])

    zip_wide = zip_wide.sort_values("ZIPCODE")
    zip_wide.to_csv(out / "mcas_zip_math_ela.csv", index=False)

    # 7) Final human-readable summary.
    summary_lines = [
        "MCAS + MassGIS ZIP build complete",
        f"Year: {args.year}",
        f"Student group: {args.all_students}",
        f"Math code: {args.math_code}",
        f"ELA code: {args.ela_code}",
        f"Grades: {args.grades if args.grades else 'ALL TEST GRADES'}",
        "",
        f"Filtered MCAS rows: {len(mcas):,}",
        f"Unique MCAS organization codes: {len(mcas_ids):,}",
        f"Safe one-ZIP matches: {len(matched_safe):,}",
        f"Unmatched MCAS IDs: {len(unmatched_ids):,}",
        f"Multi-ZIP MCAS IDs excluded: {len(multi_zip_mcas_ids):,}",
        f"ZIPs in final Math/ELA file: {zip_wide['ZIPCODE'].nunique():,}",
        "",
        "Review these files next:",
        "1. mcas_massgis_match_summary.json",
        "2. unmatched_mcas_school_ids.csv",
        "3. mcas_multi_zip_school_exceptions.csv",
        "4. mcas_zip_subject_long.csv",
        "5. mcas_zip_math_ela.csv",
    ]
    (out / "README_RESULTS.txt").write_text("\n".join(summary_lines))
    print("\n" + "\n".join(summary_lines))


def main():
    parser = argparse.ArgumentParser(
        description="Inspect or build a ZIP-level MCAS + MassGIS dataset."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect")
    p_inspect.add_argument("--mcas", required=True, type=Path)
    p_inspect.add_argument("--chunksize", type=int, default=250_000)

    p_build = sub.add_parser("build")
    p_build.add_argument("--mcas", required=True, type=Path)
    p_build.add_argument("--schools", required=True, type=Path)
    p_build.add_argument("--year", required=True)
    p_build.add_argument("--all-students", required=True)
    p_build.add_argument("--math-code", required=True)
    p_build.add_argument("--ela-code", required=True)
    p_build.add_argument("--grades", nargs="*", default=None)
    p_build.add_argument("--output", type=Path, default=Path("mcas_zip_outputs"))
    p_build.add_argument("--chunksize", type=int, default=250_000)

    args = parser.parse_args()

    if args.command == "inspect":
        inspect_mcas(args.mcas, args.chunksize)
    elif args.command == "build":
        build(args)


if __name__ == "__main__":
    main()
