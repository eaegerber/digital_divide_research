#!/usr/bin/env python3
"""
Post-Gerber extension for the 2024 Massachusetts Digital Divide project.

This script is intentionally an extension, not a replacement, of the primary
8-feature statewide analysis. It implements the follow-up questions from the
most recent Gerber meeting:

1. Quantify PC3 and inspect its loadings.
2. Keep Math/ELA participation in a PC1/PC2/PC3 sensitivity model.
3. Build interactive 3D PCA triplots with loading vectors.
4. Compare KMeans using PC1-PC2, PC1-PC3, and the full standardized feature space.
5. Explicitly inspect k=2, k=4, and k=6 profiles.
6. Diagnose why low- and high-income shares are not opposite vectors.
7. Create interactive Massachusetts and Boston-area maps with hover data.
8. Identify substantively interesting cross-domain mismatch ZIPs.
9. Optionally build a population-density overlay and true ZCTA polygon choropleths.

The analysis remains ecological: MCAS outcomes are assigned to school physical
ZIPs, not student residential ZIPs.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


PRIMARY_FEATURES = [
    "Broadband_Rate",
    "No_Computer_Rate",
    "Computer_No_Internet_Rate",
    "ACS_Median_Family_Income",
    "Low_Income_Share_lt25k",
    "High_Income_Share_150k_plus",
    "MCAS_Math_M_PLUS_E_PCT_ZIP",
    "MCAS_ELA_M_PLUS_E_PCT_ZIP",
]

PARTICIPATION_FEATURES = [
    "MCAS_Math_STU_PART_PCT_WT",
    "MCAS_ELA_STU_PART_PCT_WT",
]

PARTICIPATION_MODEL_FEATURES = PRIMARY_FEATURES + PARTICIPATION_FEATURES

DISPLAY_NAMES = {
    "Broadband_Rate": "Broadband",
    "No_Computer_Rate": "No computer",
    "Computer_No_Internet_Rate": "Computer, no internet",
    "ACS_Median_Family_Income": "Median family income",
    "Low_Income_Share_lt25k": "Low-income share",
    "High_Income_Share_150k_plus": "High-income share",
    "MCAS_Math_M_PLUS_E_PCT_ZIP": "Math attainment",
    "MCAS_ELA_M_PLUS_E_PCT_ZIP": "ELA attainment",
    "MCAS_Math_STU_PART_PCT_WT": "Math participation",
    "MCAS_ELA_STU_PART_PCT_WT": "ELA participation",
}

DOMAIN_MAP = {
    "Broadband_Rate": "Digital access",
    "No_Computer_Rate": "Digital access",
    "Computer_No_Internet_Rate": "Digital access",
    "ACS_Median_Family_Income": "Income",
    "Low_Income_Share_lt25k": "Income",
    "High_Income_Share_150k_plus": "Income",
    "MCAS_Math_M_PLUS_E_PCT_ZIP": "Education attainment",
    "MCAS_ELA_M_PLUS_E_PCT_ZIP": "Education attainment",
    "MCAS_Math_STU_PART_PCT_WT": "Participation",
    "MCAS_ELA_STU_PART_PCT_WT": "Participation",
}

CENSUS_ZCTA_CARTO_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2020/shp/"
    "cb_2020_us_zcta520_500k.zip"
)
CENSUS_2024_GAZ_ZCTA_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2024_Gazetteer/2024_Gaz_zcta_national.zip"
)
CENSUS_2024_POP_API = "https://api.census.gov/data/2024/acs/acs5"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--primary", required=True, help="411-row primary cluster_assignments CSV")
    p.add_argument("--schools-shp", default=None, help="Optional MassGIS SCHOOLS_PT.shp for town labels")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--build-polygons",
        action="store_true",
        help="Download Census ZCTA cartographic boundaries and create polygon choropleths.",
    )
    p.add_argument(
        "--build-density",
        action="store_true",
        help="Fetch 2024 ACS population + Census Gazetteer land area for population density.",
    )
    return p.parse_args()


def read_csv(path):
    df = pd.read_csv(path, dtype={"ZIPCODE": str})
    df["ZIPCODE"] = df["ZIPCODE"].str.zfill(5)
    return df


def add_town_labels(df, schools_shp):
    out = df.copy()
    out["Town"] = ""
    if not schools_shp:
        return out

    try:
        import geopandas as gpd

        schools = gpd.read_file(schools_shp)
        if "ZIPCODE" not in schools.columns:
            print("WARNING: school shapefile has no ZIPCODE column; Town labels skipped.")
            return out

        town_col = "TOWN" if "TOWN" in schools.columns else "TOWN_MAIL"
        schools["ZIPCODE"] = (
            schools["ZIPCODE"].astype(str).str.extract(r"(\d{5})", expand=False)
        )
        towns = (
            schools.dropna(subset=["ZIPCODE"])
            .groupby("ZIPCODE")[town_col]
            .agg(
                lambda x: ", ".join(
                    sorted(
                        set(
                            str(v).strip()
                            for v in x
                            if pd.notna(v) and str(v).strip()
                        )
                    )
                )
            )
            .reset_index(name="Town_from_shp")
        )
        out = out.drop(columns=["Town"], errors="ignore").merge(towns, on="ZIPCODE", how="left")
        out = out.rename(columns={"Town_from_shp": "Town"})
        out["Town"] = out["Town"].fillna("")
    except Exception as exc:
        print(f"WARNING: could not add town labels from school shapefile: {exc}")
    return out


def fit_pca(df, features, sign_rules=None):
    z = StandardScaler().fit_transform(df[features])
    pca = PCA().fit(z)
    scores = pca.transform(z)

    # PCA signs are arbitrary. Sign rules make outputs easier to compare.
    sign_rules = sign_rules or {}
    for pc_index, feature_name in sign_rules.items():
        feat_idx = features.index(feature_name)
        if pca.components_[pc_index, feat_idx] < 0:
            pca.components_[pc_index, :] *= -1
            scores[:, pc_index] *= -1

    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i+1}" for i in range(len(features))],
    )
    loadings.index.name = "Feature"

    explained = pd.DataFrame(
        {
            "PC": [f"PC{i+1}" for i in range(len(features))],
            "Explained_Variance_Ratio": pca.explained_variance_ratio_,
            "Cumulative_Explained_Variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    return z, pca, scores, loadings, explained


def domain_contributions(loadings):
    rows = []
    for pc in ["PC1", "PC2", "PC3"]:
        temp = pd.DataFrame(
            {
                "Feature": loadings.index,
                "Contribution": loadings[pc].to_numpy() ** 2,
            }
        )
        temp["Domain"] = temp["Feature"].map(DOMAIN_MAP).fillna("Other")
        g = temp.groupby("Domain", as_index=False)["Contribution"].sum()
        g["PC"] = pc
        rows.append(g)
    return pd.concat(rows, ignore_index=True)[["PC", "Domain", "Contribution"]]


def angle_degrees(v1, v2):
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return np.nan
    cos = np.dot(v1, v2) / denom
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def income_vector_diagnostic(df, loadings, outdir):
    low = "Low_Income_Share_lt25k"
    high = "High_Income_Share_150k_plus"

    corr = float(df[low].corr(df[high]))
    middle_share = 1.0 - df[low] - df[high]

    angle_2d = angle_degrees(
        loadings.loc[low, ["PC1", "PC2"]].to_numpy(),
        loadings.loc[high, ["PC1", "PC2"]].to_numpy(),
    )
    angle_3d = angle_degrees(
        loadings.loc[low, ["PC1", "PC2", "PC3"]].to_numpy(),
        loadings.loc[high, ["PC1", "PC2", "PC3"]].to_numpy(),
    )

    diagnostic = {
        "low_high_income_share_correlation": corr,
        "loading_vector_angle_PC1_PC2_degrees": angle_2d,
        "loading_vector_angle_PC1_PC2_PC3_degrees": angle_3d,
        "middle_income_share_mean": float(middle_share.mean()),
        "middle_income_share_median": float(middle_share.median()),
        "interpretation": (
            "Low-income and high-income shares are not complements because families in "
            "the middle-income ranges account for a large and varying share of each ZCTA. "
            "Therefore the two variables need not have correlation -1 or loading vectors "
            "180 degrees apart."
        ),
    }
    (outdir / "income_vector_diagnostic.json").write_text(json.dumps(diagnostic, indent=2))
    return diagnostic


def add_domain_scores(df):
    out = df.copy()
    cols = PARTICIPATION_MODEL_FEATURES
    for c in cols:
        sd = out[c].std(ddof=0)
        out[f"z_{c}"] = (out[c] - out[c].mean()) / sd if sd else 0.0

    out["SES_Score"] = (
        out["z_ACS_Median_Family_Income"]
        - out["z_Low_Income_Share_lt25k"]
        + out["z_High_Income_Share_150k_plus"]
    ) / 3

    out["Digital_Access_Score"] = (
        out["z_Broadband_Rate"]
        - out["z_No_Computer_Rate"]
        - out["z_Computer_No_Internet_Rate"]
    ) / 3

    out["Education_Score"] = (
        out["z_MCAS_Math_M_PLUS_E_PCT_ZIP"]
        + out["z_MCAS_ELA_M_PLUS_E_PCT_ZIP"]
    ) / 2

    out["Participation_Score"] = (
        out["z_MCAS_Math_STU_PART_PCT_WT"]
        + out["z_MCAS_ELA_STU_PART_PCT_WT"]
    ) / 2

    # Residual diagnostic: digital access that is unexpectedly high/low given SES + education.
    X = out[["SES_Score", "Education_Score"]].to_numpy()
    y = out["Digital_Access_Score"].to_numpy()
    model = LinearRegression().fit(X, y)
    out["Expected_Digital_Access_Score"] = model.predict(X)
    out["Digital_Access_Residual"] = y - out["Expected_Digital_Access_Score"]

    # Complementary diagnostic: education unexpectedly high/low given SES + digital access.
    X2 = out[["SES_Score", "Digital_Access_Score"]].to_numpy()
    y2 = out["Education_Score"].to_numpy()
    model2 = LinearRegression().fit(X2, y2)
    out["Expected_Education_Score"] = model2.predict(X2)
    out["Education_Residual"] = y2 - out["Expected_Education_Score"]

    return out


def mismatch_analysis(df, outdir):
    out = add_domain_scores(df)

    q25 = out[["SES_Score", "Digital_Access_Score", "Education_Score", "Broadband_Rate"]].quantile(0.25)
    q75 = out[["SES_Score", "Digital_Access_Score", "Education_Score", "Broadband_Rate"]].quantile(0.75)

    out["Mismatch_Type"] = "No extreme cross-domain mismatch"

    high_ses = out["SES_Score"] >= q75["SES_Score"]
    low_ses = out["SES_Score"] <= q25["SES_Score"]
    high_dig = out["Digital_Access_Score"] >= q75["Digital_Access_Score"]
    low_dig = out["Digital_Access_Score"] <= q25["Digital_Access_Score"]
    high_edu = out["Education_Score"] >= q75["Education_Score"]
    low_edu = out["Education_Score"] <= q25["Education_Score"]

    # Apply narrower/more substantively interesting categories first.
    out.loc[high_ses & high_edu & low_dig, "Mismatch_Type"] = (
        "High SES + high education + low digital access"
    )
    out.loc[low_ses & high_dig & low_edu, "Mismatch_Type"] = (
        "Low SES + high digital access + low education"
    )
    out.loc[low_ses & high_dig & high_edu, "Mismatch_Type"] = (
        "Low SES + high digital access + high education"
    )

    unassigned = out["Mismatch_Type"].eq("No extreme cross-domain mismatch")
    out.loc[unassigned & high_ses & low_dig, "Mismatch_Type"] = "High SES + low digital access"
    unassigned = out["Mismatch_Type"].eq("No extreme cross-domain mismatch")
    out.loc[unassigned & low_ses & high_dig, "Mismatch_Type"] = "Low SES + high digital access"

    out["High_SES_High_Edu_Low_Broadband"] = (
        high_ses & high_edu & (out["Broadband_Rate"] <= q25["Broadband_Rate"])
    )

    keep = [
        "ZIPCODE", "Town", "Latitude", "Longitude", "Mismatch_Type",
        "SES_Score", "Digital_Access_Score", "Education_Score", "Participation_Score",
        "Digital_Access_Residual", "Education_Residual",
        "Broadband_Rate", "No_Computer_Rate", "Computer_No_Internet_Rate",
        "ACS_Median_Family_Income", "Low_Income_Share_lt25k",
        "High_Income_Share_150k_plus", "MCAS_Math_M_PLUS_E_PCT_ZIP",
        "MCAS_ELA_M_PLUS_E_PCT_ZIP", "MCAS_Math_STU_PART_PCT_WT",
        "MCAS_ELA_STU_PART_PCT_WT", "High_SES_High_Edu_Low_Broadband",
    ]
    out[keep].to_csv(outdir / "mismatch_zip_analysis.csv", index=False)

    # A compact list of strongest residual anomalies for stakeholder review.
    neg_digital = out.nsmallest(15, "Digital_Access_Residual").assign(
        Residual_Flag="Unexpectedly low digital access given SES + education"
    )
    pos_digital = out.nlargest(15, "Digital_Access_Residual").assign(
        Residual_Flag="Unexpectedly high digital access given SES + education"
    )
    low_edu_resid = out.nsmallest(15, "Education_Residual").assign(
        Residual_Flag="Unexpectedly low education given SES + digital access"
    )
    high_edu_resid = out.nlargest(15, "Education_Residual").assign(
        Residual_Flag="Unexpectedly high education given SES + digital access"
    )
    residuals = pd.concat(
        [neg_digital, pos_digital, low_edu_resid, high_edu_resid], ignore_index=True
    )
    residual_keep = [
        "ZIPCODE", "Town", "Residual_Flag", "Digital_Access_Residual",
        "Education_Residual", "SES_Score", "Digital_Access_Score", "Education_Score",
        "Broadband_Rate", "ACS_Median_Family_Income",
        "MCAS_Math_M_PLUS_E_PCT_ZIP", "MCAS_ELA_M_PLUS_E_PCT_ZIP",
    ]
    residuals[residual_keep].to_csv(outdir / "residual_outlier_review.csv", index=False)

    return out


def kmeans_comparison(z, scores, model_name, outdir):
    rows = []
    label_store = {}

    spaces = {
        "PC1_PC2": scores[:, :2],
        "PC1_PC2_PC3": scores[:, :3],
        "Full_standardized_features": z,
    }

    for space_name, X in spaces.items():
        for k in range(2, 9):
            labels = KMeans(n_clusters=k, random_state=42, n_init=100).fit_predict(X)
            rows.append(
                {
                    "Model": model_name,
                    "Space": space_name,
                    "k": k,
                    "Silhouette": silhouette_score(X, labels),
                    "Cluster_Sizes": json.dumps(
                        pd.Series(labels).value_counts().sort_index().to_dict()
                    ),
                }
            )
            label_store[(space_name, k)] = labels

    result = pd.DataFrame(rows)
    return result, label_store


def save_silhouette_dimensionality(comparison, out):
    fig, ax = plt.subplots(figsize=(10, 6))
    for space, d in comparison.groupby("Space"):
        ax.plot(d["k"], d["Silhouette"], marker="o", label=space)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Participation model: cluster separation after retaining PC3")
    ax.legend(title="Clustering space")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_pc_pair_plot(df, scores, labels, pca, x_idx, y_idx, out, title):
    fig, ax = plt.subplots(figsize=(10, 7))
    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        ax.scatter(scores[mask, x_idx], scores[mask, y_idx], s=42, alpha=0.68, label=f"Cluster {cluster_id}")
    ev = pca.explained_variance_ratio_
    ax.axhline(0, linewidth=0.7)
    ax.axvline(0, linewidth=0.7)
    ax.set_xlabel(f"PC{x_idx+1} ({ev[x_idx]*100:.1f}%)")
    ax.set_ylabel(f"PC{y_idx+1} ({ev[y_idx]*100:.1f}%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_static_geo_map(df, cluster_col, out, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    for cluster_id in sorted(df[cluster_col].dropna().unique()):
        d = df[df[cluster_col] == cluster_id]
        ax.scatter(d["Longitude"], d["Latitude"], s=38, alpha=0.75, label=f"Cluster {cluster_id}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend()
    # Correct lon/lat aspect more closely around Massachusetts latitude.
    ax.set_aspect(1 / np.cos(np.deg2rad(42.2)), adjustable="box")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_pc3_geo_map(df, out, title):
    """Map the continuous participation-model PC3 score at ZCTA representative points."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        df["Longitude"],
        df["Latitude"],
        c=df["PC3_Participation"],
        s=40,
        alpha=0.82,
        cmap="coolwarm",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect(1 / np.cos(np.deg2rad(42.2)), adjustable="box")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
    cbar.set_label("PC3 score")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_3d_triplot(df, pca, scores, loadings, labels, out_html, title):
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        print(f"WARNING: plotly unavailable; 3D triplot skipped: {exc}")
        return

    plot_df = df.copy()
    plot_df["PC1_plot"] = scores[:, 0]
    plot_df["PC2_plot"] = scores[:, 1]
    plot_df["PC3_plot"] = scores[:, 2]
    plot_df["Cluster_plot"] = labels.astype(str)

    plot_df["HoverText"] = plot_df.apply(
        lambda r: (
            f"ZIP {r['ZIPCODE']}"
            + (f" — {r['Town']}" if str(r.get("Town", "")).strip() else "")
            + f"<br>Broadband: {r['Broadband_Rate']:.1%}"
            + f"<br>Median family income: ${r['ACS_Median_Family_Income']:,.0f}"
            + f"<br>Math M+E: {r['MCAS_Math_M_PLUS_E_PCT_ZIP']:.1f}%"
            + f"<br>ELA M+E: {r['MCAS_ELA_M_PLUS_E_PCT_ZIP']:.1f}%"
            + f"<br>PC1: {r['PC1_plot']:.2f}, PC2: {r['PC2_plot']:.2f}, PC3: {r['PC3_plot']:.2f}"
        ),
        axis=1,
    )

    fig = go.Figure()
    for cluster_id in sorted(plot_df["Cluster_plot"].unique()):
        d = plot_df[plot_df["Cluster_plot"] == cluster_id]
        fig.add_trace(
            go.Scatter3d(
                x=d["PC1_plot"], y=d["PC2_plot"], z=d["PC3_plot"],
                mode="markers", name=f"Cluster {cluster_id}",
                marker=dict(size=4, opacity=0.68),
                text=d["HoverText"],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    # Scale loading vectors into score space.
    spans = [
        np.nanpercentile(scores[:, i], 95) - np.nanpercentile(scores[:, i], 5)
        for i in range(3)
    ]
    scale = min(spans) * 0.75

    for feature in loadings.index:
        vec = loadings.loc[feature, ["PC1", "PC2", "PC3"]].to_numpy(dtype=float) * scale
        fig.add_trace(
            go.Scatter3d(
                x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
                mode="lines+text", name=DISPLAY_NAMES.get(feature, feature),
                line=dict(width=5),
                text=["", DISPLAY_NAMES.get(feature, feature)],
                textposition="top center",
                hovertemplate=(
                    f"{DISPLAY_NAMES.get(feature, feature)}<br>"
                    f"PC1 loading: {loadings.loc[feature, 'PC1']:.3f}<br>"
                    f"PC2 loading: {loadings.loc[feature, 'PC2']:.3f}<br>"
                    f"PC3 loading: {loadings.loc[feature, 'PC3']:.3f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        # Small cone gives a visible 3D arrowhead.
        norm = np.linalg.norm(vec)
        if norm > 0:
            direction = vec / norm
            fig.add_trace(
                go.Cone(
                    x=[vec[0]], y=[vec[1]], z=[vec[2]],
                    u=[direction[0]], v=[direction[1]], w=[direction[2]],
                    sizemode="absolute", sizeref=0.18, anchor="tip",
                    showscale=False, hoverinfo="skip", showlegend=False,
                )
            )

    ev = pca.explained_variance_ratio_
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"PC1 ({ev[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({ev[1]*100:.1f}%)",
            zaxis_title=f"PC3 ({ev[2]*100:.1f}%)",
        ),
        legend_title="KMeans on first 3 PCs",
        margin=dict(l=0, r=0, b=0, t=55),
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def create_interactive_point_map(df, cluster_col, out_html, title, boston_only=False):
    try:
        import plotly.express as px
    except Exception as exc:
        print(f"WARNING: plotly unavailable; interactive map skipped: {exc}")
        return

    d = df.copy()
    if boston_only:
        # Broad Greater Boston viewing window; users can continue zooming interactively.
        d = d[
            d["Latitude"].between(42.15, 42.60)
            & d["Longitude"].between(-71.55, -70.75)
        ].copy()

    d[cluster_col] = d[cluster_col].astype(str)
    hover_cols = {
        "ZIPCODE": True,
        "Town": True,
        "Broadband_Rate": ":.3f",
        "ACS_Median_Family_Income": ":,.0f",
        "MCAS_Math_M_PLUS_E_PCT_ZIP": ":.1f",
        "MCAS_ELA_M_PLUS_E_PCT_ZIP": ":.1f",
        "PC1_Participation": ":.2f",
        "PC2_Participation": ":.2f",
        "PC3_Participation": ":.2f",
        "Mismatch_Type": True,
        "Digital_Access_Residual": ":.2f",
        "Education_Residual": ":.2f",
    }

    fig = px.scatter_mapbox(
        d,
        lat="Latitude",
        lon="Longitude",
        color=cluster_col,
        hover_name="ZIPCODE",
        hover_data=hover_cols,
        zoom=9 if boston_only else 6.6,
        height=750,
        title=title,
    )
    fig.update_traces(marker=dict(size=9, opacity=0.80))
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=50, b=0))
    fig.write_html(out_html, include_plotlyjs="cdn")


def create_mismatch_map(df, out_html):
    try:
        import plotly.express as px
    except Exception as exc:
        print(f"WARNING: plotly unavailable; mismatch map skipped: {exc}")
        return

    d = df.copy()
    fig = px.scatter_mapbox(
        d,
        lat="Latitude",
        lon="Longitude",
        color="Mismatch_Type",
        hover_name="ZIPCODE",
        hover_data={
            "Town": True,
            "SES_Score": ":.2f",
            "Digital_Access_Score": ":.2f",
            "Education_Score": ":.2f",
            "Broadband_Rate": ":.3f",
            "ACS_Median_Family_Income": ":,.0f",
            "MCAS_Math_M_PLUS_E_PCT_ZIP": ":.1f",
            "MCAS_ELA_M_PLUS_E_PCT_ZIP": ":.1f",
            "Digital_Access_Residual": ":.2f",
            "Education_Residual": ":.2f",
        },
        zoom=6.6,
        height=760,
        title="Cross-domain mismatch ZIPs for follow-up investigation",
    )
    fig.update_traces(marker=dict(size=9, opacity=0.80))
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=50, b=0))
    fig.write_html(out_html, include_plotlyjs="cdn")


def download_if_needed(url, path):
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)
    return path


def fetch_population_density(cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 2024 ACS 5-year total population by ZCTA (B01003_001E).
    params = urllib.parse.urlencode(
        {"get": "NAME,B01003_001E", "for": "zip code tabulation area:*"}
    )
    api_url = f"{CENSUS_2024_POP_API}?{params}"
    with urllib.request.urlopen(api_url) as r:
        raw = json.loads(r.read().decode("utf-8"))
    pop = pd.DataFrame(raw[1:], columns=raw[0])
    zcta_col = "zip code tabulation area"
    pop = pop.rename(columns={zcta_col: "ZIPCODE", "B01003_001E": "ACS_2024_Population"})
    pop["ZIPCODE"] = pop["ZIPCODE"].astype(str).str.zfill(5)
    pop["ACS_2024_Population"] = pd.to_numeric(pop["ACS_2024_Population"], errors="coerce")

    # 2024 Gazetteer land area. Census notes these ZCTAs are based on 2020 blocks.
    gaz_zip = download_if_needed(CENSUS_2024_GAZ_ZCTA_URL, cache_dir / "2024_Gaz_zcta_national.zip")
    with zipfile.ZipFile(gaz_zip) as zf:
        txt_name = [n for n in zf.namelist() if n.lower().endswith(".txt")][0]
        with zf.open(txt_name) as f:
            gaz = pd.read_csv(f, sep="\t", dtype=str)
    gaz.columns = [c.strip() for c in gaz.columns]
    gaz["GEOID"] = gaz["GEOID"].astype(str).str.strip().str.zfill(5)
    gaz["ALAND_SQMI"] = pd.to_numeric(gaz["ALAND_SQMI"].str.strip(), errors="coerce")
    gaz = gaz.rename(columns={"GEOID": "ZIPCODE"})

    dens = pop.merge(gaz[["ZIPCODE", "ALAND_SQMI"]], on="ZIPCODE", how="left")
    dens["Population_Density_per_sqmi"] = dens["ACS_2024_Population"] / dens["ALAND_SQMI"]
    return dens


def create_density_overlay(df, out_html):
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        print(f"WARNING: plotly unavailable; density map skipped: {exc}")
        return

    d = df.dropna(subset=["Population_Density_per_sqmi", "Latitude", "Longitude"]).copy()
    z = np.log1p(d["Population_Density_per_sqmi"].to_numpy())

    fig = go.Figure()
    fig.add_trace(
        go.Densitymapbox(
            lat=d["Latitude"], lon=d["Longitude"], z=z,
            radius=18, opacity=0.55, name="log population density",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scattermapbox(
            lat=d["Latitude"], lon=d["Longitude"], mode="markers",
            marker=dict(size=8, color=d["K2_3PC"], opacity=0.85),
            text=[
                f"ZIP {r.ZIPCODE} — {r.Town}<br>Density: {r.Population_Density_per_sqmi:,.0f}/sq mi<br>Cluster: {r.K2_3PC}"
                for _, r in d.iterrows()
            ],
            hovertemplate="%{text}<extra></extra>", name="3-PC k=2 cluster",
        )
    )
    fig.update_layout(
        title="Population density heatmap with 3-PC k=2 cluster overlay",
        mapbox_style="open-street-map",
        mapbox_center=dict(lat=42.25, lon=-71.8), mapbox_zoom=6.5,
        margin=dict(l=0, r=0, t=50, b=0), height=760,
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def build_zcta_choropleths(df, cache_dir, outdir):
    try:
        import geopandas as gpd
        import plotly.express as px
    except Exception as exc:
        print(f"WARNING: polygon choropleths skipped: {exc}")
        return

    zpath = download_if_needed(
        CENSUS_ZCTA_CARTO_URL,
        Path(cache_dir) / "cb_2020_us_zcta520_500k.zip",
    )
    zcta = gpd.read_file(f"zip://{zpath}")
    zip_col = "ZCTA5CE20" if "ZCTA5CE20" in zcta.columns else "GEOID20"
    zcta["ZIPCODE"] = zcta[zip_col].astype(str).str.zfill(5)
    wanted = set(df["ZIPCODE"])
    zcta = zcta[zcta["ZIPCODE"].isin(wanted)].copy()
    geo = zcta.merge(
        df[["ZIPCODE", "K2_3PC", "PC3_Participation", "Mismatch_Type", "Town"]],
        on="ZIPCODE", how="left",
    )

    # Convert to WGS84 for Plotly.
    geo = geo.to_crs(epsg=4326)
    geojson = json.loads(geo.to_json())

    fig = px.choropleth_mapbox(
        geo,
        geojson=geojson,
        locations=geo.index,
        color=geo["K2_3PC"].astype(str),
        hover_name="ZIPCODE",
        hover_data={"Town": True, "PC3_Participation": ":.2f"},
        mapbox_style="open-street-map",
        center={"lat": 42.25, "lon": -71.8},
        zoom=6.5,
        opacity=0.60,
        title="ZCTA polygon choropleth: 3-PC k=2 clusters",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=760)
    fig.write_html(outdir / "12_zcta_cluster_choropleth.html", include_plotlyjs="cdn")

    fig2 = px.choropleth_mapbox(
        geo,
        geojson=geojson,
        locations=geo.index,
        color="PC3_Participation",
        hover_name="ZIPCODE",
        hover_data={"Town": True, "Mismatch_Type": True},
        mapbox_style="open-street-map",
        center={"lat": 42.25, "lon": -71.8},
        zoom=6.5,
        opacity=0.65,
        title="ZCTA polygon choropleth: PC3 score (participation model)",
    )
    fig2.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=760)
    fig2.write_html(outdir / "13_zcta_pc3_choropleth.html", include_plotlyjs="cdn")


def write_summary(outdir, primary_pca, part_pca, income_diag, comp_part, profile_table, df):
    ev8 = primary_pca.explained_variance_ratio_
    ev10 = part_pca.explained_variance_ratio_

    k2_3 = comp_part[(comp_part["Space"] == "PC1_PC2_PC3") & (comp_part["k"] == 2)].iloc[0]
    k4_3 = comp_part[(comp_part["Space"] == "PC1_PC2_PC3") & (comp_part["k"] == 4)].iloc[0]
    k6_3 = comp_part[(comp_part["Space"] == "PC1_PC2_PC3") & (comp_part["k"] == 6)].iloc[0]

    mismatch_counts = df["Mismatch_Type"].value_counts().to_dict()

    text = "# Post-Gerber PC3 / geography extension\n\n"
    text += "## Key PCA findings\n\n"
    text += f"- Original 8-feature model: PC1={ev8[0]*100:.2f}%, PC2={ev8[1]*100:.2f}%, PC3={ev8[2]*100:.2f}%; cumulative PC1-PC3={ev8[:3].sum()*100:.2f}%.\n"
    text += f"- Participation-retained 10-feature model: PC1={ev10[0]*100:.2f}%, PC2={ev10[1]*100:.2f}%, PC3={ev10[2]*100:.2f}%; cumulative PC1-PC3={ev10[:3].sum()*100:.2f}%.\n"
    text += "- In the participation model, PC2 is dominated by Math/ELA participation; PC3 is driven primarily by digital-access contrasts, with additional education/income contribution.\n\n"
    text += "## Why low-income and high-income vectors are not opposite\n\n"
    text += f"- Correlation between low-income share and high-income share: {income_diag['low_high_income_share_correlation']:.3f}.\n"
    text += f"- Angle between the loading vectors in PC1-PC2: {income_diag['loading_vector_angle_PC1_PC2_degrees']:.1f} degrees.\n"
    text += f"- Angle in PC1-PC2-PC3: {income_diag['loading_vector_angle_PC1_PC2_PC3_degrees']:.1f} degrees.\n"
    text += f"- Mean middle-income share (neither <25k nor >=150k): {income_diag['middle_income_share_mean']*100:.1f}%.\n"
    text += "- Therefore these two shares are not complements and should not be expected to point exactly 180 degrees apart.\n\n"
    text += "## KMeans after retaining PC3 (participation model)\n\n"
    text += f"- k=2 silhouette: {k2_3['Silhouette']:.3f}.\n"
    text += f"- k=4 silhouette: {k4_3['Silhouette']:.3f}.\n"
    text += f"- k=6 silhouette: {k6_3['Silhouette']:.3f}.\n"
    text += "- k=2 remains the strongest separation. k=4 and k=6 can be inspected as exploratory subgroups, but they should not be promoted as better-supported statewide cluster counts solely because they are more granular.\n"
    text += "- In the current 3-PC participation result, k=4 creates one very small participation-outlier cluster, which is a warning that the extra granularity may be isolating unusual observations rather than broad population types.\n\n"
    text += "## Cross-domain mismatch analysis\n\n"
    for k, v in mismatch_counts.items():
        text += f"- {k}: {v}\n"
    text += "\nThese mismatch categories are screening flags, not causal or policy conclusions. Use them to choose ZIPs for deeper data-quality, infrastructure, and policy investigation.\n\n"
    text += "## Geography\n\n"
    text += "- Interactive Massachusetts and Boston-area point maps include ZIP/town hover information, PC1-PC3, raw domain measures, and mismatch flags.\n"
    text += "- Optional ZCTA polygon choropleths use the Census 2020 cartographic ZCTA boundary file for visualization and can be generated with `--build-polygons`.\n"
    text += "- Optional population density combines 2024 ACS B01003 population with 2024 Census Gazetteer ZCTA land area and can be generated with `--build-density`.\n\n"
    text += "## Methodological caution\n\n"
    text += "MCAS outcomes are aggregated to the physical ZIP of each school, not student residence. This remains a school-location ecological analysis.\n"

    (outdir / "POST_GERBER_PC3_SUMMARY.md").write_text(text)


def main():
    args = parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = outdir / "cache"

    df = read_csv(args.primary)
    required = list(dict.fromkeys(PARTICIPATION_MODEL_FEATURES + ["Latitude", "Longitude"]))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Primary dataset missing required columns: {missing}")

    df = df.dropna(subset=PARTICIPATION_MODEL_FEATURES + ["Latitude", "Longitude"]).copy()
    df = add_town_labels(df, args.schools_shp).reset_index(drop=True)

    # Original primary 8-feature model.
    z8, pca8, scores8, load8, exp8 = fit_pca(
        df, PRIMARY_FEATURES,
        sign_rules={0: "ACS_Median_Family_Income", 2: "Computer_No_Internet_Rate"},
    )
    load8.to_csv(outdir / "primary_8_feature_pca_loadings.csv")
    exp8.to_csv(outdir / "primary_8_feature_pca_explained_variance.csv", index=False)
    domain_contributions(load8).to_csv(outdir / "primary_8_feature_domain_contributions_pc1_pc3.csv", index=False)
    income_diag = income_vector_diagnostic(df, load8, outdir)

    # Participation-retained PC3 model requested by Gerber.
    z10, pca10, scores10, load10, exp10 = fit_pca(
        df, PARTICIPATION_MODEL_FEATURES,
        sign_rules={0: "ACS_Median_Family_Income", 1: "MCAS_ELA_STU_PART_PCT_WT", 2: "Computer_No_Internet_Rate"},
    )
    load10.to_csv(outdir / "participation_10_feature_pca_loadings.csv")
    exp10.to_csv(outdir / "participation_10_feature_pca_explained_variance.csv", index=False)
    domain_contributions(load10).to_csv(outdir / "participation_10_feature_domain_contributions_pc1_pc3.csv", index=False)

    df["PC1_Primary8"] = scores8[:, 0]
    df["PC2_Primary8"] = scores8[:, 1]
    df["PC3_Primary8"] = scores8[:, 2]
    df["PC1_Participation"] = scores10[:, 0]
    df["PC2_Participation"] = scores10[:, 1]
    df["PC3_Participation"] = scores10[:, 2]

    # KMeans comparisons.
    comp8, labels8 = kmeans_comparison(z8, scores8, "Primary8", outdir)
    comp10, labels10 = kmeans_comparison(z10, scores10, "Participation10", outdir)
    comp = pd.concat([comp8, comp10], ignore_index=True)
    comp.to_csv(outdir / "kmeans_dimensionality_comparison.csv", index=False)

    # Explicit PC1-PC3 cluster labels for k=2/4/6 in participation model.
    for k in [2, 4, 6]:
        df[f"K{k}_3PC"] = labels10[("PC1_PC2_PC3", k)]

    # Compare 2-PC and 3-PC k=2 partitions and full-space k=2.
    ari = {
        "ARI_2PC_vs_3PC_k2": float(adjusted_rand_score(labels10[("PC1_PC2", 2)], labels10[("PC1_PC2_PC3", 2)])),
        "ARI_3PC_vs_full10D_k2": float(adjusted_rand_score(labels10[("PC1_PC2_PC3", 2)], labels10[("Full_standardized_features", 2)])),
        "ARI_2PC_vs_full10D_k2": float(adjusted_rand_score(labels10[("PC1_PC2", 2)], labels10[("Full_standardized_features", 2)])),
    }
    (outdir / "k2_partition_agreement.json").write_text(json.dumps(ari, indent=2))

    df = mismatch_analysis(df, outdir)

    # Cluster profiles for the specific k values Gerber asked about.
    profile_rows = []
    profile_vars = [
        "SES_Score", "Digital_Access_Score", "Education_Score", "Participation_Score",
        "PC1_Participation", "PC2_Participation", "PC3_Participation",
        "Broadband_Rate", "No_Computer_Rate", "Computer_No_Internet_Rate",
        "ACS_Median_Family_Income", "Low_Income_Share_lt25k", "High_Income_Share_150k_plus",
        "MCAS_Math_M_PLUS_E_PCT_ZIP", "MCAS_ELA_M_PLUS_E_PCT_ZIP",
        "MCAS_Math_STU_PART_PCT_WT", "MCAS_ELA_STU_PART_PCT_WT",
    ]
    for k in [2, 4, 6]:
        c = f"K{k}_3PC"
        for cluster_id, g in df.groupby(c):
            row = {"k": k, "Cluster": int(cluster_id), "N": int(len(g))}
            for var in profile_vars:
                row[var] = float(g[var].mean())
            profile_rows.append(row)
    profile_table = pd.DataFrame(profile_rows)
    profile_table.to_csv(outdir / "k2_k4_k6_3pc_cluster_profiles.csv", index=False)

    # Key static plots.
    save_silhouette_dimensionality(
        comp10,
        outdir / "01_silhouette_2pc_vs_3pc_vs_full.png",
    )
    save_pc_pair_plot(
        df, scores10, df["K2_3PC"].to_numpy(), pca10, 0, 2,
        outdir / "02_participation_pc1_vs_pc3_k2.png",
        "Participation-retained PCA: PC1 vs PC3 (k=2 on first 3 PCs)",
    )
    save_pc_pair_plot(
        df, scores10, df["K4_3PC"].to_numpy(), pca10, 1, 2,
        outdir / "03_participation_pc2_vs_pc3_k4.png",
        "Participation-retained PCA: PC2 vs PC3 (exploratory k=4)",
    )
    save_static_geo_map(
        df, "K2_3PC", outdir / "04_map_k2_first3pcs.png",
        "Massachusetts school-location ZIP/ZCTAs: k=2 on first 3 PCs",
    )
    save_static_geo_map(
        df, "K4_3PC", outdir / "05_map_k4_first3pcs.png",
        "Massachusetts school-location ZIP/ZCTAs: exploratory k=4 on first 3 PCs",
    )
    save_pc3_geo_map(
        df, outdir / "06_pc3_geographic_point_map.png",
        "Massachusetts school-location ZIP/ZCTAs: continuous PC3 score",
    )

    # Interactive 3D triplots and maps.
    create_3d_triplot(
        df, pca8, scores8, load8, labels8[("PC1_PC2_PC3", 2)],
        outdir / "07_primary8_3d_triplot.html",
        "Primary 8-feature PCA triplot: PC1 / PC2 / PC3",
    )
    create_3d_triplot(
        df, pca10, scores10, load10, df["K2_3PC"].to_numpy(),
        outdir / "08_participation_3d_triplot.html",
        "Participation-retained PCA triplot: PC1 / PC2 / PC3",
    )
    create_interactive_point_map(
        df, "K2_3PC", outdir / "09_interactive_massachusetts_k2_map.html",
        "Interactive Massachusetts map: 3-PC k=2 with ZIP-level diagnostics",
        boston_only=False,
    )
    create_interactive_point_map(
        df, "K2_3PC", outdir / "10_interactive_boston_k2_map.html",
        "Interactive Greater Boston map: 3-PC k=2 with ZIP-level diagnostics",
        boston_only=True,
    )
    create_mismatch_map(df, outdir / "11_interactive_mismatch_map.html")

    # Optional population density.
    if args.build_density:
        try:
            density = fetch_population_density(cache_dir)
            density.to_csv(outdir / "population_density_2024_zcta.csv", index=False)
            df = df.merge(density[["ZIPCODE", "ACS_2024_Population", "ALAND_SQMI", "Population_Density_per_sqmi"]], on="ZIPCODE", how="left")
            create_density_overlay(df, outdir / "14_population_density_vs_k2_map.html")
        except Exception as exc:
            print(f"WARNING: population density step failed: {exc}")

    # Optional polygon maps. These require a one-time ~64 MB Census boundary download.
    if args.build_polygons:
        try:
            build_zcta_choropleths(df, cache_dir, outdir)
        except Exception as exc:
            print(f"WARNING: ZCTA polygon step failed: {exc}")

    # Save enriched analysis table.
    df.to_csv(outdir / "post_gerber_pc3_enriched_411.csv", index=False)

    summary = {
        "n": int(len(df)),
        "primary8": {
            "pc1": float(pca8.explained_variance_ratio_[0]),
            "pc2": float(pca8.explained_variance_ratio_[1]),
            "pc3": float(pca8.explained_variance_ratio_[2]),
            "pc1_pc2_pc3": float(pca8.explained_variance_ratio_[:3].sum()),
        },
        "participation10": {
            "pc1": float(pca10.explained_variance_ratio_[0]),
            "pc2": float(pca10.explained_variance_ratio_[1]),
            "pc3": float(pca10.explained_variance_ratio_[2]),
            "pc1_pc2_pc3": float(pca10.explained_variance_ratio_[:3].sum()),
        },
        "income_vector_diagnostic": income_diag,
        "k2_agreement": ari,
        "mismatch_counts": {str(k): int(v) for k, v in df["Mismatch_Type"].value_counts().items()},
    }
    (outdir / "post_gerber_pc3_summary.json").write_text(json.dumps(summary, indent=2))

    write_summary(outdir, pca8, pca10, income_diag, comp10, profile_table, df)

    print("\nPOST-GERBER EXTENSION COMPLETE")
    print("Rows:", len(df))
    print(
        "Primary8 PC1/PC2/PC3:",
        ", ".join(f"{x*100:.2f}%" for x in pca8.explained_variance_ratio_[:3]),
    )
    print(
        "Participation10 PC1/PC2/PC3:",
        ", ".join(f"{x*100:.2f}%" for x in pca10.explained_variance_ratio_[:3]),
    )
    print("Outputs:", outdir)


if __name__ == "__main__":
    main()
