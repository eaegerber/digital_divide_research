#!/usr/bin/env python3
"""
November 2 Gerber follow-up analysis for the Massachusetts Digital Divide project.

This file is intentionally a SECOND extension. It imports the existing
post_gerber_pc3_extension.py rather than replacing it.

It addresses the Nov. 2 meeting requests:
1. Inspect k=3, k=4, and k=5 cluster assignments and profiles.
2. Create a k=5 3D PCA triplot and k=3/k=4/k=5 pair plots.
3. Run a broader KMeans model-selection grid (k, feature space, scaling).
4. Make an ACS broadband-subscription map (NOT infrastructure coverage).
5. Refine the continuous PC3 geography map with a robust symmetric color range.
6. Audit ZIP / coordinate / town-label provenance and flag 01434 as Devens.
7. Produce an Ayer / Devens / Shirley comparison table.
8. Document exactly how the mismatch map and residual flags are constructed.
9. Report row counts and unique ZIP/ZCTA counts.
10. Generate candidate ZIP labels for qualitative follow-up by cluster.

IMPORTANT:
- The current Broadband_Rate is an ACS household broadband-subscription measure.
  It is not FCC/MBI infrastructure availability/coverage.
- Latitude/Longitude in the primary dataset come from the Census ZCTA Gazetteer
  representative point, not from school coordinates.
- The legacy Town label comes from the MassGIS school shapefile and should be
  interpreted as a school-location label, not a canonical postal-place label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# Reuse the tested helpers/constants already in the project.
from post_gerber_pc3_extension import (
    PRIMARY_FEATURES,
    PARTICIPATION_FEATURES,
    PARTICIPATION_MODEL_FEATURES,
    DISPLAY_NAMES,
    read_csv,
    add_town_labels,
    fit_pca,
    add_domain_scores,
    mismatch_analysis,
    create_3d_triplot,
)

KNOWN_POSTAL_PLACE_OVERRIDES = {
    # Massachusetts sources identify 01434 as Devens; Ayer is also an acceptable
    # postal name. Keep the school-derived label separately rather than overwriting it.
    "01434": "DEVENS",
}

DEFAULT_INVESTIGATION_ZIPS = ["01432", "01434", "01464"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--primary",
        required=True,
        help="Primary complete-case CSV (currently 411 ZIP/ZCTA rows).",
    )
    p.add_argument(
        "--master",
        default=None,
        help="Optional merged master CSV (currently 413 ZIP/ZCTA rows).",
    )
    p.add_argument(
        "--schools-shp",
        default=None,
        help="Optional MassGIS SCHOOLS_PT.shp; used only to recreate school-town labels.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output directory.",
    )
    p.add_argument(
        "--zip-place-crosswalk",
        default=None,
        help=(
            "Optional CSV with ZIPCODE and Postal_Place columns. "
            "If supplied, it supersedes the small known-place override table."
        ),
    )
    return p.parse_args()


def clean_zip(s):
    return (
        s.astype("string")
        .str.extract(r"(\d{5})", expand=False)
        .str.zfill(5)
    )


def add_label_audit_fields(df, zip_place_crosswalk=None):
    """
    Preserve the school-derived label and add a separate display/postal label.

    The legacy Town field is produced by grouping the MassGIS school shapefile
    on ZIPCODE and taking TOWN/TOWN_MAIL. It is therefore not guaranteed to be
    the canonical ZIP place name.

    Coordinates are already Census ZCTA representative points from
    build_zcta_geography_2024.py.
    """
    out = df.copy()
    out["ZIPCODE"] = clean_zip(out["ZIPCODE"])

    # Never silently treat the school-derived Town field as a postal label.
    out["School_Town_Label"] = out.get("Town", "").fillna("").astype(str)

    out["Postal_Place"] = ""
    out["Postal_Place_Source"] = ""

    if zip_place_crosswalk:
        x = pd.read_csv(zip_place_crosswalk, dtype={"ZIPCODE": str})
        if not {"ZIPCODE", "Postal_Place"}.issubset(x.columns):
            raise ValueError(
                "--zip-place-crosswalk must contain ZIPCODE and Postal_Place columns."
            )
        x["ZIPCODE"] = clean_zip(x["ZIPCODE"])
        x = x[["ZIPCODE", "Postal_Place"]].drop_duplicates("ZIPCODE")
        out = out.merge(x, on="ZIPCODE", how="left", suffixes=("", "_crosswalk"))
        out["Postal_Place"] = out["Postal_Place_crosswalk"].fillna(out["Postal_Place"])
        out = out.drop(columns=["Postal_Place_crosswalk"])
        out.loc[out["Postal_Place"].astype(str).str.len() > 0, "Postal_Place_Source"] = (
            "user-supplied ZIP/place crosswalk"
        )

    for z, place in KNOWN_POSTAL_PLACE_OVERRIDES.items():
        mask = out["ZIPCODE"].eq(z) & out["Postal_Place"].astype(str).eq("")
        out.loc[mask, "Postal_Place"] = place
        out.loc[mask, "Postal_Place_Source"] = "known audited override"

    out["Display_Place"] = np.where(
        out["Postal_Place"].astype(str).str.strip().ne(""),
        out["Postal_Place"],
        out["School_Town_Label"],
    )

    out["Label_Audit_Flag"] = ""
    disagrees = (
        out["Postal_Place"].astype(str).str.strip().ne("")
        & out["School_Town_Label"].astype(str).str.strip().ne("")
        & out["Postal_Place"].str.upper().ne(out["School_Town_Label"].str.upper())
    )
    out.loc[disagrees, "Label_Audit_Flag"] = (
        "postal/display place differs from school-town label"
    )

    return out


def row_count_summary(primary, master=None):
    rows = [
        {
            "Dataset": "Primary complete-case analysis",
            "Rows": int(len(primary)),
            "Unique_ZIP_ZCTA": int(primary["ZIPCODE"].nunique()),
            "Duplicate_ZIP_rows": int(primary["ZIPCODE"].duplicated().sum()),
        }
    ]
    if master is not None:
        rows.append(
            {
                "Dataset": "Merged master dataset",
                "Rows": int(len(master)),
                "Unique_ZIP_ZCTA": int(master["ZIPCODE"].nunique()),
                "Duplicate_ZIP_rows": int(master["ZIPCODE"].duplicated().sum()),
            }
        )
    return pd.DataFrame(rows)


def fit_participation_pca(df):
    return fit_pca(
        df,
        PARTICIPATION_MODEL_FEATURES,
        sign_rules={
            0: "ACS_Median_Family_Income",
            1: "MCAS_ELA_STU_PART_PCT_WT",
            2: "Computer_No_Internet_Rate",
        },
    )


def add_pca_scores(df, scores):
    out = df.copy()
    out["PC1_Participation"] = scores[:, 0]
    out["PC2_Participation"] = scores[:, 1]
    out["PC3_Participation"] = scores[:, 2]
    return out


def kmeans_labels(X, k, seed=42):
    return KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=100,
    ).fit_predict(X)


def cluster_solution_table(df, X3, ks=(2, 3, 4, 5, 6)):
    out = df.copy()
    rows = []

    for k in ks:
        labels = kmeans_labels(X3, k)
        out[f"K{k}_3PC"] = labels
        counts = pd.Series(labels).value_counts().sort_index()

        rows.append(
            {
                "k": k,
                "Silhouette": float(silhouette_score(X3, labels)),
                "Calinski_Harabasz": float(calinski_harabasz_score(X3, labels)),
                "Davies_Bouldin": float(davies_bouldin_score(X3, labels)),
                "Min_Cluster_N": int(counts.min()),
                "Max_Cluster_N": int(counts.max()),
                "Cluster_Sizes": json.dumps(counts.to_dict()),
            }
        )

    return out, pd.DataFrame(rows)


def cluster_stability(X, k, seeds=range(20)):
    """
    Simple initialization stability diagnostic:
    compare each one-start KMeans solution against a high-n_init reference.
    ARI is label-permutation invariant.
    """
    reference = KMeans(
        n_clusters=k, random_state=42, n_init=100
    ).fit_predict(X)

    aris = []
    for seed in seeds:
        labels = KMeans(
            n_clusters=k, random_state=int(seed), n_init=1
        ).fit_predict(X)
        aris.append(adjusted_rand_score(reference, labels))

    return {
        "k": int(k),
        "ARI_mean": float(np.mean(aris)),
        "ARI_min": float(np.min(aris)),
        "ARI_max": float(np.max(aris)),
    }


def profile_clusters(df, ks=(3, 4, 5)):
    profile_vars = [
        "SES_Score",
        "Digital_Access_Score",
        "Education_Score",
        "Participation_Score",
        "PC1_Participation",
        "PC2_Participation",
        "PC3_Participation",
        "Broadband_Rate",
        "No_Computer_Rate",
        "Computer_No_Internet_Rate",
        "ACS_Median_Family_Income",
        "Low_Income_Share_lt25k",
        "High_Income_Share_150k_plus",
        "MCAS_Math_M_PLUS_E_PCT_ZIP",
        "MCAS_ELA_M_PLUS_E_PCT_ZIP",
        "MCAS_Math_STU_PART_PCT_WT",
        "MCAS_ELA_STU_PART_PCT_WT",
    ]

    rows = []
    for k in ks:
        c = f"K{k}_3PC"
        for cluster_id, g in df.groupby(c):
            row = {
                "k": int(k),
                "Cluster": int(cluster_id),
                "N": int(len(g)),
            }
            for var in profile_vars:
                row[var] = float(g[var].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def representative_zips(df, ks=(3, 4, 5), n_each=5):
    """
    Give Gerber concrete labels to inspect:
    - nearest ZIPs to each cluster centroid (representative)
    - farthest ZIPs from the centroid (within-cluster extremes)
    """
    rows = []
    pc_cols = ["PC1_Participation", "PC2_Participation", "PC3_Participation"]

    for k in ks:
        c = f"K{k}_3PC"
        for cluster_id, g in df.groupby(c):
            centroid = g[pc_cols].mean().to_numpy()
            dist = np.sqrt(((g[pc_cols].to_numpy() - centroid) ** 2).sum(axis=1))
            tmp = g[
                [
                    "ZIPCODE",
                    "Display_Place",
                    "School_Town_Label",
                    "Postal_Place",
                    "Broadband_Rate",
                    "ACS_Median_Family_Income",
                    "MCAS_Math_M_PLUS_E_PCT_ZIP",
                    "MCAS_ELA_M_PLUS_E_PCT_ZIP",
                    "PC1_Participation",
                    "PC2_Participation",
                    "PC3_Participation",
                    "Mismatch_Type",
                ]
            ].copy()
            tmp["Distance_to_Cluster_Centroid"] = dist

            for label, subset in [
                ("representative", tmp.nsmallest(n_each, "Distance_to_Cluster_Centroid")),
                ("within-cluster extreme", tmp.nlargest(n_each, "Distance_to_Cluster_Centroid")),
            ]:
                subset = subset.copy()
                subset.insert(0, "Selection_Type", label)
                subset.insert(0, "Cluster", int(cluster_id))
                subset.insert(0, "k", int(k))
                rows.append(subset)

    return pd.concat(rows, ignore_index=True)


def model_selection_grid(df, scores):
    """
    Grid over k, clustering representation, and scaling.

    IMPORTANT:
    KMeans is distance-based. The primary feature-space comparison should use
    scaled variables. 'none' is included only as a diagnostic to show what
    happens when income/dollar variables dominate Euclidean distance.
    """
    features = PARTICIPATION_MODEL_FEATURES

    scalers = {
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "minmax": MinMaxScaler(),
        "none_DIAGNOSTIC_ONLY": None,
    }

    spaces = {
        "PC1_PC2": scores[:, :2],
        "PC1_PC2_PC3": scores[:, :3],
        "PC1_PC2_PC3_PC4": scores[:, :4],
    }

    rows = []

    # PCA spaces are already based on standardized features.
    for space_name, X in spaces.items():
        for k in range(2, 9):
            labels = kmeans_labels(X, k)
            counts = pd.Series(labels).value_counts()
            rows.append(
                {
                    "Representation": space_name,
                    "Scaling": "PCA_from_standardized_features",
                    "k": k,
                    "Silhouette": float(silhouette_score(X, labels)),
                    "Calinski_Harabasz": float(calinski_harabasz_score(X, labels)),
                    "Davies_Bouldin": float(davies_bouldin_score(X, labels)),
                    "Min_Cluster_N": int(counts.min()),
                }
            )

    # Direct feature-space sensitivity to scaling choice.
    raw = df[features].to_numpy(dtype=float)
    for scaler_name, scaler in scalers.items():
        X = raw if scaler is None else scaler.fit_transform(raw)
        for k in range(2, 9):
            labels = kmeans_labels(X, k)
            counts = pd.Series(labels).value_counts()
            rows.append(
                {
                    "Representation": "Full_10_features",
                    "Scaling": scaler_name,
                    "k": k,
                    "Silhouette": float(silhouette_score(X, labels)),
                    "Calinski_Harabasz": float(calinski_harabasz_score(X, labels)),
                    "Davies_Bouldin": float(davies_bouldin_score(X, labels)),
                    "Min_Cluster_N": int(counts.min()),
                }
            )

    result = pd.DataFrame(rows)
    result["Recommended_for_inference"] = ~result["Scaling"].eq(
        "none_DIAGNOSTIC_ONLY"
    )
    return result



def fit_pca_kde_outliers(df, scores, loadings, bandwidth_grid=None):
    """
    Model-based density diagnostic in PC1-PC3 space.

    This complements (does not replace) the interpretable mismatch labels.
    KDE estimates where observations lie in dense vs sparse regions of the
    standardized PCA representation.

    Bandwidth is selected by cross-validated held-out log likelihood.
    Lower density / higher negative log density = more unusual multivariate
    PCA profile.
    """
    X = scores[:, :3]
    bandwidth_grid = bandwidth_grid or np.linspace(0.25, 2.0, 15)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    for bw in bandwidth_grid:
        fold_scores = []
        for train_idx, test_idx in cv.split(X):
            kde = KernelDensity(kernel="gaussian", bandwidth=float(bw))
            kde.fit(X[train_idx])
            fold_scores.append(float(kde.score(X[test_idx]) / len(test_idx)))
        rows.append(
            {
                "Bandwidth": float(bw),
                "Mean_Heldout_LogDensity": float(np.mean(fold_scores)),
            }
        )

    cv_table = pd.DataFrame(rows)
    best_bw = float(
        cv_table.loc[
            cv_table["Mean_Heldout_LogDensity"].idxmax(), "Bandwidth"
        ]
    )

    kde = KernelDensity(kernel="gaussian", bandwidth=best_bw).fit(X)
    log_density = kde.score_samples(X)

    out = df.copy()
    out["PCA_KDE_LogDensity"] = log_density
    out["PCA_KDE_OutlierScore"] = -log_density
    out["PCA_KDE_OutlierPercentile"] = (
        out["PCA_KDE_OutlierScore"].rank(pct=True, method="average")
    )

    # Which PC contributes the largest standardized displacement from the origin?
    pcs = out[
        ["PC1_Participation", "PC2_Participation", "PC3_Participation"]
    ].to_numpy(dtype=float)
    dominant_idx = np.argmax(np.abs(pcs), axis=1)
    out["Dominant_PC_Axis"] = [f"PC{i+1}" for i in dominant_idx]

    # Give a loading-based interpretation of that dominant axis.
    axis_interpretation = {}
    for pc in ["PC1", "PC2", "PC3"]:
        top = loadings[pc].abs().nlargest(4).index.tolist()
        axis_interpretation[pc] = ", ".join(
            DISPLAY_NAMES.get(x, x) for x in top
        )
    out["Dominant_PC_Top_Loadings"] = out["Dominant_PC_Axis"].map(
        axis_interpretation
    )

    return out, cv_table, best_bw


def save_pca_kde_density_map(df, out):
    """
    Empirical density view in PC1-PC2 space. Points are colored by KDE outlier
    score estimated in full PC1-PC3 space.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        df["PC1_Participation"],
        df["PC2_Participation"],
        c=df["PCA_KDE_OutlierScore"],
        s=42,
        alpha=0.82,
    )

    top = df.nlargest(12, "PCA_KDE_OutlierScore")
    for _, r in top.iterrows():
        ax.annotate(
            f"{r['ZIPCODE']} {r['Display_Place']}",
            (r["PC1_Participation"], r["PC2_Participation"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    ax.axhline(0, linewidth=0.7)
    ax.axvline(0, linewidth=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        "PCA density/outlier diagnostic\n"
        "(color = KDE outlier score estimated in PC1-PC3 space)"
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("KDE outlier score (-log density)")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def kde_outlier_review(df, n=30):
    cols = [
        "ZIPCODE",
        "Display_Place",
        "School_Town_Label",
        "Postal_Place",
        "PCA_KDE_OutlierScore",
        "PCA_KDE_OutlierPercentile",
        "Dominant_PC_Axis",
        "Dominant_PC_Top_Loadings",
        "PC1_Participation",
        "PC2_Participation",
        "PC3_Participation",
        "Broadband_Rate",
        "ACS_Median_Family_Income",
        "MCAS_Math_M_PLUS_E_PCT_ZIP",
        "MCAS_ELA_M_PLUS_E_PCT_ZIP",
        "Mismatch_Type",
        "Digital_Access_Residual",
    ]
    return df.nlargest(n, "PCA_KDE_OutlierScore")[
        [c for c in cols if c in df.columns]
    ].copy()


def save_k_comparison_plot(metrics, out):
    d = metrics[metrics["k"].isin([2, 3, 4, 5, 6])].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(d["k"], d["Silhouette"], marker="o")
    for _, r in d.iterrows():
        ax.annotate(
            f"{r['Silhouette']:.3f}",
            (r["k"], r["Silhouette"]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Participation model, first 3 PCs: k=2 through k=6")
    ax.set_xticks([2, 3, 4, 5, 6])
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_cluster_pair_plot(df, k, x, y, out):
    fig, ax = plt.subplots(figsize=(9, 7))
    c = f"K{k}_3PC"
    for cluster_id in sorted(df[c].unique()):
        g = df[df[c] == cluster_id]
        ax.scatter(g[x], g[y], s=38, alpha=0.7, label=f"Cluster {cluster_id}")
    ax.axhline(0, linewidth=0.7)
    ax.axvline(0, linewidth=0.7)
    ax.set_xlabel(x.replace("_Participation", ""))
    ax.set_ylabel(y.replace("_Participation", ""))
    ax.set_title(f"k={k} clustering on first 3 participation-model PCs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_refined_pc3_map(df, out):
    """
    Robust range prevents one extreme observation (notably 01434) from making
    most other ZIPs look like nearly the same shade.

    We keep the raw PC3 value in the data; clipping here affects visualization only.
    """
    vals = df["PC3_Participation"].to_numpy(dtype=float)

    # Symmetric robust limit around zero using the 98th percentile of |PC3|.
    lim = float(np.nanpercentile(np.abs(vals), 98))
    display_vals = np.clip(vals, -lim, lim)

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        df["Longitude"],
        df["Latitude"],
        c=display_vals,
        s=42,
        alpha=0.85,
        cmap="coolwarm",
        vmin=-lim,
        vmax=lim,
    )

    # Label the most extreme raw PC3 observations for audit.
    extreme = df.assign(abs_pc3=np.abs(vals)).nlargest(8, "abs_pc3")
    for _, r in extreme.iterrows():
        ax.annotate(
            f"{r['ZIPCODE']} {r['Display_Place']}",
            (r["Longitude"], r["Latitude"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlabel("Longitude (Census ZCTA representative point)")
    ax.set_ylabel("Latitude (Census ZCTA representative point)")
    ax.set_title(
        "PC3 geography — robust symmetric display range\n"
        "(colors clipped at 98th percentile of |PC3|; raw values unchanged)"
    )
    ax.set_aspect(1 / np.cos(np.deg2rad(42.2)), adjustable="box")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.82, extend="both")
    cbar.set_label("PC3 score (display-clipped only)")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_broadband_subscription_map(df, out):
    """
    This is ACS household broadband subscription/adoption, not infrastructure coverage.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        df["Longitude"],
        df["Latitude"],
        c=df["Broadband_Rate"],
        s=42,
        alpha=0.85,
    )
    ax.set_xlabel("Longitude (Census ZCTA representative point)")
    ax.set_ylabel("Latitude (Census ZCTA representative point)")
    ax.set_title(
        "ACS household broadband subscription rate by school-location ZIP/ZCTA\n"
        "(adoption/subscription proxy — not FCC/MBI infrastructure availability)"
    )
    ax.set_aspect(1 / np.cos(np.deg2rad(42.2)), adjustable="box")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
    cbar.set_label("Broadband subscription rate")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_interactive_broadband_subscription_map(df, out_html):
    try:
        import plotly.express as px
    except Exception as exc:
        print(f"WARNING: interactive broadband map skipped: {exc}")
        return

    d = df.copy()
    fig = px.scatter_mapbox(
        d,
        lat="Latitude",
        lon="Longitude",
        color="Broadband_Rate",
        hover_name="ZIPCODE",
        hover_data={
            "Display_Place": True,
            "School_Town_Label": True,
            "Broadband_Rate": ":.3f",
            "No_Computer_Rate": ":.3f",
            "Computer_No_Internet_Rate": ":.3f",
            "ACS_Median_Family_Income": ":,.0f",
            "PC3_Participation": ":.2f",
            "Mismatch_Type": True,
        },
        zoom=6.6,
        height=760,
        title=(
            "ACS household broadband subscription rate "
            "(not infrastructure availability/coverage)"
        ),
    )
    fig.update_traces(marker=dict(size=9, opacity=0.82))
    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=55, b=0),
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def geography_audit_table(df):
    cols = [
        "ZIPCODE",
        "Display_Place",
        "School_Town_Label",
        "Postal_Place",
        "Postal_Place_Source",
        "Label_Audit_Flag",
        "Latitude",
        "Longitude",
        "PC3_Participation",
        "Broadband_Rate",
        "ACS_Total_Households",
        "ACS_Total_Families",
        "ACS_Median_Family_Income",
        "Mismatch_Type",
    ]
    available = [c for c in cols if c in df.columns]
    return df[available].sort_values("ZIPCODE").copy()


def local_case_study(df, zips=DEFAULT_INVESTIGATION_ZIPS):
    cols = [
        "ZIPCODE",
        "Display_Place",
        "School_Town_Label",
        "Postal_Place",
        "Latitude",
        "Longitude",
        "ACS_Total_Households",
        "ACS_Total_Families",
        "Broadband_Rate",
        "No_Computer_Rate",
        "Computer_No_Internet_Rate",
        "ACS_Median_Family_Income",
        "Low_Income_Share_lt25k",
        "High_Income_Share_150k_plus",
        "MCAS_Math_M_PLUS_E_PCT_ZIP",
        "MCAS_ELA_M_PLUS_E_PCT_ZIP",
        "MCAS_Math_STU_PART_PCT_WT",
        "MCAS_ELA_STU_PART_PCT_WT",
        "SES_Score",
        "Digital_Access_Score",
        "Education_Score",
        "Participation_Score",
        "PC1_Participation",
        "PC2_Participation",
        "PC3_Participation",
        "Digital_Access_Residual",
        "Mismatch_Type",
        "K3_3PC",
        "K4_3PC",
        "K5_3PC",
    ]
    return df[df["ZIPCODE"].isin(zips)][[c for c in cols if c in df.columns]].copy()


def mismatch_methodology_markdown(df):
    q25 = df[["SES_Score", "Digital_Access_Score", "Education_Score"]].quantile(0.25)
    q75 = df[["SES_Score", "Digital_Access_Score", "Education_Score"]].quantile(0.75)

    lines = [
        "# Mismatch map methodology",
        "",
        "## What data are used?",
        "",
        "The mismatch analysis uses the same 411 complete-case school-location ZIP/ZCTA observations as the participation-retained PCA extension.",
        "",
        "Four standardized domain scores are constructed:",
        "",
        "- **SES score** = mean(z median family income, -z low-income share, z high-income share).",
        "- **Digital-access score** = mean(z broadband subscription rate, -z no-computer rate, -z computer-without-internet rate).",
        "- **Education score** = mean(z Math attainment, z ELA attainment).",
        "- **Participation score** = mean(z Math participation, z ELA participation).",
        "",
        "## Threshold-based mismatch labels",
        "",
        "High/low is defined using the 75th/25th percentile of the relevant domain score within the 411-row analysis sample.",
        "",
        f"- SES 25th percentile: {q25['SES_Score']:.3f}",
        f"- SES 75th percentile: {q75['SES_Score']:.3f}",
        f"- Digital access 25th percentile: {q25['Digital_Access_Score']:.3f}",
        f"- Digital access 75th percentile: {q75['Digital_Access_Score']:.3f}",
        f"- Education 25th percentile: {q25['Education_Score']:.3f}",
        f"- Education 75th percentile: {q75['Education_Score']:.3f}",
        "",
        "The code first assigns narrower three-domain combinations (for example high SES + high education + low digital access), then assigns broader two-domain mismatches to observations that remain unassigned.",
        "",
        "These categories are screening flags, not learned classes and not causal findings.",
        "",
        "## Residual-based mismatch diagnostic",
        "",
        "A linear regression predicts the digital-access score from SES and education. The digital-access residual is:",
        "",
        "`observed digital access - predicted digital access`",
        "",
        "Large negative residuals indicate places with lower digital access than would be expected from their SES and education profile; large positive residuals indicate the reverse.",
        "",
        "A second regression predicts education from SES and digital access and produces an analogous education residual.",
        "",
        "## Important interpretation caveats",
        "",
        "- MCAS is aggregated to the physical ZIP of schools, not student home ZIP.",
        "- ACS broadband variables measure household subscription/adoption, not infrastructure availability.",
        "- Small ZCTA denominators can make ACS estimates unusually volatile; flagged ZIPs should be audited before substantive interpretation.",
        "- The threshold labels depend on this sample's quartiles; they are exploratory, not externally validated cutoffs.",
    ]
    return "\n".join(lines) + "\n"


def write_findings_summary(outdir, counts, metrics, stability, case):
    k3 = metrics.loc[metrics["k"].eq(3)].iloc[0]
    k4 = metrics.loc[metrics["k"].eq(4)].iloc[0]
    k5 = metrics.loc[metrics["k"].eq(5)].iloc[0]

    text = f"""# November 2 Gerber follow-up outputs

## Dataset size

{counts.to_markdown(index=False)}

## k=3 / k=4 / k=5

The first-three-PC participation-model silhouette scores are:

- k=3: {k3['Silhouette']:.3f}
- k=4: {k4['Silhouette']:.3f}
- k=5: {k5['Silhouette']:.3f}

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

{case.to_markdown(index=False)}

The Devens record should be treated as an audit priority because it combines a
very large PC3 score with an unusually low ACS broadband-subscription rate and
a small ACS household denominator relative to neighboring ZIPs.
"""

    (outdir / "NOV2_FOLLOWUP_FINDINGS.md").write_text(text)


def main():
    args = parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_csv(args.primary)
    master = read_csv(args.master) if args.master else None

    # Recreate/retain the school-town label if a shapefile is available.
    if args.schools_shp:
        df = add_town_labels(df, args.schools_shp)
    elif "Town" not in df.columns:
        df["Town"] = ""

    required = PARTICIPATION_MODEL_FEATURES + ["Latitude", "Longitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Primary dataset missing required columns: {missing}")

    df = df.dropna(subset=required).reset_index(drop=True)
    df = add_label_audit_fields(df, args.zip_place_crosswalk)

    counts = row_count_summary(df, master)
    counts.to_csv(outdir / "00_dataset_row_counts.csv", index=False)

    # PCA and domain/mismatch scores.
    z10, pca10, scores10, load10, exp10 = fit_participation_pca(df)
    df = add_pca_scores(df, scores10)
    df = add_domain_scores(df)

    # Reuse the existing threshold mismatch logic without needing its output files.
    # mismatch_analysis writes files, so use a small temporary subdirectory and
    # keep the resulting dataframe.
    mismatch_dir = outdir / "_mismatch_temp"
    mismatch_dir.mkdir(exist_ok=True)
    df = mismatch_analysis(df, mismatch_dir)

    # Cluster solutions emphasized in the Nov. 2 meeting.
    X3 = scores10[:, :3]
    df, metrics = cluster_solution_table(df, X3, ks=(2, 3, 4, 5, 6))
    metrics.to_csv(outdir / "01_k2_to_k6_3pc_metrics.csv", index=False)

    stability = pd.DataFrame(
        [cluster_stability(X3, k) for k in [2, 3, 4, 5, 6]]
    )
    stability.to_csv(outdir / "02_k2_to_k6_initialization_stability.csv", index=False)

    profiles = profile_clusters(df, ks=(3, 4, 5))
    profiles.to_csv(outdir / "03_k3_k4_k5_cluster_profiles.csv", index=False)

    assignments = df[
        [
            "ZIPCODE",
            "Display_Place",
            "School_Town_Label",
            "Postal_Place",
            "Latitude",
            "Longitude",
            "PC1_Participation",
            "PC2_Participation",
            "PC3_Participation",
            "K3_3PC",
            "K4_3PC",
            "K5_3PC",
            "Mismatch_Type",
        ]
    ].copy()
    assignments.to_csv(outdir / "04_k3_k4_k5_zip_assignments.csv", index=False)

    reps = representative_zips(df, ks=(3, 4, 5), n_each=5)
    reps.to_csv(outdir / "05_cluster_labels_to_investigate.csv", index=False)

    grid = model_selection_grid(df, scores10)
    grid.to_csv(outdir / "06_kmeans_model_selection_grid.csv", index=False)

    # More model-based multivariate outlier diagnostic requested in the Nov. 2
    # discussion: KDE density in PC1-PC3 space.
    df, kde_cv, best_kde_bw = fit_pca_kde_outliers(df, scores10, load10)
    kde_cv.to_csv(outdir / "06b_pca_kde_bandwidth_cv.csv", index=False)
    kde_outlier_review(df, n=30).to_csv(
        outdir / "06c_pca_kde_outlier_review.csv", index=False
    )

    audit = geography_audit_table(df)
    audit.to_csv(outdir / "07_zip_coordinate_town_label_audit.csv", index=False)

    case = local_case_study(df)
    case.to_csv(outdir / "08_ayer_devens_shirley_comparison.csv", index=False)

    (outdir / "09_MISMATCH_MAP_METHODOLOGY.md").write_text(
        mismatch_methodology_markdown(df)
    )

    # Plots.
    save_k_comparison_plot(metrics, outdir / "10_k2_to_k6_silhouette.png")
    for k in [3, 4, 5]:
        save_cluster_pair_plot(
            df, k,
            "PC1_Participation", "PC3_Participation",
            outdir / f"11_k{k}_pc1_vs_pc3.png",
        )

    # k=5 interactive 3D triplot requested in meeting.
    create_3d_triplot(
        df,
        pca10,
        scores10,
        load10,
        df["K5_3PC"].to_numpy(),
        outdir / "12_k5_participation_3d_triplot.html",
        "Participation-retained PCA triplot: exploratory k=5",
    )

    save_refined_pc3_map(df, outdir / "13_pc3_geographic_map_refined.png")
    save_broadband_subscription_map(
        df, outdir / "14_acs_broadband_subscription_map.png"
    )
    create_interactive_broadband_subscription_map(
        df, outdir / "15_acs_broadband_subscription_map.html"
    )
    save_pca_kde_density_map(
        df, outdir / "15b_pca_kde_density_outlier_map.png"
    )

    # Save enriched table with provenance-safe labels, k=3/k=4/k=5,
    # and KDE outlier scores.
    df.to_csv(outdir / "16_nov2_followup_enriched_411.csv", index=False)

    write_findings_summary(outdir, counts, metrics, stability, case)

    # Clean temporary mismatch products; canonical methodology/output lives above.
    for p in mismatch_dir.glob("*"):
        p.unlink()
    mismatch_dir.rmdir()

    print("\nNOVEMBER 2 GERBER FOLLOW-UP COMPLETE")
    print(counts.to_string(index=False))
    print("\n3-PC k metrics:")
    print(metrics.to_string(index=False))
    print(f"\nSelected KDE bandwidth: {best_kde_bw:.3f}")
    print("\nOutputs:", outdir)


if __name__ == "__main__":
    main()
