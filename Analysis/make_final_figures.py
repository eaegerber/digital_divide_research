#!/usr/bin/env python
"""
Create publication-ready figures for the 2024 Massachusetts Digital Divide analysis.

Primary model:
    C_math_ela_attainment

Required inputs:
    --master        Expanded 413-row ACS + MCAS master dataset
    --primary       411-row C_math_ela_attainment cluster_assignments.csv
    --reclassified  A_vs_C_reclassified_zips.csv

Optional:
    --science       F_add_science_participation/cluster_assignments.csv

Outputs:
    01_primary_pca_biplot.png
    02_primary_pca_kmeans.png
    03_primary_cluster_map.png
    04_A_vs_C_reclassification_map.png
    05_kmeans_silhouette.png
    06_pca_explained_variance.png
    07_science_participation_sensitivity.png   (if --science supplied)
"""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
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

DISPLAY_NAMES = {
    "Broadband_Rate": "Broadband",
    "No_Computer_Rate": "No computer",
    "Computer_No_Internet_Rate": "Computer, no internet",
    "ACS_Median_Family_Income": "Median family income",
    "Low_Income_Share_lt25k": "Low-income share",
    "High_Income_Share_150k_plus": "High-income share",
    "MCAS_Math_M_PLUS_E_PCT_ZIP": "Math attainment",
    "MCAS_ELA_M_PLUS_E_PCT_ZIP": "ELA attainment",
}

PRIMARY_CLUSTER_NAMES = {
    0: "Higher SES/access/attainment profile",
    1: "Lower SES/access/attainment profile",
}

SCIENCE_CLUSTER_NAMES = {
    0: "Lower combined profile",
    1: "Higher combined profile",
    2: "Low-participation outlier group",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--master", required=True)
    p.add_argument("--primary", required=True)
    p.add_argument("--reclassified", required=True)
    p.add_argument("--science", default=None)
    p.add_argument("--output", required=True)
    return p.parse_args()


def read_csv(path):
    df = pd.read_csv(path, dtype={"ZIPCODE": str})
    if "ZIPCODE" in df.columns:
        df["ZIPCODE"] = df["ZIPCODE"].str.zfill(5)
    return df


def validate(master, primary, reclassified):
    missing = [c for c in PRIMARY_FEATURES if c not in primary.columns]
    if missing:
        raise ValueError(f"Primary file is missing required columns: {missing}")

    if len(primary) != 411:
        print(f"WARNING: expected 411 primary rows, found {len(primary)}")

    n_missing = int(primary[PRIMARY_FEATURES].isna().sum().sum())
    if n_missing:
        raise ValueError(f"Primary features contain {n_missing} missing values.")

    required_primary = ["KMeans_Cluster", "PC1", "PC2", "Latitude", "Longitude"]
    missing_primary = [c for c in required_primary if c not in primary.columns]
    if missing_primary:
        raise ValueError(f"Primary file is missing: {missing_primary}")

    required_reclass = ["ZIPCODE", "Transition", "Latitude", "Longitude"]
    missing_reclass = [c for c in required_reclass if c not in reclassified.columns]
    if missing_reclass:
        raise ValueError(f"Reclassification file is missing: {missing_reclass}")

    print("\nINPUT CHECK")
    print(f"Master: {master.shape}")
    print(f"Primary: {primary.shape}")
    print(f"Reclassified: {reclassified.shape}")
    print("\nPrimary KMeans sizes:")
    print(primary["KMeans_Cluster"].value_counts().sort_index().to_string())


def fit_primary_pca(primary):
    scaler = StandardScaler()
    z = scaler.fit_transform(primary[PRIMARY_FEATURES])

    pca = PCA()
    scores = pca.fit_transform(z)

    # sklearn's PCA sign is arbitrary. Align with the already-saved analysis
    # so regenerated figures have the same PC orientation.
    for i, existing_col in enumerate(["PC1", "PC2"]):
        corr = np.corrcoef(scores[:, i], primary[existing_col])[0, 1]
        if corr < 0:
            scores[:, i] *= -1
            pca.components_[i, :] *= -1

    return z, pca, scores


def save_biplot(primary, pca, scores, out):
    fig, ax = plt.subplots(figsize=(12, 8))

    for cluster_id in sorted(primary["KMeans_Cluster"].unique()):
        mask = primary["KMeans_Cluster"].to_numpy() == cluster_id
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            alpha=0.62,
            s=38,
            label=PRIMARY_CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}"),
        )

    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)

    loadings = pca.components_.T[:, :2]

    # Scale vectors into the observation-score coordinate system.
    x_span = np.nanpercentile(scores[:, 0], 95) - np.nanpercentile(scores[:, 0], 5)
    y_span = np.nanpercentile(scores[:, 1], 95) - np.nanpercentile(scores[:, 1], 5)
    scale = min(x_span, y_span) * 0.68

    for feature, (lx, ly) in zip(PRIMARY_FEATURES, loadings):
        x_end = lx * scale
        y_end = ly * scale

        ax.annotate(
            "",
            xy=(x_end, y_end),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", linewidth=1.5),
        )

        dx = 6 if x_end >= 0 else -6
        dy = 6 if y_end >= 0 else -6

        ax.annotate(
            DISPLAY_NAMES[feature],
            xy=(x_end, y_end),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if x_end >= 0 else "right",
            va="bottom" if y_end >= 0 else "top",
            fontsize=9,
        )

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 — joint SES/access/attainment gradient ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_title("Primary PCA biplot: socioeconomic, digital-access, and education features")
    ax.legend(title="Primary KMeans segmentation", loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_kmeans_pca(primary, pca, scores, out):
    fig, ax = plt.subplots(figsize=(10, 7))

    for cluster_id in sorted(primary["KMeans_Cluster"].unique()):
        mask = primary["KMeans_Cluster"].to_numpy() == cluster_id
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            alpha=0.68,
            s=42,
            label=PRIMARY_CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}"),
        )

    ev = pca.explained_variance_ratio_
    ax.axvline(0, linewidth=0.7)
    ax.axhline(0, linewidth=0.7)
    ax.set_xlabel(f"PC1 — joint SES/access/attainment gradient ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_title("Primary PCA and KMeans segmentation")
    ax.legend(title="Profile")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def try_add_massachusetts_outline(ax):
    """
    Add the official Census 2024 Massachusetts state boundary when geopandas
    and internet access are available. If unavailable, leave the point map
    unchanged and print a warning.
    """
    try:
        import geopandas as gpd

        url = "https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_state_500k.zip"

        with tempfile.TemporaryDirectory() as td:
            zpath = Path(td) / "cb_2024_us_state_500k.zip"
            urllib.request.urlretrieve(url, zpath)
            states = gpd.read_file(f"zip://{zpath}")
            ma = states.loc[states["STUSPS"] == "MA"]

            if len(ma):
                ma.boundary.plot(ax=ax, linewidth=1.2)
                return True
    except Exception as exc:
        print(f"WARNING: Massachusetts outline could not be added: {exc}")

    return False


def format_map_axes(ax, title):
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")


def save_cluster_map(primary, out):
    fig, ax = plt.subplots(figsize=(10, 8))
    outlined = try_add_massachusetts_outline(ax)

    for cluster_id in sorted(primary["KMeans_Cluster"].unique()):
        d = primary.loc[primary["KMeans_Cluster"] == cluster_id]
        ax.scatter(
            d["Longitude"],
            d["Latitude"],
            s=42,
            alpha=0.78,
            label=PRIMARY_CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}"),
        )

    format_map_axes(
        ax,
        "Primary KMeans profiles across Massachusetts school-location ZIP/ZCTAs",
    )
    ax.legend(title="Profile", loc="best")

    if not outlined:
        ax.text(
            0.01,
            0.01,
            "State outline unavailable; points shown at Census ZCTA representative coordinates.",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_reclassification_map(primary, reclassified, master, out):
    base = primary[["ZIPCODE", "Latitude", "Longitude"]].copy()
    base["Reclassification"] = "Stable corresponding group"

    transitions = reclassified[["ZIPCODE", "Transition"]].copy()
    base = base.merge(transitions, on="ZIPCODE", how="left")

    base.loc[base["Transition"].notna(), "Reclassification"] = base.loc[
        base["Transition"].notna(), "Transition"
    ]

    # Audit flag for display only; it does not determine reclassification.
    count_cols = [
        "ZIPCODE",
        "MCAS_Math_STU_CNT_SUM_ALL_VALID",
        "MCAS_ELA_STU_CNT_SUM_ALL_VALID",
    ]
    available = [c for c in count_cols if c in master.columns]

    if len(available) == 3:
        counts = master[available].copy()
        counts["Min_Math_ELA_Students"] = counts[
            [
                "MCAS_Math_STU_CNT_SUM_ALL_VALID",
                "MCAS_ELA_STU_CNT_SUM_ALL_VALID",
            ]
        ].min(axis=1)
        base = base.merge(
            counts[["ZIPCODE", "Min_Math_ELA_Students"]],
            on="ZIPCODE",
            how="left",
        )

    fig, ax = plt.subplots(figsize=(10, 8))
    outlined = try_add_massachusetts_outline(ax)

    categories = [
        "Stable corresponding group",
        "Higher SES/access → lower combined profile",
        "Lower SES/access → higher combined profile",
    ]

    for category in categories:
        d = base.loc[base["Reclassification"] == category]
        if len(d):
            ax.scatter(
                d["Longitude"],
                d["Latitude"],
                s=40 if category == "Stable corresponding group" else 64,
                alpha=0.48 if category == "Stable corresponding group" else 0.90,
                label=f"{category} (n={len(d)})",
            )

    format_map_axes(
        ax,
        "ZIPs reclassified after adding Math/ELA attainment",
    )
    ax.legend(title="A baseline → C primary model", loc="best")

    if not outlined:
        ax.text(
            0.01,
            0.01,
            "State outline unavailable; points shown at Census ZCTA representative coordinates.",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_silhouette(z, out):
    ks = list(range(2, 9))
    vals = []

    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=100)
        labels = model.fit_predict(z)
        vals.append(silhouette_score(z, labels))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ks, vals, marker="o")
    ax.scatter([2], [vals[0]], s=90, label=f"Selected k=2 ({vals[0]:.3f})")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Primary-model KMeans cluster separation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_explained_variance(pca, out):
    ev = pca.explained_variance_ratio_
    cumulative = np.cumsum(ev)
    pcs = np.arange(1, len(ev) + 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(pcs, ev, label="Individual explained variance")
    ax.plot(pcs, cumulative, marker="o", label="Cumulative explained variance")

    ax.annotate(
        f"PC1 = {ev[0]*100:.1f}%",
        xy=(1, ev[0]),
        xytext=(8, 8),
        textcoords="offset points",
    )
    ax.annotate(
        f"PC1+PC2 = {cumulative[1]*100:.1f}%",
        xy=(2, cumulative[1]),
        xytext=(8, 8),
        textcoords="offset points",
    )

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Primary-model PCA explained variance")
    ax.set_xticks(pcs)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_science_sensitivity(science, out):
    required = ["PC1", "PC2", "KMeans_Cluster"]
    missing = [c for c in required if c not in science.columns]
    if missing:
        raise ValueError(f"Science sensitivity file is missing: {missing}")

    fig, ax = plt.subplots(figsize=(10, 7))

    for cluster_id in sorted(science["KMeans_Cluster"].unique()):
        d = science.loc[science["KMeans_Cluster"] == cluster_id]
        ax.scatter(
            d["PC1"],
            d["PC2"],
            s=42,
            alpha=0.72,
            label=SCIENCE_CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}"),
        )

    ax.axhline(0, linewidth=0.7)
    ax.axvline(0, linewidth=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2 — participation-dominated dimension")
    ax.set_title("Sensitivity: Science participation creates a small outlier cluster")
    ax.legend(title="Sensitivity KMeans result", loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    master = read_csv(args.master)
    primary = read_csv(args.primary)
    reclassified = read_csv(args.reclassified)

    validate(master, primary, reclassified)

    z, pca, scores = fit_primary_pca(primary)

    print("\nPRIMARY PCA")
    print(f"PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
    print(f"PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
    print(f"PC1+PC2: {pca.explained_variance_ratio_[:2].sum()*100:.2f}%")

    save_biplot(
        primary, pca, scores,
        outdir / "01_primary_pca_biplot.png",
    )
    save_kmeans_pca(
        primary, pca, scores,
        outdir / "02_primary_pca_kmeans.png",
    )
    save_cluster_map(
        primary,
        outdir / "03_primary_cluster_map.png",
    )
    save_reclassification_map(
        primary, reclassified, master,
        outdir / "04_A_vs_C_reclassification_map.png",
    )
    save_silhouette(
        z,
        outdir / "05_kmeans_silhouette.png",
    )
    save_explained_variance(
        pca,
        outdir / "06_pca_explained_variance.png",
    )

    if args.science:
        science = read_csv(args.science)
        save_science_sensitivity(
            science,
            outdir / "07_science_participation_sensitivity.png",
        )

    summary = {
        "master_rows": int(len(master)),
        "primary_rows": int(len(primary)),
        "primary_features": PRIMARY_FEATURES,
        "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
        "pc2_explained_variance": float(pca.explained_variance_ratio_[1]),
        "pc1_pc2_cumulative": float(pca.explained_variance_ratio_[:2].sum()),
        "cluster_sizes": {
            str(k): int(v)
            for k, v in primary["KMeans_Cluster"].value_counts().sort_index().items()
        },
        "reclassified_rows": int(len(reclassified)),
    }

    with open(outdir / "figure_build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved final figures to:", outdir)
    for p in sorted(outdir.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
