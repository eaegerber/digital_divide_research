#!/usr/bin/env python3
"""
Statewide Massachusetts Digital Divide Analysis (2024)

Primary design:
- Unit: MCAS school-location ZIP matched to ACS 2024 ZCTA
- Unsupervised feature space:
    Broadband_Rate
    No_Computer_Rate
    Computer_No_Internet_Rate
    ACS_Median_Family_Income
    Low_Income_Share_lt25k
    High_Income_Share_150k_plus
- Education outcomes are NOT used to form the primary clusters.
  They are summarized afterward to interpret how school performance varies
  across the socioeconomic/digital-access clusters.

Outputs:
- analysis_dataset_all.csv
- analysis_dataset_complete_cases.csv
- descriptive_statistics.csv
- correlations_core.csv
- correlations_core_with_mcas.csv
- pca_explained_variance.csv
- pca_loadings.csv
- pca_scores.csv
- kmeans_model_selection.csv
- cluster_profiles.csv
- cluster_mcas_summary.csv
- cluster_assignments.csv
- hierarchical_comparison.json
- analysis_summary.json
- PNG figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


CORE_FEATURES = [
    "Broadband_Rate",
    "No_Computer_Rate",
    "Computer_No_Internet_Rate",
    "ACS_Median_Family_Income",
    "Low_Income_Share_lt25k",
    "High_Income_Share_150k_plus",
]

MCAS_OUTCOMES = [
    "MCAS_Math_M_PLUS_E_PCT_ZIP",
    "MCAS_ELA_M_PLUS_E_PCT_ZIP",
    "MCAS_Math_AVG_SCALED_SCORE_WT",
    "MCAS_ELA_AVG_SCALED_SCORE_WT",
]

RATE_FEATURES = [
    "Broadband_Rate",
    "No_Computer_Rate",
    "Computer_No_Internet_Rate",
    "Low_Income_Share_lt25k",
    "High_Income_Share_150k_plus",
]


def validate_columns(df: pd.DataFrame) -> None:
    required = ["ZIPCODE"] + CORE_FEATURES + MCAS_OUTCOMES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Input file is missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


def save_basic_diagnostics(df: pd.DataFrame, outdir: Path) -> None:
    desc_cols = CORE_FEATURES + MCAS_OUTCOMES
    df[desc_cols].describe().T.to_csv(outdir / "descriptive_statistics.csv")

    df[CORE_FEATURES].corr().to_csv(outdir / "correlations_core.csv")
    df[CORE_FEATURES + MCAS_OUTCOMES].corr().to_csv(
        outdir / "correlations_core_with_mcas.csv"
    )


def run_pca(complete: pd.DataFrame, outdir: Path):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(complete[CORE_FEATURES])

    pca = PCA()
    scores = pca.fit_transform(X_scaled)

    explained = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(CORE_FEATURES))],
        "Explained_Variance_Ratio": pca.explained_variance_ratio_,
        "Cumulative_Explained_Variance": np.cumsum(
            pca.explained_variance_ratio_
        ),
    })
    explained.to_csv(outdir / "pca_explained_variance.csv", index=False)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=CORE_FEATURES,
        columns=[f"PC{i+1}" for i in range(len(CORE_FEATURES))],
    )
    loadings.index.name = "Feature"
    loadings.to_csv(outdir / "pca_loadings.csv")

    score_df = complete[["ZIPCODE"]].copy()
    for i in range(scores.shape[1]):
        score_df[f"PC{i+1}"] = scores[:, i]
    score_df.to_csv(outdir / "pca_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(CORE_FEATURES) + 1)
    ax.bar(x, pca.explained_variance_ratio_)
    ax.plot(x, np.cumsum(pca.explained_variance_ratio_), marker="o")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA explained variance")
    ax.set_xticks(x)
    fig.tight_layout()
    fig.savefig(outdir / "pca_explained_variance.png", dpi=180)
    plt.close(fig)

    return X_scaled, pca, scores, loadings, explained


def select_kmeans(X_scaled: np.ndarray, outdir: Path):
    rows = []
    models = {}

    for k in range(2, 9):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=50,
        )
        labels = model.fit_predict(X_scaled)
        models[k] = (model, labels)

        rows.append({
            "k": k,
            "silhouette": silhouette_score(X_scaled, labels),
            "calinski_harabasz": calinski_harabasz_score(X_scaled, labels),
            "davies_bouldin": davies_bouldin_score(X_scaled, labels),
            "inertia": model.inertia_,
        })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir / "kmeans_model_selection.csv", index=False)

    # Primary data-driven candidate: highest silhouette.
    # Keep the full table so the research interpretation can consider
    # separation, parsimony, profile interpretability, and stability together.
    selected_k = int(
        metrics.loc[metrics["silhouette"].idxmax(), "k"]
    )
    selected_model, selected_labels = models[selected_k]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics["k"], metrics["silhouette"], marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("KMeans cluster separation")
    ax.set_xticks(metrics["k"])
    fig.tight_layout()
    fig.savefig(outdir / "kmeans_silhouette.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics["k"], metrics["inertia"], marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Within-cluster sum of squares (inertia)")
    ax.set_title("KMeans elbow diagnostic")
    ax.set_xticks(metrics["k"])
    fig.tight_layout()
    fig.savefig(outdir / "kmeans_elbow.png", dpi=180)
    plt.close(fig)

    return selected_k, selected_model, selected_labels, metrics


def cluster_profiles(
    complete: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    outdir: Path,
):
    clustered = complete.copy()
    clustered["Cluster"] = labels

    for i in range(scores.shape[1]):
        clustered[f"PC{i+1}"] = scores[:, i]

    clustered.to_csv(outdir / "cluster_assignments.csv", index=False)

    profile = (
        clustered.groupby("Cluster")[CORE_FEATURES]
        .agg(["count", "mean", "median", "std"])
    )
    profile.to_csv(outdir / "cluster_profiles.csv")

    mcas_summary = (
        clustered.groupby("Cluster")[MCAS_OUTCOMES]
        .agg(["count", "mean", "median", "std"])
    )
    mcas_summary.to_csv(outdir / "cluster_mcas_summary.csv")

    return clustered, profile, mcas_summary


def hierarchical_check(
    X_scaled: np.ndarray,
    kmeans_labels: np.ndarray,
    selected_k: int,
    outdir: Path,
):
    ward = AgglomerativeClustering(
        n_clusters=selected_k,
        linkage="ward",
    )
    ward_labels = ward.fit_predict(X_scaled)

    ari = adjusted_rand_score(kmeans_labels, ward_labels)

    result = {
        "selected_k": selected_k,
        "kmeans_vs_ward_adjusted_rand_index": float(ari),
        "interpretation_note": (
            "ARI measures agreement between KMeans and Ward hierarchical "
            "clustering after accounting for chance; 1 indicates identical "
            "partitions, values near 0 indicate little agreement."
        ),
    }

    with open(outdir / "hierarchical_comparison.json", "w") as f:
        json.dump(result, f, indent=2)

    return ward_labels, ari


def make_pca_plot(
    clustered: pd.DataFrame,
    outdir: Path,
):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        clustered["PC1"],
        clustered["PC2"],
        c=clustered["Cluster"],
        alpha=0.75,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Massachusetts ZIP/ZCTA clusters in PCA space")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(outdir / "pca_clusters.png", dpi=180)
    plt.close(fig)


def make_cluster_profile_plot(
    clustered: pd.DataFrame,
    outdir: Path,
):
    # Plot standardized cluster means so variables with different scales
    # can be compared in one figure.
    temp = clustered.copy()
    scaled = StandardScaler().fit_transform(temp[CORE_FEATURES])
    scaled_df = pd.DataFrame(
        scaled,
        columns=CORE_FEATURES,
        index=temp.index,
    )
    scaled_df["Cluster"] = temp["Cluster"].values

    means = scaled_df.groupby("Cluster")[CORE_FEATURES].mean().T

    fig, ax = plt.subplots(figsize=(10, 6))
    means.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean standardized value (z-score)")
    ax.set_title("Cluster profiles: socioeconomic and digital-access features")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(outdir / "cluster_profiles_standardized.png", dpi=180)
    plt.close(fig)


def make_mcas_cluster_plot(
    clustered: pd.DataFrame,
    outdir: Path,
):
    cols = [
        "MCAS_Math_M_PLUS_E_PCT_ZIP",
        "MCAS_ELA_M_PLUS_E_PCT_ZIP",
    ]
    means = clustered.groupby("Cluster")[cols].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    means.plot(kind="bar", ax=ax)
    ax.set_ylabel("Students meeting/exceeding expectations (%)")
    ax.set_title("MCAS performance by unsupervised cluster")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(outdir / "cluster_mcas_means.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="acs_2024_outputs/mcas_acs_2024_merged.csv",
        help="Merged MCAS + ACS CSV",
    )
    parser.add_argument(
        "--output",
        default="statewide_analysis_2024",
        help="Output directory",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, dtype={"ZIPCODE": str})
    df["ZIPCODE"] = df["ZIPCODE"].astype("string").str.zfill(5)

    validate_columns(df)

    # Keep the full merged file and explicitly flag rows eligible for the
    # six-feature primary unsupervised analysis. Do not overwrite the source.
    df["Primary_Analysis_Eligible"] = df[CORE_FEATURES].notna().all(axis=1)

    df.to_csv(outdir / "analysis_dataset_all.csv", index=False)

    complete = df.loc[df["Primary_Analysis_Eligible"]].copy()
    complete.to_csv(
        outdir / "analysis_dataset_complete_cases.csv",
        index=False,
    )

    save_basic_diagnostics(complete, outdir)

    X_scaled, pca, scores, loadings, explained = run_pca(
        complete,
        outdir,
    )

    selected_k, kmeans_model, labels, metrics = select_kmeans(
        X_scaled,
        outdir,
    )

    clustered, profile, mcas_summary = cluster_profiles(
        complete,
        labels,
        scores,
        outdir,
    )

    ward_labels, ari = hierarchical_check(
        X_scaled,
        labels,
        selected_k,
        outdir,
    )
    clustered["Ward_Cluster"] = ward_labels
    clustered.to_csv(outdir / "cluster_assignments.csv", index=False)

    make_pca_plot(clustered, outdir)
    make_cluster_profile_plot(clustered, outdir)
    make_mcas_cluster_plot(clustered, outdir)

    exclusions = df.loc[
        ~df["Primary_Analysis_Eligible"],
        ["ZIPCODE"] + CORE_FEATURES
    ]
    exclusions.to_csv(
        outdir / "primary_analysis_exclusions.csv",
        index=False,
    )

    summary = {
        "input_rows": int(len(df)),
        "primary_analysis_rows": int(len(complete)),
        "excluded_rows_for_missing_core_features": int(
            len(df) - len(complete)
        ),
        "excluded_zipcodes": exclusions["ZIPCODE"].tolist(),
        "core_features": CORE_FEATURES,
        "education_outcomes_not_used_for_clustering": MCAS_OUTCOMES,
        "selected_k_by_highest_silhouette": selected_k,
        "selected_k_silhouette": float(
            metrics.loc[
                metrics["k"] == selected_k,
                "silhouette"
            ].iloc[0]
        ),
        "pca_pc1_explained_variance": float(
            explained.loc[0, "Explained_Variance_Ratio"]
        ),
        "pca_pc2_explained_variance": float(
            explained.loc[1, "Explained_Variance_Ratio"]
        ),
        "pca_first_two_cumulative_variance": float(
            explained.loc[1, "Cumulative_Explained_Variance"]
        ),
        "kmeans_vs_ward_adjusted_rand_index": float(ari),
        "notes": [
            "Primary clustering uses only socioeconomic and digital-access variables.",
            "MCAS outcomes are summarized after clustering and do not determine cluster membership.",
            "Rates are standardized before PCA and clustering; raw values are retained for interpretation.",
            "Rows missing any core feature are excluded rather than imputed.",
        ],
    }

    with open(outdir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSTATEWIDE 2024 ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Input rows: {len(df):,}")
    print(f"Primary complete-case rows: {len(complete):,}")
    print(
        "Excluded ZIPs:",
        ", ".join(exclusions["ZIPCODE"].tolist())
        if len(exclusions)
        else "None",
    )
    print(f"Selected KMeans k (highest silhouette): {selected_k}")
    print(
        "Silhouette:",
        f"{summary['selected_k_silhouette']:.4f}",
    )
    print(
        "PC1 variance:",
        f"{100 * summary['pca_pc1_explained_variance']:.2f}%",
    )
    print(
        "PC2 variance:",
        f"{100 * summary['pca_pc2_explained_variance']:.2f}%",
    )
    print(
        "First two PCs cumulative:",
        f"{100 * summary['pca_first_two_cumulative_variance']:.2f}%",
    )
    print(
        "KMeans vs Ward ARI:",
        f"{summary['kmeans_vs_ward_adjusted_rand_index']:.4f}",
    )
    print(f"\nOutputs saved to: {outdir}/")


if __name__ == "__main__":
    main()
