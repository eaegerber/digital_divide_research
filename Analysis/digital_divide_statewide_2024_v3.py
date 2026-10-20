#!/usr/bin/env python3
"""
Expanded statewide Massachusetts digital-divide unsupervised analysis (2024).

This version preserves the original six-feature baseline and adds the
Gerber-requested sensitivity analyses:

1) Baseline PCA/clustering:
   - Broadband_Rate
   - No_Computer_Rate
   - Computer_No_Internet_Rate
   - ACS_Median_Family_Income
   - Low_Income_Share_lt25k
   - High_Income_Share_150k_plus

2) Detailed-income sensitivity PCA:
   - baseline digital features
   - median family income
   - all 16 B19101 income-bin shares

3) Stepwise all-domain PCA sequence to isolate what changes the latent structure:
   C) add Math/ELA attainment
   D) add Math/ELA participation
   D2) hold D's features fixed but restrict to the Science-complete sample
   E) add Science attainment
   F) add Science participation

4) Fully expanded all-domain sensitivity PCA:
   - selected detailed digital-access variables
   - median family income + all 16 income shares
   - MCAS Math / ELA / Science Meeting-or-Exceeding %
   - MCAS Math / ELA / Science participation %

For each available specification the script:
- standardizes features
- runs PCA and saves explained variance / loadings / scores
- creates a PC1/PC2 biplot
- evaluates KMeans k=2..8
- selects the highest-silhouette k
- runs 100 single-initialization KMeans seeds as an initialization-stability test
- runs bootstrap KMeans stability
- compares KMeans, Ward hierarchical clustering, and Gaussian Mixture clustering
- saves cluster profiles and assignment comparisons
- optionally adds ZCTA latitude/longitude diagnostics and geographic plots

Important:
- Different specifications may have different complete-case sample sizes.
- The all-16-income-share specifications are intentionally sensitivity analyses:
  the shares are compositional and sum to ~1, so they contain exact/redundant
  information and give the income domain substantial weight in PCA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


BASELINE_FEATURES = [
    "Broadband_Rate",
    "No_Computer_Rate",
    "Computer_No_Internet_Rate",
    "ACS_Median_Family_Income",
    "Low_Income_Share_lt25k",
    "High_Income_Share_150k_plus",
]

INCOME_SHARES_16 = [
    "Income_Share_lt10k",
    "Income_Share_10k_14999",
    "Income_Share_15k_19999",
    "Income_Share_20k_24999",
    "Income_Share_25k_29999",
    "Income_Share_30k_34999",
    "Income_Share_35k_39999",
    "Income_Share_40k_44999",
    "Income_Share_45k_49999",
    "Income_Share_50k_59999",
    "Income_Share_60k_74999",
    "Income_Share_75k_99999",
    "Income_Share_100k_124999",
    "Income_Share_125k_149999",
    "Income_Share_150k_199999",
    "Income_Share_200k_plus",
]

DETAILED_DIGITAL_FEATURES = [
    "Broadband_Rate",
    "No_Computer_Rate",
    "Computer_No_Internet_Rate",
    "Internet_Subscription_Rate",
    "Cellular_Only_Rate",
    "Cable_Fiber_DSL_Rate",
    "Satellite_Rate",
    "No_Internet_Access_Rate",
]

EDUCATION_M_PLUS_E = [
    "MCAS_Math_M_PLUS_E_PCT_ZIP",
    "MCAS_ELA_M_PLUS_E_PCT_ZIP",
    "MCAS_Science_M_PLUS_E_PCT_ZIP",
]

PARTICIPATION_FEATURES = [
    "MCAS_Math_STU_PART_PCT_WT",
    "MCAS_ELA_STU_PART_PCT_WT",
    "MCAS_Science_STU_PART_PCT_WT",
]

SUPPORTING_EDUCATION = [
    "MCAS_Math_AVG_SCALED_SCORE_WT",
    "MCAS_ELA_AVG_SCALED_SCORE_WT",
    "MCAS_Science_AVG_SCALED_SCORE_WT",
    "MCAS_Math_AVG_SGP_WT",
    "MCAS_ELA_AVG_SGP_WT",
    "MCAS_Science_AVG_SGP_WT",
]

DOMAIN_MAP = {}
for c in BASELINE_FEATURES:
    if "Income" in c or "income" in c:
        DOMAIN_MAP[c] = "Income"
    elif c.startswith("ACS_Median"):
        DOMAIN_MAP[c] = "Income"
    else:
        DOMAIN_MAP[c] = "Digital access"
for c in INCOME_SHARES_16:
    DOMAIN_MAP[c] = "Income"
for c in DETAILED_DIGITAL_FEATURES:
    DOMAIN_MAP[c] = "Digital access"
for c in EDUCATION_M_PLUS_E:
    DOMAIN_MAP[c] = "Education attainment"
for c in PARTICIPATION_FEATURES:
    DOMAIN_MAP[c] = "Participation"
for c in SUPPORTING_EDUCATION:
    DOMAIN_MAP[c] = "Education supporting"


def zfill5(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.extract(r"(\d+)", expand=False)
        .str.zfill(5)
    )


def unique_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def available_or_missing(df: pd.DataFrame, features: list[str]):
    available = [c for c in features if c in df.columns]
    missing = [c for c in features if c not in df.columns]
    return available, missing


def specifications(df: pd.DataFrame):
    math_ela_attainment = [
        "MCAS_Math_M_PLUS_E_PCT_ZIP",
        "MCAS_ELA_M_PLUS_E_PCT_ZIP",
    ]
    math_ela_participation = [
        "MCAS_Math_STU_PART_PCT_WT",
        "MCAS_ELA_STU_PART_PCT_WT",
    ]
    science_attainment = ["MCAS_Science_M_PLUS_E_PCT_ZIP"]
    science_participation = ["MCAS_Science_STU_PART_PCT_WT"]

    specs = {
        "A_baseline": {
            "features": BASELINE_FEATURES,
            "filter_features": [],
        },
        "B_detailed_income": {
            "features": (
                [
                    "Broadband_Rate",
                    "No_Computer_Rate",
                    "Computer_No_Internet_Rate",
                    "ACS_Median_Family_Income",
                ]
                + INCOME_SHARES_16
            ),
            "filter_features": [],
        },
        "C_math_ela_attainment": {
            "features": BASELINE_FEATURES + math_ela_attainment,
            "filter_features": [],
        },
        "D_math_ela_participation": {
            "features": (
                BASELINE_FEATURES
                + math_ela_attainment
                + math_ela_participation
            ),
            "filter_features": [],
        },
        # Same PCA variables as D, but restrict to rows that also have Science.
        # This isolates the effect of the 411 -> ~394 sample restriction.
        "D2_math_ela_participation_science_complete": {
            "features": (
                BASELINE_FEATURES
                + math_ela_attainment
                + math_ela_participation
            ),
            "filter_features": science_attainment + science_participation,
        },
        # Starting from D2's same sample, add Science attainment only.
        "E_add_science_attainment": {
            "features": (
                BASELINE_FEATURES
                + math_ela_attainment
                + math_ela_participation
                + science_attainment
            ),
            "filter_features": science_participation,
        },
        # Then add Science participation. This reproduces the old C feature set.
        "F_add_science_participation": {
            "features": (
                BASELINE_FEATURES
                + EDUCATION_M_PLUS_E
                + PARTICIPATION_FEATURES
            ),
            "filter_features": [],
        },
        "G_expanded_all_domain": {
            "features": (
                DETAILED_DIGITAL_FEATURES
                + ["ACS_Median_Family_Income"]
                + INCOME_SHARES_16
                + EDUCATION_M_PLUS_E
                + PARTICIPATION_FEATURES
            ),
            "filter_features": [],
        },
    }

    result = {}
    for name, spec in specs.items():
        requested = unique_preserve_order(spec["features"])
        filter_features = unique_preserve_order(spec.get("filter_features", []))
        needed = unique_preserve_order(requested + filter_features)
        available, missing = available_or_missing(df, needed)

        result[name] = {
            "requested": requested,
            "filter_features": filter_features,
            "available": available,
            "missing": missing,
            "runnable": len(missing) == 0,
        }
    return result


def complete_cases(df: pd.DataFrame, features: list[str]):
    mask = df[features].notna().all(axis=1)
    return df.loc[mask].copy(), df.loc[~mask, ["ZIPCODE"] + features].copy()


def run_pca(df: pd.DataFrame, features: list[str], outdir: Path):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    pca = PCA()
    scores = pca.fit_transform(X)

    pc_names = [f"PC{i+1}" for i in range(len(features))]
    explained = pd.DataFrame(
        {
            "PC": pc_names,
            "Explained_Variance_Ratio": pca.explained_variance_ratio_,
            "Cumulative_Explained_Variance": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    )
    explained.to_csv(outdir / "pca_explained_variance.csv", index=False)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=pc_names,
    )
    loadings.index.name = "Feature"
    loadings.to_csv(outdir / "pca_loadings.csv")

    # Squared loadings sum to 1 within each component and are convenient for
    # summarizing which conceptual domain contributes to each PC direction.
    contribution = loadings.pow(2)
    contribution["Domain"] = [
        DOMAIN_MAP.get(c, "Other") for c in contribution.index
    ]
    domain_contrib = (
        contribution.groupby("Domain")[pc_names]
        .sum()
        .reset_index()
    )
    domain_contrib.to_csv(outdir / "pca_domain_contributions.csv", index=False)

    score_df = df[["ZIPCODE"]].copy()
    for i, pc in enumerate(pc_names):
        score_df[pc] = scores[:, i]
    score_df.to_csv(outdir / "pca_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(features) + 1)
    ax.bar(x, pca.explained_variance_ratio_)
    ax.plot(x, np.cumsum(pca.explained_variance_ratio_), marker="o")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA explained variance")
    fig.tight_layout()
    fig.savefig(outdir / "pca_explained_variance.png", dpi=180)
    plt.close(fig)

    return X, scaler, pca, scores, loadings, explained, domain_contrib


def canonical_partition_hash(labels: np.ndarray) -> str:
    """Label-invariant hash based on pairwise co-clustering."""
    labels = np.asarray(labels)
    same = labels[:, None] == labels[None, :]
    upper = same[np.triu_indices(len(labels), k=1)]
    packed = np.packbits(upper.astype(np.uint8))
    return hashlib.sha1(packed.tobytes()).hexdigest()


def kmeans_model_selection(X: np.ndarray, outdir: Path):
    rows = []
    models = {}

    max_k = min(8, len(X) - 1)
    for k in range(2, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=100,
        )
        labels = model.fit_predict(X)
        models[k] = (model, labels)

        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(X, labels),
                "calinski_harabasz": calinski_harabasz_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
                "inertia": model.inertia_,
            }
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir / "kmeans_model_selection.csv", index=False)

    selected_k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
    selected_model, selected_labels = models[selected_k]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics["k"], metrics["silhouette"], marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("KMeans cluster separation")
    fig.tight_layout()
    fig.savefig(outdir / "kmeans_silhouette.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics["k"], metrics["inertia"], marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("KMeans elbow diagnostic")
    fig.tight_layout()
    fig.savefig(outdir / "kmeans_elbow.png", dpi=180)
    plt.close(fig)

    return selected_k, selected_model, selected_labels, metrics


def initialization_stability(
    X: np.ndarray,
    selected_k: int,
    reference_labels: np.ndarray,
    outdir: Path,
    n_runs: int,
):
    rows = []
    hashes = []

    for seed in range(n_runs):
        model = KMeans(
            n_clusters=selected_k,
            random_state=seed,
            n_init=1,
        )
        labels = model.fit_predict(X)
        hashes.append(canonical_partition_hash(labels))
        rows.append(
            {
                "seed": seed,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X, labels),
                "ARI_vs_reference": adjusted_rand_score(
                    reference_labels, labels
                ),
                "partition_hash": hashes[-1],
                "cluster_sizes": json.dumps(
                    pd.Series(labels)
                    .value_counts()
                    .sort_index()
                    .to_dict()
                ),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(outdir / "kmeans_100_seed_stability.csv", index=False)

    summary = {
        "n_runs": n_runs,
        "mean_ARI_vs_reference": float(result["ARI_vs_reference"].mean()),
        "min_ARI_vs_reference": float(result["ARI_vs_reference"].min()),
        "pct_runs_ARI_1": float(
            100 * np.mean(np.isclose(result["ARI_vs_reference"], 1.0))
        ),
        "distinct_label_invariant_partitions": int(result["partition_hash"].nunique()),
    }
    (outdir / "kmeans_100_seed_stability_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return result, summary


def bootstrap_stability(
    X: np.ndarray,
    selected_k: int,
    reference_labels: np.ndarray,
    outdir: Path,
    n_reps: int,
    random_state: int = 2024,
):
    rng = np.random.default_rng(random_state)
    rows = []

    for rep in range(n_reps):
        idx = rng.integers(0, len(X), size=len(X))
        X_boot = X[idx]

        model = KMeans(
            n_clusters=selected_k,
            random_state=rep,
            n_init=25,
        )
        model.fit(X_boot)

        # Evaluate the bootstrap-fitted centroids on the original full sample.
        full_labels = model.predict(X)
        rows.append(
            {
                "replicate": rep,
                "ARI_vs_reference": adjusted_rand_score(
                    reference_labels, full_labels
                ),
                "silhouette_on_full_sample": silhouette_score(
                    X, full_labels
                ),
                "cluster_sizes_on_full_sample": json.dumps(
                    pd.Series(full_labels)
                    .value_counts()
                    .sort_index()
                    .to_dict()
                ),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(outdir / "kmeans_bootstrap_stability.csv", index=False)

    summary = {
        "n_replicates": n_reps,
        "mean_ARI_vs_reference": float(result["ARI_vs_reference"].mean()),
        "median_ARI_vs_reference": float(result["ARI_vs_reference"].median()),
        "min_ARI_vs_reference": float(result["ARI_vs_reference"].min()),
        "pct_replicates_ARI_ge_0_8": float(
            100 * np.mean(result["ARI_vs_reference"] >= 0.8)
        ),
    }
    (outdir / "kmeans_bootstrap_stability_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return result, summary


def compare_cluster_methods(
    X: np.ndarray,
    selected_k: int,
    kmeans_labels: np.ndarray,
    outdir: Path,
):
    ward = AgglomerativeClustering(
        n_clusters=selected_k,
        linkage="ward",
    )
    ward_labels = ward.fit_predict(X)

    gmm = GaussianMixture(
        n_components=selected_k,
        covariance_type="full",
        random_state=42,
        n_init=25,
    )
    gmm_labels = gmm.fit_predict(X)

    labels_by_method = {
        "KMeans": kmeans_labels,
        "Ward": ward_labels,
        "GMM": gmm_labels,
    }

    metrics_rows = []
    for method, labels in labels_by_method.items():
        metrics_rows.append(
            {
                "method": method,
                "k": selected_k,
                "silhouette": silhouette_score(X, labels),
                "calinski_harabasz": calinski_harabasz_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
                "cluster_sizes": json.dumps(
                    pd.Series(labels)
                    .value_counts()
                    .sort_index()
                    .to_dict()
                ),
            }
        )

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(outdir / "cluster_method_metrics.csv", index=False)

    methods = list(labels_by_method)
    ari_rows = []
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            ari_rows.append(
                {
                    "method_a": a,
                    "method_b": b,
                    "adjusted_rand_index": adjusted_rand_score(
                        labels_by_method[a],
                        labels_by_method[b],
                    ),
                }
            )
    ari = pd.DataFrame(ari_rows)
    ari.to_csv(outdir / "cluster_method_ari.csv", index=False)

    return ward_labels, gmm_labels, metrics, ari


def save_cluster_outputs(
    df: pd.DataFrame,
    features: list[str],
    scores: np.ndarray,
    kmeans_labels: np.ndarray,
    ward_labels: np.ndarray,
    gmm_labels: np.ndarray,
    outdir: Path,
):
    out = df.copy()
    out["KMeans_Cluster"] = kmeans_labels
    out["Ward_Cluster"] = ward_labels
    out["GMM_Cluster"] = gmm_labels
    for i in range(scores.shape[1]):
        out[f"PC{i+1}"] = scores[:, i]

    out["KMeans_Ward_Agree"] = (
        canonicalize_to_reference(ward_labels, kmeans_labels)
        == kmeans_labels
    )
    out["KMeans_GMM_Agree"] = (
        canonicalize_to_reference(gmm_labels, kmeans_labels)
        == kmeans_labels
    )
    out.to_csv(outdir / "cluster_assignments.csv", index=False)

    # Raw feature profiles for each method.
    for method_col in ["KMeans_Cluster", "Ward_Cluster", "GMM_Cluster"]:
        profile = (
            out.groupby(method_col)[features]
            .agg(["count", "mean", "median", "std"])
        )
        profile.to_csv(
            outdir / f"{method_col.lower()}_profiles.csv"
        )

    return out


def canonicalize_to_reference(labels: np.ndarray, reference: np.ndarray):
    """
    Relabel clusters to maximize simple agreement with reference labels.
    Uses a small exhaustive permutation for k <= 8.
    """
    import itertools

    labels = np.asarray(labels)
    reference = np.asarray(reference)
    source_vals = sorted(np.unique(labels))
    ref_vals = sorted(np.unique(reference))

    if len(source_vals) != len(ref_vals):
        return labels.copy()

    best = labels.copy()
    best_match = -1

    for perm in itertools.permutations(ref_vals):
        mapping = dict(zip(source_vals, perm))
        relabeled = np.array([mapping[x] for x in labels])
        match = np.mean(relabeled == reference)
        if match > best_match:
            best_match = match
            best = relabeled

    return best


def make_pca_cluster_plot(
    scores: np.ndarray,
    labels: np.ndarray,
    method_name: str,
    outpath: Path,
):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        scores[:, 0],
        scores[:, 1],
        c=labels,
        alpha=0.75,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{method_name} clusters in PCA space")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def make_biplot(
    scores: np.ndarray,
    loadings: pd.DataFrame,
    labels: np.ndarray,
    outpath: Path,
    max_arrows: int = 15,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        scores[:, 0],
        scores[:, 1],
        c=labels,
        alpha=0.55,
    )

    pc12 = loadings[["PC1", "PC2"]].copy()
    pc12["vector_norm"] = np.sqrt(pc12["PC1"] ** 2 + pc12["PC2"] ** 2)
    selected = pc12.sort_values("vector_norm", ascending=False).head(max_arrows)

    x_span = np.nanpercentile(scores[:, 0], 95) - np.nanpercentile(scores[:, 0], 5)
    y_span = np.nanpercentile(scores[:, 1], 95) - np.nanpercentile(scores[:, 1], 5)
    scale = 0.35 * min(x_span, y_span)

    for feature, row in selected.iterrows():
        x = row["PC1"] * scale
        y = row["PC2"] * scale
        ax.arrow(
            0, 0, x, y,
            head_width=max(scale * 0.015, 0.02),
            length_includes_head=True,
            alpha=0.8,
        )
        ax.text(x * 1.06, y * 1.06, feature, fontsize=8)

    ax.axhline(0, linewidth=0.7)
    ax.axvline(0, linewidth=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA biplot: observations + strongest feature vectors")
    legend = ax.legend(*scatter.legend_elements(), title="KMeans cluster")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def merge_geography(df: pd.DataFrame, geo_path: Path | None):
    if geo_path is None:
        return df, False

    geo = pd.read_csv(geo_path, dtype={"ZIPCODE": str})
    geo["ZIPCODE"] = zfill5(geo["ZIPCODE"])

    required = ["ZIPCODE", "Latitude", "Longitude"]
    missing = [c for c in required if c not in geo.columns]
    if missing:
        raise ValueError(
            f"Geography file missing columns: {missing}"
        )

    # Avoid duplicate coordinate columns if rerunning.
    base = df.drop(
        columns=[c for c in ["Latitude", "Longitude"] if c in df.columns],
        errors="ignore",
    )
    merged = base.merge(
        geo[required].drop_duplicates("ZIPCODE"),
        on="ZIPCODE",
        how="left",
        validate="one_to_one",
    )
    return merged, True


def geography_plots(clustered: pd.DataFrame, outdir: Path):
    if not {"Latitude", "Longitude"}.issubset(clustered.columns):
        return
    valid = clustered.dropna(subset=["Latitude", "Longitude"]).copy()
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        valid["Longitude"],
        valid["Latitude"],
        c=valid["KMeans_Cluster"],
        alpha=0.8,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("KMeans clusters by ZCTA representative point")
    legend = ax.legend(*sc.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(outdir / "map_points_kmeans.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        valid["Longitude"],
        valid["Latitude"],
        c=valid["PC1"],
        alpha=0.8,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("PC1 across Massachusetts ZCTA representative points")
    fig.colorbar(sc, ax=ax, label="PC1")
    fig.tight_layout()
    fig.savefig(outdir / "map_points_pc1.png", dpi=180)
    plt.close(fig)

    # Highlight KMeans/Ward disagreements after optimal label matching.
    ward_aligned = canonicalize_to_reference(
        valid["Ward_Cluster"].to_numpy(),
        valid["KMeans_Cluster"].to_numpy(),
    )
    disagree = ward_aligned != valid["KMeans_Cluster"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        valid["Longitude"],
        valid["Latitude"],
        c=disagree.astype(int),
        alpha=0.8,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("KMeans vs Ward assignment disagreement")
    fig.colorbar(sc, ax=ax, label="Disagreement indicator")
    fig.tight_layout()
    fig.savefig(outdir / "map_points_kmeans_ward_disagreement.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        valid["PC1"],
        valid["PC2"],
        c=valid["Longitude"],
        alpha=0.75,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA space colored by longitude (east/west)")
    fig.colorbar(sc, ax=ax, label="Longitude")
    fig.tight_layout()
    fig.savefig(outdir / "pca_colored_by_longitude.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        valid["PC1"],
        valid["PC2"],
        c=valid["Latitude"],
        alpha=0.75,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA space colored by latitude (north/south)")
    fig.colorbar(sc, ax=ax, label="Latitude")
    fig.tight_layout()
    fig.savefig(outdir / "pca_colored_by_latitude.png", dpi=180)
    plt.close(fig)


def run_specification(
    name: str,
    base_df: pd.DataFrame,
    features: list[str],
    spec_dir: Path,
    n_seed_runs: int,
    bootstrap_reps: int,
    filter_features: list[str] | None = None,
):
    spec_dir.mkdir(parents=True, exist_ok=True)

    filter_features = filter_features or []
    complete_case_requirements = unique_preserve_order(features + filter_features)
    complete, exclusions = complete_cases(base_df, complete_case_requirements)
    complete.to_csv(spec_dir / "analysis_complete_cases.csv", index=False)
    exclusions.to_csv(spec_dir / "analysis_exclusions.csv", index=False)

    if len(complete) < 20:
        raise ValueError(
            f"{name}: only {len(complete)} complete rows; too few to analyze."
        )

    complete[features].describe().T.to_csv(
        spec_dir / "descriptive_statistics.csv"
    )
    complete[features].corr().to_csv(
        spec_dir / "feature_correlations.csv"
    )

    X, scaler, pca, scores, loadings, explained, domain_contrib = run_pca(
        complete, features, spec_dir
    )

    selected_k, km_model, km_labels, selection = kmeans_model_selection(
        X, spec_dir
    )

    seed_runs, seed_summary = initialization_stability(
        X,
        selected_k,
        km_labels,
        spec_dir,
        n_runs=n_seed_runs,
    )

    boot_runs, boot_summary = bootstrap_stability(
        X,
        selected_k,
        km_labels,
        spec_dir,
        n_reps=bootstrap_reps,
    )

    ward_labels, gmm_labels, method_metrics, ari = compare_cluster_methods(
        X,
        selected_k,
        km_labels,
        spec_dir,
    )

    clustered = save_cluster_outputs(
        complete,
        features,
        scores,
        km_labels,
        ward_labels,
        gmm_labels,
        spec_dir,
    )

    make_pca_cluster_plot(
        scores, km_labels, "KMeans", spec_dir / "pca_kmeans.png"
    )
    make_pca_cluster_plot(
        scores, ward_labels, "Ward", spec_dir / "pca_ward.png"
    )
    make_pca_cluster_plot(
        scores, gmm_labels, "Gaussian Mixture", spec_dir / "pca_gmm.png"
    )
    make_biplot(
        scores,
        loadings,
        km_labels,
        spec_dir / "pca_biplot.png",
        max_arrows=(len(features) if len(features) <= 10 else 15),
    )

    geography_plots(clustered, spec_dir)

    summary = {
        "specification": name,
        "features": features,
        "filter_only_complete_case_features": filter_features,
        "input_rows": int(len(base_df)),
        "complete_case_rows": int(len(complete)),
        "excluded_rows": int(len(base_df) - len(complete)),
        "selected_k_by_silhouette": int(selected_k),
        "selected_k_silhouette": float(
            selection.loc[
                selection["k"] == selected_k, "silhouette"
            ].iloc[0]
        ),
        "pc1_explained_variance": float(
            explained.loc[0, "Explained_Variance_Ratio"]
        ),
        "pc2_explained_variance": float(
            explained.loc[1, "Explained_Variance_Ratio"]
        ),
        "first_two_cumulative_variance": float(
            explained.loc[1, "Cumulative_Explained_Variance"]
        ),
        "initialization_stability": seed_summary,
        "bootstrap_stability": boot_summary,
        "cluster_method_ari": ari.to_dict(orient="records"),
        "notes": [
            "PCA/clustering uses standardized complete cases for this specification.",
            "Some specifications use filter-only variables to hold the analytic sample constant without including those variables in PCA/clustering.",
            "No missing feature values are imputed.",
            "Detailed-income specifications include all 16 income shares, which are compositional and redundant by construction.",
        ],
    }
    (spec_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="acs_2024_outputs_expanded/mcas_acs_2024_expanded.csv",
    )
    p.add_argument(
        "--geography",
        default=None,
        help="Optional analysis_zcta_coordinates.csv from build_zcta_geography_2024.py",
    )
    p.add_argument(
        "--output",
        default="statewide_analysis_2024_v3",
    )
    p.add_argument(
        "--seed-runs",
        type=int,
        default=100,
    )
    p.add_argument(
        "--bootstrap-reps",
        type=int,
        default=100,
    )
    args = p.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, dtype={"ZIPCODE": str})
    df["ZIPCODE"] = zfill5(df["ZIPCODE"])

    geo_path = Path(args.geography) if args.geography else None
    df, has_geography = merge_geography(df, geo_path)
    df.to_csv(outdir / "analysis_master_input.csv", index=False)

    specs = specifications(df)
    (outdir / "specification_availability.json").write_text(
        json.dumps(specs, indent=2)
    )

    all_summaries = {}
    for name, info in specs.items():
        if not info["runnable"]:
            print(
                f"SKIPPING {name}: missing columns: "
                + ", ".join(info["missing"])
            )
            continue

        print("\n" + "=" * 80)
        print(f"RUNNING {name}")
        print("=" * 80)

        summary = run_specification(
            name,
            df,
            info["requested"],
            outdir / name,
            n_seed_runs=args.seed_runs,
            bootstrap_reps=args.bootstrap_reps,
            filter_features=info.get("filter_features", []),
        )
        all_summaries[name] = summary

        print(
            f"{name}: n={summary['complete_case_rows']}, "
            f"k={summary['selected_k_by_silhouette']}, "
            f"silhouette={summary['selected_k_silhouette']:.3f}, "
            f"PC1={100*summary['pc1_explained_variance']:.1f}%, "
            f"PC1+PC2={100*summary['first_two_cumulative_variance']:.1f}%"
        )

    project_summary = {
        "input": str(args.input),
        "geography_included": has_geography,
        "specifications": all_summaries,
        "skipped_specifications": {
            name: info["missing"]
            for name, info in specs.items()
            if not info["runnable"]
        },
    }
    (outdir / "project_summary.json").write_text(
        json.dumps(project_summary, indent=2)
    )

    print("\nEXPANDED UNSUPERVISED ANALYSIS COMPLETE")
    print(f"Outputs: {outdir}/")


if __name__ == "__main__":
    main()
