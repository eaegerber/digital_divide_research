from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


INPUT = (
    "statewide_analysis_2024_v3/"
    "C_math_ela_attainment/"
    "cluster_assignments.csv"
)

OUTPUT = (
    "final_results_2024/final_figures/"
    "01_primary_pca_biplot_FINAL.png"
)

FEATURES = [
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

CLUSTER_NAMES = {
    0: "Higher SES/access/attainment profile",
    1: "Lower SES/access/attainment profile",
}

# Fine-tuned offsets for label placement.
# Values are x/y offsets in display points from the arrow endpoint.
LABEL_OFFSETS = {
    "Computer_No_Internet_Rate": (-14, 36),
    "No_Computer_Rate": (-14, 20),
    "Low_Income_Share_lt25k": (-14, -2),

    "MCAS_ELA_M_PLUS_E_PCT_ZIP": (15, 40),
    "MCAS_Math_M_PLUS_E_PCT_ZIP": (15, 24),
    "High_Income_Share_150k_plus": (15, 7),
    "ACS_Median_Family_Income": (15, -10),

    "Broadband_Rate": (12, -18),
}


def main():
    df = pd.read_csv(INPUT, dtype={"ZIPCODE": str})
    df["ZIPCODE"] = df["ZIPCODE"].str.zfill(5)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    if df[FEATURES].isna().any().any():
        raise ValueError("Primary PCA features contain missing values.")

    # Recompute the exact standardized PCA used by the primary specification.
    X = StandardScaler().fit_transform(df[FEATURES])

    pca = PCA()
    scores = pca.fit_transform(X)

    # PCA signs are mathematically arbitrary.
    # Align the newly computed PCs with the PC1/PC2 orientation already saved
    # in cluster_assignments.csv so all final figures use the same direction.
    for i, saved_pc in enumerate(["PC1", "PC2"]):
        corr = np.corrcoef(scores[:, i], df[saved_pc])[0, 1]

        if corr < 0:
            scores[:, i] *= -1
            pca.components_[i, :] *= -1

    loadings = pca.components_.T[:, :2]

    fig, ax = plt.subplots(figsize=(13, 8.5))

    # Plot observations.
    for cluster_id in sorted(df["KMeans_Cluster"].unique()):
        mask = df["KMeans_Cluster"].to_numpy() == cluster_id

        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            s=42,
            alpha=0.62,
            label=CLUSTER_NAMES[cluster_id],
        )

    # Reference axes.
    ax.axhline(0, linewidth=0.8, alpha=0.7)
    ax.axvline(0, linewidth=0.8, alpha=0.7)

    # Scale loading vectors into observation-score space.
    x_range = (
        np.percentile(scores[:, 0], 95)
        - np.percentile(scores[:, 0], 5)
    )

    y_range = (
        np.percentile(scores[:, 1], 95)
        - np.percentile(scores[:, 1], 5)
    )

    vector_scale = min(x_range, y_range) * 0.67

    for feature, (loading_x, loading_y) in zip(FEATURES, loadings):
        end_x = loading_x * vector_scale
        end_y = loading_y * vector_scale

        # Arrow.
        ax.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1.6,
            ),
        )

        offset_x, offset_y = LABEL_OFFSETS[feature]

        # Label.
        ax.annotate(
            DISPLAY_NAMES[feature],
            xy=(end_x, end_y),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=9.5,
            ha="left" if offset_x >= 0 else "right",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="none",
                alpha=0.78,
            ),
        )

    pc1 = pca.explained_variance_ratio_[0] * 100
    pc2 = pca.explained_variance_ratio_[1] * 100

    ax.set_xlabel(
        f"PC1 — joint SES/access/attainment gradient ({pc1:.1f}%)",
        fontsize=11,
    )

    ax.set_ylabel(
        f"PC2 ({pc2:.1f}%)",
        fontsize=11,
    )

    ax.set_title(
        "Primary PCA: socioeconomic status, digital access, and educational attainment",
        fontsize=14,
    )

    ax.legend(
        title="Primary KMeans segmentation",
        loc="upper right",
        fontsize=9.5,
    )

    ax.margins(x=0.05, y=0.08)

    fig.text(
        0.5,
        0.015,
        "Arrows show standardized feature loadings; points represent school-location ZIP/ZCTA observations.",
        ha="center",
        fontsize=9,
    )

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(
        OUTPUT,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", OUTPUT)
    print(f"PC1 explained variance: {pc1:.1f}%")
    print(f"PC2 explained variance: {pc2:.1f}%")
    print(f"PC1 + PC2: {pc1 + pc2:.1f}%")


if __name__ == "__main__":
    main()
