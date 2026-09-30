"""
Module 5 (Lego Block: Unsupervised Behavioral State Clustering)
Responsible for clustering 4D PCA conversational geometry into 4 behavioral archetypes:
State 0: Deep Dive (Analytical, constructive)
State 1: Inquiry (Questions, clarifications)
State 2: Friction (Hostile, argumentative, agitated)
State 3: Casual Banter (Short, low-effort, humor/noise)
"""

import os
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


INPUT_PCA_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "pca_transformed.csv"
)
OUTPUT_CLUSTERED_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "clustered_comments.csv"
)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")

CANONICAL_LABELS = {
    0: "Deep Dive (Analytical)",
    1: "Inquiry (Clarification)",
    2: "Friction (Hostility/Tension)",
    3: "Casual Banter (Low-Effort/Noise)"
}


def canonicalize_clusters(df: pd.DataFrame, raw_labels: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Deterministically maps raw hierarchical clusters to canonical behavioral archetypes:
    - State 0 (Deep Dive): Cluster with the highest mean PC1 (Discourse Depth)
    - State 2 (Friction): Cluster with the highest mean PC2 (Conversational Tension)
    - State 1 (Inquiry): Cluster with the highest average question marks among remaining
    - State 3 (Casual Banter): Remaining cluster (low depth, baseline tension)
    """
    df = df.copy()
    df["raw_cluster"] = raw_labels

    cluster_stats = df.groupby("raw_cluster").agg({
        "PC1": "mean",
        "PC2": "mean",
        "q_marks": "mean",
        "polarity": "mean",
        "word_count": "mean"
    })

    remaining = set(range(len(centers)))

    # 1. State 0: Deep Dive (highest PC1)
    deep_dive_raw = cluster_stats.loc[list(remaining), "PC1"].idxmax()
    remaining.remove(deep_dive_raw)

    # 2. State 2: Friction / Hostility (highest PC2)
    friction_raw = cluster_stats.loc[list(remaining), "PC2"].idxmax()
    remaining.remove(friction_raw)

    # 3. State 1: Inquiry (highest question marks among remaining)
    inquiry_raw = cluster_stats.loc[list(remaining), "q_marks"].idxmax()
    remaining.remove(inquiry_raw)

    # 4. State 3: Casual Banter / Noise (the final remaining)
    banter_raw = remaining.pop()

    raw_to_canonical = {
        deep_dive_raw: 0,
        inquiry_raw: 1,
        friction_raw: 2,
        banter_raw: 3
    }

    canonical_labels = np.array([raw_to_canonical[r] for r in raw_labels])
    return canonical_labels, raw_to_canonical


def plot_cluster_map(df: pd.DataFrame, output_path: str):
    """
    Plots the 2D Hierarchical cluster map with colored convex boundary hulls,
    styled centroid markers, and quadrant lines, focused on the core 99% data
    density so all 4 clusters are clearly visible and distinguished.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    palette = {
        0: ("#009E73", "#e6f5f0", "State 0: Deep Dive (Analytical)"),
        1: ("#0072B2", "#e6f0fa", "State 1: Inquiry (Clarification)"),
        2: ("#D55E00", "#fdf0e7", "State 2: Friction (Hostility/Tension)"),
        3: ("#7570B3", "#f0eff7", "State 3: Casual Banter (Noise/Banter)")
    }

    # Core viewport bounds: captures 99% of comments without squishing clusters
    xlim = (-4.2, 6.2)
    ylim = (-3.0, 4.2)

    for state, (edge_color, fill_color, label) in palette.items():
        sub = df[df["state"] == state]
        pts = sub[["PC1", "PC2"]].values

        # Filter core points within visual boundary
        core_mask = (pts[:, 0] >= xlim[0]) & (pts[:, 0] <= xlim[1]) & (pts[:, 1] >= ylim[0]) & (pts[:, 1] <= ylim[1])
        core_pts = pts[core_mask]

        # Scatter points with clear visibility
        ax.scatter(core_pts[:, 0], core_pts[:, 1], c=edge_color, label=f"{label} (n={len(sub):,})",
                   alpha=0.35, s=28, edgecolors="none")

        # Robust convex hull around core 92% closest to centroid
        if len(core_pts) >= 6:
            cx, cy = core_pts[:, 0].mean(), core_pts[:, 1].mean()
            dists = np.hypot(core_pts[:, 0] - cx, core_pts[:, 1] - cy)
            hull_pts_sub = core_pts[dists <= np.percentile(dists, 92)]
            try:
                hull = ConvexHull(hull_pts_sub)
                hp = hull_pts_sub[hull.vertices]
                ax.fill(hp[:, 0], hp[:, 1], color=fill_color, alpha=0.55, zorder=2)
                ax.plot(np.append(hp[:, 0], hp[0, 0]), np.append(hp[:, 1], hp[0, 1]),
                        color=edge_color, linestyle="--", linewidth=1.6, alpha=0.85, zorder=3)
            except Exception:
                pass

        # Centroid with glowing halo
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        ax.scatter(cx, cy, c="white", s=260, edgecolors="black", linewidth=2.2, zorder=6)
        ax.scatter(cx, cy, c=edge_color, s=140, edgecolors="none", zorder=7)

    # Centroid annotations with clear non-colliding positions
    annots = {
        0: (45, 25),
        1: (-55, -28),
        2: (0, 35),
        3: (0, -32)
    }
    for state, (edge_color, _, label) in palette.items():
        sub = df[df["state"] == state]
        cx, cy = sub["PC1"].mean(), sub["PC2"].mean()
        ox, oy = annots[state]
        short_name = label.split(": ")[1].split(" (")[0]
        ax.annotate(
            f"Centroid {state}: {short_name}\n({cx:+.2f}, {cy:+.2f})",
            xy=(cx, cy), xytext=(ox, oy), textcoords="offset points",
            ha="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.95, edgecolor=edge_color, lw=1.5),
            arrowprops=dict(arrowstyle="->", color=edge_color, lw=1.3),
            zorder=10
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_title("Hierarchical Behavioral State Space (Ward Linkage, K=4)\nFocused Latent Manifold Projection", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("PC1: Intellectual & Discourse Depth (Word Count + Lexical Richness)", fontsize=11, fontweight="bold")
    ax.set_ylabel("PC2: Conversational Tension & Emotional Valence (Polarity + Subjectivity)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(title="Behavioral Archetype", loc="upper right", frameon=True, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[✓] Hierarchical cluster map saved to: {output_path}")


def plot_hierarchical_dendrogram(X: np.ndarray, output_path: str):
    """
    Renders an Agglomerative Hierarchical Dendrogram (Ward Linkage) with a horizontal cut threshold at K=4.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.random.seed(42)
    sample_indices = np.random.choice(len(X), size=120, replace=False)
    X_sample = X[sample_indices]

    linkage_matrix = linkage(X_sample, method="ward")

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
    dendrogram(
        linkage_matrix,
        ax=ax,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=9,
        show_contracted=True
    )
    cut_dist = (linkage_matrix[-3, 2] + linkage_matrix[-4, 2]) / 2.0
    ax.axhline(cut_dist, color="#d95f02", linestyle="--", linewidth=2, label=f"K=4 Cluster Cut (Threshold Distance={cut_dist:.2f})")
    ax.set_title("Hierarchical Agglomerative Clustering Dendrogram (Ward Linkage)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Cluster Leaf Groups (Sampled Trajectory Comments)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ward Euclidean Merge Distance", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[✓] Hierarchical dendrogram saved to: {output_path}")


def run_clustering_pipeline(
    input_csv: str = INPUT_PCA_CSV,
    output_csv: str = OUTPUT_CLUSTERED_CSV,
    n_clusters: int = 4
) -> pd.DataFrame:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input PCA file not found: {input_csv}. Run Module 4 first.")

    df = pd.read_csv(input_csv)
    X_raw = df[["PC1", "PC2", "PC3", "PC4"]].values

    # Robust Winsorization (1st to 99th percentile clipping) across 4 dimensions
    # Prevents zero-inflated punctuation kurtosis (outliers up to +32 sigma)
    # from dominating squared Euclidean distance in Ward linkage
    p1 = np.percentile(X_raw, 1, axis=0)
    p99 = np.percentile(X_raw, 99, axis=0)
    X = np.clip(X_raw, p1, p99)

    print(f"Fitting Hierarchical Agglomerative Clustering (Ward Linkage, K={n_clusters}) on 4D Winsorized PCA Space...")
    agg_model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    raw_labels = agg_model.fit_predict(X)

    # Compute cluster centers of mass
    centers = np.array([X[raw_labels == i].mean(axis=0) for i in range(n_clusters)])

    # Compute validation metrics
    sil = silhouette_score(X, raw_labels, sample_size=3000, random_state=42)
    ch_score = calinski_harabasz_score(X, raw_labels)
    print(f"  [✓] 4D Silhouette Score: {sil:.4f}")
    print(f"  [✓] 4D Calinski-Harabasz Index: {ch_score:.1f}")

    # Re-order and canonicalize clusters into standard states 0, 1, 2, 3
    canonical_labels, mapping = canonicalize_clusters(df, raw_labels, centers)
    df["state"] = canonical_labels
    df["state_label"] = df["state"].map(CANONICAL_LABELS)

    # Print Archetype Profiles
    print("\nArchetype Cluster Profiles (Hierarchical Ward):")
    profile = df.groupby("state").agg({
        "word_count": "mean",
        "polarity": "mean",
        "subjectivity": "mean",
        "q_marks": "mean",
        "excl_marks": "mean",
        "PC1": "mean",
        "PC2": "mean",
        "PC3": "mean",
        "PC4": "mean",
        "comment_id": "count"
    }).rename(columns={"comment_id": "comment_count"}).round(3)
    profile["label"] = profile.index.map(CANONICAL_LABELS)
    print(profile[["label", "comment_count", "word_count", "polarity", "q_marks", "PC1", "PC2", "PC3", "PC4"]])

    # Show Subreddit distribution across states
    print("\nCommunity State Distribution (% of comments):")
    dist = pd.crosstab(df["subreddit"], df["state_label"], normalize="index") * 100
    print(dist.round(1))

    # Save serialized model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(agg_model, os.path.join(MODEL_DIR, "hierarchical_model.joblib"))

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[✓] Saved clustered dataset to: {output_csv}")

    # Generate Visuals: Hierarchical Cluster Map & Dendrogram
    map_plot_path = os.path.join(REPORTS_DIR, "cluster_map.png")
    plot_cluster_map(df, map_plot_path)

    dend_plot_path = os.path.join(REPORTS_DIR, "hierarchical_dendrogram.png")
    plot_hierarchical_dendrogram(X, dend_plot_path)

    # Remove obsolete comparison file if present
    obsolete_comp = os.path.join(REPORTS_DIR, "clustering_algorithm_comparison.png")
    if os.path.exists(obsolete_comp):
        try:
            os.remove(obsolete_comp)
        except Exception:
            pass

    return df


if __name__ == "__main__":
    run_clustering_pipeline()


if __name__ == "__main__":
    run_clustering_pipeline()
