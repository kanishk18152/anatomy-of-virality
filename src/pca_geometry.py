"""
Module 4 (Lego Block: Latent Geometry & PCA Dimensionality Reduction)
Responsible for standardizing NLP metrics, fitting PCA with 4 components,
enforcing canonical eigenvector sign alignment, and exporting 4D coordinates.

The 4 Principal Components capture 83.9% of total variance:
  PC1 (34.1%) — Analytical Depth: Word Count, Root-TTR
  PC2 (18.2%) — Emotional Valence: Sentiment Polarity, Subjectivity
  PC3 (16.2%) — Agitation Intensity: Exclamation Density
  PC4 (15.3%) — Inquisitive Engagement: Question Density, Polarity

Generates 4 publication-ready PCA figures:
  1. pca_scree_plot.png             — Scree plot & cumulative variance (why 4 PCs)
  2. pca_correlation_heatmap.png    — Feature correlation matrix (why PCA is needed)
  3. pca_loadings_bar.png           — Feature loading magnitudes per PC (all 4)
  4. pca_biplot.png                 — 2D biplot (PC1 vs PC2) with loading vectors
"""

import os
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


INPUT_FEATURES_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "comment_features.csv"
)
OUTPUT_PCA_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "pca_transformed.csv"
)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")

FEATURE_COLUMNS = [
    "polarity",
    "subjectivity",
    "word_count",
    "lexical_diversity",
    "q_marks",
    "excl_marks"
]

FEATURE_LABELS = {
    "polarity": "Sentiment\nPolarity",
    "subjectivity": "Subjectivity",
    "word_count": "Word\nCount",
    "lexical_diversity": "Root-TTR\n(Guiraud)",
    "q_marks": "Question\nDensity",
    "excl_marks": "Exclamation\nDensity"
}

PC_NAMES = {
    0: "PC1: Analytical Depth",
    1: "PC2: Emotional Valence",
    2: "PC3: Agitation Intensity",
    3: "PC4: Inquisitive Engagement"
}

PC_SHORT_NAMES = {
    0: "PC1_Analytical_Depth",
    1: "PC2_Emotional_Valence",
    2: "PC3_Agitation_Intensity",
    3: "PC4_Inquisitive_Engagement"
}


class AlignedPCA:
    """
    Wrapper around scikit-learn PCA that enforces deterministic,
    interpretable sign orientation for conversational dynamics:
    - PC1 (Analytical Depth):        Positive loading on word_count & lexical_diversity
    - PC2 (Emotional Valence):       Positive loading on polarity (positive = warm)
    - PC3 (Agitation Intensity):     Positive loading on excl_marks
    - PC4 (Inquisitive Engagement):  Positive loading on q_marks
    """
    def __init__(self, n_components: int = 4, random_state: int = 42):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.n_components = n_components
        self.sign_multipliers = np.ones(n_components)
        self.feature_names = FEATURE_COLUMNS

    def fit(self, X: pd.DataFrame) -> "AlignedPCA":
        X_scaled = self.scaler.fit_transform(X[self.feature_names])
        self.pca.fit(X_scaled)

        # Check and enforce canonical sign alignment
        components = self.pca.components_.copy()
        wc_idx = self.feature_names.index("word_count")
        pol_idx = self.feature_names.index("polarity")
        excl_idx = self.feature_names.index("excl_marks")
        q_idx = self.feature_names.index("q_marks")

        # Sign alignment anchors for each component
        anchors = [
            (0, wc_idx),    # PC1: word_count should be positive
            (1, pol_idx),   # PC2: polarity should be positive
            (2, excl_idx),  # PC3: excl_marks should be positive
            (3, q_idx),     # PC4: q_marks should be positive
        ]

        for pc_idx, feat_idx in anchors:
            if pc_idx < self.n_components:
                if components[pc_idx, feat_idx] < 0:
                    components[pc_idx] *= -1
                    self.sign_multipliers[pc_idx] = -1.0
                else:
                    self.sign_multipliers[pc_idx] = 1.0

        self.aligned_components_ = components
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X[self.feature_names])
        X_pca = self.pca.transform(X_scaled)
        # Apply sign corrections
        return X_pca * self.sign_multipliers

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_

    @property
    def loadings(self) -> pd.DataFrame:
        col_names = [PC_SHORT_NAMES.get(i, f"PC{i+1}") for i in range(self.n_components)]
        return pd.DataFrame(
            self.aligned_components_.T,
            index=self.feature_names,
            columns=col_names
        )


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: Scree Plot & Cumulative Variance Explained
# ═══════════════════════════════════════════════════════════════════════

def plot_scree(df: pd.DataFrame, scaler: StandardScaler, output_path: str):
    """
    Fits a full 6-component PCA to generate a scree plot showing
    individual and cumulative explained variance, with a clear visual
    justification for selecting 4 components.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X_scaled = scaler.transform(df[FEATURE_COLUMNS])
    full_pca = PCA(n_components=len(FEATURE_COLUMNS), random_state=42)
    full_pca.fit(X_scaled)

    var_ratio = full_pca.explained_variance_ratio_
    cumulative = np.cumsum(var_ratio)
    pcs = np.arange(1, len(var_ratio) + 1)

    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)

    # Bar chart: individual variance (highlight retained PCs)
    bar_colors = ["#2b5c8f" if i < 4 else "#c0c7ce" for i in range(len(var_ratio))]
    bars = ax1.bar(pcs, var_ratio * 100, color=bar_colors, edgecolor="#1a1a2e",
                   width=0.6, zorder=3, label="Individual Variance")
    ax1.set_xlabel("Principal Component", fontsize=12, fontweight="semibold")
    ax1.set_ylabel("Individual Variance Explained (%)", fontsize=11, fontweight="semibold", color="#2b5c8f")
    ax1.set_xticks(pcs)
    ax1.set_xticklabels([f"PC{i}" for i in pcs], fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#2b5c8f")

    # Add percentage labels on bars
    for bar, val in zip(bars, var_ratio):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9,
                 fontweight="bold", color="#2b5c8f")

    # Line chart: cumulative variance (secondary y-axis)
    ax2 = ax1.twinx()
    ax2.plot(pcs, cumulative * 100, color="#d95f02", marker="o", linewidth=2.5,
             markersize=8, zorder=4, label="Cumulative Variance")
    ax2.set_ylabel("Cumulative Variance Explained (%)", fontsize=11, fontweight="semibold", color="#d95f02")
    ax2.tick_params(axis="y", labelcolor="#d95f02")
    ax2.set_ylim(0, 105)

    # Threshold line at PC4 cumulative
    ax2.axhline(y=cumulative[3] * 100, color="#d95f02", linestyle="--", alpha=0.5, linewidth=1)
    ax2.text(5.5, cumulative[3] * 100 + 1.5, f"{cumulative[3]*100:.1f}% at PC4",
             fontsize=10, color="#d95f02", fontweight="bold", ha="right")

    # Visual marker for the "elbow" at PC4
    ax1.axvline(x=4.5, color="#e74c3c", linestyle=":", linewidth=1.5, alpha=0.6)
    ax1.text(4.7, max(var_ratio) * 100 * 0.9, "← Retain 4 PCs",
             fontsize=10, color="#e74c3c", fontweight="bold", fontstyle="italic")

    ax1.set_title("Scree Plot: Variance Explained per Principal Component",
                  fontsize=13, fontweight="bold", pad=14)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax1.set_axisbelow(True)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[✓] PCA Scree Plot saved to: {output_path}")

    return var_ratio, cumulative


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Feature Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(df: pd.DataFrame, output_path: str):
    """
    Renders a correlation heatmap across the 6 raw NLP features to visually
    demonstrate multicollinearity (the motivating problem PCA solves).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    corr = df[FEATURE_COLUMNS].corr()

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)

    # Custom diverging colormap
    cmap = plt.cm.RdBu_r

    labels = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLUMNS]
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_yticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(labels, fontsize=9, fontweight="semibold", rotation=0, ha="center")
    ax.set_yticklabels(labels, fontsize=9, fontweight="semibold")

    # Annotate correlation values
    for i in range(len(FEATURE_COLUMNS)):
        for j in range(len(FEATURE_COLUMNS)):
            val = corr.values[i, j]
            text_color = "white" if abs(val) > 0.55 else "black"
            fontw = "bold" if abs(val) > 0.3 else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=text_color, fontweight=fontw)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Pearson Correlation (r)", fontsize=10, fontweight="semibold")

    ax.set_title("Feature Correlation Matrix (Pre-PCA)\nMulticollinearity Diagnostic",
                 fontsize=13, fontweight="bold", pad=14)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[✓] Correlation Heatmap saved to: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: PCA Loadings Bar Chart (All 4 PCs)
# ═══════════════════════════════════════════════════════════════════════

def plot_loadings_bar(model: AlignedPCA, output_path: str):
    """
    Horizontal grouped bar chart showing the magnitude and direction of
    each feature's loading on all 4 principal components.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    loadings = model.loadings

    features = loadings.index.tolist()
    labels = [FEATURE_LABELS.get(f, f).replace("\n", " ") for f in features]
    n_pcs = model.n_components
    n_feats = len(features)

    # Colors for each PC
    pc_colors = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3"]
    pc_labels_display = [
        "PC1: Analytical Depth (34.1%)",
        "PC2: Emotional Valence (18.2%)",
        "PC3: Agitation Intensity (16.2%)",
        "PC4: Inquisitive Engagement (15.3%)"
    ]

    y_pos = np.arange(n_feats)
    total_height = 0.75
    bar_height = total_height / n_pcs

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    for pc_idx in range(n_pcs):
        vals = loadings.iloc[:, pc_idx].values
        offset = (pc_idx - (n_pcs - 1) / 2) * bar_height
        bars = ax.barh(y_pos + offset, vals, bar_height * 0.9,
                       label=pc_labels_display[pc_idx],
                       color=pc_colors[pc_idx], edgecolor="#1a1a2e",
                       alpha=0.85, zorder=3)

        # Add value labels for significant loadings
        for bar, val in zip(bars, vals):
            if abs(val) > 0.15:
                x_offset = 0.02 if val >= 0 else -0.02
                ha = "left" if val >= 0 else "right"
                ax.text(val + x_offset, bar.get_y() + bar.get_height()/2,
                        f"{val:+.2f}", ha=ha, va="center", fontsize=7.5,
                        fontweight="bold", color=pc_colors[pc_idx])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.5, fontweight="semibold")
    ax.set_xlabel("Loading Coefficient (Eigenvector Weight)", fontsize=11, fontweight="semibold")
    ax.axvline(0, color="gray", linewidth=1.2, linestyle="-")
    ax.set_title("PCA Feature Loadings: Which Linguistic Traits Drive Each Component?",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(loc="lower right", frameon=True, fontsize=9, ncol=1)
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(-1.0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[✓] PCA Loadings Bar Chart (4 PCs) saved to: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: PCA Biplot (PC1 vs PC2 — Primary Clustering Plane)
# ═══════════════════════════════════════════════════════════════════════

def plot_pca_biplot(df: pd.DataFrame, model: AlignedPCA, output_path: str):
    """
    Renders an executive-grade 2D PCA Biplot with data points colored by subreddit
    and vector overlays representing feature loadings.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    # Subreddit color map
    palette = {"science": "#2b5c8f", "politics": "#d95f02", "cringe": "#7570b3"}
    for sub, color in palette.items():
        sub_data = df[df["subreddit"] == sub]
        if not sub_data.empty:
            ax.scatter(
                sub_data["PC1"], sub_data["PC2"],
                c=color, label=f"r/{sub}", alpha=0.35, s=20, edgecolors="none"
            )

    # Plot loading vectors
    loadings = model.loadings
    scale_factor = 3.5  # Visual scaling for loading arrows
    pc1_col = loadings.columns[0]
    pc2_col = loadings.columns[1]
    for feat in loadings.index:
        x = loadings.loc[feat, pc1_col] * scale_factor
        y = loadings.loc[feat, pc2_col] * scale_factor
        label = FEATURE_LABELS.get(feat, feat).replace("\n", " ")
        ax.arrow(0, 0, x, y, color="#e7298a", width=0.03, head_width=0.15, alpha=0.85, length_includes_head=True)
        ax.text(x * 1.15, y * 1.15, label, color="#333333", fontsize=9, fontweight="bold", ha="center")

    var_pc1 = model.explained_variance_ratio_[0] * 100
    var_pc2 = model.explained_variance_ratio_[1] * 100

    ax.set_title("Latent Conversational Geometry (PCA Biplot: PC1 vs PC2)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(f"PC1: Analytical Depth ({var_pc1:.1f}% variance)", fontsize=11, fontweight="semibold")
    ax.set_ylabel(f"PC2: Emotional Valence ({var_pc2:.1f}% variance)", fontsize=11, fontweight="semibold")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(title="Community", loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[✓] PCA Biplot (PC1 vs PC2) saved to: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5: PCA Biplot (PC3 vs PC4 — Secondary Dimensions)
# ═══════════════════════════════════════════════════════════════════════

def plot_pca_biplot_pc3_pc4(df: pd.DataFrame, model: AlignedPCA, output_path: str):
    """
    2D scatter plot of PC3 vs PC4 with loading vectors, showing the
    secondary conversational dimensions: Agitation vs Inquisitiveness.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    # Subreddit color map
    palette = {"science": "#2b5c8f", "politics": "#d95f02", "cringe": "#7570b3"}
    for sub, color in palette.items():
        sub_data = df[df["subreddit"] == sub]
        if not sub_data.empty:
            ax.scatter(
                sub_data["PC3"], sub_data["PC4"],
                c=color, label=f"r/{sub}", alpha=0.35, s=20, edgecolors="none"
            )

    # Plot loading vectors for PC3 and PC4
    loadings = model.loadings
    scale_factor = 3.5
    pc3_col = loadings.columns[2]
    pc4_col = loadings.columns[3]
    for feat in loadings.index:
        x = loadings.loc[feat, pc3_col] * scale_factor
        y = loadings.loc[feat, pc4_col] * scale_factor
        label = FEATURE_LABELS.get(feat, feat).replace("\n", " ")
        ax.arrow(0, 0, x, y, color="#e7298a", width=0.03, head_width=0.15, alpha=0.85, length_includes_head=True)
        ax.text(x * 1.15, y * 1.15, label, color="#333333", fontsize=9, fontweight="bold", ha="center")

    var_pc3 = model.explained_variance_ratio_[2] * 100
    var_pc4 = model.explained_variance_ratio_[3] * 100

    ax.set_title("Secondary Conversational Dimensions (PCA Biplot: PC3 vs PC4)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(f"PC3: Agitation Intensity ({var_pc3:.1f}% variance)", fontsize=11, fontweight="semibold")
    ax.set_ylabel(f"PC4: Inquisitive Engagement ({var_pc4:.1f}% variance)", fontsize=11, fontweight="semibold")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(title="Community", loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[✓] PCA Biplot (PC3 vs PC4) saved to: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# MASTER PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run_pca_pipeline(
    input_csv: str = INPUT_FEATURES_CSV,
    output_csv: str = OUTPUT_PCA_CSV
) -> Tuple[pd.DataFrame, AlignedPCA]:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input features file not found: {input_csv}. Run Module 3 first.")

    df = pd.read_csv(input_csv)
    print(f"Fitting Aligned PCA (4 components) on {len(df):,} comments across {len(FEATURE_COLUMNS)} dimensions...")

    model = AlignedPCA(n_components=4)
    coords = model.fit_transform(df)

    df["PC1"] = np.round(coords[:, 0], 4)
    df["PC2"] = np.round(coords[:, 1], 4)
    df["PC3"] = np.round(coords[:, 2], 4)
    df["PC4"] = np.round(coords[:, 3], 4)

    # Print mathematical findings
    evr = model.explained_variance_ratio_
    print(f"\nExplained Variance Ratio:")
    for i in range(4):
        print(f"  {PC_NAMES[i]}:  {evr[i] * 100:.2f}%")
    print(f"  Total 4D Explained Variance: {sum(evr) * 100:.2f}%")

    print("\nAligned Feature Loadings Matrix (4 PCs):")
    print(model.loadings.round(3))

    # Save model and artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "pca_pipeline.joblib")
    joblib.dump(model, model_path)
    print(f"[✓] Saved fitted PCA model to: {model_path}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Saved PCA coordinates dataset (4D) to: {output_csv}")

    # ── Generate all 5 PCA Figures ──
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Figure 1: Scree Plot (fit full 6-component PCA for this)
    scree_path = os.path.join(REPORTS_DIR, "pca_scree_plot.png")
    plot_scree(df, model.scaler, scree_path)

    # Figure 2: Correlation Heatmap
    corr_path = os.path.join(REPORTS_DIR, "pca_correlation_heatmap.png")
    plot_correlation_heatmap(df, corr_path)

    # Figure 3: Loadings Bar Chart (all 4 PCs)
    loadings_path = os.path.join(REPORTS_DIR, "pca_loadings_bar.png")
    plot_loadings_bar(model, loadings_path)

    # Figure 4: Biplot PC1 vs PC2
    biplot_path = os.path.join(REPORTS_DIR, "pca_biplot.png")
    plot_pca_biplot(df, model, biplot_path)

    # Figure 5: Biplot PC3 vs PC4
    biplot34_path = os.path.join(REPORTS_DIR, "pca_biplot_pc3_pc4.png")
    plot_pca_biplot_pc3_pc4(df, model, biplot34_path)

    return df, model


if __name__ == "__main__":
    run_pca_pipeline()
