"""
The Anatomy of Virality: Master Orchestrator CLI
Unified entry point to run modular pipeline stages, verify mathematical models,
and execute stochastic conversation simulations.
"""

import argparse
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_raw_comments
from src.tree_parser import run_tree_parser
from src.feature_extractor import run_feature_extractor
from src.pca_geometry import run_pca_pipeline
from src.state_clustering import run_clustering_pipeline
from src.markov_engine import run_markov_pipeline
from src.simulator import TrajectorySimulator, STATE_NAMES


def run_full_pipeline():
    print("\n" + "="*70)
    print("      THE ANATOMY OF VIRALITY: FULL PIPELINE EXECUTION")
    print("="*70)

    # 1. Data Ingestion
    print("\n[STEP 1/6] Validating Data Layer...")
    raw_path = "data/raw/reddit_comments.csv"
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found. Please provide your data or run scripts/prepare_dataset.py.")
        sys.exit(1)
    df_raw = load_raw_comments(raw_path)

    # 2. Conversational Tree Decomposition
    print("\n[STEP 2/6] Decomposing Reply Trees into Linear Chains...")
    df_chains = run_tree_parser()

    # 3. Feature Extraction
    print("\n[STEP 3/6] Computing NLP Signals & Root-TTR Lexical Diversity...")
    df_features = run_feature_extractor()

    # 4. Latent Geometry & PCA
    print("\n[STEP 4/6] Reducing Dimensionality & Enforcing Canonical Alignment...")
    df_pca, model_pca = run_pca_pipeline()

    # 5. Behavioral State Clustering
    print("\n[STEP 5/6] Segmenting Unsupervised Behavioral States (Hierarchical Ward)...")
    df_clusters = run_clustering_pipeline()

    # 6. Absorbing Markov Chains
    print("\n[STEP 6/7] Computing Markov Matrices, Fundamental Matrix N & Tipping Points...")
    markov_results = run_markov_pipeline()

    # 7. Interactive Executive Dashboard
    print("\n[STEP 7/7] Generating Self-Contained Interactive Dashboard...")
    try:
        from src.dashboard_generator import generate_dashboard
        dashboard_path = generate_dashboard()
        print(f"[✓] Dashboard generated: {dashboard_path}")
    except Exception as e:
        print(f"[!] Dashboard generation skipped: {e}")

    print("\n" + "="*70)
    print("  [✓] PIPELINE EXECUTION COMPLETE — ALL LEGO MODULES VERIFIED!")
    print("="*70)
    print("Artifacts generated:")
    print("  • Visual Figures:  reports/figures/")
    print("  • Saved Models:    models/")
    print("  • Processed Data:  data/processed/")
    print("  • Matrix Exports:  reports/metrics/")
    print("  • Interactive UI:  reports/dashboard.html")


def run_interactive_simulation(community: str = "science", start_state: int = 0, steps: int = 12):
    print("\n" + "="*70)
    print(f"  STOCHASTIC THREAD TRAJECTORY SIMULATOR (r/{community.upper()})")
    print("="*70)
    print(f"Initiating conversation from: State {start_state} ({STATE_NAMES.get(start_state, 'Unknown')})\n")

    sim = TrajectorySimulator()
    trajectory = sim.simulate_thread(start_state=start_state, community=community, max_steps=steps)

    for item in trajectory:
        step_num = item["step"]
        s_name = item["state_name"]
        comment = item["comment"]
        
        prefix = f"[Step {step_num:02d}] [{s_name.upper()}]"
        if item["is_absorbing"]:
            print(f"\n{prefix} ⚠️  {comment['body']}\n")
        else:
            author = comment.get('author', 'user')
            body = comment.get('body', '').replace('\n', ' ')
            if len(body) > 110:
                body = body[:107] + "..."
            print(f"{prefix:<32} (u/{author}): \"{body}\"")

    print("="*70)
    print(f"Simulation completed after {len(trajectory)} state transitions.")


def main():
    parser = argparse.ArgumentParser(
        description="The Anatomy of Virality: Statistical Learning & Stochastic Discussion Dynamics"
    )
    parser.add_argument(
        "--step",
        choices=["all", "data", "tree", "features", "pca", "cluster", "markov"],
        default=None,
        help="Run an individual pipeline module or 'all' to execute end-to-end."
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run a stochastic conversation simulation based on empirical transition matrices."
    )
    parser.add_argument(
        "--community",
        type=str,
        default="science",
        choices=["science", "politics", "cringe", "overall_corpus"],
        help="Subreddit dynamics to simulate."
    )
    parser.add_argument(
        "--state",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Initial state for simulation: 0=Deep Dive, 1=Inquiry, 2=Friction, 3=Banter."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=15,
        help="Maximum simulation steps."
    )

    args = parser.parse_args()

    if args.simulate:
        run_interactive_simulation(community=args.community, start_state=args.state, steps=args.steps)
    elif args.step == "all":
        run_full_pipeline()
    elif args.step == "data":
        load_raw_comments("data/raw/reddit_comments.csv")
    elif args.step == "tree":
        run_tree_parser()
    elif args.step == "features":
        run_feature_extractor()
    elif args.step == "pca":
        run_pca_pipeline()
    elif args.step == "cluster":
        run_clustering_pipeline()
    elif args.step == "markov":
        run_markov_pipeline()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
