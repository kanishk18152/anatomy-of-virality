"""
Module 6 (Lego Block: Absorbing Markov Chains & Trajectory Dynamics)
Responsible for:
1. Converting sequential comment chains into discrete-time stochastic transitions.
2. Formulating the Absorbing Markov Chain with State 4 = Thread Death.
3. Inverting (I - Q) to compute the Fundamental Matrix N and expected lifetime tau = N * 1.
4. Calculating community-level transition dynamics (r/science vs. r/politics vs. r/cringe).
5. Detecting managerial tipping points and derailment thresholds.
"""

import os
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_CLUSTERED_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "clustered_comments.csv"
)
OUTPUT_METRICS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "metrics")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")

STATE_NAMES = {
    0: "Deep Dive",
    1: "Inquiry",
    2: "Friction",
    3: "Banter",
    4: "Thread Death (Absorbing)"
}

TRANSIENT_NAMES = [STATE_NAMES[i] for i in range(4)]
ALL_NAMES = [STATE_NAMES[i] for i in range(5)]


def extract_markov_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts chronological state transitions (s_t -> s_{t+1}) from linear chains,
    appending an explicit absorbing transition to State 4 (Thread Death) at chain termination.
    Propagates transition_weight to eliminate tree-decomposition pseudoreplication,
    and thread_id to enable thread-level cluster block-bootstrapping.
    """
    transitions = []
    
    for chain_id, chain_group in df.groupby("chain_id"):
        sorted_chain = chain_group.sort_values("step_index")
        states = sorted_chain["state"].tolist()
        sub = sorted_chain["subreddit"].iloc[0]
        tid = sorted_chain["thread_id"].iloc[0] if "thread_id" in sorted_chain.columns else chain_id.split("_")[0]
        has_weights = "transition_weight" in sorted_chain.columns

        # Intra-chain transitions between transient states
        for i in range(len(states) - 1):
            w = float(sorted_chain["transition_weight"].iloc[i]) if has_weights else 1.0
            transitions.append({
                "from_state": states[i],
                "to_state": states[i+1],
                "subreddit": sub,
                "chain_id": chain_id,
                "thread_id": tid,
                "transition_weight": w
            })

        # Terminal transition to absorbing State 4 (Thread Death)
        if len(states) > 0:
            w_term = float(sorted_chain["transition_weight"].iloc[-1]) if has_weights else 1.0
            transitions.append({
                "from_state": states[-1],
                "to_state": 4,
                "subreddit": sub,
                "chain_id": chain_id,
                "thread_id": tid,
                "transition_weight": w_term
            })

    return pd.DataFrame(transitions)


def compute_absorbing_markov_properties(
    transitions_df: pd.DataFrame,
    n_transient: int = 4
) -> Dict[str, Any]:
    """
    Constructs the 5x5 absorbing transition matrix P = [[Q, R], [0, 1]],
    inverts (I - Q) to get Fundamental Matrix N, and computes expected lifetime tau = N * 1.
    Uses multiplicity weights when present to eliminate pseudoreplication.
    """
    if "transition_weight" in transitions_df.columns:
        # Multiplicity-weighted cross-tabulation (unbiased tree estimation)
        ct = pd.pivot_table(
            transitions_df,
            index="from_state",
            columns="to_state",
            values="transition_weight",
            aggfunc="sum",
            fill_value=0.0
        ).reindex(index=range(n_transient + 1), columns=range(n_transient + 1), fill_value=0.0)
    else:
        # Standard unweighted cross-tabulate transition counts
        ct = pd.crosstab(
            transitions_df["from_state"],
            transitions_df["to_state"],
            dropna=False
        ).reindex(index=range(n_transient + 1), columns=range(n_transient + 1), fill_value=0.0)

    # Convert counts to probabilities (row-stochastic)
    row_sums = ct.sum(axis=1)
    # Avoid zero division for absorbing state
    row_sums[n_transient] = 1.0
    ct.loc[n_transient, n_transient] = 1.0  # Absorbing self-loop: State 4 -> State 4

    P = ct.div(row_sums, axis=0).fillna(0.0).to_numpy(copy=True)
    # Enforce exact absorbing row [0, 0, 0, 0, 1]
    P[n_transient, :] = 0.0
    P[n_transient, n_transient] = 1.0

    # C3 Fix: If a transient state was never observed in transitions, its row
    # sums to 0, violating the row-stochastic invariant. Route unvisited
    # transient states directly to absorption (State 4) so P remains valid.
    for i in range(n_transient):
        if P[i, :].sum() < 1e-10:
            P[i, :] = 0.0
            P[i, n_transient] = 1.0

    # Partition: P = [[Q, R], [0, I]]
    Q = P[:n_transient, :n_transient]
    R = P[:n_transient, n_transient:]

    # Fundamental Matrix N = (I - Q)^(-1)
    I = np.eye(n_transient)
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        import warnings
        warnings.warn(
            "(I - Q) is singular — a closed communicating class exists. "
            "Falling back to pseudo-inverse; expected lifetimes may be unreliable."
        )
        N = np.linalg.pinv(I - Q)

    # Expected comments to thread death starting from state i: tau = N * 1
    tau = N.dot(np.ones(n_transient))

    # Absorption probability B = N * R (in 1-absorbing case, this equals 1.0)
    B = N.dot(R)

    # Format into readable DataFrames
    df_P = pd.DataFrame(P, index=ALL_NAMES, columns=ALL_NAMES)
    df_Q = pd.DataFrame(Q, index=TRANSIENT_NAMES, columns=TRANSIENT_NAMES)
    df_N = pd.DataFrame(N, index=TRANSIENT_NAMES, columns=TRANSIENT_NAMES)
    series_tau = pd.Series(tau, index=TRANSIENT_NAMES, name="Expected_Comments_To_Death")

    return {
        "P_matrix": df_P,
        "Q_matrix": df_Q,
        "R_matrix": R,
        "Fundamental_Matrix_N": df_N,
        "Expected_Comments_To_Death": series_tau,
        "Absorption_B": B
    }


def bootstrap_markov_properties(
    transitions_df: pd.DataFrame,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Non-parametric thread-level block bootstrap for uncertainty quantification.
    Resamples entire discussion trees with replacement to preserve intra-thread correlation,
    deriving 95% confidence intervals for transition probabilities (P) and expected lifetimes (tau).
    """
    np.random.seed(random_state)
    if "thread_id" not in transitions_df.columns:
        raise ValueError("transitions_df must contain 'thread_id' for thread-level block bootstrapping.")

    thread_groups = {tid: df for tid, df in transitions_df.groupby("thread_id")}
    unique_threads = list(thread_groups.keys())
    n_threads = len(unique_threads)

    if n_threads < 2:
        raise ValueError("Cannot bootstrap with fewer than 2 distinct threads.")

    tau_bootstraps = []
    P_bootstraps = []

    for _ in range(n_bootstraps):
        sampled_tids = np.random.choice(unique_threads, size=n_threads, replace=True)
        boot_df = pd.concat([thread_groups[tid] for tid in sampled_tids], ignore_index=True)
        boot_res = compute_absorbing_markov_properties(boot_df)
        
        tau_bootstraps.append(boot_res["Expected_Comments_To_Death"].values)
        P_bootstraps.append(boot_res["P_matrix"].values)

    tau_arr = np.array(tau_bootstraps)  # Shape: (B, 4)
    P_arr = np.array(P_bootstraps)      # Shape: (B, 5, 5)

    low_pct = 100 * (alpha / 2.0)
    high_pct = 100 * (1.0 - alpha / 2.0)

    # Compute statistics for expected lifetime tau
    tau_mean = np.mean(tau_arr, axis=0)
    tau_std = np.std(tau_arr, axis=0)
    tau_ci_lower = np.percentile(tau_arr, low_pct, axis=0)
    tau_ci_upper = np.percentile(tau_arr, high_pct, axis=0)

    tau_inference = pd.DataFrame({
        "Mean_tau": np.round(tau_mean, 2),
        "Std_tau": np.round(tau_std, 2),
        "CI_Lower_95": np.round(tau_ci_lower, 2),
        "CI_Upper_95": np.round(tau_ci_upper, 2)
    }, index=TRANSIENT_NAMES)

    # Compute statistics for transition matrix P
    P_mean = np.mean(P_arr, axis=0)
    P_std = np.std(P_arr, axis=0)
    P_ci_lower = np.percentile(P_arr, low_pct, axis=0)
    P_ci_upper = np.percentile(P_arr, high_pct, axis=0)

    return {
        "tau_inference": tau_inference,
        "P_mean": pd.DataFrame(P_mean, index=ALL_NAMES, columns=ALL_NAMES),
        "P_std": pd.DataFrame(P_std, index=ALL_NAMES, columns=ALL_NAMES),
        "P_ci_lower": pd.DataFrame(P_ci_lower, index=ALL_NAMES, columns=ALL_NAMES),
        "P_ci_upper": pd.DataFrame(P_ci_upper, index=ALL_NAMES, columns=ALL_NAMES)
    }


def permutation_test_matrix_homogeneity(
    transitions_df: pd.DataFrame,
    sub_a: str,
    sub_b: str,
    n_permutations: int = 1000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Non-parametric permutation test for Markov matrix homogeneity across two communities.
    Tests H0: P_A = P_B against H1: P_A != P_B using the Frobenius distance ||P_A - P_B||_F.
    Shuffles community assignments across discussion threads to form the empirical null distribution.
    """
    np.random.seed(random_state)
    subset = transitions_df[transitions_df["subreddit"].isin([sub_a, sub_b])]
    
    # Pre-group by thread_id
    thread_to_sub = subset.groupby("thread_id")["subreddit"].first().to_dict()
    thread_groups = {tid: df for tid, df in subset.groupby("thread_id")}
    
    threads_a = [tid for tid, sub in thread_to_sub.items() if sub == sub_a]
    threads_b = [tid for tid, sub in thread_to_sub.items() if sub == sub_b]
    n_a = len(threads_a)
    all_threads = np.array(threads_a + threads_b)

    # Compute observed baseline difference
    P_a_obs = compute_absorbing_markov_properties(subset[subset["subreddit"] == sub_a])["P_matrix"].values
    P_b_obs = compute_absorbing_markov_properties(subset[subset["subreddit"] == sub_b])["P_matrix"].values
    d_obs = np.linalg.norm(P_a_obs - P_b_obs, ord="fro")

    # Permutation sampling under H0
    permuted_diffs = []
    for _ in range(n_permutations):
        shuffled = np.random.permutation(all_threads)
        perm_tids_a = shuffled[:n_a]
        perm_tids_b = shuffled[n_a:]

        df_perm_a = pd.concat([thread_groups[tid] for tid in perm_tids_a], ignore_index=True)
        df_perm_b = pd.concat([thread_groups[tid] for tid in perm_tids_b], ignore_index=True)

        P_perm_a = compute_absorbing_markov_properties(df_perm_a)["P_matrix"].values
        P_perm_b = compute_absorbing_markov_properties(df_perm_b)["P_matrix"].values

        d_perm = np.linalg.norm(P_perm_a - P_perm_b, ord="fro")
        permuted_diffs.append(d_perm)

    permuted_diffs = np.array(permuted_diffs)
    p_value = np.mean(permuted_diffs >= d_obs)

    return {
        "comparison": f"r/{sub_a} vs r/{sub_b}",
        "frobenius_distance_obs": round(float(d_obs), 4),
        "p_value": round(float(p_value), 4),
        "is_significant_05": bool(p_value < 0.05),
        "is_significant_01": bool(p_value < 0.01),
        "threads_a": n_a,
        "threads_b": len(threads_b)
    }


def plot_transition_heatmaps(results_by_community: Dict[str, Dict[str, Any]], output_path: str):
    """
    Renders side-by-side heatmaps of the transient transition matrix Q for each community.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    communities = list(results_by_community.keys())
    fig, axes = plt.subplots(1, len(communities), figsize=(5 * len(communities), 4.5), dpi=300)

    if len(communities) == 1:
        axes = [axes]

    for ax, comm in zip(axes, communities):
        Q = results_by_community[comm]["Q_matrix"].values
        cax = ax.imshow(Q, cmap="YlOrRd", vmin=0, vmax=0.6)
        
        # Axis labels
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(TRANSIENT_NAMES, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(TRANSIENT_NAMES, fontsize=9)
        ax.set_title(f"r/{comm} Transition Flow (Q)", fontsize=11, fontweight="bold", pad=10)

        # Annotate cell values
        for i in range(4):
            for j in range(4):
                val = Q[i, j]
                color = "white" if val > 0.3 else "black"
                ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center", color=color, fontsize=8, fontweight="bold")

    fig.colorbar(cax, ax=axes, orientation="horizontal", fraction=0.05, pad=0.25, label="Transition Probability")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[✓] Transition heatmaps saved to: {output_path}")


def plot_lifetime_comparison(
    results_by_community: Dict[str, Dict[str, Any]],
    bootstrap_by_community: Dict[str, Dict[str, Any]],
    output_path: str
):
    """
    Bar chart comparing expected thread life expectancy (comments remaining) by initial state,
    with 95% confidence interval error bars derived from thread-level block bootstrapping.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    communities = [c for c in results_by_community.keys() if c != "Overall Corpus"]
    x = np.arange(len(TRANSIENT_NAMES))
    width = 0.25
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    for i, comm in enumerate(communities):
        lifetimes = results_by_community[comm]["Expected_Comments_To_Death"].values
        offset = (i - 1) * width
        
        # Extract 95% CI error bounds if available
        yerr = None
        if comm in bootstrap_by_community:
            inf = bootstrap_by_community[comm]["tau_inference"]
            lower_err = np.maximum(0, lifetimes - inf["CI_Lower_95"].values)
            upper_err = np.maximum(0, inf["CI_Upper_95"].values - lifetimes)
            yerr = np.array([lower_err, upper_err])

        rects = ax.bar(
            x + offset, lifetimes, width,
            yerr=yerr, capsize=4,
            label=f"r/{comm}", color=colors[i % len(colors)], alpha=0.85
        )
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 6 if yerr is not None else 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Expected Remaining Comments to Thread Death (tau)", fontsize=11, fontweight="semibold")
    ax.set_title("Conversational Life Expectancy with 95% Bootstrap CIs", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(TRANSIENT_NAMES, fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(title="Community", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[✓] Expected thread lifetime visual with CIs saved to: {output_path}")


def run_markov_pipeline(input_csv: str = INPUT_CLUSTERED_CSV) -> Dict[str, Any]:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Clustered dataset not found: {input_csv}. Run Module 5 first.")

    df = pd.read_csv(input_csv)
    print(f"Extracting state trajectories from {df['chain_id'].nunique():,} chains...")

    transitions_df = extract_markov_transitions(df)
    print(f"[✓] Total transitions extracted: {len(transitions_df):,}")

    # Compute Markov dynamics across communities
    results = {}
    bootstrap_results = {}
    
    # 1. Overall Corpus
    print("\n--- [ OVERALL CORPUS MARKOV DYNAMICS ] ---")
    overall = compute_absorbing_markov_properties(transitions_df)
    results["Overall Corpus"] = overall

    print("\nEmpirical Transition Probability Matrix P (Multiplicity-Weighted):")
    print((overall["P_matrix"] * 100).round(1).astype(str) + "%")

    print("\nFundamental Matrix N (Expected visits before Thread Death):")
    print(overall["Fundamental_Matrix_N"].round(2))

    print("\nExpected Comments Remaining until Thread Death (tau = N * 1):")
    print(overall["Expected_Comments_To_Death"].round(2))

    # Overall Bootstrap
    print("\nRunning Thread Block Bootstrap (B=500 iterations)...")
    boot_overall = bootstrap_markov_properties(transitions_df, n_bootstraps=500)
    bootstrap_results["Overall Corpus"] = boot_overall
    print("Overall 95% Confidence Intervals for tau:")
    print(boot_overall["tau_inference"])

    # 2. Individual Subreddit Breakdown & Bootstrapping
    for sub in df["subreddit"].unique():
        sub_transitions = transitions_df[transitions_df["subreddit"] == sub]
        sub_res = compute_absorbing_markov_properties(sub_transitions)
        results[sub] = sub_res
        
        # Subreddit Bootstrap
        boot_sub = bootstrap_markov_properties(sub_transitions, n_bootstraps=500)
        bootstrap_results[sub] = boot_sub

    # 3. Permutation Tests for Cross-Community Matrix Homogeneity
    print("\n=======================================================")
    print("      STATISTICAL HYPOTHESIS TESTING: HOMOGENEITY     ")
    print("=======================================================")
    sub_pairs = [("science", "politics"), ("science", "cringe"), ("politics", "cringe")]
    permutation_records = []
    
    available_subs = set(df["subreddit"].unique())
    for s_a, s_b in sub_pairs:
        if s_a in available_subs and s_b in available_subs:
            test_res = permutation_test_matrix_homogeneity(transitions_df, s_a, s_b, n_permutations=500)
            permutation_records.append(test_res)
            sig_marker = "***" if test_res["is_significant_01"] else ("**" if test_res["is_significant_05"] else "(Not Sig)")
            print(f"  • {test_res['comparison']}: Frobenius Dist = {test_res['frobenius_distance_obs']:.4f}, p-value = {test_res['p_value']:.4f} {sig_marker}")

    # 4. Managerial Tipping Point Analysis with 95% CIs
    print("\n=======================================================")
    print("      EXECUTIVE & MANAGERIAL TIPPING POINT ANALYSIS    ")
    print("=======================================================")
    for sub in ["science", "politics", "cringe"]:
        if sub in results:
            P_comm = results[sub]["P_matrix"]
            tau_comm = results[sub]["Expected_Comments_To_Death"]
            inf_tau = bootstrap_results[sub]["tau_inference"]
            P_lower = bootstrap_results[sub]["P_ci_lower"]
            P_upper = bootstrap_results[sub]["P_ci_upper"]
            
            p_ff = P_comm.loc["Friction", "Friction"] * 100
            p_ff_low = P_lower.loc["Friction", "Friction"] * 100
            p_ff_high = P_upper.loc["Friction", "Friction"] * 100

            p_fd = P_comm.loc["Friction", "Thread Death (Absorbing)"] * 100
            p_id = P_comm.loc["Inquiry", "Thread Death (Absorbing)"] * 100

            print(f"\n[ r/{sub.upper()} TIPPING DYNAMICS (with 95% CIs) ]:")
            print(f"  • Friction Escalation Trap: {p_ff:.1f}% [95% CI: {p_ff_low:.1f}% - {p_ff_high:.1f}%]")
            print(f"  • Immediate Burnout Hazard P(Friction -> Death): {p_fd:.1f}%")
            print(f"  • Inquiry Burnout Hazard P(Inquiry -> Death):    {p_id:.1f}%")
            print(f"  • Expected Life from Deep Dive: {tau_comm['Deep Dive']:.2f} cmts [95% CI: {inf_tau.loc['Deep Dive', 'CI_Lower_95']} - {inf_tau.loc['Deep Dive', 'CI_Upper_95']}]")
            print(f"  • Expected Life from Friction:  {tau_comm['Friction']:.2f} cmts [95% CI: {inf_tau.loc['Friction', 'CI_Lower_95']} - {inf_tau.loc['Friction', 'CI_Upper_95']}]")

    # Save metrics to CSV
    os.makedirs(OUTPUT_METRICS_DIR, exist_ok=True)
    for name, res in results.items():
        clean_name = name.lower().replace(" ", "_")
        res["P_matrix"].to_csv(os.path.join(OUTPUT_METRICS_DIR, f"{clean_name}_P_matrix.csv"))
        res["Fundamental_Matrix_N"].to_csv(os.path.join(OUTPUT_METRICS_DIR, f"{clean_name}_N_matrix.csv"))
        res["Expected_Comments_To_Death"].to_csv(os.path.join(OUTPUT_METRICS_DIR, f"{clean_name}_lifetime.csv"))

    # Save bootstrap inference & hypothesis tests
    for name, b_res in bootstrap_results.items():
        clean_name = name.lower().replace(" ", "_")
        b_res["tau_inference"].to_csv(os.path.join(OUTPUT_METRICS_DIR, f"{clean_name}_tau_inference_95ci.csv"))
    
    if permutation_records:
        pd.DataFrame(permutation_records).to_csv(
            os.path.join(OUTPUT_METRICS_DIR, "community_homogeneity_tests.csv"), index=False
        )

    # Generate Visuals
    plot_transition_heatmaps(
        {k: v for k, v in results.items() if k != "Overall Corpus"},
        os.path.join(REPORTS_DIR, "markov_transition_heatmaps.png")
    )
    plot_lifetime_comparison(results, bootstrap_results, os.path.join(REPORTS_DIR, "expected_thread_lifetime.png"))

    return {
        "results": results,
        "bootstrap_results": bootstrap_results,
        "permutation_tests": permutation_records
    }


if __name__ == "__main__":
    run_markov_pipeline()
