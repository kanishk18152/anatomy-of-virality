"""
Streamlit Web Dashboard for The Anatomy of Virality
Run via: streamlit run app.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    import streamlit as st
except ImportError:
    st = None

from src.simulator import TrajectorySimulator, STATE_NAMES, STATE_COLORS


def main():
    if st is None:
        print("Streamlit is not installed. To run the web app: pip install streamlit && streamlit run app.py")
        print("Alternatively, open reports/dashboard.html directly in any web browser!")
        return

    st.set_page_config(
        page_title="The Anatomy of Virality",
        page_icon="🔬",
        layout="wide"
    )

    st.title("🔬 The Anatomy of Virality: Discussion Lifecycle Engine")
    st.markdown(
        "**Statistical Learning Methods for Managers** | Unsupervised Latent Geometry (PCA), "
        "Behavioral Archetypes (K-Means), and Absorbing Markov Chains ($N = (I-Q)^{-1}$)."
    )

    st.sidebar.header("🕹️ Simulation & Controls")
    community = st.sidebar.selectbox("Select Community Physics", ["science", "politics", "cringe", "overall_corpus"])
    start_state = st.sidebar.selectbox(
        "Initial Comment Archetype",
        [0, 1, 2, 3],
        format_func=lambda x: f"State {x}: {STATE_NAMES[x]}"
    )
    max_steps = st.sidebar.slider("Maximum Steps", min_value=5, max_value=25, value=12)

    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Live Trajectory Simulator", "📐 Latent Geometry (PCA)", "📊 Markov Matrices (P & N)", "💡 Executive Insights"])

    with tab1:
        st.subheader("Stochastic Conversation Trajectory")
        st.write("Simulate how discussions evolve step-by-step from different starting archetypes.")

        if st.button("▶ Run New Simulation", type="primary"):
            sim = TrajectorySimulator()
            traj = sim.simulate_thread(start_state=start_state, community=community, max_steps=max_steps)

            for item in traj:
                step = item["step"]
                name = item["state_name"]
                color = item["color"]
                comment = item["comment"]

                if item["is_absorbing"]:
                    st.error(f"**Step {step} • {name.upper()}** — Thread terminated (No further replies)")
                else:
                    with st.container():
                        st.markdown(
                            f"<div style='border-left: 4px solid {color}; padding: 10px 15px; margin: 8px 0; background: rgba(255,255,255,0.03); border-radius: 4px;'>"
                            f"<span style='color:{color}; font-weight:700;'>Step {step} • {name.upper()}</span> | <i>u/{comment.get('author','user')} in r/{comment.get('subreddit', community)}</i><br>"
                            f"\"{comment.get('body','')}\"<br>"
                            f"<small style='opacity:0.7;'>Score: {comment.get('score',0)} | Words: {comment.get('word_count',0)} | Polarity: {comment.get('polarity',0.0)}</small>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

    with tab2:
        st.subheader("Latent Conversational Space (PCA Biplot)")
        biplot_path = "reports/figures/pca_biplot.png"
        if os.path.exists(biplot_path):
            st.image(biplot_path, caption="2D PCA Biplot: Intellectual Depth (PC1) vs. Conversational Tension (PC2)")
        
        cluster_map_path = "reports/figures/cluster_map.png"
        if os.path.exists(cluster_map_path):
            st.image(cluster_map_path, caption="K-Means Behavioral Clusters (K=4 Archetypes)")

    with tab3:
        st.subheader(f"Stochastic Dynamics for r/{community.upper()}")
        p_path = f"reports/metrics/{community}_P_matrix.csv"
        n_path = f"reports/metrics/{community}_N_matrix.csv"
        tau_path = f"reports/metrics/{community}_lifetime.csv"

        col1, col2 = st.columns(2)
        if os.path.exists(p_path):
            with col1:
                st.markdown("**Transition Probability Matrix P**")
                p_df = pd.read_csv(p_path, index_col=0)
                st.dataframe((p_df * 100).round(1).astype(str) + "%")

        if os.path.exists(n_path):
            with col2:
                st.markdown("**Fundamental Matrix N = (I - Q)⁻¹**")
                n_df = pd.read_csv(n_path, index_col=0)
                st.dataframe(n_df.round(2))

        if os.path.exists(tau_path):
            st.markdown("**Expected Comments to Thread Death (τ)**")
            tau_df = pd.read_csv(tau_path, index_col=0)
            st.bar_chart(tau_df)

    with tab4:
        st.subheader("Executive & Managerial Findings")
        st.markdown("""
        ### Key Managerial Takeaways
        1. **The 2-Comment Friction Cliff**: When a conversation enters Friction, the empirical probability of back-to-back friction reaches **38.0%** in volatile communities. Automated intervention (slow-mode or pinned clarification) must occur within **1 reply** before toxic feedback loops lock in.
        2. **Inquiry as the Retention Anchor**: Inquiries boast the lowest immediate burnout hazard (**14.7%–15.3%**) and sustain an expected **5.34 comments** before termination.
        3. **Cross-Community Physics**: Discussions in structured communities like `r/science` sustain higher intellectual depth and lower friction traps, whereas unmoderated or sarcastic forums suffer rapid conversational decay.
        """)


if __name__ == "__main__":
    main()
