"""
Lego Block: Stochastic Conversation Trajectory Simulator
Simulates step-by-step thread evolutions using empirical Markov transition probabilities
and samples authentic comments representing each conversational state.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


CLUSTERED_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "clustered_comments.csv"
)
METRICS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "metrics")

STATE_NAMES = {
    0: "Deep Dive",
    1: "Inquiry",
    2: "Friction",
    3: "Casual Banter",
    4: "Thread Death (Absorbing)"
}

STATE_COLORS = {
    0: "#1b9e77",  # Emerald Green
    1: "#386cb0",  # Blue
    2: "#d95f02",  # Fiery Orange / Red
    3: "#7570b3",  # Purple
    4: "#666666"   # Gray
}


class TrajectorySimulator:
    def __init__(self, clustered_csv: str = CLUSTERED_CSV):
        if not os.path.exists(clustered_csv):
            raise FileNotFoundError(f"Clustered dataset not found: {clustered_csv}. Run Module 5 first.")

        self.df = pd.read_csv(clustered_csv)
        self.communities = ["overall_corpus"] + [s.lower() for s in self.df["subreddit"].unique()]
        self.P_matrices = {}

        # Load P matrices
        for comm in self.communities:
            path = os.path.join(METRICS_DIR, f"{comm}_P_matrix.csv")
            if os.path.exists(path):
                self.P_matrices[comm] = pd.read_csv(path, index_col=0)

    def sample_representative_comment(self, state: int, community: str = "overall_corpus") -> Dict[str, Any]:
        """
        Samples a real comment from the dataset matching the given state and community.
        """
        subset = self.df[self.df["state"] == state]
        if community != "overall_corpus" and community in self.df["subreddit"].unique():
            comm_subset = subset[subset["subreddit"] == community]
            if not comm_subset.empty:
                subset = comm_subset

        if subset.empty:
            return {"body": "[No matching text]", "author": "anonymous", "score": 0}

        row = subset.sample(1).iloc[0]
        return {
            "comment_id": row.get("comment_id", ""),
            "author": row.get("author", "user"),
            "body": str(row.get("body", "")),
            "score": int(row.get("score", 0)),
            "polarity": float(row.get("polarity", 0.0)),
            "word_count": int(row.get("word_count", 0)),
            "subreddit": row.get("subreddit", community)
        }

    def simulate_thread(
        self,
        start_state: int = 0,
        community: str = "science",
        max_steps: int = 15,
        random_seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Runs a discrete-time Markov chain simulation from start_state until
        absorption into State 4 (Thread Death) or max_steps.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        comm_key = community.lower().replace(" ", "_")
        if comm_key not in self.P_matrices:
            comm_key = "overall_corpus"

        P_df = self.P_matrices[comm_key]
        states_order = [0, 1, 2, 3, 4]

        trajectory = []
        current_state = start_state
        step = 0

        # Initial comment
        init_comment = self.sample_representative_comment(current_state, comm_key)
        trajectory.append({
            "step": step,
            "state_id": current_state,
            "state_name": STATE_NAMES[current_state],
            "color": STATE_COLORS[current_state],
            "comment": init_comment,
            "is_absorbing": False
        })

        # Run state transitions
        while current_state != 4 and step < max_steps:
            row_probs = P_df.iloc[current_state].values.astype(float)
            # Normalize to exactly 1.0 to prevent float rounding crashes in np.random.choice
            row_probs = row_probs / row_probs.sum()
            next_state = int(np.random.choice(states_order, p=row_probs))
            step += 1

            if next_state == 4:
                trajectory.append({
                    "step": step,
                    "state_id": 4,
                    "state_name": STATE_NAMES[4],
                    "color": STATE_COLORS[4],
                    "comment": {
                        "body": "── [THREAD DEATH: Inactivity / No further replies] ──",
                        "author": "system",
                        "score": 0,
                        "polarity": 0.0,
                        "word_count": 0,
                        "subreddit": comm_key
                    },
                    "is_absorbing": True
                })
                break
            else:
                comment = self.sample_representative_comment(next_state, comm_key)
                trajectory.append({
                    "step": step,
                    "state_id": next_state,
                    "state_name": STATE_NAMES[next_state],
                    "color": STATE_COLORS[next_state],
                    "comment": comment,
                    "is_absorbing": False
                })
                current_state = next_state

        return trajectory


if __name__ == "__main__":
    sim = TrajectorySimulator()
    print("Simulating thread in r/science starting from 'Deep Dive':")
    path = sim.simulate_thread(start_state=0, community="science", random_seed=42)
    for p in path:
        print(f"Step {p['step']}: [{p['state_name']}] {p['comment']['body'][:80]}...")
