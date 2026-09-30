import unittest
import numpy as np
import pandas as pd
from src.markov_engine import (
    extract_markov_transitions,
    compute_absorbing_markov_properties,
    bootstrap_markov_properties,
    permutation_test_matrix_homogeneity
)


class TestMarkovEngine(unittest.TestCase):
    def test_absorbing_markov_calculus(self):
        df = pd.DataFrame([
            {"chain_id": "c1", "thread_id": "t1", "step_index": 0, "state": 0, "subreddit": "test", "transition_weight": 0.5},
            {"chain_id": "c1", "thread_id": "t1", "step_index": 1, "state": 1, "subreddit": "test", "transition_weight": 1.0},
            {"chain_id": "c1", "thread_id": "t1", "step_index": 2, "state": 2, "subreddit": "test", "transition_weight": 1.0},
            {"chain_id": "c2", "thread_id": "t2", "step_index": 0, "state": 2, "subreddit": "test", "transition_weight": 1.0},
            {"chain_id": "c2", "thread_id": "t2", "step_index": 1, "state": 2, "subreddit": "test", "transition_weight": 1.0},
            {"chain_id": "c2", "thread_id": "t2", "step_index": 2, "state": 3, "subreddit": "test", "transition_weight": 1.0},
        ])

        transitions = extract_markov_transitions(df)
        self.assertEqual(len(transitions), 6)
        self.assertTrue("transition_weight" in transitions.columns)
        self.assertTrue("thread_id" in transitions.columns)

        metrics = compute_absorbing_markov_properties(transitions, n_transient=4)
        P = metrics["P_matrix"].values
        N = metrics["Fundamental_Matrix_N"].values
        tau = metrics["Expected_Comments_To_Death"].values

        np.testing.assert_allclose(P.sum(axis=1), np.ones(5), atol=1e-5)
        self.assertEqual(P[4, 4], 1.0)
        self.assertTrue(np.all(N >= 0.0))
        self.assertTrue(np.all(tau >= 1.0))

    def test_bootstrap_and_permutation_testing(self):
        # Create multi-thread dataset for 2 communities
        records = []
        for t_idx in range(10):
            sub = "comm_A" if t_idx < 5 else "comm_B"
            for step in range(3):
                records.append({
                    "chain_id": f"t{t_idx}_ch0",
                    "thread_id": f"t{t_idx}",
                    "step_index": step,
                    "state": (step + t_idx) % 4,
                    "subreddit": sub,
                    "transition_weight": 1.0
                })
        df = pd.DataFrame(records)
        transitions = extract_markov_transitions(df)

        # 1. Test Block Bootstrap
        boot_res = bootstrap_markov_properties(transitions, n_bootstraps=20, alpha=0.05, random_state=42)
        tau_inf = boot_res["tau_inference"]
        self.assertEqual(len(tau_inf), 4)
        self.assertTrue(np.all(tau_inf["CI_Lower_95"] <= tau_inf["CI_Upper_95"]))

        # 2. Test Permutation Test
        perm_res = permutation_test_matrix_homogeneity(transitions, "comm_A", "comm_B", n_permutations=20, random_state=42)
        self.assertTrue(0.0 <= perm_res["p_value"] <= 1.0)
        self.assertGreaterEqual(perm_res["frobenius_distance_obs"], 0.0)


if __name__ == "__main__":
    unittest.main()
