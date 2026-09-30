import unittest
import pandas as pd
from src.tree_parser import decompose_thread_to_chains


class TestTreeParser(unittest.TestCase):
    def test_tree_decomposition_logic(self):
        data = [
            {"comment_id": "c1", "parent_id": "thread_1", "thread_id": "thread_1", "subreddit": "science", "body": "Root reply", "score": 10, "created_utc": 10},
            {"comment_id": "c2", "parent_id": "c1", "thread_id": "thread_1", "subreddit": "science", "body": "Deep follow up", "score": 5, "created_utc": 20},
            {"comment_id": "c3", "parent_id": "c2", "thread_id": "thread_1", "subreddit": "science", "body": "Branch A end", "score": 2, "created_utc": 30},
            {"comment_id": "c4", "parent_id": "c2", "thread_id": "thread_1", "subreddit": "science", "body": "Branch B end", "score": 3, "created_utc": 40},
            {"comment_id": "c5", "parent_id": "thread_1", "thread_id": "thread_1", "subreddit": "science", "body": "Short branch start", "score": 1, "created_utc": 15},
            {"comment_id": "c6", "parent_id": "c5", "thread_id": "thread_1", "subreddit": "science", "body": "Short branch end", "score": 1, "created_utc": 25},
        ]
        df = pd.DataFrame(data)

        chains = decompose_thread_to_chains(df, "thread_1", min_chain_len=3)
        chains_df = pd.DataFrame(chains)

        self.assertEqual(chains_df["chain_id"].nunique(), 2)
        for _, group in chains_df.groupby("chain_id"):
            self.assertEqual(len(group), 3)
            timestamps = group["created_utc"].tolist()
            self.assertEqual(timestamps, sorted(timestamps))
            self.assertEqual(group["step_index"].tolist(), [0, 1, 2])
            self.assertEqual(group["is_terminal_comment"].tolist(), [False, False, True])

        # Verify path multiplicity weighting:
        # Edge (c1 -> c2) is shared across 2 paths -> weight = 0.5
        c1_steps = chains_df[chains_df["comment_id"] == "c1"]
        for _, row in c1_steps.iterrows():
            self.assertAlmostEqual(row["transition_weight"], 0.5, places=3)
        # Leaves c3 and c4 appear in exactly 1 chain -> terminal weight = 1.0
        leaf_steps = chains_df[chains_df["comment_id"].isin(["c3", "c4"])]
        for _, row in leaf_steps.iterrows():
            self.assertAlmostEqual(row["transition_weight"], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
