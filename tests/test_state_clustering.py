import unittest
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from src.state_clustering import canonicalize_clusters


class TestStateClustering(unittest.TestCase):
    def test_canonicalize_clusters(self):
        n_per_cluster = 30
        c0 = pd.DataFrame({"PC1": np.random.normal(3, 0.2, n_per_cluster), "PC2": np.random.normal(0, 0.2, n_per_cluster), "q_marks": 0, "polarity": 0.1, "word_count": 80})
        c1 = pd.DataFrame({"PC1": np.random.normal(0, 0.2, n_per_cluster), "PC2": np.random.normal(0, 0.2, n_per_cluster), "q_marks": 3, "polarity": 0.0, "word_count": 25})
        c2 = pd.DataFrame({"PC1": np.random.normal(0, 0.2, n_per_cluster), "PC2": np.random.normal(3, 0.2, n_per_cluster), "q_marks": 0, "polarity": -0.5, "word_count": 30})
        c3 = pd.DataFrame({"PC1": np.random.normal(-2, 0.2, n_per_cluster), "PC2": np.random.normal(-1, 0.2, n_per_cluster), "q_marks": 0, "polarity": 0.0, "word_count": 5})

        df = pd.concat([c0, c1, c2, c3], ignore_index=True)
        agg = AgglomerativeClustering(n_clusters=4, linkage="ward")
        X = df[["PC1", "PC2"]].values
        raw_labels = agg.fit_predict(X)
        centers = np.array([X[raw_labels == i].mean(axis=0) for i in range(4)])

        canonical_labels, _ = canonicalize_clusters(df, raw_labels, centers)
        self.assertEqual(set(canonical_labels), {0, 1, 2, 3})


if __name__ == "__main__":
    unittest.main()
