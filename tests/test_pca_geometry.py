import unittest
import numpy as np
import pandas as pd
from src.pca_geometry import AlignedPCA


class TestPCAGeometry(unittest.TestCase):
    def test_aligned_pca_properties(self):
        np.random.seed(42)
        n = 150
        df = pd.DataFrame({
            "polarity": np.random.uniform(-1, 1, n),
            "subjectivity": np.random.uniform(0, 1, n),
            "word_count": np.random.exponential(40, n),
            "lexical_diversity": np.random.uniform(1, 8, n),
            "q_marks": np.random.poisson(0.5, n),
            "excl_marks": np.random.poisson(0.3, n)
        })

        model = AlignedPCA(n_components=4)
        coords = model.fit_transform(df)

        self.assertEqual(coords.shape, (n, 4))
        loadings = model.loadings
        # PC1: word_count should be positive (Analytical Depth)
        self.assertGreater(loadings.loc["word_count", loadings.columns[0]], 0)
        # PC2: polarity should be positive (Emotional Valence)
        self.assertGreater(loadings.loc["polarity", loadings.columns[1]], 0)
        # PC3: excl_marks should be positive (Agitation Intensity)
        self.assertGreater(loadings.loc["excl_marks", loadings.columns[2]], 0)
        # PC4: q_marks should be positive (Inquisitive Engagement)
        self.assertGreater(loadings.loc["q_marks", loadings.columns[3]], 0)
        # Variance ratios should sum to <= 1.0
        self.assertLessEqual(sum(model.explained_variance_ratio_), 1.0)


if __name__ == "__main__":
    unittest.main()
