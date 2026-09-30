import unittest
from src.feature_extractor import extract_comment_features


class TestFeatureExtractor(unittest.TestCase):
    def test_feature_extraction_basics(self):
        # "What is the primary mechanism behind this result? Could it be quantum tunneling?"
        # 13 words, 2 question marks → q_marks = 2/13 ≈ 0.1538
        inquiry_text = "What is the primary mechanism behind this result? Could it be quantum tunneling?"
        feat_inquiry = extract_comment_features(inquiry_text)
        self.assertAlmostEqual(feat_inquiry["q_marks"], 2 / 13, places=3)
        self.assertAlmostEqual(feat_inquiry["excl_marks"], 0.0, places=3)
        self.assertGreater(feat_inquiry["word_count"], 5)

        # "This is completely ridiculous and infuriating! Absolutely absurd nonsense!"
        # 9 words, 2 exclamation marks → excl_marks = 2/9 ≈ 0.2222
        hostile_text = "This is completely ridiculous and infuriating! Absolutely absurd nonsense!"
        feat_hostile = extract_comment_features(hostile_text)
        self.assertAlmostEqual(feat_hostile["excl_marks"], 2 / 9, places=3)
        self.assertLess(feat_hostile["polarity"], 0.0)
        self.assertGreater(feat_hostile["subjectivity"], 0.5)

        long_rich_text = "Photosynthesis converts carbon dioxide and water into glucose and oxygen through chlorophyll pigments using sunlight energy efficiently."
        feat_rich = extract_comment_features(long_rich_text)
        self.assertGreater(feat_rich["lexical_diversity"], 2.0)

        feat_empty = extract_comment_features("")
        self.assertEqual(feat_empty["word_count"], 0)
        self.assertEqual(feat_empty["polarity"], 0.0)


if __name__ == "__main__":
    unittest.main()
