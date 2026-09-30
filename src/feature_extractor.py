"""
Module 3 (Lego Block: Linguistic Feature Extraction & Bias Mitigation)
Responsible for computing NLP metrics (sentiment polarity, subjectivity,
Root-TTR lexical diversity, inquiry, and agitation signals) for each sequenced comment.
"""

import math
import os
import re
from typing import Dict, Any
import pandas as pd
from textblob import TextBlob


INPUT_CHAINS_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "conversation_chains.csv"
)
OUTPUT_FEATURES_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "comment_features.csv"
)

FEATURE_COLUMNS = [
    "polarity",
    "subjectivity",
    "word_count",
    "lexical_diversity",
    "q_marks",
    "excl_marks"
]


def extract_comment_features(text: str) -> Dict[str, float]:
    """
    Computes 6 core linguistic metrics for a single comment body:
    1. Polarity: [-1.0, 1.0] (Tone from hostile to constructive)
    2. Subjectivity: [0.0, 1.0] (Emotional/opinionated vs factual)
    3. Word Count: Total number of whitespace-delimited tokens
    4. Lexical Diversity: Root Type-Token Ratio (Unique Words / sqrt(Total Words))
       Mitigates Zipf's Law length bias where long comments falsely look impoverished.
    5. Inquiry Density: Question marks per word (normalized by comment length)
    6. Agitation Density: Exclamation marks per word (normalized by comment length)
    """
    clean_text = str(text).strip() if pd.notnull(text) else ""
    if not clean_text or clean_text == "__STRUCTURAL_PLACEHOLDER__":
        return {
            "polarity": 0.0,
            "subjectivity": 0.0,
            "word_count": 0,
            "lexical_diversity": 0.0,
            "q_marks": 0.0,
            "excl_marks": 0.0
        }

    # Tokenize words using regex for clean alphanumeric tokens
    tokens = re.findall(r"\b\w+\b", clean_text.lower())
    wc = len(tokens)
    unique_wc = len(set(tokens))

    # Root-TTR (Guiraud's Index) to prevent word-count penalty
    root_ttr = (unique_wc / math.sqrt(wc)) if wc > 0 else 0.0

    # Punctuation density (per-word ratios, not raw counts)
    # This prevents longer comments from artificially inflating tension signals
    q_count = clean_text.count("?")
    excl_count = clean_text.count("!")
    q_density = q_count / max(wc, 1)
    excl_density = excl_count / max(wc, 1)

    # Sentiment analysis via TextBlob
    blob = TextBlob(clean_text)
    polarity = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)

    return {
        "polarity": round(polarity, 4),
        "subjectivity": round(subjectivity, 4),
        "word_count": wc,
        "lexical_diversity": round(root_ttr, 4),
        "q_marks": round(q_density, 4),
        "excl_marks": round(excl_density, 4)
    }


def extract_features_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature extraction across all rows of the conversational chains DataFrame.
    """
    print(f"Extracting linguistic features for {len(df):,} sequenced comments...")
    feature_records = []
    
    for idx, text in enumerate(df["body"]):
        feat = extract_comment_features(text)
        feature_records.append(feat)
        if (idx + 1) % 2000 == 0:
            print(f"  Processed {idx + 1:,} / {len(df):,} comments...")

    feat_df = pd.DataFrame(feature_records)
    
    # Concatenate features with original metadata
    result_df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)

    print("\nFeature Summary Statistics:")
    print(result_df[FEATURE_COLUMNS].describe().round(3))
    return result_df


def run_feature_extractor(
    input_csv: str = INPUT_CHAINS_CSV,
    output_csv: str = OUTPUT_FEATURES_CSV
) -> pd.DataFrame:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input chains file not found: {input_csv}. Run Module 2 first.")

    chains_df = pd.read_csv(input_csv)
    enriched_df = extract_features_from_dataframe(chains_df)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    enriched_df.to_csv(output_csv, index=False)
    print(f"\n[✓] Saved enriched features dataset to: {output_csv}")
    return enriched_df


if __name__ == "__main__":
    run_feature_extractor()
