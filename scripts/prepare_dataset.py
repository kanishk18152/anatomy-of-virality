"""
Module 1 Adapter: Real Reddit Corpus Processor
Converts the downloaded Cornell ConvoKit Reddit dataset into our standardized
Module 1 schema contract (data/raw/reddit_comments.csv).
"""

import json
import os
import csv
from collections import defaultdict


INPUT_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "reddit-corpus-small", "utterances.json")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "reddit_comments.csv")

# Contrasting communities for benchmarking:
# r/science (academic depth, structured discussion)
# r/politics (ideological tension, heated debate, high friction)
# r/cringe (sarcasm, casual banter, mockery)
DEFAULT_SUBREDDITS = ["science", "politics", "cringe"]


def process_corpus(subreddits=DEFAULT_SUBREDDITS, min_thread_comments=15, max_thread_comments=150):
    print(f"Reading {INPUT_JSON}...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        utterances = json.load(f)

    print(f"Total raw utterances in corpus: {len(utterances):,}")
    
    # Group ALL comments by (subreddit, thread_id), including deleted ones as
    # structural placeholders so tree_parser.py can preserve parent-child linkage.
    # Raw counts are tracked separately to filter threads by their true size.
    thread_comments_all = defaultdict(list)   # includes structural placeholders
    thread_raw_counts = defaultdict(int)       # raw count before any filtering
    
    for u in utterances:
        meta = u.get("meta", {})
        sub = meta.get("subreddit", "").lower()
        if sub in [s.lower() for s in subreddits]:
            thread_id = u.get("root", "")
            text = u.get("text", "").strip()
            thread_raw_counts[(sub, thread_id)] += 1

            reply_to = u.get("reply_to")
            parent_id = reply_to if reply_to else thread_id

            is_deleted = (not text or text in ["[deleted]", "[removed]"])
            is_bot = u.get("user") in ["AutoModerator", "[deleted]"]

            # Retain deleted/bot comments as structural placeholders with a
            # sentinel body so tree_parser preserves the hierarchy but
            # feature_extractor and clustering can skip them downstream.
            if is_deleted or is_bot:
                body = "__STRUCTURAL_PLACEHOLDER__"
            else:
                body = text

            thread_comments_all[(sub, thread_id)].append({
                "comment_id": u.get("id"),
                "parent_id": parent_id,
                "thread_id": thread_id,
                "subreddit": sub,
                "author": u.get("user", "[deleted]"),
                "body": body,
                "score": meta.get("score", 0),
                "created_utc": u.get("timestamp", 0)
            })

    # Filter threads by RAW comment count (before deleted-node removal)
    # to avoid artificially shrinking threads below the threshold.
    filtered_rows = []
    threads_kept = defaultdict(int)

    for (sub, thread_id), comments in thread_comments_all.items():
        raw_count = thread_raw_counts[(sub, thread_id)]
        if min_thread_comments <= raw_count <= max_thread_comments:
            # Sort comments chronologically
            comments.sort(key=lambda x: x["created_utc"])
            filtered_rows.extend(comments)
            threads_kept[sub] += 1

    print("\nExtraction Summary by Subreddit:")
    for sub, count in threads_kept.items():
        sub_comments = sum(1 for r in filtered_rows if r["subreddit"] == sub)
        print(f"  r/{sub}: {count} threads, {sub_comments:,} comments")

    print(f"\nTotal filtered comments ready for analysis: {len(filtered_rows):,}")

    # Write to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = ["comment_id", "parent_id", "thread_id", "subreddit", "author", "body", "score", "created_utc"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"[✓] Saved standardized CSV to: {OUTPUT_CSV}")
    return OUTPUT_CSV


if __name__ == "__main__":
    process_corpus()
