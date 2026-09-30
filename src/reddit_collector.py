"""
Lego Block: Real Reddit Data Collector
Fetches real discussion trees from target subreddits (r/science vs. r/wallstreetbets)
using Reddit's public JSON endpoints, preserving full recursive comment hierarchies.
"""

import json
import os
import sys
import time
import urllib.request
from typing import List, Dict, Any
import csv


SUBREDDITS = ["science", "wallstreetbets"]
HEADERS = {
    "User-Agent": "academic:anatomy_of_virality:v1.0 (by /u/researcher_slmmr)"
}

OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "reddit_comments.csv")


def fetch_json(url: str) -> Any:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_comments_recursive(comment_node: Dict[str, Any], thread_id: str, subreddit: str) -> List[Dict[str, Any]]:
    """
    Recursively unpacks Reddit comment trees and captures exact parent-child links.
    """
    extracted = []
    data = comment_node.get("data", {})
    body = data.get("body", "")
    author = data.get("author", "")

    # Filter out empty, deleted, removed, or bot comments
    if body and body not in ["[deleted]", "[removed]"] and author not in ["AutoModerator", "[deleted]"]:
        extracted.append({
            "comment_id": data.get("id", ""),
            "parent_id": data.get("parent_id", ""),  # e.g. t1_... or t3_...
            "thread_id": thread_id,
            "subreddit": subreddit.lower(),
            "author": author,
            "body": body,
            "score": data.get("score", 0),
            "created_utc": int(data.get("created_utc", 0))
        })

    # Recurse into nested replies
    replies = data.get("replies")
    if isinstance(replies, dict):
        reply_children = replies.get("data", {}).get("children", [])
        for child in reply_children:
            if child.get("kind") == "t1":  # t1 = comment
                extracted.extend(extract_comments_recursive(child, thread_id, subreddit))

    return extracted


def harvest_subreddit_threads(subreddit: str, max_threads: int = 30) -> List[Dict[str, Any]]:
    """
    Fetches the top monthly discussions for a subreddit and extracts their full trees.
    """
    print(f"\n[+] Harvesting {max_threads} active threads from r/{subreddit}...")
    listing_url = f"https://www.reddit.com/r/{subreddit}/top.json?limit=50&t=month"
    
    try:
        listing_data = fetch_json(listing_url)
    except Exception as e:
        print(f"[-] Failed to fetch listing for r/{subreddit}: {e}")
        return []

    posts = listing_data.get("data", {}).get("children", [])
    all_comments = []
    collected_threads = 0

    for post in posts:
        post_data = post.get("data", {})
        num_comments = post_data.get("num_comments", 0)
        thread_id = post_data.get("id", "")
        title = post_data.get("title", "")[:60]

        # Target threads with conversational depth (30 to 150 comments)
        if 25 <= num_comments <= 200 and not post_data.get("stickied", False):
            thread_url = f"https://www.reddit.com/r/{subreddit}/comments/{thread_id}.json?limit=100"
            try:
                thread_json = fetch_json(thread_url)
                if len(thread_json) >= 2:
                    comment_forest = thread_json[1].get("data", {}).get("children", [])
                    thread_comments = []
                    for comment_tree in comment_forest:
                        if comment_tree.get("kind") == "t1":
                            thread_comments.extend(
                                extract_comments_recursive(comment_tree, thread_id, subreddit)
                            )
                    
                    if len(thread_comments) >= 15:
                        all_comments.extend(thread_comments)
                        collected_threads += 1
                        print(f"  ({collected_threads}/{max_threads}) [{subreddit}] {title}... ({len(thread_comments)} comments)")
                
                # Polite rate limiting between thread calls
                time.sleep(1.2)
                
            except Exception as e:
                print(f"  [!] Skipped thread {thread_id}: {e}")
                time.sleep(2.0)

            if collected_threads >= max_threads:
                break

    return all_comments


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    all_rows = []

    for sub in SUBREDDITS:
        comments = harvest_subreddit_threads(sub, max_threads=25)
        all_rows.extend(comments)

    if not all_rows:
        print("\n[-] No comments retrieved. Check network connectivity.")
        sys.exit(1)

    # Save to CSV adhering to Module 1 contract
    keys = ["comment_id", "parent_id", "thread_id", "subreddit", "author", "body", "score", "created_utc"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[✓] Successfully collected {len(all_rows):,} comments across {len(set(r['thread_id'] for r in all_rows))} threads.")
    print(f"[✓] Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
