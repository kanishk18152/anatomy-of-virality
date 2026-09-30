"""
Module 2 (Lego Block: Conversational Tree Decomposition & Linear Path Extraction)
Responsible for parsing hierarchical Reddit discussion trees into linear
root-to-leaf conversational chains using NetworkX directed graphs.
"""

import os
from typing import List, Dict, Any
import networkx as nx
import pandas as pd
from src.data_loader import load_raw_comments


OUTPUT_CHAINS_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "conversation_chains.csv"
)
PLACEHOLDER_BODY = "__STRUCTURAL_PLACEHOLDER__"


def decompose_thread_to_chains(
    thread_df: pd.DataFrame,
    thread_id: str,
    min_chain_len: int = 3
) -> List[Dict[str, Any]]:
    """
    Constructs a directed graph of replies for a single thread and extracts
    all simple paths from the submission root to terminal leaf comments.
    
    Structural placeholder nodes (deleted/bot comments retained for hierarchy)
    participate in graph construction but are excluded from the final chain output.
    """
    G = nx.DiGraph()
    root_node = f"root_{thread_id}"
    G.add_node(root_node, created_utc=0)

    # Add all comment nodes (including structural placeholders)
    comment_lookup = {}
    for _, row in thread_df.iterrows():
        cid = str(row["comment_id"])
        comment_lookup[cid] = row.to_dict()
        G.add_node(cid)

    # Add edges parent -> child
    for cid, row_data in comment_lookup.items():
        pid = str(row_data["parent_id"])
        if pid == str(thread_id) or pid not in comment_lookup:
            G.add_edge(root_node, cid)
        else:
            G.add_edge(pid, cid)

    # Skip cycle detection entirely — if cycles exist from corrupt metadata,
    # affected nodes will be unreachable from root and safely skipped by
    # the NetworkXNoPath handler below.

    # Find terminal leaf nodes (out-degree == 0, excluding root)
    leaves = [n for n in G.nodes() if G.out_degree(n) == 0 and n != root_node]
    
    # 1. Collect all qualifying chains first
    raw_chains = []
    for leaf in leaves:
        try:
            path = nx.shortest_path(G, source=root_node, target=leaf)
            comment_ids = path[1:]  # Omit synthetic root

            # Build chain from real comments only (skip structural placeholders)
            chain_comments = [
                comment_lookup[cid] for cid in comment_ids
                if comment_lookup[cid].get("body") != PLACEHOLDER_BODY
            ]

            if len(chain_comments) >= min_chain_len:
                chain_comments.sort(key=lambda x: x["created_utc"])
                raw_chains.append(chain_comments)

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    if not raw_chains:
        return []

    # 2. Compute edge multiplicity across all chains in this thread to eliminate pseudoreplication.
    # Every shared edge (u -> v) is weighted by 1 / (number of paths sharing it),
    # ensuring every unique conversation step receives exactly unit weight (1.0).
    from collections import Counter
    edge_counts = Counter()
    terminal_counts = Counter()

    for ch in raw_chains:
        for i in range(len(ch) - 1):
            edge_counts[(ch[i]["comment_id"], ch[i+1]["comment_id"])] += 1
        terminal_counts[ch[-1]["comment_id"]] += 1

    # 3. Format into structured records with inverse multiplicity weights
    extracted_chains = []
    for chain_idx, chain_comments in enumerate(raw_chains):
        chain_uid = f"{thread_id}_ch{chain_idx}"
        chain_len = len(chain_comments)

        for step, c in enumerate(chain_comments):
            record = dict(c)
            record["chain_id"] = chain_uid
            record["step_index"] = step
            record["chain_length"] = chain_len
            record["is_terminal_comment"] = (step == chain_len - 1)

            if step < chain_len - 1:
                edge_key = (c["comment_id"], chain_comments[step + 1]["comment_id"])
                record["transition_weight"] = round(1.0 / edge_counts[edge_key], 6)
            else:
                record["transition_weight"] = round(1.0 / terminal_counts[c["comment_id"]], 6)

            extracted_chains.append(record)

    return extracted_chains


def extract_all_chains(
    df: pd.DataFrame,
    min_chain_len: int = 3
) -> pd.DataFrame:
    """
    Decomposes all threads in the DataFrame into linear root-to-leaf chains.
    """
    all_chain_records = []
    grouped = df.groupby("thread_id")
    print(f"Decomposing {len(grouped)} discussion trees into linear reply chains (min_length={min_chain_len})...")

    for thread_id, thread_df in grouped:
        chains = decompose_thread_to_chains(thread_df, str(thread_id), min_chain_len=min_chain_len)
        all_chain_records.extend(chains)

    chains_df = pd.DataFrame(all_chain_records)
    if chains_df.empty:
        raise ValueError("No conversational chains met the minimum length requirement.")

    print(f"[✓] Extracted {chains_df['chain_id'].nunique():,} distinct conversational chains.")
    print(f"[✓] Total sequenced comments across chains: {len(chains_df):,}")

    # Display breakdown by subreddit
    sub_breakdown = chains_df.groupby("subreddit")["chain_id"].nunique()
    print("\nChains per Subreddit:")
    for sub, count in sub_breakdown.items():
        print(f"  r/{sub}: {count:,} chains")

    return chains_df


def run_tree_parser(input_csv: str = None, output_csv: str = OUTPUT_CHAINS_CSV) -> pd.DataFrame:
    if input_csv is None:
        input_csv = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "reddit_comments.csv")

    raw_df = load_raw_comments(input_csv)
    chains_df = extract_all_chains(raw_df, min_chain_len=3)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    chains_df.to_csv(output_csv, index=False)
    print(f"[✓] Saved linear reply chains to: {output_csv}")
    return chains_df


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else None
    run_tree_parser(src)
