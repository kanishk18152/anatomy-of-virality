# Data Layer Specification (Module 1 Contract)

This document defines the schema contract and guidelines for obtaining and dropping your raw dataset into the project.

---

## 1. Where to Place the Data
Once you extract your data (CSV, Parquet, or JSON Lines), place it here:
```
data/raw/reddit_comments.csv
```
*(Or `.parquet` / `.jsonl`)*

---

## 2. Required Schema Contract

Every row must represent a single comment with these exact columns:

| Column Name | Data Type | Required | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `comment_id` | `String` | Yes | Unique ID of the comment (e.g., `ep1a2b3` or `t1_ep1a2b3`). |
| `parent_id` | `String` | Yes | Parent pointer. `t3_...` = root post; `t1_...` = reply to another comment. **Crucial for tree reconstruction**. |
| `thread_id` | `String` | Yes | Submission/Thread ID (or `link_id`). Used to group comments by conversation. |
| `subreddit` | `String` | Yes | Subreddit name (e.g., `science`, `wallstreetbets`). Used for cross-community benchmarking. |
| `author` | `String` | Optional | Author handle (used to verify bot filtering). |
| `body` | `String` | Yes | Raw text of the comment for NLP feature extraction. |
| `score` | `Integer` | Optional | Net karma/upvotes. |
| `created_utc` | `Integer` | Yes | Unix epoch timestamp. Used to order comments chronologically along reply chains. |

---

## 3. How to Run the Query in Google Cloud BigQuery

1. Open the [Google Cloud BigQuery Console](https://console.cloud.google.com/bigquery).
2. Ensure you are in a project (BigQuery gives **1 TB/month of free query scans**).
3. Paste the contents of `sql/extract_reddit.sql`.
4. Check the query validator in the top right: it will show approximately **15–25 GB** scanned.
5. Click **Run**.
6. Once the query completes:
   - Click **Save Results** $\to$ **CSV (local file)** or **Export to Google Drive / Cloud Storage**.
   - If downloading locally, save it directly to:
     `anatomy-of-virality/data/raw/reddit_comments.csv`

---

## 4. Quality Rules Built Into the Contract
- **No deleted comments**: Rows with `[deleted]` or `[removed]` are excluded so they do not break text NLP.
- **No AutoModerator bots**: Filtered so automated sticky rules don't skew Markov state transitions.
- **Thread Length**: Threads are filtered to 30–120 comments, ensuring enough conversational depth without blowing up memory.
