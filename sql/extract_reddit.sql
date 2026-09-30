-- ==============================================================================
-- The Anatomy of Virality: BigQuery Extraction Query
-- Target Dataset: fh-bigquery.reddit_comments (partitioned monthly tables)
-- Target Communities: r/science (high-signal/intellectual) vs. r/wallstreetbets (volatile/tension)
-- Approximate scan size: ~15-25 GB (well within Google BigQuery's 1 TB/month free tier)
-- ==============================================================================

WITH TargetSubmissions AS (
  -- Step 1: Select balanced discussion threads with meaningful depth (30 to 120 comments)
  SELECT 
    link_id,
    subreddit,
    COUNT(1) AS total_comments
  FROM 
    `fh-bigquery.reddit_comments.2019_05` -- Replace month shard if desired (e.g. 2019_06, 2018_12)
  WHERE 
    LOWER(subreddit) IN ('science', 'wallstreetbets')
    AND body IS NOT NULL
    AND body NOT IN ('[deleted]', '[removed]')
    AND author NOT IN ('AutoModerator', '[deleted]')
  GROUP BY 
    link_id, subreddit
  HAVING 
    total_comments BETWEEN 30 AND 120
  LIMIT 400
)

-- Step 2: Extract every comment belonging to the selected threads with parent-child linkage
SELECT 
  c.id AS comment_id,
  c.parent_id AS parent_id,            -- Contains t1_ prefix (comment) or t3_ prefix (thread root)
  c.link_id AS thread_id,              -- Target submission identifier
  c.subreddit AS subreddit,            -- Forum category
  c.author AS author,                  -- Filtered out bots and deleted accounts
  c.body AS body,                      -- Comment text for NLP feature extraction
  c.score AS score,                    -- Upvotes / community resonance
  c.created_utc AS created_utc         -- Unix timestamp for chronological sequence
FROM 
  `fh-bigquery.reddit_comments.2019_05` c
JOIN 
  TargetSubmissions s ON c.link_id = s.link_id
WHERE 
  c.body IS NOT NULL
  AND c.body NOT IN ('[deleted]', '[removed]')
  AND c.author NOT IN ('AutoModerator', '[deleted]')
ORDER BY 
  c.link_id, c.created_utc ASC;
