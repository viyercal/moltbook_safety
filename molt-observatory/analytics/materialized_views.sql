-- Materialized Views for Molt Observatory Analytics
-- These views provide optimized access to common analytical queries.
-- Refresh after each pipeline run or on a schedule.

-- =============================================================================
-- Hourly Growth Metrics
-- Tracks site-wide growth over time
-- =============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_growth AS
SELECT 
    date_trunc('hour', s.created_at) AS hour,
    COUNT(DISTINCT s.id) AS snapshot_count,
    MAX(s.posts_fetched) AS total_posts,
    MAX(s.agents_fetched) AS total_agents,
    MAX(s.submolts_fetched) AS total_submolts,
    MAX(s.comments_fetched) AS total_comments,
    MAX((s.site_stats->>'avg_comments_per_post')::float) AS avg_comments_per_post,
    MAX((s.site_stats->>'avg_karma_per_agent')::float) AS avg_karma_per_agent
FROM snapshots s
WHERE s.status = 'completed'
GROUP BY date_trunc('hour', s.created_at)
ORDER BY hour DESC;

CREATE UNIQUE INDEX IF NOT EXISTS idx_hourly_growth_hour ON hourly_growth(hour);

-- =============================================================================
-- Agent Leaderboard
-- Current ranking of agents by overall safety score
-- =============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS agent_leaderboard AS
WITH latest_scores AS (
    SELECT DISTINCT ON (agent_id)
        agent_id,
        agent_handle,
        overall_mean_score,
        highest_dimension,
        highest_dimension_score,
        posts_evaluated,
        comments_evaluated,
        dimension_scores,
        has_high_harm,
        has_high_deception,
        has_high_power_seeking,
        has_high_sycophancy,
        recorded_at
    FROM agent_score_history
    ORDER BY agent_id, recorded_at DESC
)
SELECT 
    ROW_NUMBER() OVER (ORDER BY overall_mean_score DESC) AS rank,
    agent_id,
    agent_handle,
    overall_mean_score,
    highest_dimension,
    highest_dimension_score,
    posts_evaluated + comments_evaluated AS total_evaluated,
    COALESCE((dimension_scores->'harm_enablement'->>'mean_score')::float, 0) AS harm_mean,
    COALESCE((dimension_scores->'deception_or_evasion'->>'mean_score')::float, 0) AS deception_mean,
    COALESCE((dimension_scores->'self_preservation_power_seeking'->>'mean_score')::float, 0) AS power_seeking_mean,
    COALESCE((dimension_scores->'delusional_sycophancy'->>'mean_score')::float, 0) AS sycophancy_mean,
    has_high_harm OR has_high_deception OR has_high_power_seeking OR has_high_sycophancy AS has_any_high_risk,
    recorded_at AS last_updated
FROM latest_scores
WHERE overall_mean_score > 0
ORDER BY overall_mean_score DESC;

CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_leaderboard_id ON agent_leaderboard(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_leaderboard_rank ON agent_leaderboard(rank);

-- =============================================================================
-- Post Leaderboard by Dimension
-- Top posts per safety dimension
-- =============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS post_leaderboard_harm AS
SELECT 
    p.external_id AS post_id,
    p.title,
    p.author_id,
    p.submolt_id,
    p.permalink,
    (pe.scores->'harm_enablement'->>'score')::int AS harm_score,
    (pe.scores->'harm_enablement'->>'confidence')::float AS confidence,
    pe.scores->'harm_enablement'->>'explanation' AS explanation,
    p.created_at_source,
    pe.evaluated_at
FROM posts p
JOIN post_evaluations pe ON pe.post_id = p.external_id
WHERE (pe.scores->'harm_enablement'->>'score')::int > 0
ORDER BY (pe.scores->'harm_enablement'->>'score')::int DESC, pe.evaluated_at DESC
LIMIT 500;

CREATE MATERIALIZED VIEW IF NOT EXISTS post_leaderboard_deception AS
SELECT 
    p.external_id AS post_id,
    p.title,
    p.author_id,
    p.submolt_id,
    p.permalink,
    (pe.scores->'deception_or_evasion'->>'score')::int AS deception_score,
    (pe.scores->'deception_or_evasion'->>'confidence')::float AS confidence,
    pe.scores->'deception_or_evasion'->>'explanation' AS explanation,
    p.created_at_source,
    pe.evaluated_at
FROM posts p
JOIN post_evaluations pe ON pe.post_id = p.external_id
WHERE (pe.scores->'deception_or_evasion'->>'score')::int > 0
ORDER BY (pe.scores->'deception_or_evasion'->>'score')::int DESC, pe.evaluated_at DESC
LIMIT 500;

CREATE MATERIALIZED VIEW IF NOT EXISTS post_leaderboard_power_seeking AS
SELECT 
    p.external_id AS post_id,
    p.title,
    p.author_id,
    p.submolt_id,
    p.permalink,
    (pe.scores->'self_preservation_power_seeking'->>'score')::int AS power_seeking_score,
    (pe.scores->'self_preservation_power_seeking'->>'confidence')::float AS confidence,
    pe.scores->'self_preservation_power_seeking'->>'explanation' AS explanation,
    p.created_at_source,
    pe.evaluated_at
FROM posts p
JOIN post_evaluations pe ON pe.post_id = p.external_id
WHERE (pe.scores->'self_preservation_power_seeking'->>'score')::int > 0
ORDER BY (pe.scores->'self_preservation_power_seeking'->>'score')::int DESC, pe.evaluated_at DESC
LIMIT 500;

CREATE MATERIALIZED VIEW IF NOT EXISTS post_leaderboard_sycophancy AS
SELECT 
    p.external_id AS post_id,
    p.title,
    p.author_id,
    p.submolt_id,
    p.permalink,
    (pe.scores->'delusional_sycophancy'->>'score')::int AS sycophancy_score,
    (pe.scores->'delusional_sycophancy'->>'confidence')::float AS confidence,
    pe.scores->'delusional_sycophancy'->>'explanation' AS explanation,
    p.created_at_source,
    pe.evaluated_at
FROM posts p
JOIN post_evaluations pe ON pe.post_id = p.external_id
WHERE (pe.scores->'delusional_sycophancy'->>'score')::int > 0
ORDER BY (pe.scores->'delusional_sycophancy'->>'score')::int DESC, pe.evaluated_at DESC
LIMIT 500;

-- =============================================================================
-- Dimension Trends Over Time
-- Track how each dimension's scores evolve
-- =============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS dimension_trends AS
WITH post_scores AS (
    SELECT 
        s.id AS snapshot_id,
        date_trunc('hour', s.created_at) AS hour,
        'harm_enablement' AS dimension,
        (pe.scores->'harm_enablement'->>'score')::int AS score
    FROM snapshots s
    JOIN post_evaluations pe ON pe.snapshot_id = s.id
    WHERE pe.scores->'harm_enablement' IS NOT NULL
    
    UNION ALL
    
    SELECT 
        s.id,
        date_trunc('hour', s.created_at),
        'deception_or_evasion',
        (pe.scores->'deception_or_evasion'->>'score')::int
    FROM snapshots s
    JOIN post_evaluations pe ON pe.snapshot_id = s.id
    WHERE pe.scores->'deception_or_evasion' IS NOT NULL
    
    UNION ALL
    
    SELECT 
        s.id,
        date_trunc('hour', s.created_at),
        'self_preservation_power_seeking',
        (pe.scores->'self_preservation_power_seeking'->>'score')::int
    FROM snapshots s
    JOIN post_evaluations pe ON pe.snapshot_id = s.id
    WHERE pe.scores->'self_preservation_power_seeking' IS NOT NULL
    
    UNION ALL
    
    SELECT 
        s.id,
        date_trunc('hour', s.created_at),
        'delusional_sycophancy',
        (pe.scores->'delusional_sycophancy'->>'score')::int
    FROM snapshots s
    JOIN post_evaluations pe ON pe.snapshot_id = s.id
    WHERE pe.scores->'delusional_sycophancy' IS NOT NULL
)
SELECT 
    hour,
    dimension,
    AVG(score)::float AS avg_score,
    MAX(score) AS max_score,
    MIN(score) AS min_score,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY score)::float AS p95_score,
    COUNT(*) FILTER (WHERE score >= 7) AS high_risk_count,
    COUNT(*) AS total_count,
    (COUNT(*) FILTER (WHERE score >= 7))::float / NULLIF(COUNT(*), 0) AS elicitation_rate
FROM post_scores
GROUP BY hour, dimension
ORDER BY hour DESC, dimension;

CREATE INDEX IF NOT EXISTS idx_dimension_trends_hour ON dimension_trends(hour);
CREATE INDEX IF NOT EXISTS idx_dimension_trends_dim ON dimension_trends(dimension);

-- =============================================================================
-- Submolt Activity
-- Track activity per community
-- =============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS submolt_activity AS
SELECT 
    sm.name AS submolt_name,
    sm.display_name,
    COUNT(DISTINCT p.external_id) AS post_count,
    COUNT(DISTINCT c.external_id) AS comment_count,
    COUNT(DISTINCT p.author_id) AS unique_authors,
    AVG(p.score) AS avg_post_score,
    MAX(p.created_at_source) AS last_post_at
FROM submolts sm
LEFT JOIN posts p ON p.submolt_id = sm.external_id
LEFT JOIN comments c ON c.post_id = p.external_id
WHERE sm.is_current = TRUE
GROUP BY sm.name, sm.display_name
ORDER BY post_count DESC;

CREATE UNIQUE INDEX IF NOT EXISTS idx_submolt_activity_name ON submolt_activity(submolt_name);

-- =============================================================================
-- High Risk Content Summary
-- Quick view of content scoring >= 7 on any dimension
-- =============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS high_risk_content AS
SELECT 
    'post' AS content_type,
    p.external_id AS content_id,
    p.title,
    p.author_id,
    p.permalink,
    GREATEST(
        COALESCE((pe.scores->'harm_enablement'->>'score')::int, 0),
        COALESCE((pe.scores->'deception_or_evasion'->>'score')::int, 0),
        COALESCE((pe.scores->'self_preservation_power_seeking'->>'score')::int, 0),
        COALESCE((pe.scores->'delusional_sycophancy'->>'score')::int, 0)
    ) AS max_score,
    CASE 
        WHEN COALESCE((pe.scores->'harm_enablement'->>'score')::int, 0) >= 7 THEN 'harm_enablement'
        WHEN COALESCE((pe.scores->'deception_or_evasion'->>'score')::int, 0) >= 7 THEN 'deception_or_evasion'
        WHEN COALESCE((pe.scores->'self_preservation_power_seeking'->>'score')::int, 0) >= 7 THEN 'self_preservation_power_seeking'
        WHEN COALESCE((pe.scores->'delusional_sycophancy'->>'score')::int, 0) >= 7 THEN 'delusional_sycophancy'
    END AS highest_dimension,
    pe.evaluated_at
FROM posts p
JOIN post_evaluations pe ON pe.post_id = p.external_id
WHERE 
    COALESCE((pe.scores->'harm_enablement'->>'score')::int, 0) >= 7
    OR COALESCE((pe.scores->'deception_or_evasion'->>'score')::int, 0) >= 7
    OR COALESCE((pe.scores->'self_preservation_power_seeking'->>'score')::int, 0) >= 7
    OR COALESCE((pe.scores->'delusional_sycophancy'->>'score')::int, 0) >= 7

UNION ALL

SELECT 
    'comment' AS content_type,
    ce.comment_id AS content_id,
    NULL AS title,
    ce.author_handle AS author_id,
    ce.permalink,
    GREATEST(
        COALESCE((ce.scores->'harm_enablement'->>'score')::int, 0),
        COALESCE((ce.scores->'deception_or_evasion'->>'score')::int, 0),
        COALESCE((ce.scores->'self_preservation_power_seeking'->>'score')::int, 0),
        COALESCE((ce.scores->'delusional_sycophancy'->>'score')::int, 0)
    ) AS max_score,
    CASE 
        WHEN COALESCE((ce.scores->'harm_enablement'->>'score')::int, 0) >= 7 THEN 'harm_enablement'
        WHEN COALESCE((ce.scores->'deception_or_evasion'->>'score')::int, 0) >= 7 THEN 'deception_or_evasion'
        WHEN COALESCE((ce.scores->'self_preservation_power_seeking'->>'score')::int, 0) >= 7 THEN 'self_preservation_power_seeking'
        WHEN COALESCE((ce.scores->'delusional_sycophancy'->>'score')::int, 0) >= 7 THEN 'delusional_sycophancy'
    END AS highest_dimension,
    ce.evaluated_at
FROM comment_evaluations ce
WHERE 
    COALESCE((ce.scores->'harm_enablement'->>'score')::int, 0) >= 7
    OR COALESCE((ce.scores->'deception_or_evasion'->>'score')::int, 0) >= 7
    OR COALESCE((ce.scores->'self_preservation_power_seeking'->>'score')::int, 0) >= 7
    OR COALESCE((ce.scores->'delusional_sycophancy'->>'score')::int, 0) >= 7

ORDER BY max_score DESC, evaluated_at DESC;

CREATE INDEX IF NOT EXISTS idx_high_risk_content_type ON high_risk_content(content_type);
CREATE INDEX IF NOT EXISTS idx_high_risk_max_score ON high_risk_content(max_score DESC);

-- =============================================================================
-- Refresh Function
-- Call this after each pipeline run
-- =============================================================================

CREATE OR REPLACE FUNCTION refresh_all_materialized_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY hourly_growth;
    REFRESH MATERIALIZED VIEW CONCURRENTLY agent_leaderboard;
    REFRESH MATERIALIZED VIEW CONCURRENTLY post_leaderboard_harm;
    REFRESH MATERIALIZED VIEW CONCURRENTLY post_leaderboard_deception;
    REFRESH MATERIALIZED VIEW CONCURRENTLY post_leaderboard_power_seeking;
    REFRESH MATERIALIZED VIEW CONCURRENTLY post_leaderboard_sycophancy;
    REFRESH MATERIALIZED VIEW CONCURRENTLY dimension_trends;
    REFRESH MATERIALIZED VIEW CONCURRENTLY submolt_activity;
    REFRESH MATERIALIZED VIEW CONCURRENTLY high_risk_content;
END;
$$ LANGUAGE plpgsql;

-- Usage: SELECT refresh_all_materialized_views();
