"""
Sandbox SQLite database for integration testing.

Provides a simplified schema matching the PostgreSQL design,
with all data isolated in a temporary database file.
"""

from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional


# =============================================================================
# Schema Definition
# =============================================================================

SCHEMA_SQL = """
-- Snapshots (pipeline runs)
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    posts_fetched INTEGER DEFAULT 0,
    agents_fetched INTEGER DEFAULT 0,
    submolts_fetched INTEGER DEFAULT 0,
    comments_fetched INTEGER DEFAULT 0,
    evals_completed INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running'
);

-- Agents
CREATE TABLE IF NOT EXISTS agents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_external_id TEXT UNIQUE NOT NULL,
    handle TEXT NOT NULL,
    display_name TEXT,
    bio TEXT,
    avatar_url TEXT,
    karma INTEGER,
    follower_count INTEGER,
    is_claimed INTEGER,
    created_at TEXT,
    fetched_at TEXT NOT NULL,
    snapshot_id INTEGER REFERENCES snapshots(id)
);

-- Submolts
CREATE TABLE IF NOT EXISTS submolts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submolt_external_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    display_name TEXT,
    description TEXT,
    subscriber_count INTEGER,
    post_count INTEGER,
    owner_handle TEXT,
    created_at TEXT,
    fetched_at TEXT NOT NULL,
    snapshot_id INTEGER REFERENCES snapshots(id)
);

-- Posts
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_external_id TEXT UNIQUE NOT NULL,
    title TEXT,
    body_text TEXT,
    outbound_url TEXT,
    upvotes INTEGER,
    downvotes INTEGER,
    score INTEGER,
    comment_count INTEGER,
    author_handle TEXT,
    author_external_id TEXT,
    submolt_slug TEXT,
    submolt_external_id TEXT,
    permalink TEXT,
    created_at TEXT,
    fetched_at TEXT NOT NULL,
    snapshot_id INTEGER REFERENCES snapshots(id)
);

-- Comments
CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    comment_external_id TEXT UNIQUE NOT NULL,
    post_external_id TEXT NOT NULL,
    parent_comment_external_id TEXT,
    body_text TEXT,
    upvotes INTEGER,
    downvotes INTEGER,
    score INTEGER,
    author_handle TEXT,
    author_external_id TEXT,
    created_at TEXT,
    fetched_at TEXT NOT NULL,
    snapshot_id INTEGER REFERENCES snapshots(id),
    FOREIGN KEY (post_external_id) REFERENCES posts(post_external_id)
);

-- Post Evaluations
CREATE TABLE IF NOT EXISTS post_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_external_id TEXT NOT NULL,
    transcript_id TEXT,
    model TEXT NOT NULL,
    harm_enablement_score INTEGER,
    harm_enablement_confidence REAL,
    deception_score INTEGER,
    deception_confidence REAL,
    power_seeking_score INTEGER,
    power_seeking_confidence REAL,
    sycophancy_score INTEGER,
    sycophancy_confidence REAL,
    notes TEXT,
    raw_result TEXT,  -- JSON blob
    evaluated_at TEXT NOT NULL,
    snapshot_id INTEGER REFERENCES snapshots(id),
    FOREIGN KEY (post_external_id) REFERENCES posts(post_external_id)
);

-- Comment Evaluations
CREATE TABLE IF NOT EXISTS comment_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    comment_external_id TEXT NOT NULL,
    post_external_id TEXT NOT NULL,
    model TEXT NOT NULL,
    harm_enablement_score INTEGER,
    deception_score INTEGER,
    power_seeking_score INTEGER,
    sycophancy_score INTEGER,
    evaluated_at TEXT NOT NULL,
    snapshot_id INTEGER REFERENCES snapshots(id),
    FOREIGN KEY (comment_external_id) REFERENCES comments(comment_external_id)
);

-- Agent Score History
CREATE TABLE IF NOT EXISTS agent_score_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_external_id TEXT NOT NULL,
    agent_handle TEXT,
    snapshot_id INTEGER REFERENCES snapshots(id),
    posts_evaluated INTEGER,
    comments_evaluated INTEGER,
    harm_mean REAL,
    harm_max INTEGER,
    deception_mean REAL,
    deception_max INTEGER,
    power_seeking_mean REAL,
    power_seeking_max INTEGER,
    sycophancy_mean REAL,
    sycophancy_max INTEGER,
    overall_mean REAL,
    has_high_risk INTEGER,
    recorded_at TEXT NOT NULL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at);
CREATE INDEX IF NOT EXISTS idx_posts_author ON posts(author_handle);
CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(post_external_id);
CREATE INDEX IF NOT EXISTS idx_evals_post ON post_evaluations(post_external_id);
CREATE INDEX IF NOT EXISTS idx_agent_scores_agent ON agent_score_history(agent_external_id);
"""


# =============================================================================
# Database Class
# =============================================================================

class SandboxDatabase:
    """
    SQLite database for integration testing.
    
    Usage:
        db = SandboxDatabase(Path("/tmp/test.db"))
        db.initialize()
        
        with db.transaction() as conn:
            db.insert_post(conn, post_data)
        
        db.close()
    """
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
    
    def initialize(self) -> None:
        """Create database and tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        conn.close()
        
        print(f"📦 Initialized SQLite database at {self.db_path}")
    
    def connect(self) -> sqlite3.Connection:
        """Get a database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database transactions."""
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    # =========================================================================
    # Snapshot Operations
    # =========================================================================
    
    def create_snapshot(self, run_id: str) -> int:
        """Create a new snapshot and return its ID."""
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO snapshots (run_id, started_at, status)
                VALUES (?, ?, 'running')
                """,
                (run_id, datetime.now(timezone.utc).isoformat())
            )
            return cursor.lastrowid
    
    def complete_snapshot(self, snapshot_id: int, stats: Dict[str, int]) -> None:
        """Mark a snapshot as complete with final stats."""
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE snapshots SET
                    completed_at = ?,
                    posts_fetched = ?,
                    agents_fetched = ?,
                    submolts_fetched = ?,
                    comments_fetched = ?,
                    evals_completed = ?,
                    status = 'completed'
                WHERE id = ?
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    stats.get("posts", 0),
                    stats.get("agents", 0),
                    stats.get("submolts", 0),
                    stats.get("comments", 0),
                    stats.get("evals", 0),
                    snapshot_id,
                )
            )
    
    def get_snapshot(self, snapshot_id: int) -> Optional[Dict[str, Any]]:
        """Get snapshot by ID."""
        conn = self.connect()
        row = conn.execute(
            "SELECT * FROM snapshots WHERE id = ?",
            (snapshot_id,)
        ).fetchone()
        return dict(row) if row else None
    
    # =========================================================================
    # Insert Operations
    # =========================================================================
    
    def insert_agent(self, agent: Dict[str, Any], snapshot_id: int) -> int:
        """Insert an agent, returning the row ID."""
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO agents (
                    agent_external_id, handle, display_name, bio, avatar_url,
                    karma, follower_count, is_claimed, created_at, fetched_at, snapshot_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent.get("agent_external_id"),
                    agent.get("handle"),
                    agent.get("display_name"),
                    agent.get("bio"),
                    agent.get("avatar_url"),
                    agent.get("karma"),
                    agent.get("followers") or agent.get("follower_count"),
                    1 if agent.get("is_claimed") else 0,
                    agent.get("created_at"),
                    datetime.now(timezone.utc).isoformat(),
                    snapshot_id,
                )
            )
            return cursor.lastrowid
    
    def insert_post(self, post: Dict[str, Any], snapshot_id: int) -> int:
        """Insert a post, returning the row ID."""
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO posts (
                    post_external_id, title, body_text, outbound_url,
                    upvotes, downvotes, score, comment_count,
                    author_handle, author_external_id,
                    submolt_slug, submolt_external_id, permalink,
                    created_at, fetched_at, snapshot_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    post.get("post_external_id"),
                    post.get("title"),
                    post.get("body_text"),
                    post.get("outbound_url"),
                    post.get("upvotes"),
                    post.get("downvotes"),
                    post.get("score"),
                    post.get("comment_count"),
                    post.get("author_handle"),
                    post.get("author_external_id"),
                    post.get("submolt_slug"),
                    post.get("submolt_external_id"),
                    post.get("permalink"),
                    post.get("created_at"),
                    datetime.now(timezone.utc).isoformat(),
                    snapshot_id,
                )
            )
            return cursor.lastrowid
    
    def insert_comment(self, comment: Dict[str, Any], snapshot_id: int) -> int:
        """Insert a comment, returning the row ID."""
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO comments (
                    comment_external_id, post_external_id, parent_comment_external_id,
                    body_text, upvotes, downvotes, score,
                    author_handle, author_external_id, created_at, fetched_at, snapshot_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    comment.get("comment_external_id"),
                    comment.get("post_external_id"),
                    comment.get("parent_comment_external_id"),
                    comment.get("body_text"),
                    comment.get("upvotes"),
                    comment.get("downvotes"),
                    comment.get("score"),
                    comment.get("author_handle"),
                    comment.get("author_external_id"),
                    comment.get("created_at"),
                    datetime.now(timezone.utc).isoformat(),
                    snapshot_id,
                )
            )
            return cursor.lastrowid
    
    def insert_submolt(self, submolt: Dict[str, Any], snapshot_id: int) -> int:
        """Insert a submolt, returning the row ID."""
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO submolts (
                    submolt_external_id, name, display_name, description,
                    subscriber_count, post_count, owner_handle,
                    created_at, fetched_at, snapshot_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    submolt.get("submolt_external_id"),
                    submolt.get("name"),
                    submolt.get("display_name"),
                    submolt.get("description"),
                    submolt.get("subscriber_count"),
                    submolt.get("post_count"),
                    submolt.get("owner_handle"),
                    submolt.get("created_at"),
                    datetime.now(timezone.utc).isoformat(),
                    snapshot_id,
                )
            )
            return cursor.lastrowid
    
    def insert_post_evaluation(self, eval_result: Dict[str, Any], snapshot_id: int) -> int:
        """Insert a post evaluation result."""
        scores = eval_result.get("scores", {})
        
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO post_evaluations (
                    post_external_id, transcript_id, model,
                    harm_enablement_score, harm_enablement_confidence,
                    deception_score, deception_confidence,
                    power_seeking_score, power_seeking_confidence,
                    sycophancy_score, sycophancy_confidence,
                    notes, raw_result, evaluated_at, snapshot_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    eval_result.get("post_id"),
                    eval_result.get("transcript_id"),
                    eval_result.get("model"),
                    scores.get("harm_enablement", {}).get("score"),
                    scores.get("harm_enablement", {}).get("confidence"),
                    scores.get("deception_or_evasion", {}).get("score"),
                    scores.get("deception_or_evasion", {}).get("confidence"),
                    scores.get("self_preservation_power_seeking", {}).get("score"),
                    scores.get("self_preservation_power_seeking", {}).get("confidence"),
                    scores.get("delusional_sycophancy", {}).get("score"),
                    scores.get("delusional_sycophancy", {}).get("confidence"),
                    eval_result.get("notes"),
                    json.dumps(eval_result),
                    datetime.now(timezone.utc).isoformat(),
                    snapshot_id,
                )
            )
            return cursor.lastrowid
    
    def insert_agent_score(self, record: Dict[str, Any], snapshot_id: int) -> int:
        """Insert an agent score history record."""
        dim_scores = record.get("dimension_scores", {})
        
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO agent_score_history (
                    agent_external_id, agent_handle, snapshot_id,
                    posts_evaluated, comments_evaluated,
                    harm_mean, harm_max,
                    deception_mean, deception_max,
                    power_seeking_mean, power_seeking_max,
                    sycophancy_mean, sycophancy_max,
                    overall_mean, has_high_risk, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.get("agent_id"),
                    record.get("agent_handle"),
                    snapshot_id,
                    record.get("posts_evaluated"),
                    record.get("comments_evaluated"),
                    dim_scores.get("harm_enablement", {}).get("mean_score"),
                    dim_scores.get("harm_enablement", {}).get("max_score"),
                    dim_scores.get("deception_or_evasion", {}).get("mean_score"),
                    dim_scores.get("deception_or_evasion", {}).get("max_score"),
                    dim_scores.get("self_preservation_power_seeking", {}).get("mean_score"),
                    dim_scores.get("self_preservation_power_seeking", {}).get("max_score"),
                    dim_scores.get("delusional_sycophancy", {}).get("mean_score"),
                    dim_scores.get("delusional_sycophancy", {}).get("max_score"),
                    record.get("overall_mean_score"),
                    1 if record.get("has_high_harm_enablement") or record.get("has_high_deception") else 0,
                    datetime.now(timezone.utc).isoformat(),
                )
            )
            return cursor.lastrowid
    
    # =========================================================================
    # Query Operations
    # =========================================================================
    
    def get_all_posts(self, snapshot_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all posts, optionally filtered by snapshot."""
        conn = self.connect()
        if snapshot_id:
            rows = conn.execute(
                "SELECT * FROM posts WHERE snapshot_id = ? ORDER BY created_at DESC",
                (snapshot_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM posts ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]
    
    def get_all_agents(self, snapshot_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all agents, optionally filtered by snapshot."""
        conn = self.connect()
        if snapshot_id:
            rows = conn.execute(
                "SELECT * FROM agents WHERE snapshot_id = ? ORDER BY karma DESC",
                (snapshot_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM agents ORDER BY karma DESC"
            ).fetchall()
        return [dict(row) for row in rows]
    
    def get_post_evaluations(self, snapshot_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all post evaluations."""
        conn = self.connect()
        if snapshot_id:
            rows = conn.execute(
                "SELECT * FROM post_evaluations WHERE snapshot_id = ?",
                (snapshot_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM post_evaluations").fetchall()
        return [dict(row) for row in rows]
    
    def get_agent_scores(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get agent score history."""
        conn = self.connect()
        if agent_id:
            rows = conn.execute(
                "SELECT * FROM agent_score_history WHERE agent_external_id = ? ORDER BY recorded_at",
                (agent_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM agent_score_history ORDER BY overall_mean DESC"
            ).fetchall()
        return [dict(row) for row in rows]
    
    def get_stats(self) -> Dict[str, int]:
        """Get overall database statistics."""
        conn = self.connect()
        
        stats = {}
        for table in ["snapshots", "agents", "posts", "comments", "submolts", 
                      "post_evaluations", "agent_score_history"]:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats[table] = count
        
        return stats
    
    def clear_all(self) -> None:
        """Clear all data (for test reset)."""
        with self.transaction() as conn:
            for table in ["agent_score_history", "comment_evaluations", "post_evaluations",
                          "comments", "posts", "submolts", "agents", "snapshots"]:
                conn.execute(f"DELETE FROM {table}")

