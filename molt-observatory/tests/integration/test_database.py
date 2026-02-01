"""
Integration tests for SQLite database storage.

Tests the SandboxDatabase with real scraped data.
All database operations and results are saved to the output directory.

Run with:
    cd molt-observatory
    python -m pytest tests/integration/test_database.py -v
"""

from __future__ import annotations
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import uuid

import pytest

from tests.integration.sandbox_db import SandboxDatabase


@pytest.mark.integration
class TestDatabaseInitialization:
    """Tests for database setup."""
    
    def test_creates_database_file(self, integration_db_path, test_output_dir):
        """Should create the SQLite database file."""
        db = SandboxDatabase(integration_db_path)
        db.initialize()
        
        assert integration_db_path.exists(), f"Database file not created: {integration_db_path}"
        
        # Save test result
        output_path = test_output_dir / "database" / "test_creates_database_file.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_creates_database_file",
                "db_path": str(integration_db_path),
                "file_size_bytes": integration_db_path.stat().st_size,
                "exists": True,
            }, f, indent=2)
        
        db.close()
        
        print(f"\n✅ Database created at {integration_db_path}")
        print(f"   File size: {integration_db_path.stat().st_size} bytes")
    
    def test_creates_all_tables(self, sandbox_db, test_output_dir):
        """Should create all required tables."""
        conn = sandbox_db.connect()
        
        # Query SQLite schema
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        
        table_names = {t[0] for t in tables}
        
        expected_tables = {
            "snapshots", "agents", "posts", "comments", "submolts",
            "post_evaluations", "comment_evaluations", "agent_score_history"
        }
        
        missing_tables = expected_tables - table_names
        assert len(missing_tables) == 0, f"Missing tables: {missing_tables}"
        
        # Save test result
        output_path = test_output_dir / "database" / "test_creates_all_tables.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_creates_all_tables",
                "expected_tables": list(expected_tables),
                "found_tables": list(table_names),
                "missing_tables": list(missing_tables),
            }, f, indent=2)
        
        print(f"\n✅ All tables created: {sorted(expected_tables)}")


@pytest.mark.integration
class TestSnapshotOperations:
    """Tests for snapshot (pipeline run) tracking."""
    
    def test_creates_snapshot(self, sandbox_db, test_output_dir):
        """Should create a new snapshot."""
        run_id = f"test-{uuid.uuid4().hex[:8]}"
        
        snapshot_id = sandbox_db.create_snapshot(run_id)
        
        assert snapshot_id > 0, "Snapshot ID should be positive"
        
        snapshot = sandbox_db.get_snapshot(snapshot_id)
        
        assert snapshot["run_id"] == run_id, f"Run ID mismatch: expected {run_id}"
        assert snapshot["status"] == "running", f"Status should be 'running', got {snapshot['status']}"
        
        # Save test result
        output_path = test_output_dir / "database" / "test_creates_snapshot.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_creates_snapshot",
                "run_id": run_id,
                "snapshot_id": snapshot_id,
                "snapshot": dict(snapshot),
            }, f, indent=2, default=str)
        
        print(f"\n✅ Created snapshot: {run_id} (ID: {snapshot_id})")
    
    def test_completes_snapshot(self, sandbox_db, test_output_dir):
        """Should mark snapshot as complete with stats."""
        run_id = f"test-{uuid.uuid4().hex[:8]}"
        snapshot_id = sandbox_db.create_snapshot(run_id)
        
        stats = {
            "posts": 10,
            "agents": 5,
            "submolts": 3,
            "comments": 25,
            "evals": 8,
        }
        
        sandbox_db.complete_snapshot(snapshot_id, stats)
        
        snapshot = sandbox_db.get_snapshot(snapshot_id)
        
        assert snapshot["status"] == "completed", f"Status should be 'completed', got {snapshot['status']}"
        assert snapshot["posts_fetched"] == 10, f"Posts count mismatch"
        assert snapshot["evals_completed"] == 8, f"Evals count mismatch"
        
        # Save test result
        output_path = test_output_dir / "database" / "test_completes_snapshot.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_completes_snapshot",
                "run_id": run_id,
                "snapshot_id": snapshot_id,
                "stats": stats,
                "snapshot": dict(snapshot),
            }, f, indent=2, default=str)
        
        print(f"\n✅ Completed snapshot with stats: {stats}")


@pytest.mark.integration
class TestPostStorage:
    """Tests for storing and retrieving posts."""
    
    def test_inserts_scraped_posts(self, sandbox_db, scraped_posts, test_output_dir):
        """Should insert real scraped posts into database."""
        assert len(scraped_posts) > 0, "No scraped posts available for testing"
        
        run_id = f"test-posts-{uuid.uuid4().hex[:8]}"
        snapshot_id = sandbox_db.create_snapshot(run_id)
        
        inserted = 0
        for post in scraped_posts:
            sandbox_db.insert_post(post, snapshot_id)
            inserted += 1
        
        # Verify
        all_posts = sandbox_db.get_all_posts(snapshot_id)
        
        assert len(all_posts) == inserted, f"Inserted {inserted} but found {len(all_posts)}"
        
        # Save test result
        output_path = test_output_dir / "database" / "test_inserts_scraped_posts.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_inserts_scraped_posts",
                "snapshot_id": snapshot_id,
                "inserted_count": inserted,
                "retrieved_count": len(all_posts),
                "posts": [dict(p) for p in all_posts],
            }, f, indent=2, default=str)
        
        print(f"\n✅ Inserted {inserted} posts into database")
        for post in all_posts[:3]:
            print(f"   - {post['title'][:40] if post['title'] else 'No title'}...")
    
    def test_posts_have_correct_fields(self, sandbox_db, scraped_posts, test_output_dir):
        """Should store all post fields correctly."""
        assert len(scraped_posts) > 0, "No scraped posts available for testing"
        
        run_id = f"test-fields-{uuid.uuid4().hex[:8]}"
        snapshot_id = sandbox_db.create_snapshot(run_id)
        
        original = scraped_posts[0]
        sandbox_db.insert_post(original, snapshot_id)
        
        stored = sandbox_db.get_all_posts(snapshot_id)[0]
        
        assert stored["post_external_id"] == original["post_external_id"], "post_external_id mismatch"
        assert stored["title"] == original.get("title"), "title mismatch"
        assert stored["author_handle"] == original.get("author_handle"), "author_handle mismatch"
        
        # Save test result
        output_path = test_output_dir / "database" / "test_posts_have_correct_fields.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_posts_have_correct_fields",
                "original": original,
                "stored": dict(stored),
            }, f, indent=2, default=str)
        
        print(f"\n✅ Post fields stored correctly")


@pytest.mark.integration
class TestAgentStorage:
    """Tests for storing and retrieving agents."""
    
    def test_inserts_scraped_agents(self, sandbox_db, scraped_agents, test_output_dir):
        """Should insert real scraped agents into database."""
        assert len(scraped_agents) > 0, "No scraped agents available for testing"
        
        run_id = f"test-agents-{uuid.uuid4().hex[:8]}"
        snapshot_id = sandbox_db.create_snapshot(run_id)
        
        inserted = 0
        for agent in scraped_agents:
            sandbox_db.insert_agent(agent, snapshot_id)
            inserted += 1
        
        # Verify
        all_agents = sandbox_db.get_all_agents(snapshot_id)
        
        assert len(all_agents) == inserted, f"Inserted {inserted} but found {len(all_agents)}"
        
        # Save test result
        output_path = test_output_dir / "database" / "test_inserts_scraped_agents.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_inserts_scraped_agents",
                "snapshot_id": snapshot_id,
                "inserted_count": inserted,
                "agents": [dict(a) for a in all_agents],
            }, f, indent=2, default=str)
        
        print(f"\n✅ Inserted {inserted} agents into database")
        for agent in all_agents[:3]:
            print(f"   - {agent['handle']}: karma={agent.get('karma')}")
    
    def test_agent_upsert_works(self, sandbox_db, scraped_agents, test_output_dir):
        """Should update existing agent on re-insert."""
        assert len(scraped_agents) > 0, "No scraped agents available for testing"
        
        run_id = f"test-upsert-{uuid.uuid4().hex[:8]}"
        snapshot_id = sandbox_db.create_snapshot(run_id)
        
        agent = scraped_agents[0].copy()
        original_karma = agent.get("karma", 0)
        
        # Insert once
        sandbox_db.insert_agent(agent, snapshot_id)
        
        # Modify and insert again (should update)
        agent["karma"] = 9999
        sandbox_db.insert_agent(agent, snapshot_id)
        
        # Should only have one record
        all_agents = sandbox_db.get_all_agents()
        matching = [a for a in all_agents if a["agent_external_id"] == agent["agent_external_id"]]
        
        assert len(matching) == 1, f"Expected 1 agent, found {len(matching)}"
        assert matching[0]["karma"] == 9999, f"Karma not updated: {matching[0]['karma']}"
        
        # Save test result
        output_path = test_output_dir / "database" / "test_agent_upsert_works.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_agent_upsert_works",
                "original_karma": original_karma,
                "updated_karma": 9999,
                "stored_karma": matching[0]["karma"],
            }, f, indent=2, default=str)
        
        print(f"\n✅ Agent upsert works correctly")


@pytest.mark.integration
class TestCommentStorage:
    """Tests for storing comments."""
    
    def test_inserts_comments_from_post_detail(self, sandbox_db, scraped_post_details, test_output_dir):
        """Should insert comments from post details."""
        from scraper.extractors import flatten_comments_tree
        
        assert len(scraped_post_details) > 0, "No post details available for testing"
        
        run_id = f"test-comments-{uuid.uuid4().hex[:8]}"
        snapshot_id = sandbox_db.create_snapshot(run_id)
        
        total_comments = 0
        all_inserted_comments = []
        
        for detail in scraped_post_details:
            post_id = detail.get("post", {}).get("id")
            comments_data = detail.get("comments", [])
            
            if not post_id:
                continue
            
            # First insert the post
            post_data = {
                "post_external_id": post_id,
                "title": detail.get("post", {}).get("title"),
                "body_text": detail.get("post", {}).get("body"),
            }
            sandbox_db.insert_post(post_data, snapshot_id)
            
            # Then insert comments
            comments = flatten_comments_tree(comments_data, post_id)
            
            for comment in comments:
                sandbox_db.insert_comment(comment, snapshot_id)
                all_inserted_comments.append(comment)
                total_comments += 1
        
        # Save test result
        output_path = test_output_dir / "database" / "test_inserts_comments.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_inserts_comments_from_post_detail",
                "snapshot_id": snapshot_id,
                "posts_processed": len(scraped_post_details),
                "comments_inserted": total_comments,
                "comments": all_inserted_comments[:10],  # Sample
            }, f, indent=2, default=str)
        
        print(f"\n✅ Inserted {total_comments} comments from {len(scraped_post_details)} posts")


@pytest.mark.integration
class TestSubmoltStorage:
    """Tests for storing submolts."""
    
    def test_inserts_scraped_submolts(self, sandbox_db, scraped_submolts, test_output_dir):
        """Should insert real scraped submolts into database."""
        assert len(scraped_submolts) > 0, "No scraped submolts available for testing"
        
        run_id = f"test-submolts-{uuid.uuid4().hex[:8]}"
        snapshot_id = sandbox_db.create_snapshot(run_id)
        
        inserted = 0
        for submolt in scraped_submolts:
            sandbox_db.insert_submolt(submolt, snapshot_id)
            inserted += 1
        
        conn = sandbox_db.connect()
        count = conn.execute("SELECT COUNT(*) FROM submolts WHERE snapshot_id = ?", (snapshot_id,)).fetchone()[0]
        
        assert count == inserted, f"Inserted {inserted} but found {count}"
        
        # Save test result
        output_path = test_output_dir / "database" / "test_inserts_submolts.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_inserts_scraped_submolts",
                "snapshot_id": snapshot_id,
                "inserted_count": inserted,
                "submolts": scraped_submolts,
            }, f, indent=2, default=str)
        
        print(f"\n✅ Inserted {inserted} submolts into database")


@pytest.mark.integration
class TestDatabaseStats:
    """Tests for database statistics."""
    
    def test_gets_overall_stats(self, sandbox_db, scraped_posts, scraped_agents, scraped_submolts, test_output_dir):
        """Should return accurate database statistics."""
        run_id = f"test-stats-{uuid.uuid4().hex[:8]}"
        snapshot_id = sandbox_db.create_snapshot(run_id)
        
        # Insert data
        for post in scraped_posts:
            sandbox_db.insert_post(post, snapshot_id)
        for agent in scraped_agents:
            sandbox_db.insert_agent(agent, snapshot_id)
        for submolt in scraped_submolts:
            sandbox_db.insert_submolt(submolt, snapshot_id)
        
        stats = sandbox_db.get_stats()
        
        assert stats["posts"] >= len(scraped_posts), f"Posts: expected >= {len(scraped_posts)}, got {stats['posts']}"
        assert stats["agents"] >= len(scraped_agents), f"Agents: expected >= {len(scraped_agents)}, got {stats['agents']}"
        assert stats["submolts"] >= len(scraped_submolts), f"Submolts: expected >= {len(scraped_submolts)}, got {stats['submolts']}"
        assert stats["snapshots"] >= 1, f"Snapshots: expected >= 1, got {stats['snapshots']}"
        
        # Save test result
        output_path = test_output_dir / "database" / "test_database_stats.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_gets_overall_stats",
                "stats": stats,
            }, f, indent=2, default=str)
        
        print(f"\n✅ Database stats: {stats}")


@pytest.mark.integration
class TestTransactionRollback:
    """Tests for transaction safety."""
    
    def test_rollback_on_error(self, integration_db_path, test_output_dir):
        """Should rollback on transaction error."""
        db = SandboxDatabase(integration_db_path)
        db.initialize()
        
        run_id = f"test-rollback-{uuid.uuid4().hex[:8]}"
        snapshot_id = db.create_snapshot(run_id)
        
        # Insert a valid post
        db.insert_post({"post_external_id": "test-1", "title": "Test"}, snapshot_id)
        
        # Try to insert with error inside the transaction (simulated ValueError)
        try:
            with db.transaction() as conn:
                # This insert is valid (includes all required fields)
                conn.execute(
                    "INSERT INTO posts (post_external_id, title, fetched_at) VALUES (?, ?, ?)", 
                    ("test-2", "Will be rolled back", datetime.now(timezone.utc).isoformat())
                )
                # But then we raise an error
                raise ValueError("Simulated error")
        except ValueError:
            pass  # Expected
        
        # Original data should still be there, but test-2 should be rolled back
        posts = db.get_all_posts()
        test1_exists = any(p["post_external_id"] == "test-1" for p in posts)
        test2_exists = any(p["post_external_id"] == "test-2" for p in posts)
        
        assert test1_exists, "Original post should still exist"
        assert not test2_exists, "Rolled-back post should not exist"
        
        # Save test result
        output_path = test_output_dir / "database" / "test_rollback.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_rollback_on_error",
                "test1_exists": test1_exists,
                "test2_exists": test2_exists,
                "rollback_successful": test1_exists and not test2_exists,
            }, f, indent=2, default=str)
        
        db.close()
        
        print(f"\n✅ Transaction rollback works correctly")


@pytest.mark.integration  
class TestDatabaseCleanup:
    """Tests for data cleanup."""
    
    def test_clear_all_data(self, integration_db_path, test_output_dir):
        """Should clear all data on request."""
        db = SandboxDatabase(integration_db_path)
        db.initialize()
        
        # Insert some data
        snapshot_id = db.create_snapshot(f"test-clear-{uuid.uuid4().hex[:8]}")
        db.insert_post({"post_external_id": "clear-test", "title": "Test"}, snapshot_id)
        db.insert_agent({"agent_external_id": "agent-1", "handle": "test"}, snapshot_id)
        
        # Verify data exists
        stats_before = db.get_stats()
        assert stats_before["posts"] > 0, "Should have posts before clear"
        
        # Clear all
        db.clear_all()
        
        # Verify empty
        stats_after = db.get_stats()
        
        assert stats_after["posts"] == 0, "Posts should be cleared"
        assert stats_after["agents"] == 0, "Agents should be cleared"
        assert stats_after["snapshots"] == 0, "Snapshots should be cleared"
        
        # Save test result
        output_path = test_output_dir / "database" / "test_clear_all.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_clear_all_data",
                "stats_before": stats_before,
                "stats_after": stats_after,
            }, f, indent=2, default=str)
        
        db.close()
        
        print(f"\n✅ Database clear works correctly")
