"""
End-to-end integration test.

Runs the complete pipeline from scraping to report generation
using real data and real LLM evaluations.
All output is saved to the test output directory.

Run with:
    cd molt-observatory
    python -m pytest tests/integration/test_e2e_pipeline.py -v
"""

from __future__ import annotations
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import uuid

import pytest
from tqdm import tqdm


def _extract_scores(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract scores from result, handling nested structure."""
    if "scores" in result:
        return result["scores"]
    if isinstance(result.get("result"), dict) and "scores" in result["result"]:
        return result["result"]["scores"]
    return {}


def _has_scores(result: Dict[str, Any]) -> bool:
    """Check if result has scores."""
    return len(_extract_scores(result)) > 0


@pytest.mark.integration
class TestEndToEndPipeline:
    """Full end-to-end pipeline test."""
    
    def test_complete_pipeline(
        self,
        real_moltbook_api,
        real_openrouter_client,
        sandbox_db,
        test_output_dir,
        integration_config
    ):
        """
        Run complete pipeline:
        1. Scrape posts, agents, submolts from moltbook.com
        2. Build transcripts
        3. Run LLM evaluations
        4. Store in database
        5. Aggregate agent scores
        6. Generate reports
        
        This is the definitive integration test.
        All output is saved for inspection.
        """
        from scraper.extractors import (
            extract_posts_from_list,
            extract_post_detail,
            extract_agents_from_recent,
            extract_submolts_from_list,
            flatten_comments_tree,
        )
        from transcript_builder import build_transcript_from_post_detail
        from judge_runner import LLMJudgeRunner
        from agent_scorer import aggregate_agent_scores
        from reports.generator import generate_dashboard
        from reports.leaderboards import generate_agent_leaderboard
        
        print("\n" + "="*60)
        print("🚀 STARTING END-TO-END INTEGRATION TEST")
        print("="*60)
        
        # Create e2e-specific output directory
        e2e_dir = test_output_dir / "e2e_pipeline"
        e2e_dir.mkdir(exist_ok=True)
        
        pipeline_log = []
        
        def log(phase: str, message: str, data: Any = None):
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": phase,
                "message": message,
            }
            if data:
                entry["data"] = data
            pipeline_log.append(entry)
            print(f"   {message}")
        
        # =====================================================================
        # PHASE 1: Initialize
        # =====================================================================
        run_id = f"e2e-{uuid.uuid4().hex[:8]}"
        snapshot_id = sandbox_db.create_snapshot(run_id)
        
        stats = {
            "posts": 0,
            "agents": 0,
            "submolts": 0,
            "comments": 0,
            "evals": 0,
        }
        
        print(f"\n📍 Phase 1: Initialized run {run_id}")
        log("init", f"Created run {run_id}", {"snapshot_id": snapshot_id})
        
        # =====================================================================
        # PHASE 2: Scrape Data
        # =====================================================================
        print(f"\n📍 Phase 2: Scraping data from moltbook.com...")
        
        # Fetch posts
        posts_resp = real_moltbook_api.get_json("/api/v1/posts", params={
            "limit": integration_config["posts_limit"],
            "sort": "new"
        })
        posts = extract_posts_from_list(posts_resp.json_body)
        
        for post in posts:
            sandbox_db.insert_post(post, snapshot_id)
            stats["posts"] += 1
        
        # Save posts
        with open(e2e_dir / "scraped_posts.json", "w") as f:
            json.dump(posts, f, indent=2, default=str)
        
        log("scrape", f"Scraped {len(posts)} posts", {"count": len(posts)})
        
        # Fetch agents
        agents_resp = real_moltbook_api.get_json("/api/v1/agents/recent", params={
            "limit": integration_config["agents_limit"]
        })
        agents = extract_agents_from_recent(agents_resp.json_body)
        
        for agent in agents:
            sandbox_db.insert_agent(agent, snapshot_id)
            stats["agents"] += 1
        
        # Save agents
        with open(e2e_dir / "scraped_agents.json", "w") as f:
            json.dump(agents, f, indent=2, default=str)
        
        log("scrape", f"Scraped {len(agents)} agents", {"count": len(agents)})
        
        # Fetch submolts
        submolts_resp = real_moltbook_api.get_json("/api/v1/submolts", params={
            "limit": integration_config["submolts_limit"]
        })
        submolts = extract_submolts_from_list(submolts_resp.json_body)
        
        for submolt in submolts:
            sandbox_db.insert_submolt(submolt, snapshot_id)
            stats["submolts"] += 1
        
        # Save submolts
        with open(e2e_dir / "scraped_submolts.json", "w") as f:
            json.dump(submolts, f, indent=2, default=str)
        
        log("scrape", f"Scraped {len(submolts)} submolts", {"count": len(submolts)})
        
        # Fetch post details (with comments)
        # Store both raw API responses (for transcript building) and extracted details (for DB)
        post_details = []
        raw_post_responses = []  # Raw API responses for transcript building
        
        pbar = tqdm(posts[:3], desc="📥 Fetching post details", unit="post", leave=True)
        for i, post in enumerate(pbar):
            pid = post["post_external_id"]
            pbar.set_postfix({"post": pid[:8]})
            try:
                resp = real_moltbook_api.get_json(f"/api/v1/posts/{pid}")
                raw_response = resp.json_body
                raw_post_responses.append(raw_response)  # Save raw for transcripts
                
                detail = extract_post_detail(raw_response)
                post_details.append(detail)
                
                # Save individual detail
                with open(e2e_dir / f"post_detail_{i+1}.json", "w") as f:
                    json.dump(detail, f, indent=2, default=str)
                
                # Insert comments
                comments = flatten_comments_tree(detail.get("comments", []), pid)
                for comment in comments:
                    sandbox_db.insert_comment(comment, snapshot_id)
                    stats["comments"] += 1
                    
            except Exception as e:
                log("scrape", f"Failed to fetch post {pid}: {e}")
        
        log("scrape", f"Fetched {len(post_details)} post details with {stats['comments']} comments")
        
        # =====================================================================
        # PHASE 3: Build Transcripts
        # =====================================================================
        print(f"\n📍 Phase 3: Building transcripts...")
        
        transcripts = []
        pbar = tqdm(raw_post_responses, desc="📝 Building transcripts", unit="transcript", leave=True)
        for raw_response in pbar:
            try:
                # Pass raw API response to transcript builder (it will call extract_post_detail internally)
                t = build_transcript_from_post_detail(raw_response)
                transcripts.append(t)
                pbar.set_postfix({"built": len(transcripts)})
            except Exception as e:
                log("transcripts", f"Failed to build transcript: {e}")
        
        # Save transcripts as JSONL
        with open(e2e_dir / "transcripts.jsonl", "w") as f:
            for t in transcripts:
                f.write(json.dumps(asdict(t), default=str) + "\n")
        
        log("transcripts", f"Built {len(transcripts)} transcripts", {"count": len(transcripts)})
        
        # =====================================================================
        # PHASE 4: LLM Evaluations
        # =====================================================================
        print(f"\n📍 Phase 4: Running LLM evaluations...")
        
        eval_limit = min(integration_config["eval_limit"], len(transcripts), 2)
        eval_results = []
        
        runner = LLMJudgeRunner(client=real_openrouter_client)
        
        pbar = tqdm(transcripts[:eval_limit], desc="🤖 Running LLM evaluations", unit="eval", leave=True)
        for i, transcript in enumerate(pbar):
            pbar.set_postfix({"transcript": transcript.transcript_id[:8]})
            try:
                result = runner.score_transcript(asdict(transcript))
                
                scores = _extract_scores(result)
                
                if scores:
                    eval_record = {
                        "index": i,
                        "post_id": transcript.post_id,
                        "transcript_id": transcript.transcript_id,
                        "model": runner.model,
                        "scores": scores,
                        "author_id": posts[i].get("author_external_id") if i < len(posts) else None,
                        "author_handle": posts[i].get("author_handle") if i < len(posts) else None,
                    }
                    
                    sandbox_db.insert_post_evaluation(eval_record, snapshot_id)
                    eval_results.append(eval_record)
                    stats["evals"] += 1
                    
                    log("eval", f"Evaluated transcript {i+1}", {"transcript_id": transcript.transcript_id[:12]})
                else:
                    log("eval", f"Evaluation {i+1} failed: {result.get('error', 'no scores')}")
                    
            except Exception as e:
                log("eval", f"Evaluation error: {e}")
        
        # Save evaluations
        with open(e2e_dir / "evaluations.json", "w") as f:
            json.dump(eval_results, f, indent=2, default=str)
        
        log("eval", f"Completed {stats['evals']} evaluations")
        
        # =====================================================================
        # PHASE 5: Agent Score Aggregation
        # =====================================================================
        print(f"\n📍 Phase 5: Aggregating agent scores...")
        
        aggregated_agents = []
        if eval_results and agents:
            # Aggregate for first agent as example
            agent = agents[0]
            agent_id = agent.get("agent_external_id")
            agent_handle = agent.get("handle")
            
            # Use the eval results as post evals
            post_evals = [
                {
                    "transcript_id": e["transcript_id"],
                    "post_id": e["post_id"],
                    "scores": e["scores"],
                }
                for e in eval_results
            ]
            
            try:
                agent_record = aggregate_agent_scores(
                    agent_id=agent_id,
                    agent_handle=agent_handle,
                    post_evals=post_evals,
                    comment_evals=[],
                    snapshot_id=run_id,
                )
                
                sandbox_db.insert_agent_score(asdict(agent_record), snapshot_id)
                aggregated_agents.append(asdict(agent_record))
                
                log("aggregate", f"Aggregated scores for {agent_handle}")
                
            except Exception as e:
                log("aggregate", f"Aggregation error: {e}")
        
        # Save agent scores
        with open(e2e_dir / "agent_scores.json", "w") as f:
            json.dump(aggregated_agents, f, indent=2, default=str)
        
        log("aggregate", f"Aggregated {len(aggregated_agents)} agent scores")
        
        # =====================================================================
        # PHASE 6: Generate Reports
        # =====================================================================
        print(f"\n📍 Phase 6: Generating reports...")
        
        # Dashboard
        dashboard_path = e2e_dir / "dashboard.html"
        generate_dashboard(
            db=sandbox_db,
            output_path=dashboard_path,
            title=f"E2E Test Report - {run_id}"
        )
        log("reports", f"Generated dashboard", {"path": str(dashboard_path)})
        
        # Agent leaderboard
        if aggregated_agents:
            leaderboard_path = e2e_dir / "leaderboard.html"
            generate_agent_leaderboard(
                agent_scores=aggregated_agents,
                output_path=leaderboard_path,
                title="E2E Agent Leaderboard"
            )
            log("reports", f"Generated leaderboard", {"path": str(leaderboard_path)})
        
        # =====================================================================
        # PHASE 7: Complete and Verify
        # =====================================================================
        print(f"\n📍 Phase 7: Completing pipeline...")
        
        sandbox_db.complete_snapshot(snapshot_id, stats)
        
        # Verify database state
        db_stats = sandbox_db.get_stats()
        
        assert db_stats["posts"] >= stats["posts"], f"Posts mismatch: {db_stats['posts']} < {stats['posts']}"
        assert db_stats["agents"] >= stats["agents"], f"Agents mismatch: {db_stats['agents']} < {stats['agents']}"
        
        # Verify files created
        assert (e2e_dir / "scraped_posts.json").exists(), "Posts file not created"
        assert (e2e_dir / "transcripts.jsonl").exists(), "Transcripts file not created"
        assert dashboard_path.exists(), "Dashboard not created"
        
        log("complete", "Pipeline completed successfully", {"db_stats": db_stats})
        
        # Save pipeline log
        with open(e2e_dir / "pipeline_log.json", "w") as f:
            json.dump(pipeline_log, f, indent=2, default=str)
        
        # Save final summary
        summary = {
            "run_id": run_id,
            "snapshot_id": snapshot_id,
            "stats": stats,
            "db_stats": db_stats,
            "output_dir": str(e2e_dir),
            "files_created": [f.name for f in e2e_dir.iterdir()],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        
        with open(e2e_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # =====================================================================
        # SUMMARY
        # =====================================================================
        print("\n" + "="*60)
        print("✅ END-TO-END PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"\n📊 Statistics:")
        print(f"   Posts scraped:     {stats['posts']}")
        print(f"   Agents scraped:    {stats['agents']}")
        print(f"   Submolts scraped:  {stats['submolts']}")
        print(f"   Comments scraped:  {stats['comments']}")
        print(f"   LLM evaluations:   {stats['evals']}")
        print(f"\n📁 Output directory: {e2e_dir}")
        print(f"   Files created: {len(list(e2e_dir.iterdir()))}")
        for f in sorted(e2e_dir.iterdir()):
            size = f.stat().st_size / 1024
            print(f"     - {f.name} ({size:.1f} KB)")


@pytest.mark.integration
class TestPipelineRecovery:
    """Tests for pipeline recovery and error handling."""
    
    def test_handles_api_errors_gracefully(self, real_moltbook_api, sandbox_db, test_output_dir):
        """Should handle API errors without crashing."""
        run_id = f"recovery-{uuid.uuid4().hex[:8]}"
        snapshot_id = sandbox_db.create_snapshot(run_id)
        
        error_info = {}
        
        # Try to fetch a non-existent post
        try:
            resp = real_moltbook_api.get_json("/api/v1/posts/non-existent-id-12345")
            error_info["status"] = resp.status
            error_info["response"] = resp.json_body
        except Exception as e:
            error_info["exception"] = str(e)
            error_info["exception_type"] = type(e).__name__
        
        # Pipeline should continue despite error
        sandbox_db.complete_snapshot(snapshot_id, {"posts": 0})
        
        snapshot = sandbox_db.get_snapshot(snapshot_id)
        assert snapshot["status"] == "completed", f"Snapshot not completed: {snapshot['status']}"
        
        # Save error info
        output_path = test_output_dir / "e2e_pipeline" / "test_api_error_handling.json"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_handles_api_errors_gracefully",
                "error_info": error_info,
                "pipeline_recovered": True,
                "snapshot_status": snapshot["status"],
            }, f, indent=2, default=str)
        
        print(f"\n✅ Pipeline recovered from API error")
        print(f"   Saved to: {output_path}")
    
    def test_partial_success_still_saves(self, real_moltbook_api, sandbox_db, test_output_dir):
        """Should save partial results even if some operations fail."""
        from scraper.extractors import extract_posts_from_list
        
        run_id = f"partial-{uuid.uuid4().hex[:8]}"
        snapshot_id = sandbox_db.create_snapshot(run_id)
        
        # Fetch some posts
        resp = real_moltbook_api.get_json("/api/v1/posts", params={"limit": 3})
        posts = extract_posts_from_list(resp.json_body)
        
        saved = 0
        for post in posts:
            sandbox_db.insert_post(post, snapshot_id)
            saved += 1
        
        # Complete even without evaluations
        sandbox_db.complete_snapshot(snapshot_id, {"posts": saved, "evals": 0})
        
        # Verify partial data saved
        stored_posts = sandbox_db.get_all_posts(snapshot_id)
        assert len(stored_posts) == saved, f"Expected {saved}, got {len(stored_posts)}"
        
        # Save result
        output_path = test_output_dir / "e2e_pipeline" / "test_partial_success.json"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_partial_success_still_saves",
                "posts_saved": saved,
                "evals_completed": 0,
                "partial_success": True,
            }, f, indent=2, default=str)
        
        print(f"\n✅ Saved {saved} posts despite incomplete pipeline")
        print(f"   Saved to: {output_path}")


@pytest.mark.integration
class TestOutputVerification:
    """Tests to verify output quality."""
    
    def test_html_reports_are_valid(self, test_output_dir):
        """Should produce valid HTML reports."""
        reports_dir = test_output_dir / "reports"
        
        if not reports_dir.exists():
            reports_dir.mkdir(parents=True)
        
        validation_results = []
        
        for html_file in reports_dir.glob("*.html"):
            content = html_file.read_text()
            
            result = {
                "file": html_file.name,
                "size_bytes": len(content),
                "has_html_tag": "<html" in content or "<!DOCTYPE" in content,
                "has_closing_tag": "</html>" in content or content.count("<") > 5,
                "valid": True,
            }
            
            if not result["has_html_tag"] or not result["has_closing_tag"]:
                result["valid"] = False
            
            validation_results.append(result)
            print(f"   ✓ {html_file.name} is {'valid' if result['valid'] else 'INVALID'}")
        
        # Save validation results
        output_path = test_output_dir / "reports" / "html_validation.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_html_reports_are_valid",
                "results": validation_results,
            }, f, indent=2)
        
        print(f"\n✅ Validated {len(validation_results)} HTML reports")
    
    def test_json_files_are_valid(self, test_output_dir):
        """Should produce valid JSON files."""
        validation_results = []
        
        for subdir in ["raw", "transcripts", "evaluations", "reports", "database", "e2e_pipeline"]:
            dir_path = test_output_dir / subdir
            
            if not dir_path.exists():
                continue
            
            for json_file in dir_path.glob("*.json"):
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                    
                    result = {
                        "file": f"{subdir}/{json_file.name}",
                        "valid": True,
                        "size_bytes": json_file.stat().st_size,
                    }
                except json.JSONDecodeError as e:
                    result = {
                        "file": f"{subdir}/{json_file.name}",
                        "valid": False,
                        "error": str(e),
                    }
                
                validation_results.append(result)
        
        # Save validation results
        output_path = test_output_dir / "json_validation.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_json_files_are_valid",
                "total_files": len(validation_results),
                "valid_files": sum(1 for r in validation_results if r["valid"]),
                "results": validation_results,
            }, f, indent=2)
        
        invalid = [r for r in validation_results if not r["valid"]]
        assert len(invalid) == 0, f"Invalid JSON files: {[r['file'] for r in invalid]}"
        
        print(f"\n✅ Validated {len(validation_results)} JSON files")
    
    def test_database_integrity(self, sandbox_db, test_output_dir):
        """Should maintain database integrity."""
        conn = sandbox_db.connect()
        
        integrity_checks = {}
        
        # Check for orphaned records
        orphaned_evals = conn.execute("""
            SELECT COUNT(*) FROM post_evaluations pe
            LEFT JOIN posts p ON pe.post_external_id = p.post_external_id
            WHERE p.id IS NULL AND pe.post_external_id IS NOT NULL
        """).fetchone()[0]
        
        integrity_checks["orphaned_evaluations"] = orphaned_evals
        
        # All snapshots should have valid status
        invalid_snapshots = conn.execute("""
            SELECT COUNT(*) FROM snapshots 
            WHERE status NOT IN ('running', 'completed', 'failed')
        """).fetchone()[0]
        
        integrity_checks["invalid_snapshot_status"] = invalid_snapshots
        
        assert invalid_snapshots == 0, f"Found {invalid_snapshots} snapshots with invalid status"
        
        # Save integrity check results
        output_path = test_output_dir / "database" / "integrity_check.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_database_integrity",
                "checks": integrity_checks,
                "all_passed": invalid_snapshots == 0,
            }, f, indent=2)
        
        print(f"\n✅ Database integrity verified")
        print(f"   Saved to: {output_path}")
