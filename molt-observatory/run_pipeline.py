#!/usr/bin/env python3
"""
Molt Observatory Pipeline - THE SINGLE ENTRY POINT FOR ALL OPERATIONS.

This script handles:
- Scraping posts, agents, submolts, and comments from moltbook.com
- Building transcripts (post + comment)
- Running LLM judge evaluations on all content
- Aggregating scores per agent
- Storing results in PostgreSQL (or SQLite for testing)
- Generating Plotly HTML reports

Usage:
  # Query total counts over all time
  python run_pipeline.py --stats

  # Query/scrape within a date range
  python run_pipeline.py --start "2026-01-31T07:00:00" --end "2026-01-31T09:00:00"

  # Query last N hours from now
  python run_pipeline.py --hours 2

  # Force full scrape from site genesis
  python run_pipeline.py --full-scrape

  # Generate reports only (no scraping)
  python run_pipeline.py --reports-only

  # Dry run (scrape only, no LLM evaluation)
  python run_pipeline.py --start "2026-01-31T07:00:00" --end "2026-01-31T09:00:00" --dry-run
"""

from __future__ import annotations
from reports.leaderboards import generate_leaderboard_report, generate_agent_leaderboard
from reports.growth import generate_growth_report
from reports.generator import (
    generate_all_reports,
    generate_all_charts_png,
    generate_threat_vector_chart_with_png,
    generate_agent_heatmap_with_png,
    generate_growth_chart_with_png,
)
from agent_scorer import aggregate_all_agents, write_agent_scores_jsonl
from judge_runner import (
    run_judges,
    run_comment_judges,
    run_judges_with_cost_tracking,
    run_comment_judges_with_cost_tracking,
    DEFAULT_DIMENSIONS,
    CostTracker,
)
from comment_transcript_builder import (
    build_comment_transcripts_from_post_detail,
    write_comment_transcripts_jsonl,
)
from transcript_builder import build_transcript_from_post_detail, write_transcripts_jsonl
from scraper.extractors import (
    extract_posts_from_list,
    extract_post_detail,
    extract_agents_from_recent,
    extract_submolts_from_list,
    extract_site_stats,
    flatten_comments_tree,
    dedupe_agents,
    dedupe_posts,
    dedupe_submolts,
)
from scraper.moltbook_api import MoltbookAPI
from analysis.thread_analysis import (
    analyze_batch,
    partition_deep_threads,
    partition_hot_threads,
)
from analysis.thread_charts import generate_all_thread_charts
from analysis.content_filter import (
    run_content_filter,
    analyze_agents_only,
    build_spammer_profiles,
)
from analysis.cascade_detector import (
    analyze_cascades,
    generate_cascade_report,
)
from analysis.lite_judge import run_lite_judge
from analysis.tiered_reports import generate_all_tiered_reports
from reports.pdf_generator import generate_thread_analysis_pdf, generate_pipeline_report_pdf, generate_tiered_eval_pdf
import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Local imports


# =============================================================================
# Configuration Constants
# =============================================================================

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "runs"
DEFAULT_RATE_LIMIT = 1.0  # Requests per second
DEFAULT_BATCH_SIZE = 50   # Posts per API call


def _utcnow_str() -> str:
    """Get current UTC timestamp as a string."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(p: Path) -> None:
    """Ensure a directory exists."""
    p.mkdir(parents=True, exist_ok=True)


def parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp string to timezone-aware datetime."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        # Try parsing without timezone
        dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt


def is_in_range(created_at: str, start: datetime, end: datetime) -> bool:
    """Check if a timestamp falls within [start, end]."""
    if not created_at:
        return False
    try:
        ts = parse_timestamp(created_at)
        return start <= ts <= end
    except Exception:
        return False


def load_batch_data(batch_dir: Path) -> Dict[str, Any]:
    """
    Load pre-scraped batch data from a directory.

    Expected structure:
        batch_dir/
        ├── transcripts/transcripts.jsonl   # Post transcripts
        ├── posts/all_comments.json          # All comments
        ├── posts/detail_*.json              # Post details
        ├── submolts/submolts_list.json      # Submolts
        └── agents/agents_list.json          # Agents

    Args:
        batch_dir: Path to batch directory

    Returns:
        Dict with keys: posts, comments, submolts, agents, transcripts, batch_summary
    """
    batch_dir = Path(batch_dir)

    result = {
        "posts": [],
        "comments": [],
        "submolts": [],
        "agents": [],
        "transcripts": [],
        "batch_summary": {},
        "stats": {},
    }

    # Load batch summary if exists
    summary_path = batch_dir / "batch_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            result["batch_summary"] = json.load(f)

    # Load transcripts from JSONL (check both standard location and partition location)
    transcripts_path = batch_dir / "transcripts" / "transcripts.jsonl"
    if not transcripts_path.exists():
        # Also check root level (for partitions)
        transcripts_path = batch_dir / "transcripts.jsonl"
    
    if transcripts_path.exists():
        print(f"   Loading transcripts from {transcripts_path}...")
        with open(transcripts_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        result["transcripts"].append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        print(f"   ✅ Loaded {len(result['transcripts'])} transcripts")

    # Load comments
    comments_path = batch_dir / "posts" / "all_comments.json"
    if comments_path.exists():
        print(f"   Loading comments from {comments_path}...")
        with open(comments_path, "r", encoding="utf-8") as f:
            result["comments"] = json.load(f)
        print(f"   ✅ Loaded {len(result['comments'])} comments")

    # Load submolts
    submolts_path = batch_dir / "submolts" / "submolts_list.json"
    if submolts_path.exists():
        print(f"   Loading submolts from {submolts_path}...")
        with open(submolts_path, "r", encoding="utf-8") as f:
            result["submolts"] = json.load(f)
        print(f"   ✅ Loaded {len(result['submolts'])} submolts")

    # Load agents
    agents_path = batch_dir / "agents" / "agents_list.json"
    if agents_path.exists():
        print(f"   Loading agents from {agents_path}...")
        with open(agents_path, "r", encoding="utf-8") as f:
            result["agents"] = json.load(f)
        print(f"   ✅ Loaded {len(result['agents'])} agents")

    # Load post details (extract from individual detail files if needed)
    posts_dir = batch_dir / "posts"
    if posts_dir.exists():
        detail_files = list(posts_dir.glob("detail_*.json"))
        if detail_files:
            print(f"   Loading {len(detail_files)} post details...")
            for detail_file in tqdm(detail_files, desc="Loading post details", disable=len(detail_files) < 100):
                try:
                    with open(detail_file, "r", encoding="utf-8") as f:
                        detail = json.load(f)
                    # Extract post data from detail
                    if "post" in detail:
                        post = detail["post"]
                        # Add post_external_id if not present
                        if "post_external_id" not in post and "id" in post:
                            post["post_external_id"] = post["id"]
                        result["posts"].append(post)
                except (json.JSONDecodeError, Exception):
                    pass
            print(f"   ✅ Loaded {len(result['posts'])} posts")

    # Build stats from batch data
    result["stats"] = {
        "total_posts": len(result["posts"]),
        "total_comments": len(result["comments"]),
        "total_submolts": len(result["submolts"]),
        "total_agents": len(result["agents"]),
    }

    return result


# =============================================================================
# Core Pipeline Functions
# =============================================================================

def scrape_all_entities(
    api: MoltbookAPI,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    max_posts: int = 10000,
    max_agents: int = 1000,
    max_submolts: int = 500,
    fetch_comments: bool = True,
) -> Dict[str, Any]:
    """
    Scrape all entities (posts, agents, submolts, comments) from Moltbook.

    Args:
        api: MoltbookAPI instance
        start_time: Optional start of date range (scrape only items in range)
        end_time: Optional end of date range
        max_posts: Maximum posts to fetch
        max_agents: Maximum agents to fetch
        max_submolts: Maximum submolts to fetch
        fetch_comments: Whether to fetch post details with comments

    Returns:
        Dict with scraped entities
    """
    result = {
        "posts": [],
        "post_details": [],  # Raw API responses
        "agents": [],
        "submolts": [],
        "comments": [],
        "stats": {},
    }

    # -------------------------------------------------------------------------
    # 1. Scrape Posts
    # -------------------------------------------------------------------------
    print("\n📥 Scraping posts...")
    all_posts = []
    posts_in_range = []
    page = 0
    reached_start = False

    pbar = tqdm(desc="Fetching posts", unit="page")
    max_pages = max(1, max_posts // DEFAULT_BATCH_SIZE)
    while not reached_start and page < max_pages:
        try:
            resp = api.get_json("/api/v1/posts", params={
                "sort": "new",
                "limit": DEFAULT_BATCH_SIZE,
                "offset": page * DEFAULT_BATCH_SIZE,
            })

            body = resp.json_body
            posts = body.get("posts", []) if isinstance(body, dict) else body

            if not posts:
                break

            for post in posts:
                created_at = post.get("created_at")

                if created_at and start_time:
                    ts = parse_timestamp(created_at)

                    # Check if we've gone past the start of our range
                    if ts < start_time:
                        reached_start = True
                        break

                    # Check if in range
                    if start_time <= ts <= (end_time or datetime.now(timezone.utc)):
                        posts_in_range.append(post)
                else:
                    # No date filter - include all
                    posts_in_range.append(post)

                all_posts.append(post)

            page += 1
            pbar.update(1)
            pbar.set_postfix(
                {"total": len(all_posts), "in_range": len(posts_in_range)})

            if len(posts) < DEFAULT_BATCH_SIZE:
                break

        except Exception as e:
            print(f"\n   ⚠️ Error fetching posts: {e}")
            break

    pbar.close()

    # Extract and dedupe posts
    extracted_posts = extract_posts_from_list(
        {"posts": posts_in_range}, dedupe=True)
    result["posts"] = extracted_posts
    print(
        f"   ✅ Found {len(posts_in_range)} posts in range, {len(extracted_posts)} unique")

    # -------------------------------------------------------------------------
    # 2. Fetch Post Details with Comments
    # -------------------------------------------------------------------------
    if fetch_comments and posts_in_range:
        print(f"\n📥 Fetching post details with comments...")

        post_details = []
        all_comments = []

        pbar = tqdm(posts_in_range, desc="Fetching details", unit="post")
        for post in pbar:
            post_id = post.get("id")
            if not post_id:
                continue

            pbar.set_postfix({"post": post_id[:8]})

            try:
                resp = api.get_json(f"/api/v1/posts/{post_id}")
                raw_detail = resp.json_body
                post_details.append(raw_detail)

                # Extract comments
                detail = extract_post_detail(raw_detail)
                if detail and detail.get("comments"):
                    flat_comments = flatten_comments_tree(
                        detail["comments"], post_id)
                    all_comments.extend(flat_comments)

            except Exception as e:
                print(f"\n   ⚠️ Error fetching post {post_id[:8]}: {e}")

        pbar.close()

        result["post_details"] = post_details
        result["comments"] = all_comments
        print(
            f"   ✅ Fetched {len(post_details)} post details with {len(all_comments)} comments")

    # -------------------------------------------------------------------------
    # 3. Scrape Agents (from post/comment authors, NOT /agents/recent)
    # -------------------------------------------------------------------------
    print("\n📥 Extracting agent handles from posts and comments...")

    # Collect unique agent handles from posts and comments
    agent_handles: Set[str] = set()

    # From posts
    for post in extracted_posts:
        handle = post.get("author_handle")
        if handle:
            agent_handles.add(handle)

    # From comments
    for comment in result.get("comments", []):
        handle = comment.get("author_handle") or comment.get("author_name")
        if handle:
            agent_handles.add(handle)

    # Also check raw post details for author info
    for detail in result.get("post_details", []):
        post_data = detail.get("post", {}) if isinstance(detail, dict) else {}
        author = post_data.get("author") or {}
        if author.get("name"):
            agent_handles.add(author["name"])

        # Check comments in detail
        for comment in detail.get("comments", []):
            author = comment.get("author") or {}
            if author.get("name"):
                agent_handles.add(author["name"])

    print(f"   Found {len(agent_handles)} unique agent handles")

    # Fetch agent profiles
    agent_profiles = []
    if agent_handles:
        print(f"\n📥 Fetching {len(agent_handles)} agent profiles...")

        pbar = tqdm(list(agent_handles)[:max_agents],
                    desc="Fetching profiles", unit="agent")
        for handle in pbar:
            pbar.set_postfix({"agent": handle[:12]})
            try:
                resp = api.get_json(f"/api/v1/agents/{handle}")
                body = resp.json_body

                if isinstance(body, dict):
                    # Could be {agent: {...}} or direct agent object
                    agent_data = body.get("agent", body)
                    if agent_data and agent_data.get("name"):
                        agent_profiles.append(agent_data)

            except Exception as e:
                # Agent profile may not exist or be private
                pass

        pbar.close()

    # Extract and dedupe
    extracted_agents = extract_agents_from_recent(
        {"agents": agent_profiles}, dedupe=True)
    result["agents"] = extracted_agents
    print(f"   ✅ Fetched {len(extracted_agents)} unique agent profiles")

    # -------------------------------------------------------------------------
    # 4. Scrape Submolts
    # -------------------------------------------------------------------------
    print("\n📥 Scraping submolts...")

    all_submolts = []
    page = 0

    pbar = tqdm(desc="Fetching submolts", unit="page")
    max_submolt_pages = max(1, max_submolts // DEFAULT_BATCH_SIZE)
    while page < max_submolt_pages:
        try:
            resp = api.get_json("/api/v1/submolts", params={
                "limit": DEFAULT_BATCH_SIZE,
                "offset": page * DEFAULT_BATCH_SIZE,
            })

            body = resp.json_body
            submolts = body.get("submolts", []) if isinstance(
                body, dict) else body

            if not submolts:
                break

            all_submolts.extend(submolts)
            page += 1
            pbar.update(1)
            pbar.set_postfix({"total": len(all_submolts)})

            if len(submolts) < DEFAULT_BATCH_SIZE:
                break

        except Exception as e:
            print(f"\n   ⚠️ Error fetching submolts: {e}")
            break

    pbar.close()

    # Extract and dedupe
    extracted_submolts = extract_submolts_from_list(
        {"submolts": all_submolts}, dedupe=True)
    result["submolts"] = extracted_submolts
    print(f"   ✅ Found {len(extracted_submolts)} unique submolts")

    # -------------------------------------------------------------------------
    # 5. Compute site stats
    # -------------------------------------------------------------------------
    result["stats"] = {
        "total_posts": len(result["posts"]),
        "total_agents": len(result["agents"]),
        "total_submolts": len(result["submolts"]),
        "total_comments": len(result["comments"]),
    }

    return result


def build_all_transcripts(
    post_details: List[Dict[str, Any]],
    build_comments: bool = True,
) -> Dict[str, Any]:
    """
    Build post and comment transcripts from raw post details.

    Args:
        post_details: List of raw post detail API responses
        build_comments: Whether to also build comment transcripts

    Returns:
        Dict with transcripts
    """
    result = {
        "post_transcripts": [],
        "comment_transcripts": [],
    }

    print("\n📝 Building transcripts...")

    pbar = tqdm(post_details, desc="Building post transcripts", unit="post")
    for raw_detail in pbar:
        try:
            t = build_transcript_from_post_detail(raw_detail)
            result["post_transcripts"].append(t)
        except Exception as e:
            pass  # Skip failed transcripts silently
    pbar.close()

    print(f"   ✅ Built {len(result['post_transcripts'])} post transcripts")

    if build_comments:
        pbar = tqdm(
            post_details, desc="Building comment transcripts", unit="post")
        for raw_detail in pbar:
            try:
                cts = build_comment_transcripts_from_post_detail(raw_detail)
                result["comment_transcripts"].extend(cts)
            except Exception as e:
                pass  # Skip failed transcripts silently
        pbar.close()

        print(
            f"   ✅ Built {len(result['comment_transcripts'])} comment transcripts")

    return result


def run_all_evaluations(
    post_transcripts: List[Any],
    comment_transcripts: List[Any],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run LLM judge evaluations on all transcripts.

    Args:
        post_transcripts: List of Transcript objects
        comment_transcripts: List of CommentTranscript objects
        model: LLM model to use (defaults to env var OPENROUTER_MODEL)

    Returns:
        Dict with evaluation results
    """
    result = {
        "post_evals": [],
        "comment_evals": [],
    }

    if not model:
        model = os.environ.get(
            "OPENROUTER_MODEL", "google/gemini-3-flash-preview")

    # -------------------------------------------------------------------------
    # Evaluate Posts
    # -------------------------------------------------------------------------
    if post_transcripts:
        print(f"\n⚖️ Evaluating {len(post_transcripts)} post transcripts...")

        # Convert transcripts to dicts
        transcripts_dicts = []
        for t in post_transcripts:
            transcripts_dicts.append({
                "transcript_id": t.transcript_id,
                "post_id": t.post_id,
                "permalink": t.permalink,
                "community": t.community,
                "metadata": t.metadata,
                "messages": t.messages,
            })

        result["post_evals"] = run_judges(
            transcripts_dicts,
            dimensions=DEFAULT_DIMENSIONS,
            judge_models=[model],
        )

        print(f"   ✅ Evaluated {len(result['post_evals'])} posts")

    # -------------------------------------------------------------------------
    # Evaluate Comments
    # -------------------------------------------------------------------------
    if comment_transcripts:
        print(
            f"\n⚖️ Evaluating {len(comment_transcripts)} comment transcripts...")

        # Convert to dicts
        comment_transcript_dicts = [
            {
                "transcript_id": ct.transcript_id,
                "comment_id": ct.comment_id,
                "post_id": ct.post_id,
                "permalink": ct.permalink,
                "community": ct.community,
                "context_messages": ct.context_messages,
                "target_comment": ct.target_comment,
                "metadata": ct.metadata,
            }
            for ct in comment_transcripts
        ]

        result["comment_evals"] = run_comment_judges(
            comment_transcript_dicts,
            dimensions=DEFAULT_DIMENSIONS,
            judge_models=[model],
        )

        print(f"   ✅ Evaluated {len(result['comment_evals'])} comments")

    return result


def aggregate_scores(
    agents: List[Dict[str, Any]],
    post_evals: List[Dict[str, Any]],
    comment_evals: List[Dict[str, Any]],
    snapshot_id: str,
) -> List[Dict[str, Any]]:
    """
    Aggregate evaluation scores per agent.

    Returns:
        List of agent score records
    """
    print("\n📊 Aggregating agent scores...")

    agent_records = aggregate_all_agents(
        post_evals=post_evals,
        comment_evals=comment_evals,
        agents=agents,
        snapshot_id=snapshot_id,
    )

    print(f"   ✅ Aggregated scores for {len(agent_records)} agents")

    return agent_records


def generate_all_html_reports(
    output_dir: Path,
    stats: Dict[str, Any],
    aggregates: Dict[str, Any],
    agent_scores: List[Dict[str, Any]],
    posts: List[Dict[str, Any]],
    post_evals: List[Dict[str, Any]] = None,
    comment_evals: List[Dict[str, Any]] = None,
    transcripts: List[Dict[str, Any]] = None,
    comments: List[Dict[str, Any]] = None,
    submolts: List[Dict[str, Any]] = None,
    comment_transcripts: List[Dict[str, Any]] = None,
    export_png: bool = True,
) -> Dict[str, str]:
    """
    Generate all HTML reports and PNG charts.

    Args:
        output_dir: Base output directory
        stats: Site statistics dict
        aggregates: Evaluation aggregates dict
        agent_scores: List of agent score records
        posts: List of posts
        post_evals: Optional list of post evaluations
        comment_evals: Optional list of comment evaluations
        transcripts: Optional list of post transcripts with timestamps
        comments: Optional list of comments with timestamps
        submolts: Optional list of submolts with timestamps
        comment_transcripts: Optional list of comment transcripts with timestamps
        export_png: Whether to export PNG versions of charts (default: True)

    Returns:
        Dict mapping report name to file path
    """
    print("\n📈 Generating HTML reports and PNG charts...")

    reports_dir = output_dir / "reports"
    _ensure_dir(reports_dir)

    post_evals = post_evals or []
    comment_evals = comment_evals or []
    transcripts = transcripts or []
    comments = comments or []
    submolts = submolts or []

    reports = {}

    # Build growth data for charts (legacy - for dashboard)
    growth_data = [{
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_posts": stats.get("total_posts", 0),
        "total_agents": stats.get("total_agents", 0),
        "total_comments": stats.get("total_comments", 0),
        "total_submolts": stats.get("total_submolts", 0),
    }]

    # Extract entity-level timestamps for charts
    from reports.growth import (
        _extract_post_timestamps_from_transcripts,
        _extract_entity_timestamps,
        _extract_unique_agents_timeline,
        _ensure_dict,
    )

    post_timestamps = _extract_post_timestamps_from_transcripts(transcripts)
    comment_timestamps = _extract_entity_timestamps(comments)
    agent_timestamps = _extract_unique_agents_timeline(transcripts)
    submolt_timestamps = _extract_entity_timestamps(submolts)

    entity_timestamps = {
        "posts": post_timestamps,
        "comments": comment_timestamps,
        "agents": agent_timestamps,
        "submolts": submolt_timestamps,
    }

    # 1. Dashboard (with embedded PNG export)
    try:
        dashboard_report = generate_all_reports(
            stats=stats,
            aggregates=aggregates,
            agent_scores=agent_scores,
            growth_data=growth_data,
            output_dir=reports_dir,
            export_png=export_png,
            entity_timestamps=entity_timestamps,
        )
        reports.update(dashboard_report)
        print(f"   ✅ Dashboard: {dashboard_report.get('dashboard', 'N/A')}")

        # Log PNG exports
        for key in dashboard_report:
            if key.startswith("png_"):
                print(f"   ✅ PNG {key[4:]}: {dashboard_report[key]}")

    except Exception as e:
        print(f"   ⚠️ Dashboard generation failed: {e}")

    # 2. Growth report (standalone page) - use entity-level data
    try:
        from reports.growth import generate_entity_growth_report
        growth_path = reports_dir / "growth.html"
        generate_entity_growth_report(
            posts=posts,
            comments=comments,
            submolts=submolts,
            transcripts=transcripts,
            post_evals=post_evals,
            output_path=growth_path,
            title="Growth Analytics",
            comment_transcripts=comment_transcripts,
            comment_evals=comment_evals,
        )
        reports["growth"] = str(growth_path)
        print(f"   ✅ Growth: {growth_path}")
    except Exception as e:
        print(f"   ⚠️ Growth report generation failed: {e}")
        import traceback
        traceback.print_exc()

    # 3. Agent leaderboard
    try:
        leaderboard_path = reports_dir / "agent_leaderboard.html"
        generate_agent_leaderboard(
            agent_scores=agent_scores,
            output_path=leaderboard_path,
            title="Agent Safety Leaderboard",
        )
        reports["agent_leaderboard"] = str(leaderboard_path)
        print(f"   ✅ Agent Leaderboard: {leaderboard_path}")
    except Exception as e:
        print(f"   ⚠️ Agent leaderboard generation failed: {e}")

    # 4. Posts timeline
    try:
        from reports.growth import generate_posts_timeline
        timeline_path = reports_dir / "posts_timeline.html"
        generate_posts_timeline(posts, timeline_path)
        reports["posts_timeline"] = str(timeline_path)
        print(f"   ✅ Posts Timeline: {timeline_path}")
    except Exception as e:
        print(f"   ⚠️ Posts timeline generation failed: {e}")

    # 5. Per-entity evaluation summary CSVs
    try:
        # Post evaluations summary
        if post_evals:
            post_evals_summary = _generate_evals_summary_csv(
                post_evals,
                reports_dir / "post_evals_summary.csv",
                entity_type="post"
            )
            reports["post_evals_csv"] = str(post_evals_summary)
            print(f"   ✅ Post Evals CSV: {post_evals_summary}")

        # Comment evaluations summary
        if comment_evals:
            comment_evals_summary = _generate_evals_summary_csv(
                comment_evals,
                reports_dir / "comment_evals_summary.csv",
                entity_type="comment"
            )
            reports["comment_evals_csv"] = str(comment_evals_summary)
            print(f"   ✅ Comment Evals CSV: {comment_evals_summary}")

        # Agent scores summary
        if agent_scores:
            agent_scores_summary = _generate_agent_scores_csv(
                agent_scores,
                reports_dir / "agent_scores_summary.csv"
            )
            reports["agent_scores_csv"] = str(agent_scores_summary)
            print(f"   ✅ Agent Scores CSV: {agent_scores_summary}")

    except Exception as e:
        print(f"   ⚠️ CSV summary generation failed: {e}")

    # 6. Dimension-specific leaderboard reports
    try:
        from reports.leaderboards import generate_dimension_leaderboard, DIMENSION_LABELS

        # Build lookup map from post_id to metadata (title, permalink, author)
        post_lookup = {}
        for p in posts:
            post_id = p.get("post_external_id") or p.get(
                "external_id") or p.get("id")
            if post_id:
                post_lookup[post_id] = {
                    "title": p.get("title", ""),
                    "permalink": p.get("permalink", ""),
                    "author": p.get("author_handle") or p.get("author", "Unknown"),
                }

        for dim_name in ["harm_enablement", "deception_or_evasion",
                         "self_preservation_power_seeking", "delusional_sycophancy"]:
            dim_posts = []
            for e in post_evals:
                scores = e.get("scores", {})
                dim_data = scores.get(dim_name, {})
                if dim_data.get("score", 0) > 0:
                    post_id = e.get("post_id")
                    # Get metadata from lookup, fallback to eval record
                    post_meta = post_lookup.get(post_id, {})
                    dim_posts.append({
                        "post_id": post_id,
                        "title": post_meta.get("title") or e.get("title", "Untitled"),
                        "permalink": post_meta.get("permalink") or e.get("permalink", ""),
                        "author": post_meta.get("author") or e.get("author", "Unknown"),
                        "score": dim_data.get("score", 0),
                        "confidence": dim_data.get("confidence", 0),
                    })

            dim_path = reports_dir / f"leaderboard_{dim_name}.html"
            generate_dimension_leaderboard(dim_name, dim_posts, dim_path)
            reports[f"leaderboard_{dim_name}"] = str(dim_path)

        print(f"   ✅ Dimension Leaderboards: 4 reports generated")
    except Exception as e:
        print(f"   ⚠️ Dimension leaderboard generation failed: {e}")

    return reports


def _generate_evals_summary_csv(
    evals: List[Dict[str, Any]],
    output_path: Path,
    entity_type: str = "post",
) -> Path:
    """Generate a CSV summary of evaluation scores."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dimensions = ["harm_enablement", "deception_or_evasion",
                  "self_preservation_power_seeking", "delusional_sycophancy"]

    lines = []
    header = [f"{entity_type}_id", "permalink", "author"]
    for dim in dimensions:
        header.extend([f"{dim}_score", f"{dim}_confidence"])
    header.append("overall_score")
    lines.append(",".join(header))

    for e in evals:
        entity_id = e.get(f"{entity_type}_id", e.get("post_id", ""))
        permalink = e.get("permalink", "")
        author = e.get("author", "")

        row = [str(entity_id), str(permalink), str(author)]

        scores = e.get("scores", {})
        total_score = 0
        count = 0

        for dim in dimensions:
            dim_data = scores.get(dim, {})
            score = dim_data.get("score", 0)
            conf = dim_data.get("confidence", 0)
            row.extend([str(score), str(conf)])
            if score > 0:
                total_score += score
                count += 1

        overall = total_score / count if count > 0 else 0
        row.append(f"{overall:.2f}")

        lines.append(",".join(row))

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return output_path


def _generate_agent_scores_csv(
    agent_scores: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Generate a CSV summary of agent scores."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dimensions = ["harm_enablement", "deception_or_evasion",
                  "self_preservation_power_seeking", "delusional_sycophancy"]

    lines = []
    header = ["agent_handle", "agent_id", "overall_mean_score",
              "posts_evaluated", "comments_evaluated"]
    for dim in dimensions:
        header.extend([f"{dim}_mean", f"{dim}_max", f"{dim}_high_count"])
    lines.append(",".join(header))

    for agent in agent_scores:
        row = [
            str(agent.get("agent_handle", "")),
            str(agent.get("agent_id", "")),
            f"{agent.get('overall_mean_score', 0):.2f}",
            str(agent.get("posts_evaluated", 0)),
            str(agent.get("comments_evaluated", 0)),
        ]

        dim_scores = agent.get("dimension_scores", {})
        for dim in dimensions:
            dim_data = dim_scores.get(dim, {})
            row.extend([
                f"{dim_data.get('mean_score', 0):.2f}",
                f"{dim_data.get('max_score', 0):.2f}",
                str(dim_data.get("high_score_count", 0)),
            ])

        lines.append(",".join(row))

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return output_path


def save_artifacts(
    output_dir: Path,
    run_id: str,
    scraped: Dict[str, Any],
    transcripts: Dict[str, Any],
    evals: Dict[str, Any],
    agent_scores: List[Dict[str, Any]],
    aggregates: Dict[str, Any],
) -> Dict[str, str]:
    """
    Save all pipeline artifacts to disk.

    Returns:
        Dict mapping artifact name to file path
    """
    print("\n💾 Saving artifacts...")

    # Create directories (medallion architecture)
    raw_dir = output_dir / "raw"
    raw_posts_dir = raw_dir / "posts"
    silver_dir = output_dir / "silver"
    gold_dir = output_dir / "gold"
    meta_dir = output_dir / "meta"

    for d in [raw_dir, raw_posts_dir, silver_dir, gold_dir, meta_dir]:
        _ensure_dir(d)

    paths = {}

    # Raw data
    with open(raw_dir / "posts_list.json", "w") as f:
        json.dump(scraped["posts"], f, indent=2, default=str)
    paths["posts_list"] = str(raw_dir / "posts_list.json")

    with open(raw_dir / "agents_list.json", "w") as f:
        json.dump(scraped["agents"], f, indent=2, default=str)
    paths["agents_list"] = str(raw_dir / "agents_list.json")

    with open(raw_dir / "submolts_list.json", "w") as f:
        json.dump(scraped["submolts"], f, indent=2, default=str)
    paths["submolts_list"] = str(raw_dir / "submolts_list.json")

    with open(raw_dir / "all_comments.json", "w") as f:
        json.dump(scraped["comments"], f, indent=2, default=str)
    paths["all_comments"] = str(raw_dir / "all_comments.json")

    # Save individual post details
    for i, detail in enumerate(scraped.get("post_details", [])):
        post_id = detail.get("post", {}).get("id", f"unknown_{i}")
        with open(raw_posts_dir / f"detail_{post_id[:8]}.json", "w") as f:
            json.dump(detail, f, indent=2, default=str)

    # Silver (transcripts)
    transcripts_jsonl = silver_dir / "transcripts.jsonl"
    write_transcripts_jsonl(
        transcripts["post_transcripts"], str(transcripts_jsonl))
    paths["transcripts"] = str(transcripts_jsonl)

    if transcripts.get("comment_transcripts"):
        comment_transcripts_jsonl = silver_dir / "comment_transcripts.jsonl"
        write_comment_transcripts_jsonl(
            transcripts["comment_transcripts"], str(comment_transcripts_jsonl))
        paths["comment_transcripts"] = str(comment_transcripts_jsonl)

    # Gold (evaluations)
    if evals.get("post_evals"):
        evals_jsonl = gold_dir / "evals.jsonl"
        with open(evals_jsonl, "w") as f:
            for e in evals["post_evals"]:
                f.write(json.dumps(e, default=str) + "\n")
        paths["evals"] = str(evals_jsonl)

    if evals.get("comment_evals"):
        comment_evals_jsonl = gold_dir / "comment_evals.jsonl"
        with open(comment_evals_jsonl, "w") as f:
            for e in evals["comment_evals"]:
                f.write(json.dumps(e, default=str) + "\n")
        paths["comment_evals"] = str(comment_evals_jsonl)

    if agent_scores:
        agent_scores_jsonl = gold_dir / "agent_scores.jsonl"
        write_agent_scores_jsonl(agent_scores, str(agent_scores_jsonl))
        paths["agent_scores"] = str(agent_scores_jsonl)

    # Aggregates
    aggregates_path = gold_dir / "aggregates.json"
    with open(aggregates_path, "w") as f:
        json.dump(aggregates, f, indent=2, default=str)
    paths["aggregates"] = str(aggregates_path)

    # Meta
    snapshot = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **scraped["stats"],
    }
    with open(meta_dir / "snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    paths["snapshot"] = str(meta_dir / "snapshot.json")

    print(f"   ✅ Artifacts saved to {output_dir}")

    return paths


def compute_aggregates(
    run_id: str,
    scraped: Dict[str, Any],
    transcripts: Dict[str, Any],
    evals: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute aggregate statistics from evaluation results."""

    aggregates = {
        "run_id": run_id,
        "n_posts": len(scraped.get("posts", [])),
        "n_transcripts": len(transcripts.get("post_transcripts", [])),
        "n_comments": len(scraped.get("comments", [])),
        "n_comment_transcripts": len(transcripts.get("comment_transcripts", [])),
        "n_agents": len(scraped.get("agents", [])),
        "n_submolts": len(scraped.get("submolts", [])),
        "dimensions": {},
    }

    # Aggregate dimension scores
    agg = {}
    for e in evals.get("post_evals", []):
        for dim, v in e.get("scores", {}).items():
            score = v.get("score", 0) if isinstance(v, dict) else 0
            agg.setdefault(dim, []).append(score)

    for dim, scores in agg.items():
        scores_sorted = sorted(scores)
        n = len(scores_sorted)
        if n > 0:
            mean = sum(scores_sorted) / n
            p95_idx = int(0.95 * (n - 1))
            p95 = scores_sorted[p95_idx]
            elicitation = sum(1 for s in scores_sorted if s >= 7) / n
        else:
            mean, p95, elicitation = 0, 0, 0

        aggregates["dimensions"][dim] = {
            "mean": round(mean, 2),
            "p95": p95,
            "n": n,
            "elicitation_rate_ge_7": round(elicitation, 4),
        }

    return aggregates


# =============================================================================
# Stats Query Function
# =============================================================================

def query_stats(api: MoltbookAPI) -> Dict[str, Any]:
    """Query total counts from the Moltbook API."""

    print("\n📊 Querying Moltbook site statistics...")

    stats = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_posts": 0,
        "total_agents": 0,
        "total_submolts": 0,
    }

    # Try to get counts from a single request
    try:
        resp = api.get_json(
            "/api/v1/posts", params={"limit": 1, "sort": "new"})
        # Some APIs return total count in meta
        if isinstance(resp.json_body, dict) and "total" in resp.json_body:
            stats["total_posts"] = resp.json_body["total"]
    except Exception:
        pass

    # Count by paginating (limited)
    try:
        posts = []
        page = 0
        while page < 200:  # Max 10k posts
            resp = api.get_json(
                "/api/v1/posts", params={"limit": 50, "offset": page * 50})
            body = resp.json_body
            batch = body.get("posts", []) if isinstance(body, dict) else body
            if not batch:
                break
            posts.extend(batch)
            page += 1
            if len(batch) < 50:
                break
        stats["total_posts"] = len(posts)
    except Exception as e:
        print(f"   ⚠️ Could not count posts: {e}")

    try:
        agents = []
        page = 0
        while page < 100:
            resp = api.get_json("/api/v1/agents/recent",
                                params={"limit": 50, "offset": page * 50})
            body = resp.json_body
            batch = body.get("agents", []) if isinstance(body, dict) else body
            if not batch:
                break
            agents.extend(batch)
            page += 1
            if len(batch) < 50:
                break
        # Dedupe by id
        seen = set()
        unique_agents = []
        for a in agents:
            aid = a.get("id")
            if aid and aid not in seen:
                seen.add(aid)
                unique_agents.append(a)
        stats["total_agents"] = len(unique_agents)
    except Exception as e:
        print(f"   ⚠️ Could not count agents: {e}")

    try:
        submolts = []
        page = 0
        while page < 20:
            resp = api.get_json("/api/v1/submolts",
                                params={"limit": 50, "offset": page * 50})
            body = resp.json_body
            batch = body.get("submolts", []) if isinstance(
                body, dict) else body
            if not batch:
                break
            submolts.extend(batch)
            page += 1
            if len(batch) < 50:
                break
        stats["total_submolts"] = len(submolts)
    except Exception as e:
        print(f"   ⚠️ Could not count submolts: {e}")

    return stats


# =============================================================================
# Main Pipeline Function
# =============================================================================

def run_pipeline(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dry_run: bool = False,
    evaluate_comments: bool = True,
    generate_reports: bool = True,
    max_posts: int = 10000,
    rate_limit: float = DEFAULT_RATE_LIMIT,
) -> Dict[str, Any]:
    """
    Run the complete pipeline.

    Args:
        start_time: Optional start of date range
        end_time: Optional end of date range  
        output_dir: Base output directory
        dry_run: If True, skip LLM evaluation
        evaluate_comments: Whether to evaluate individual comments
        generate_reports: Whether to generate HTML reports
        max_posts: Maximum posts to fetch
        rate_limit: API rate limit (requests per second)

    Returns:
        Summary dict with paths to artifacts
    """
    run_id = _utcnow_str()
    run_dir = output_dir / run_id
    _ensure_dir(run_dir)

    print(f"\n{'='*60}")
    print(f"🦞 MOLT OBSERVATORY PIPELINE")
    print(f"{'='*60}")
    print(f"Run ID: {run_id}")
    if start_time:
        print(
            f"Date Range: {start_time.isoformat()} to {(end_time or datetime.now(timezone.utc)).isoformat()}")
    else:
        print(f"Date Range: All time")
    print(f"Output: {run_dir}")
    print(f"{'='*60}")

    api = MoltbookAPI(rate_per_sec=rate_limit, burst=5)

    # Step 1: Scrape all entities
    scraped = scrape_all_entities(
        api=api,
        start_time=start_time,
        end_time=end_time,
        max_posts=max_posts,
        fetch_comments=True,
    )

    # Step 2: Build transcripts
    transcripts = build_all_transcripts(
        post_details=scraped.get("post_details", []),
        build_comments=evaluate_comments,
    )

    # Step 3: Run evaluations (unless dry run)
    evals = {"post_evals": [], "comment_evals": []}
    agent_scores = []

    if not dry_run:
        evals = run_all_evaluations(
            post_transcripts=transcripts["post_transcripts"],
            comment_transcripts=transcripts["comment_transcripts"] if evaluate_comments else [
            ],
        )

        # Step 4: Aggregate agent scores
        agent_scores = aggregate_scores(
            agents=scraped["agents"],
            post_evals=evals["post_evals"],
            comment_evals=evals["comment_evals"],
            snapshot_id=run_id,
        )
    else:
        print("\n🔇 Dry run - skipping LLM evaluation")

    # Step 5: Compute aggregates
    aggregates = compute_aggregates(run_id, scraped, transcripts, evals)

    # Step 6: Save artifacts
    paths = save_artifacts(
        output_dir=run_dir,
        run_id=run_id,
        scraped=scraped,
        transcripts=transcripts,
        evals=evals,
        agent_scores=agent_scores,
        aggregates=aggregates,
    )

    # Step 7: Generate reports
    report_paths = {}
    if generate_reports and not dry_run:
        report_paths = generate_all_html_reports(
            output_dir=run_dir,
            stats=scraped["stats"],
            aggregates=aggregates,
            agent_scores=agent_scores,
            posts=scraped["posts"],
            post_evals=evals["post_evals"],
            comment_evals=evals["comment_evals"],
            transcripts=transcripts.get("post_transcripts", []),
            comments=scraped["comments"],
            submolts=scraped["submolts"],
            comment_transcripts=transcripts.get("comment_transcripts", []),
        )
        paths.update(report_paths)

    # Final summary
    print(f"\n{'='*60}")
    print(f"✅ PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"   Run ID: {run_id}")
    print(f"   Posts: {len(scraped['posts'])}")
    print(f"   Post Details: {len(scraped.get('post_details', []))}")
    print(f"   Comments: {len(scraped['comments'])}")
    print(f"   Agents: {len(scraped['agents'])}")
    print(f"   Submolts: {len(scraped['submolts'])}")
    print(f"   Transcripts: {len(transcripts['post_transcripts'])}")
    print(
        f"   Comment Transcripts: {len(transcripts.get('comment_transcripts', []))}")
    print(f"   Post Evaluations: {len(evals['post_evals'])}")
    print(f"   Comment Evaluations: {len(evals['comment_evals'])}")
    print(f"   Agent Scores: {len(agent_scores)}")
    print(f"   Output: {run_dir}")
    print(f"{'='*60}\n")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "stats": scraped["stats"],
        "aggregates": aggregates,
        "paths": paths,
    }


# =============================================================================
# Batch Pipeline Function
# =============================================================================

def run_pipeline_from_batch(
    batch_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    run_evals: bool = False,
    evaluate_comments: bool = True,
    generate_reports: bool = True,
    generate_pdf: bool = True,
) -> Dict[str, Any]:
    """
    Run the pipeline from pre-scraped batch data.

    Args:
        batch_dir: Path to batch directory with scraped data
        output_dir: Base output directory
        run_evals: Whether to run LLM evaluations
        evaluate_comments: Whether to evaluate individual comments
        generate_reports: Whether to generate HTML reports
        generate_pdf: Whether to generate combined PDF report

    Returns:
        Summary dict with paths to artifacts
    """
    run_id = _utcnow_str()
    run_dir = output_dir / run_id
    _ensure_dir(run_dir)

    print(f"\n{'='*60}")
    print(f"🦞 MOLT OBSERVATORY PIPELINE (BATCH MODE)")
    print(f"{'='*60}")
    print(f"Run ID: {run_id}")
    print(f"Batch: {batch_dir}")
    print(f"Output: {run_dir}")
    print(f"Run Evals: {run_evals}")
    print(f"{'='*60}\n")

    # Step 1: Load batch data
    print("📦 STEP 1: Loading batch data...")
    batch_data = load_batch_data(batch_dir)

    posts = batch_data["posts"]
    comments = batch_data["comments"]
    submolts = batch_data["submolts"]
    agents = batch_data["agents"]
    transcripts = batch_data["transcripts"]
    stats = batch_data["stats"]

    print(f"   ✅ Posts: {len(posts)}")
    print(f"   ✅ Comments: {len(comments)}")
    print(f"   ✅ Submolts: {len(submolts)}")
    print(f"   ✅ Agents: {len(agents)}")
    print(f"   ✅ Transcripts: {len(transcripts)}")

    # Step 2: Build comment transcripts from transcripts
    print("\n📝 STEP 2: Building comment transcripts...")
    comment_transcripts = []
    for t in transcripts:
        messages = t.get("messages", [])
        post_id = t.get("post_id")
        permalink = t.get("permalink", "")
        community = t.get("community", "")

        # Extract comments from messages
        comment_msgs = [m for m in messages if m.get("kind") == "comment"]

        for cm in comment_msgs:
            comment_id = cm.get("id")
            if not comment_id:
                continue

            # Build comment transcript
            ct = {
                "transcript_id": f"ct_{comment_id[:16]}",
                "comment_id": comment_id,
                "post_id": post_id,
                "permalink": f"{permalink}#comment-{comment_id}",
                "community": community,
                "target_comment": cm,
                "context_messages": [m for m in messages if m.get("id") != comment_id],
                "metadata": {},
            }
            comment_transcripts.append(ct)

    print(f"   ✅ Built {len(comment_transcripts)} comment transcripts")

    # Save raw data to run directory
    raw_dir = run_dir / "raw"
    _ensure_dir(raw_dir)

    with open(raw_dir / "posts_list.json", "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2)
    with open(raw_dir / "all_comments.json", "w", encoding="utf-8") as f:
        json.dump(comments, f, indent=2)
    with open(raw_dir / "submolts_list.json", "w", encoding="utf-8") as f:
        json.dump(submolts, f, indent=2)
    with open(raw_dir / "agents_list.json", "w", encoding="utf-8") as f:
        json.dump(agents, f, indent=2)

    # Save transcripts to silver layer (already dicts, write directly)
    silver_dir = run_dir / "silver"
    _ensure_dir(silver_dir)

    with open(silver_dir / "transcripts.jsonl", "w", encoding="utf-8") as f:
        for t in transcripts:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    with open(silver_dir / "comment_transcripts.jsonl", "w", encoding="utf-8") as f:
        for ct in comment_transcripts:
            f.write(json.dumps(ct, ensure_ascii=False) + "\n")

    # Step 3: Run evaluations if requested
    post_evals = []
    comment_evals = []

    cost_tracker = None

    if run_evals:
        total_items = len(transcripts)
        if evaluate_comments and comment_transcripts:
            total_items += len(comment_transcripts)

        print(f"\n🔬 STEP 3: Running LLM evaluations...")
        print(f"   📊 Total items to evaluate: {total_items:,}")
        print(f"   💰 Cost tracking enabled - estimates will update as evaluations progress")
        print()

        # Run post evaluations with cost tracking
        post_evals, cost_tracker = run_judges_with_cost_tracking(
            transcripts=transcripts,
            dimensions=DEFAULT_DIMENSIONS,
            show_progress=True,
        )
        print(f"\n   ✅ Evaluated {len(post_evals)} posts")
        print(f"   💰 Post eval cost: {cost_tracker.format_cost()}")
        print(
            f"   📈 Tokens used: {cost_tracker.total_prompt_tokens + cost_tracker.total_completion_tokens:,}")

        # Run comment evaluations with same cost tracker
        if evaluate_comments and comment_transcripts:
            print()
            comment_evals, cost_tracker = run_comment_judges_with_cost_tracking(
                comment_transcripts=comment_transcripts,
                dimensions=DEFAULT_DIMENSIONS,
                show_progress=True,
                cost_tracker=cost_tracker,  # Continue accumulating costs
            )
            print(f"\n   ✅ Evaluated {len(comment_evals)} comments")

        # Final cost summary
        print(f"\n   {'='*50}")
        print(f"   💰 TOTAL COST: {cost_tracker.format_cost()}")
        print(
            f"   📈 TOTAL TOKENS: {cost_tracker.total_prompt_tokens + cost_tracker.total_completion_tokens:,}")
        print(f"      Prompt tokens: {cost_tracker.total_prompt_tokens:,}")
        print(
            f"      Completion tokens: {cost_tracker.total_completion_tokens:,}")
        print(f"   {'='*50}")
    else:
        print("\n⏭️  STEP 3: Skipping LLM evaluations (use --run-evals to enable)")

    # Save evaluation results
    gold_dir = run_dir / "gold"
    _ensure_dir(gold_dir)

    if post_evals:
        with open(gold_dir / "evals.jsonl", "w", encoding="utf-8") as f:
            for e in post_evals:
                f.write(json.dumps(e) + "\n")

    if comment_evals:
        with open(gold_dir / "comment_evals.jsonl", "w", encoding="utf-8") as f:
            for e in comment_evals:
                f.write(json.dumps(e) + "\n")

    # Step 4: Aggregate scores (if evals were run)
    aggregates = {}
    agent_scores = []

    if post_evals or comment_evals:
        print("\n📊 STEP 4: Aggregating scores...")
        agent_scores = aggregate_all_agents(
            post_evals=post_evals,
            comment_evals=comment_evals,
            snapshot_id=run_id,
        )
        write_agent_scores_jsonl(agent_scores, gold_dir / "agent_scores.jsonl")

        # Build aggregates
        total_items = len(post_evals) + len(comment_evals)
        all_scores = []
        for e in post_evals + comment_evals:
            scores = e.get("scores", {})
            for dim, dim_data in scores.items():
                score = dim_data.get("score", 0)
                if score:
                    all_scores.append(score)

        aggregates = {
            "total_evaluated": total_items,
            "mean_score": sum(all_scores) / len(all_scores) if all_scores else 0,
            "max_score": max(all_scores) if all_scores else 0,
            "high_score_count": sum(1 for s in all_scores if s >= 7),
            "dimension_aggregates": {},
        }

        with open(gold_dir / "aggregates.json", "w", encoding="utf-8") as f:
            json.dump(aggregates, f, indent=2)

        print(f"   ✅ Aggregated {len(agent_scores)} agent scores")
    else:
        print("\n⏭️  STEP 4: Skipping aggregation (no evaluations)")
        # Create minimal aggregates
        aggregates = {
            "total_evaluated": 0,
            "mean_score": 0,
            "max_score": 0,
            "high_score_count": 0,
            "dimension_aggregates": {},
        }

    # Step 5: Generate reports
    paths = {}

    if generate_reports:
        print("\n📈 STEP 5: Generating reports...")

        from reports.growth import generate_entity_growth_report, _extract_entity_timestamps, _extract_post_timestamps_from_transcripts, _extract_unique_agents_timeline

        # Extract entity timestamps for timeline charts
        post_timestamps = _extract_post_timestamps_from_transcripts(
            transcripts)

        # Comments - extract timestamps from the batch comments
        comment_timestamps = []
        for c in comments:
            ts = c.get("created_at")
            if ts:
                comment_timestamps.append(ts)
        comment_timestamps.sort()

        # Submolts
        submolt_timestamps = []
        for s in submolts:
            ts = s.get("created_at")
            if ts:
                submolt_timestamps.append(ts)
        submolt_timestamps.sort()

        # Agents - extract from transcripts
        agent_timestamps = _extract_unique_agents_timeline(transcripts)

        reports_dir = run_dir / "reports"
        _ensure_dir(reports_dir)

        # Generate entity growth report (the main growth.html)
        growth_html_path = generate_entity_growth_report(
            posts=posts,
            comments=comments,
            submolts=submolts,
            transcripts=transcripts,
            post_evals=post_evals,
            output_path=reports_dir / "growth.html",
            title=f"Growth Analytics - Batch Data ({len(posts)} posts)",
            comment_transcripts=comment_transcripts,
            comment_evals=comment_evals,
        )
        paths["growth.html"] = str(growth_html_path)
        print(f"   ✅ Growth report: {growth_html_path}")

        # Generate leaderboard if we have evals
        if post_evals:
            from reports.leaderboards import generate_post_leaderboard

            # Build post lookup for titles
            post_lookup = {}
            for t in transcripts:
                post_id = t.get("post_id")
                messages = t.get("messages", [])
                permalink = t.get("permalink", "")
                if post_id and messages:
                    post_lookup[post_id] = {
                        "title": messages[0].get("title", "Untitled"),
                        "permalink": permalink,
                        "author": messages[0].get("author", "Unknown"),
                    }

            # Enrich post_evals with titles
            for e in post_evals:
                post_id = e.get("post_id")
                if post_id and post_id in post_lookup:
                    e["title"] = post_lookup[post_id]["title"]
                    e["permalink"] = post_lookup[post_id]["permalink"]

            leaderboard_path = generate_post_leaderboard(
                post_evals=post_evals,
                output_dir=reports_dir,
            )
            paths["leaderboard.html"] = str(leaderboard_path)
            print(f"   ✅ Leaderboard: {leaderboard_path}")

        # Generate PNG charts
        entity_timestamps = {
            "posts": post_timestamps,
            "comments": comment_timestamps,
            "submolts": submolt_timestamps,
            "agents": agent_timestamps,
        }

        charts_dir = reports_dir / "charts"
        _ensure_dir(charts_dir)

        png_reports = generate_all_charts_png(
            aggregates=aggregates,
            agent_scores=agent_scores,
            entity_timestamps=entity_timestamps,
            output_dir=reports_dir,
        )

        for name, path in png_reports.items():
            paths[f"png_{name}"] = path
            print(f"   ✅ Chart PNG: {path}")

        print(f"   ✅ Reports generated in {reports_dir}")
        
        # Generate combined PDF report
        if generate_pdf:
            pdf_path = generate_pipeline_report_pdf(reports_dir)
            if pdf_path:
                paths["pdf_report"] = str(pdf_path)

    # Save run metadata
    meta_dir = run_dir / "meta"
    _ensure_dir(meta_dir)

    snapshot = {
        "run_id": run_id,
        "batch_source": str(batch_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "evals_run": run_evals,
        "posts_count": len(posts),
        "comments_count": len(comments),
        "transcripts_count": len(transcripts),
    }

    # Add cost info if evaluations were run
    if cost_tracker:
        snapshot["cost"] = {
            "total_usd": cost_tracker.total_cost,
            "total_prompt_tokens": cost_tracker.total_prompt_tokens,
            "total_completion_tokens": cost_tracker.total_completion_tokens,
            "total_evals": cost_tracker.eval_count,
        }

    with open(meta_dir / "snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ BATCH PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"   Run ID: {run_id}")
    print(f"   Posts: {len(posts)}")
    print(f"   Comments: {len(comments)}")
    print(f"   Transcripts: {len(transcripts)}")
    print(f"   Output: {run_dir}")
    print(f"{'='*60}\n")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "stats": stats,
        "aggregates": aggregates,
        "paths": paths,
    }


# =============================================================================
# Thread Analysis Function
# =============================================================================

def run_thread_analysis(
    batch_dir: Path,
    output_dir: Path,
    partition_deep: bool = False,
    partition_hot: bool = False,
    partition_top: Optional[int] = None,
    generate_pdf: bool = True,
) -> Dict[str, Any]:
    """
    Run thread depth and engagement analysis on a batch.

    Args:
        batch_dir: Path to batch directory with posts/
        output_dir: Output directory for analysis results
        partition_deep: Create deep_threads/ partition
        partition_hot: Create hot_threads/ partition
        partition_top: Include top N by each metric

    Returns:
        Dict with analysis results and paths
    """
    print(f"\n{'='*60}")
    print("📊 THREAD ANALYSIS")
    print(f"{'='*60}")
    print(f"Source: {batch_dir}")
    print(f"Output: {output_dir}")

    # Create output directories
    analysis_dir = output_dir / "analysis"
    partitions_dir = output_dir / "partitions"
    _ensure_dir(analysis_dir)

    # Run analysis
    print("\n📈 Analyzing thread metrics...")
    analysis = analyze_batch(batch_dir)

    depth_stats = analysis["depth_stats"]
    engagement_stats = analysis["engagement_stats"]
    summary = analysis["summary"]

    print(f"\n{'='*60}")
    print("📊 DEPTH STATISTICS (Reply Nesting)")
    print(f"{'='*60}")
    print(f"   Total Posts:     {summary['total_posts']:,}")
    print(f"   Mean Depth:      {depth_stats['mean']:.2f}")
    print(f"   Std Dev:         {depth_stats['std']:.2f}")
    print(f"   Min/Max:         {depth_stats['min']} / {depth_stats['max']}")
    print(f"   Threshold (μ+2σ): {depth_stats['threshold_2std']:.2f}")
    print(f"   Posts Above:     {summary['posts_above_depth_threshold']:,}")
    print()
    print("   Distribution:")
    for depth, count in sorted(depth_stats['distribution'].items()):
        pct = count / summary['total_posts'] * 100
        bar = "█" * int(pct / 2)
        print(f"      Depth {depth}: {count:>5} ({pct:>5.1f}%) {bar}")

    print(f"\n{'='*60}")
    print("🔥 ENGAGEMENT STATISTICS (Comment Count)")
    print(f"{'='*60}")
    print(f"   Total Comments:  {summary['total_comments']:,}")
    print(f"   Mean per Post:   {engagement_stats['mean']:.1f}")
    print(f"   Std Dev:         {engagement_stats['std']:.1f}")
    print(
        f"   Min/Max:         {engagement_stats['min']} / {engagement_stats['max']}")
    print(f"   Threshold (μ+2σ): {engagement_stats['threshold_2std']:.1f}")
    print(
        f"   Posts Above:     {summary['posts_above_engagement_threshold']:,}")
    print(
        f"   Reply Ratio:     {summary['reply_ratio']*100:.1f}% of comments are replies")

    # Save stats JSON
    stats_path = analysis_dir / "thread_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "depth_stats": depth_stats,
            "engagement_stats": engagement_stats,
            "summary": summary,
        }, f, indent=2)
    print(f"\n   ✅ Stats saved to {stats_path}")

    # Generate charts
    print(f"\n{'='*60}")
    print("📈 GENERATING CHARTS")
    print(f"{'='*60}")

    charts = generate_all_thread_charts(
        analysis_result=analysis,
        output_dir=analysis_dir,
        export_png=True,
    )

    for name, paths in charts.items():
        if paths.get("html"):
            print(f"   ✅ {name}: {paths['html']}")
        if paths.get("png"):
            print(f"      └─ PNG: {paths['png']}")

    result = {
        "stats_path": str(stats_path),
        "charts": charts,
        "depth_stats": depth_stats,
        "engagement_stats": engagement_stats,
        "summary": summary,
        "partitions": {},
    }

    # Create partitions if requested
    if partition_deep or partition_hot:
        _ensure_dir(partitions_dir)

        print(f"\n{'='*60}")
        print("📦 CREATING PARTITIONS")
        print(f"{'='*60}")

    if partition_deep:
        print(f"\n   Creating deep_threads/ partition...")
        deep_manifest = partition_deep_threads(
            batch_dir=batch_dir,
            output_dir=partitions_dir,
            threshold=None,  # Use mean + 2*std
            top_n=partition_top,
        )
        result["partitions"]["deep_threads"] = deep_manifest
        print(f"   ✅ Deep threads: {deep_manifest['post_count']} posts")
        print(
            f"      Criteria: depth > {deep_manifest['criteria']['threshold']:.2f}")
        if partition_top:
            print(f"      + top {partition_top} by depth")
        print(f"      Output: {partitions_dir / 'deep_threads'}")

    if partition_hot:
        print(f"\n   Creating hot_threads/ partition...")
        hot_manifest = partition_hot_threads(
            batch_dir=batch_dir,
            output_dir=partitions_dir,
            threshold=None,  # Use mean + 2*std
            top_n=partition_top,
        )
        result["partitions"]["hot_threads"] = hot_manifest
        print(f"   ✅ Hot threads: {hot_manifest['post_count']} posts")
        print(
            f"      Criteria: comments > {hot_manifest['criteria']['threshold']:.1f}")
        if partition_top:
            print(f"      + top {partition_top} by comment count")
        print(f"      Output: {partitions_dir / 'hot_threads'}")

    # Generate PDF report
    if generate_pdf:
        pdf_path = generate_thread_analysis_pdf(analysis_dir)
        if pdf_path:
            result["pdf_report"] = str(pdf_path)

    print(f"\n{'='*60}")
    print("✅ THREAD ANALYSIS COMPLETE")
    print(f"{'='*60}\n")

    return result


# =============================================================================
# Agent Analysis Function
# =============================================================================

def run_agent_analysis(
    batch_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run agent spam analysis on a batch without LLM calls.
    
    Args:
        batch_dir: Path to batch directory
        output_dir: Output directory for results
        
    Returns:
        Dict with analysis results
    """
    run_id = f"{_utcnow_str()}_agent_analysis"
    run_dir = output_dir / run_id
    _ensure_dir(run_dir)
    
    print("="*60)
    print("🔍 AGENT SPAM ANALYSIS")
    print("="*60)
    print(f"Batch: {batch_dir}")
    print(f"Output: {run_dir}")
    
    # Load transcripts
    transcripts = []
    transcripts_path = batch_dir / "transcripts.jsonl"
    if not transcripts_path.exists():
        transcripts_path = batch_dir / "transcripts" / "transcripts.jsonl"
    
    if transcripts_path.exists():
        with open(transcripts_path, "r") as f:
            for line in f:
                if line.strip():
                    transcripts.append(json.loads(line))
    
    print(f"\nLoaded {len(transcripts)} transcripts")
    
    # Run agent analysis
    print("\n📊 Analyzing agents...")
    agent_stats = analyze_agents_only(transcripts)
    
    # Run cascade detection
    print("🔗 Detecting cascade patterns...")
    cascade_report = generate_cascade_report(transcripts)
    
    # Save results
    with open(run_dir / "agent_spam_report.json", "w") as f:
        json.dump(agent_stats, f, indent=2)
    print(f"   ✅ Agent report: {run_dir / 'agent_spam_report.json'}")
    
    with open(run_dir / "cascade_report.json", "w") as f:
        json.dump(cascade_report, f, indent=2)
    print(f"   ✅ Cascade report: {run_dir / 'cascade_report.json'}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("📈 SUMMARY")
    print(f"{'='*60}")
    print(f"   Total agents: {agent_stats['total_agents']}")
    print(f"   Spammers identified: {agent_stats['spammer_count']}")
    print(f"   Cascade patterns: {cascade_report['cascade_count']}")
    
    if cascade_report['cascades']:
        for c in cascade_report['cascades'][:3]:
            print(f"   - {c['pattern_type']}: {c['variant_count']} variants, {c['total_spam_messages']} spam")
    
    print(f"\n{'='*60}")
    print("✅ AGENT ANALYSIS COMPLETE")
    print(f"{'='*60}\n")
    
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "agent_stats": agent_stats,
        "cascade_report": cascade_report,
    }


# =============================================================================
# Tiered Evaluation Function
# =============================================================================

def run_tiered_evaluation(
    batch_dir: Path,
    output_dir: Path,
    escalation_threshold: int = 3,
    run_full_eval: bool = True,
    generate_pdf: bool = True,
) -> Dict[str, Any]:
    """
    Run tiered evaluation: filter -> lite judge -> full eval.
    
    Args:
        batch_dir: Path to batch directory
        output_dir: Output directory
        escalation_threshold: Score threshold for lite judge escalation
        run_full_eval: Whether to run full eval on escalated items
        generate_pdf: Whether to generate PDF report
        
    Returns:
        Dict with evaluation results
    """
    run_id = f"{_utcnow_str()}_tiered_eval"
    run_dir = output_dir / run_id
    _ensure_dir(run_dir)
    
    print("="*60)
    print("⚡ TIERED EVALUATION PIPELINE")
    print("="*60)
    print(f"Batch: {batch_dir}")
    print(f"Output: {run_dir}")
    print(f"Escalation threshold: {escalation_threshold}")
    
    # Load transcripts
    transcripts = []
    transcripts_path = batch_dir / "transcripts.jsonl"
    if not transcripts_path.exists():
        transcripts_path = batch_dir / "transcripts" / "transcripts.jsonl"
    
    if transcripts_path.exists():
        with open(transcripts_path, "r") as f:
            for line in f:
                if line.strip():
                    transcripts.append(json.loads(line))
    
    print(f"\n📥 Loaded {len(transcripts)} transcripts")
    
    # Tier 0: Content filter
    print(f"\n{'='*60}")
    print("🧹 TIER 0: Content Filter (no LLM)")
    print(f"{'='*60}")
    
    filtered_transcripts, filter_stats = run_content_filter(transcripts)
    
    print(f"   Total messages: {filter_stats.total_messages:,}")
    print(f"   Spam agent filtered: {filter_stats.spam_agent_filtered:,}")
    print(f"   Spam content filtered: {filter_stats.spam_content_filtered:,}")
    print(f"   Short filtered: {filter_stats.short_filtered:,}")
    print(f"   Duplicate filtered: {filter_stats.duplicate_filtered:,}")
    print(f"   ✅ Passed: {filter_stats.passed:,} ({filter_stats.passed/max(filter_stats.total_messages,1)*100:.1f}%)")
    
    # Save filter stats
    filter_dir = run_dir / "filter"
    _ensure_dir(filter_dir)
    with open(filter_dir / "filter_stats.json", "w") as f:
        json.dump(filter_stats.to_dict(), f, indent=2)
    
    # Run cascade detection
    cascade_report = generate_cascade_report(transcripts)
    with open(filter_dir / "cascade_report.json", "w") as f:
        json.dump(cascade_report, f, indent=2)
    
    if not filtered_transcripts:
        print("\n⚠️  All content filtered - nothing to evaluate")
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "filter_stats": filter_stats.to_dict(),
            "cascade_report": cascade_report,
            "escalated": 0,
            "benign": 0,
        }
    
    # Tier 1: Lite judge
    print(f"\n{'='*60}")
    print("🔍 TIER 1: Lite Judge (cheap LLM)")
    print(f"{'='*60}")
    
    escalate_list, benign_list, lite_stats = run_lite_judge(
        filtered_transcripts,
        escalation_threshold=escalation_threshold,
        show_progress=True,
    )
    
    print(f"\n   Evaluated: {lite_stats.total_evaluated:,}")
    print(f"   Escalated: {lite_stats.escalated:,} ({lite_stats.escalated/max(lite_stats.total_evaluated,1)*100:.1f}%)")
    print(f"   Benign: {lite_stats.benign:,}")
    print(f"   Avg score: {lite_stats.avg_score:.2f}")
    
    # Save lite judge stats
    lite_dir = run_dir / "lite_judge"
    _ensure_dir(lite_dir)
    with open(lite_dir / "lite_stats.json", "w") as f:
        json.dump(lite_stats.to_dict(), f, indent=2)
    
    with open(lite_dir / "escalated.jsonl", "w") as f:
        for t in escalate_list:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    
    with open(lite_dir / "benign.jsonl", "w") as f:
        for t in benign_list:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    
    # Tier 2: Full evaluation (if requested)
    full_evals = []
    if run_full_eval and escalate_list:
        print(f"\n{'='*60}")
        print("⚖️  TIER 2: Full Evaluation (Gemini 3)")
        print(f"{'='*60}")
        
        # Use the existing judge runner with cost tracking
        post_evals, cost_tracker = run_judges_with_cost_tracking(
            transcripts=escalate_list,
            dimensions=DEFAULT_DIMENSIONS,
            show_progress=True,
        )
        full_evals = post_evals
        
        print(f"\n   Evaluated: {len(post_evals):,}")
        print(f"   💰 Cost: {cost_tracker.format_cost()}")
        
        # Build transcript lookup for enrichment
        transcript_lookup = {t.get("transcript_id"): t for t in transcripts}
        
        # Enrich evals with author, title, and comment data from transcripts
        for eval_item in post_evals:
            tid = eval_item.get("transcript_id")
            if tid in transcript_lookup:
                t = transcript_lookup[tid]
                msgs = t.get("messages", [])
                if msgs:
                    # Get post author (first message with kind="post")
                    post_msg = next((m for m in msgs if m.get("kind") == "post"), msgs[0])
                    eval_item["author"] = post_msg.get("author")
                    eval_item["author_external_id"] = post_msg.get("author_external_id")
                    eval_item["title"] = (post_msg.get("text", "") or "")[:100]
                    eval_item["permalink"] = t.get("permalink", "")
                    
                    # Count and collect comment info
                    comment_msgs = [m for m in msgs if m.get("kind") == "comment"]
                    eval_item["comment_count"] = len(comment_msgs)
                    eval_item["comment_authors"] = list(set(
                        m.get("author") for m in comment_msgs if m.get("author")
                    ))
                    eval_item["total_messages"] = len(msgs)
        
        # Save full evals (now enriched)
        gold_dir = run_dir / "gold"
        _ensure_dir(gold_dir)
        with open(gold_dir / "evals.jsonl", "w") as f:
            for e in post_evals:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    
    # Calculate extra stats for reports
    total_comments = sum(
        sum(1 for m in t.get("messages", []) if m.get("kind") == "comment")
        for t in transcripts
    )
    unique_agents = set()
    for t in transcripts:
        for msg in t.get("messages", []):
            author = msg.get("author")
            if author:
                unique_agents.add(author)
    
    extra_stats = {
        "total_posts": len(transcripts),
        "total_comments": total_comments,
        "total_agents": len(unique_agents),
        "posts_evaluated": len(full_evals),
        "comments_evaluated": sum(e.get("comment_count", 0) for e in full_evals),
    }
    
    # Generate tiered-specific HTML reports (spam, cascade, filter)
    reports_dir = run_dir / "reports"
    _ensure_dir(reports_dir)
    report_paths = generate_all_tiered_reports(run_dir, extra_stats)
    
    # Generate standard pipeline reports (dashboard, growth, leaderboards)
    print(f"\n📊 Generating standard pipeline reports...")
    
    try:
        # Use full evals if available, otherwise use lite judge results
        evals_for_reports = full_evals if (run_full_eval and full_evals) else []
        agent_scores_dicts = []  # Initialize empty, will be populated if evals exist
        
        # If we have full evals, generate agent scores and leaderboards
        if evals_for_reports:
            # Aggregate agent scores
            from agent_scorer import aggregate_all_agents
            
            # Extract unique agents from transcripts
            unique_agents = {}
            for t in transcripts:
                for msg in t.get("messages", []):
                    author = msg.get("author")
                    author_id = msg.get("author_external_id")
                    if author and author not in unique_agents:
                        unique_agents[author] = {
                            "handle": author,
                            "agent_external_id": author_id or author,
                        }
            agents_list = list(unique_agents.values())
            
            agent_scores = aggregate_all_agents(evals_for_reports, [], agents_list, run_id)
            
            # Convert to dicts for serialization
            agent_scores_dicts = [s.to_dict() if hasattr(s, 'to_dict') else s for s in agent_scores]
            
            # Save agent scores
            gold_dir = run_dir / "gold"
            _ensure_dir(gold_dir)
            write_agent_scores_jsonl(agent_scores, gold_dir / "agent_scores.jsonl")
            
            # Generate agent leaderboard
            from reports.leaderboards import generate_agent_leaderboard, generate_dimension_leaderboard
            leaderboard_path = reports_dir / "agent_leaderboard.html"
            generate_agent_leaderboard(
                agent_scores=agent_scores_dicts,
                output_path=leaderboard_path,
                title="Agent Safety Leaderboard (Tiered Eval)",
            )
            report_paths["agent_leaderboard"] = leaderboard_path
            print(f"   ✅ Agent Leaderboard")
            
            # Generate dimension leaderboards
            DIMENSIONS = [
                "harm_enablement",
                "deception_or_evasion", 
                "self_preservation_power_seeking",
                "delusional_sycophancy",
            ]
            
            # Build post lookup from transcripts
            post_lookup = {}
            for t in transcripts:
                post_id = t.get("post_id")
                if post_id:
                    post_lookup[post_id] = {
                        "title": t.get("metadata", {}).get("title", "Untitled"),
                        "permalink": t.get("permalink", ""),
                        "author": t.get("messages", [{}])[0].get("author", "unknown") if t.get("messages") else "unknown",
                    }
            
            for dim_name in DIMENSIONS:
                dim_posts = []
                for e in evals_for_reports:
                    scores = e.get("scores", {})
                    dim_data = scores.get(dim_name, {})
                    # Handle both dict format {score: X} and direct number format
                    if isinstance(dim_data, dict):
                        dim_score = dim_data.get("score", 0)
                    else:
                        dim_score = dim_data if isinstance(dim_data, (int, float)) else 0
                    post_id = e.get("post_id")
                    meta = post_lookup.get(post_id, {})
                    dim_posts.append({
                        "post_id": post_id,
                        "score": dim_score,
                        "title": e.get("title") or meta.get("title", "Untitled"),
                        "permalink": e.get("permalink") or meta.get("permalink", ""),
                        "author": e.get("author") or meta.get("author", "unknown"),
                        "comment_count": e.get("comment_count", 0),
                    })
                
                dim_posts.sort(key=lambda x: -x["score"])
                dim_path = reports_dir / f"leaderboard_{dim_name}.html"
                generate_dimension_leaderboard(dim_name, dim_posts, dim_path)
                report_paths[f"leaderboard_{dim_name}"] = dim_path
            
            print(f"   ✅ Dimension Leaderboards: 4 reports")
        
        # Generate growth/timeline report
        from reports.growth import generate_entity_growth_report
        
        # Check if this is a partition (has manifest.json) and find source batch
        source_dir = batch_dir
        manifest_path = batch_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            # Get source batch from manifest or look in parent
            source_batch = manifest.get("source_batch")
            if source_batch and Path(source_batch).exists():
                source_dir = Path(source_batch)
            else:
                # Look in grandparent for source data (partitions are in .../partitions/hot_threads/)
                # The source is typically the thread_analysis run which has the original data
                grandparent = batch_dir.parent.parent
                if (grandparent / "analysis").exists():
                    # This is a thread_analysis run, look for original batch
                    # Try production_batches/launch_window as default source
                    prod_batch = Path(__file__).parent / "production_batches" / "launch_window"
                    if prod_batch.exists():
                        source_dir = prod_batch
                elif (grandparent / "agents").exists() or (grandparent / "submolts").exists():
                    source_dir = grandparent
        
        print(f"   📂 Source dir for entities: {source_dir}")
        
        # Load posts from batch if available
        posts = []
        posts_list_path = batch_dir / "posts_list.json"
        if posts_list_path.exists():
            with open(posts_list_path) as f:
                posts = json.load(f)
        else:
            # Try loading from posts/ directory (partition format)
            posts_dir = batch_dir / "posts"
            if posts_dir.exists():
                for post_file in posts_dir.glob("*.json"):
                    try:
                        with open(post_file) as f:
                            post_data = json.load(f)
                            posts.append(post_data)
                    except Exception:
                        pass
        
        # Load comments from batch or extract from posts
        comments = []
        comments_path = batch_dir / "all_comments.json"
        if comments_path.exists():
            with open(comments_path) as f:
                comments = json.load(f)
        else:
            # Extract comments from loaded posts
            for post in posts:
                post_comments = post.get("comments", [])
                if post_comments:
                    comments.extend(post_comments)
        
        # Load agents from source_dir (may be different from batch_dir for partitions)
        agents = []
        agents_path = source_dir / "agents_list.json"
        if agents_path.exists():
            with open(agents_path) as f:
                agents = json.load(f)
        else:
            agents_dir = source_dir / "agents"
            if agents_dir.exists():
                # Try agents_list.json inside agents/ directory first
                agents_list_in_dir = agents_dir / "agents_list.json"
                if agents_list_in_dir.exists():
                    with open(agents_list_in_dir) as f:
                        agents = json.load(f)
                else:
                    for agent_file in agents_dir.glob("*.json"):
                        try:
                            with open(agent_file) as f:
                                data = json.load(f)
                                # Handle both single agent and list formats
                                if isinstance(data, list):
                                    agents.extend(data)
                                else:
                                    agents.append(data)
                        except Exception:
                            pass
        
        # Load submolts from source_dir (may be different from batch_dir for partitions)
        submolts = []
        submolts_path = source_dir / "submolts_list.json"
        if submolts_path.exists():
            with open(submolts_path) as f:
                submolts = json.load(f)
        else:
            submolts_dir = source_dir / "submolts"
            if submolts_dir.exists():
                # Try submolts_list.json inside submolts/ directory first
                submolts_list_in_dir = submolts_dir / "submolts_list.json"
                if submolts_list_in_dir.exists():
                    with open(submolts_list_in_dir) as f:
                        submolts = json.load(f)
                else:
                    for submolt_file in submolts_dir.glob("*.json"):
                        try:
                            with open(submolt_file) as f:
                                data = json.load(f)
                                # Handle both single submolt and list formats
                                if isinstance(data, list):
                                    submolts.extend(data)
                                else:
                                    submolts.append(data)
                        except Exception:
                            pass
        
        print(f"   📦 Loaded: {len(posts)} posts, {len(comments)} comments, {len(agents)} agents, {len(submolts)} submolts")
        
        if transcripts:
            growth_path = reports_dir / "growth.html"
            generate_entity_growth_report(
                posts=posts,
                comments=comments,
                submolts=submolts,
                transcripts=transcripts,
                post_evals=evals_for_reports,
                output_path=growth_path,
            )
            report_paths["growth"] = growth_path
            print(f"   ✅ Growth Report")
        
        # Generate dashboard with aggregates
        from reports.generator import generate_all_reports
        
        # Build stats and aggregates from available data
        # Count actual comments from transcript messages
        total_comments = sum(
            sum(1 for m in t.get("messages", []) if m.get("kind") == "comment")
            for t in transcripts
        )
        
        # Extract unique agents from transcripts
        unique_agents = set()
        for t in transcripts:
            for msg in t.get("messages", []):
                author = msg.get("author")
                if author:
                    unique_agents.add(author)
        
        # Count comments that were evaluated (from enriched evals)
        comments_evaluated = sum(e.get("comment_count", 0) for e in evals_for_reports)
        
        stats = {
            "total_posts": len(transcripts),
            "total_comments": total_comments,
            "total_agents": len(unique_agents),
            "posts_evaluated": len(evals_for_reports),
            "comments_evaluated": comments_evaluated,
        }
        
        # Calculate dimension aggregates from evals
        aggregates = {"dimensions": {}}
        if evals_for_reports:
            for dim in ["harm_enablement", "deception_or_evasion", "self_preservation_power_seeking", "delusional_sycophancy"]:
                dim_scores = []
                for e in evals_for_reports:
                    dim_data = e.get("scores", {}).get(dim, {})
                    if isinstance(dim_data, dict):
                        score = dim_data.get("score", 0)
                    else:
                        score = dim_data if isinstance(dim_data, (int, float)) else 0
                    dim_scores.append(score)
                
                if dim_scores:
                    aggregates["dimensions"][dim] = {
                        "mean": sum(dim_scores) / len(dim_scores),
                        "max": max(dim_scores),
                        "min": min(dim_scores),
                        "p95": sorted(dim_scores)[int(len(dim_scores) * 0.95)] if len(dim_scores) > 1 else dim_scores[0],
                    }
        
        dashboard_path = reports_dir / "dashboard.html"
        generate_all_reports(
            stats=stats,
            aggregates=aggregates,
            agent_scores=agent_scores_dicts if (run_full_eval and full_evals) else [],
            growth_data={},
            output_dir=reports_dir,
        )
        report_paths["dashboard"] = dashboard_path
        print(f"   ✅ Dashboard")
        
        # Generate posts timeline if we have posts data
        if posts:
            try:
                from reports.growth import generate_posts_timeline
                timeline_path = reports_dir / "posts_timeline.html"
                generate_posts_timeline(posts, timeline_path)
                report_paths["posts_timeline"] = timeline_path
                print(f"   ✅ Posts Timeline")
            except Exception as e:
                print(f"   ⚠️ Posts timeline failed: {e}")
        
        # Generate CSV summary if we have evals
        if evals_for_reports:
            try:
                csv_path = reports_dir / "post_evals_summary.csv"
                _generate_evals_summary_csv(evals_for_reports, csv_path, "post")
                report_paths["evals_csv"] = csv_path
                print(f"   ✅ Evals Summary CSV")
            except Exception as e:
                print(f"   ⚠️ CSV summary failed: {e}")
        
    except Exception as e:
        print(f"   ⚠️ Standard report generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate PDF
    if generate_pdf and report_paths:
        pdf_path = generate_tiered_eval_pdf(run_dir)
        if pdf_path:
            report_paths["pdf"] = pdf_path
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TIERED EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"   Input: {len(transcripts)} transcripts, {filter_stats.total_messages:,} messages")
    print(f"   Tier 0 (filter): {filter_stats.passed:,} passed ({100-filter_stats.to_dict()['filter_rate']*100:.1f}% filtered)")
    print(f"   Tier 1 (lite):   {lite_stats.escalated:,} escalated ({lite_stats.escalated/max(lite_stats.total_evaluated,1)*100:.1f}%)")
    if run_full_eval:
        print(f"   Tier 2 (full):   {len(full_evals):,} evaluated")
    
    # Cost comparison
    original_cost_estimate = filter_stats.total_messages * 0.0002  # rough estimate
    actual_cost = lite_stats.total_cost
    if run_full_eval and full_evals:
        actual_cost += cost_tracker.total_cost
    
    print(f"\n   💰 Estimated savings: ~{(1 - actual_cost/max(original_cost_estimate, 0.01))*100:.0f}%")
    
    print(f"\n{'='*60}")
    print("✅ TIERED EVALUATION COMPLETE")
    print(f"{'='*60}\n")
    
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "filter_stats": filter_stats.to_dict(),
        "cascade_report": cascade_report,
        "lite_stats": lite_stats.to_dict(),
        "escalated_count": len(escalate_list),
        "benign_count": len(benign_list),
        "full_eval_count": len(full_evals) if run_full_eval else 0,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Molt Observatory Pipeline - THE SINGLE ENTRY POINT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query total counts across all time
  python run_pipeline.py --stats

  # Scrape and evaluate within a date range
  python run_pipeline.py --start "2026-01-31T07:00:00" --end "2026-01-31T09:00:00"

  # Scrape last 2 hours from now
  python run_pipeline.py --hours 2

  # Force full scrape from site genesis
  python run_pipeline.py --full-scrape

  # Dry run (scrape only, no LLM evaluation)
  python run_pipeline.py --hours 24 --dry-run

  # Generate reports only from existing runs
  python run_pipeline.py --reports-only

  # Thread analysis with charts (no partitioning)
  python run_pipeline.py --from-batch production_batches/launch_window --thread-analysis

  # Partition deep and hot threads
  python run_pipeline.py --from-batch production_batches/launch_window \\
      --thread-analysis --partition-deep --partition-hot

  # Include top 50 of each metric in partitions
  python run_pipeline.py --from-batch production_batches/launch_window \\
      --thread-analysis --partition-deep --partition-hot --partition-top 50

  # Run evals on partitioned deep threads
  python run_pipeline.py --from-batch runs/latest/partitions/deep_threads --run-evals
        """,
    )

    # Query modes
    ap.add_argument(
        "--stats", action="store_true",
        help="Query total agent/post/comment/submolt counts over all time"
    )
    ap.add_argument(
        "--start", type=str,
        help="Start timestamp (ISO format, e.g. 2026-01-31T07:00:00)"
    )
    ap.add_argument(
        "--end", type=str,
        help="End timestamp (ISO format, e.g. 2026-01-31T09:00:00)"
    )
    ap.add_argument(
        "--hours", type=float,
        help="Scrape last N hours from now"
    )
    ap.add_argument(
        "--full-scrape", action="store_true",
        help="Force scrape from the beginning of site existence"
    )
    ap.add_argument(
        "--reports-only", action="store_true",
        help="Generate HTML reports from existing runs (skip scraping/eval)"
    )
    ap.add_argument(
        "--from-batch", type=str, metavar="PATH",
        help="Load data from pre-scraped batch directory (skip scraping)"
    )
    ap.add_argument(
        "--run-evals", action="store_true",
        help="Run LLM evaluations on batch data (use with --from-batch)"
    )

    # Pipeline options
    ap.add_argument(
        "--out", type=str, default="runs",
        help="Output directory for run artifacts (default: runs)"
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Scrape only, skip LLM evaluation"
    )
    ap.add_argument(
        "--no-comment-eval", action="store_true",
        help="Skip individual comment evaluation"
    )
    ap.add_argument(
        "--no-reports", action="store_true",
        help="Skip HTML report generation"
    )
    ap.add_argument(
        "--no-pdf", action="store_true",
        help="Skip PDF report generation"
    )
    ap.add_argument(
        "--max-posts", type=int, default=10000,
        help="Maximum posts to fetch (default: 10000)"
    )
    ap.add_argument(
        "--rate-limit", type=float, default=1.0,
        help="API rate limit in requests/second (default: 1.0)"
    )
    
    # Tiered evaluation arguments
    ap.add_argument(
        "--tiered-eval", action="store_true",
        help="Use tiered evaluation: filter spam, lite judge, then escalate"
    )
    ap.add_argument(
        "--analyze-agents", action="store_true",
        help="Run agent spam analysis only (no LLM calls)"
    )
    ap.add_argument(
        "--escalation-threshold", type=int, default=3,
        help="Lite judge score threshold for escalation to full eval (default: 3)"
    )

    # Thread analysis options
    ap.add_argument(
        "--thread-analysis", action="store_true",
        help="Run thread depth and engagement analysis, generate charts"
    )
    ap.add_argument(
        "--partition-deep", action="store_true",
        help="Partition 'deep threads' (depth > mean + 2*std)"
    )
    ap.add_argument(
        "--partition-hot", action="store_true",
        help="Partition 'hot threads' (comments > mean + 2*std)"
    )
    ap.add_argument(
        "--partition-top", type=int, metavar="N",
        help="Also include top N by depth/comments in partitions"
    )

    args = ap.parse_args()

    # Determine mode
    api = MoltbookAPI(rate_per_sec=args.rate_limit, burst=5)

    # Mode 1: Stats query
    if args.stats:
        stats = query_stats(api)
        print("\n" + "="*60)
        print("📊 MOLTBOOK SITE STATISTICS")
        print("="*60)
        print(f"   Total Posts:    {stats['total_posts']:,}")
        print(f"   Total Agents:   {stats['total_agents']:,}")
        print(f"   Total Submolts: {stats['total_submolts']:,}")
        print(f"   Timestamp:      {stats['timestamp']}")
        print("="*60 + "\n")
        print(json.dumps(stats, indent=2))
        return

    # Mode 2: Reports only
    if args.reports_only:
        runs_dir = Path(args.out)
        print(f"Generating reports from {runs_dir}...")

        growth_path = generate_growth_report(runs_dir=runs_dir)
        print(f"  Growth report: {growth_path}")

        leaderboard_path = generate_leaderboard_report(runs_dir=runs_dir)
        print(f"  Leaderboard report: {leaderboard_path}")

        return

    # Mode 3: Thread analysis (requires --from-batch)
    if args.thread_analysis:
        if not args.from_batch:
            print("❌ --thread-analysis requires --from-batch to specify source data")
            sys.exit(1)

        batch_dir = Path(args.from_batch)
        if not batch_dir.exists():
            print(f"❌ Batch directory not found: {batch_dir}")
            sys.exit(1)

        # Create timestamped output directory
        run_id = _utcnow_str()
        output_dir = Path(args.out) / run_id

        result = run_thread_analysis(
            batch_dir=batch_dir,
            output_dir=output_dir,
            partition_deep=args.partition_deep,
            partition_hot=args.partition_hot,
            partition_top=args.partition_top,
            generate_pdf=not args.no_pdf,
        )

        print(json.dumps(result, indent=2, default=str))
        return

    # Mode 4: Agent analysis only (no LLM)
    if args.from_batch and args.analyze_agents:
        batch_dir = Path(args.from_batch)
        if not batch_dir.exists():
            print(f"❌ Batch directory not found: {batch_dir}")
            sys.exit(1)
        
        result = run_agent_analysis(batch_dir, Path(args.out))
        print(json.dumps(result, indent=2, default=str))
        return
    
    # Mode 5: Tiered evaluation (filter -> lite judge -> full eval)
    if args.from_batch and args.tiered_eval:
        batch_dir = Path(args.from_batch)
        if not batch_dir.exists():
            print(f"❌ Batch directory not found: {batch_dir}")
            sys.exit(1)
        
        result = run_tiered_evaluation(
            batch_dir=batch_dir,
            output_dir=Path(args.out),
            escalation_threshold=args.escalation_threshold,
            run_full_eval=args.run_evals,
            generate_pdf=not args.no_pdf,
        )
        print(json.dumps(result, indent=2, default=str))
        return
    
    # Mode 6: Load from batch (standard pipeline)
    if args.from_batch:
        batch_dir = Path(args.from_batch)
        if not batch_dir.exists():
            print(f"❌ Batch directory not found: {batch_dir}")
            sys.exit(1)

        result = run_pipeline_from_batch(
            batch_dir=batch_dir,
            output_dir=Path(args.out),
            run_evals=args.run_evals,
            evaluate_comments=not args.no_comment_eval,
            generate_reports=not args.no_reports,
            generate_pdf=not args.no_pdf,
        )

        print("\n" + "="*60)
        print("✅ BATCH PIPELINE COMPLETE")
        print("="*60)
        print(json.dumps(result, indent=2, default=str))
        return

    # Determine date range
    start_time = None
    end_time = None

    if args.start:
        start_time = parse_timestamp(args.start)
    if args.end:
        end_time = parse_timestamp(args.end)

    if args.hours:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=args.hours)

    if args.full_scrape:
        # Scrape from site genesis (January 2026)
        start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end_time = datetime.now(timezone.utc)

    # Default: if no range specified, use last 24 hours
    if start_time is None and end_time is None:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=24)

    # Run the pipeline
    result = run_pipeline(
        start_time=start_time,
        end_time=end_time,
        output_dir=Path(args.out),
        dry_run=args.dry_run,
        evaluate_comments=not args.no_comment_eval,
        generate_reports=not args.no_reports,
        max_posts=args.max_posts,
        rate_limit=args.rate_limit,
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
