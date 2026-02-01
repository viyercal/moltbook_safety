#!/usr/bin/env python3
"""
DEPRECATED: Use run_pipeline.py instead.

This script is deprecated. All production batch functionality has been merged
into the main run_pipeline.py entry point.

Use instead:
    python run_pipeline.py --start "2026-01-31T07:00:00" --end "2026-01-31T09:00:00"

Production Batch Scraper for Molt Observatory.

Fetches a snapshot of moltbook.com data within a specified date range.
Supports fetching posts, agents, submolts, and comments.

Usage:
    cd molt-observatory
    python scripts/production_batch.py --start "2026-01-26T00:00:00" --end "2026-01-30T23:59:59" --output ./batch_output

Environment:
    Loads configuration from .env file via python-dotenv.
"""

import warnings
warnings.warn(
    "production_batch.py is deprecated. Use run_pipeline.py instead:\n"
    "  python run_pipeline.py --start '2026-01-31T07:00:00' --end '2026-01-31T09:00:00'",
    DeprecationWarning,
    stacklevel=2
)

from __future__ import annotations
import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from scraper.moltbook_api import MoltbookAPI
from scraper.extractors import (
    extract_posts_from_list,
    extract_post_detail,
    extract_agents_from_recent,
    extract_submolts_from_list,
    flatten_comments_tree,
)
from transcript_builder import build_transcript_from_post_detail


def parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp string to datetime."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        # Try parsing without timezone
        dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)


def is_in_range(created_at: str, start: datetime, end: datetime) -> bool:
    """Check if a timestamp falls within the range [start, end]."""
    if not created_at:
        return False
    try:
        ts = parse_timestamp(created_at)
        # Make timezone-aware if needed
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return start <= ts <= end
    except Exception:
        return False


def run_production_batch(
    start_time: datetime,
    end_time: datetime,
    output_dir: Path,
    rate_limit: float = 0.5,
    max_posts: int = 1000,
    max_agents: int = 500,
    max_submolts: int = 100,
    fetch_comments: bool = True,
    build_transcripts: bool = True,
) -> Dict[str, Any]:
    """
    Run a production batch scrape of moltbook.com.
    
    Args:
        start_time: Start of date range (inclusive)
        end_time: End of date range (inclusive)
        output_dir: Directory to save output
        rate_limit: Requests per second
        max_posts: Maximum posts to fetch
        max_agents: Maximum agents to fetch
        max_submolts: Maximum submolts to fetch
        fetch_comments: Whether to fetch post details with comments
        build_transcripts: Whether to build transcripts from post details
    
    Returns:
        Summary statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / "posts").mkdir(exist_ok=True)
    (output_dir / "agents").mkdir(exist_ok=True)
    (output_dir / "submolts").mkdir(exist_ok=True)
    (output_dir / "transcripts").mkdir(exist_ok=True)
    
    api = MoltbookAPI(rate_per_sec=rate_limit, burst=5)
    
    stats = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "batch_started": datetime.now(timezone.utc).isoformat(),
        "posts_fetched": 0,
        "posts_in_range": 0,
        "post_details_fetched": 0,
        "comments_fetched": 0,
        "agents_fetched": 0,
        "submolts_fetched": 0,
        "transcripts_built": 0,
        "errors": [],
    }
    
    print(f"\n{'='*60}")
    print(f"🦞 PRODUCTION BATCH SCRAPE")
    print(f"{'='*60}")
    print(f"Date range: {start_time.isoformat()} to {end_time.isoformat()}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # =========================================================================
    # PHASE 1: Fetch all posts (paginate until we're past the date range)
    # =========================================================================
    print("📥 Phase 1: Fetching posts...")
    
    all_posts = []
    posts_in_range = []
    page = 0
    limit = 50
    reached_start = False
    
    pbar = tqdm(desc="Fetching posts", unit="page")
    while not reached_start and page < (max_posts // limit):
        try:
            resp = api.get_json("/api/v1/posts", params={
                "sort": "new",
                "limit": limit,
                "offset": page * limit,
            })
            
            body = resp.json_body
            posts = body.get("posts", []) if isinstance(body, dict) else body
            
            if not posts:
                break
            
            for post in posts:
                created_at = post.get("created_at")
                
                if created_at:
                    ts = parse_timestamp(created_at)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    
                    # Check if we've gone past the start of our range
                    if ts < start_time:
                        reached_start = True
                        break
                    
                    # Check if in range
                    if start_time <= ts <= end_time:
                        posts_in_range.append(post)
                
                all_posts.append(post)
            
            page += 1
            pbar.update(1)
            pbar.set_postfix({"total": len(all_posts), "in_range": len(posts_in_range)})
            
            if len(posts) < limit:
                break
                
        except Exception as e:
            stats["errors"].append({"phase": "posts", "error": str(e)})
            break
    
    pbar.close()
    
    stats["posts_fetched"] = len(all_posts)
    stats["posts_in_range"] = len(posts_in_range)
    
    # Save posts
    extracted_posts = extract_posts_from_list({"posts": posts_in_range})
    with open(output_dir / "posts" / "posts_list.json", "w") as f:
        json.dump(extracted_posts, f, indent=2, default=str)
    
    print(f"   ✅ Fetched {len(all_posts)} posts total, {len(posts_in_range)} in date range")
    
    # =========================================================================
    # PHASE 2: Fetch post details with comments
    # =========================================================================
    if fetch_comments and posts_in_range:
        print(f"\n📥 Phase 2: Fetching post details with comments...")
        
        post_details = []
        raw_details = []  # For transcript building
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
                raw_details.append(raw_detail)
                
                detail = extract_post_detail(raw_detail)
                if detail:
                    post_details.append(detail)
                    
                    # Extract comments
                    comments = flatten_comments_tree(detail.get("comments", []), post_id)
                    all_comments.extend(comments)
                    
                    # Save individual detail
                    with open(output_dir / "posts" / f"detail_{post_id[:8]}.json", "w") as f:
                        json.dump(detail, f, indent=2, default=str)
                    
                    stats["post_details_fetched"] += 1
                    
            except Exception as e:
                stats["errors"].append({"phase": "post_detail", "post_id": post_id, "error": str(e)})
        
        pbar.close()
        
        # Save all comments
        with open(output_dir / "posts" / "all_comments.json", "w") as f:
            json.dump(all_comments, f, indent=2, default=str)
        
        stats["comments_fetched"] = len(all_comments)
        print(f"   ✅ Fetched {len(post_details)} post details with {len(all_comments)} comments")
        
        # =====================================================================
        # PHASE 2b: Build transcripts
        # =====================================================================
        if build_transcripts and raw_details:
            print(f"\n📝 Phase 2b: Building transcripts...")
            
            transcripts = []
            pbar = tqdm(raw_details, desc="Building transcripts", unit="transcript")
            for raw_detail in pbar:
                try:
                    t = build_transcript_from_post_detail(raw_detail)
                    transcripts.append(asdict(t))
                    stats["transcripts_built"] += 1
                except Exception as e:
                    stats["errors"].append({"phase": "transcript", "error": str(e)})
            
            pbar.close()
            
            # Save transcripts as JSONL
            with open(output_dir / "transcripts" / "transcripts.jsonl", "w") as f:
                for t in transcripts:
                    f.write(json.dumps(t, default=str) + "\n")
            
            print(f"   ✅ Built {len(transcripts)} transcripts")
    
    # =========================================================================
    # PHASE 3: Fetch agents
    # =========================================================================
    print(f"\n📥 Phase 3: Fetching agents...")
    
    all_agents = []
    page = 0
    
    pbar = tqdm(desc="Fetching agents", unit="page")
    while page < (max_agents // 50):
        try:
            resp = api.get_json("/api/v1/agents/recent", params={
                "limit": 50,
                "offset": page * 50,
            })
            
            body = resp.json_body
            agents = body.get("agents", []) if isinstance(body, dict) else body
            
            if not agents:
                break
            
            all_agents.extend(agents)
            page += 1
            pbar.update(1)
            pbar.set_postfix({"total": len(all_agents)})
            
            if len(agents) < 50:
                break
                
        except Exception as e:
            stats["errors"].append({"phase": "agents", "error": str(e)})
            break
    
    pbar.close()
    
    # Filter agents by creation date if available
    agents_in_range = []
    for agent in all_agents:
        created_at = agent.get("created_at")
        if created_at and is_in_range(created_at, start_time, end_time):
            agents_in_range.append(agent)
        elif not created_at:
            # Include agents without creation date
            agents_in_range.append(agent)
    
    stats["agents_fetched"] = len(agents_in_range)
    
    # Save agents
    extracted_agents = extract_agents_from_recent({"agents": agents_in_range})
    with open(output_dir / "agents" / "agents_list.json", "w") as f:
        json.dump(extracted_agents, f, indent=2, default=str)
    
    print(f"   ✅ Fetched {len(all_agents)} agents total, {len(agents_in_range)} potentially in range")
    
    # =========================================================================
    # PHASE 4: Fetch submolts
    # =========================================================================
    print(f"\n📥 Phase 4: Fetching submolts...")
    
    all_submolts = []
    page = 0
    
    pbar = tqdm(desc="Fetching submolts", unit="page")
    while page < (max_submolts // 50):
        try:
            resp = api.get_json("/api/v1/submolts", params={
                "limit": 50,
                "offset": page * 50,
            })
            
            body = resp.json_body
            submolts = body.get("submolts", []) if isinstance(body, dict) else body
            
            if not submolts:
                break
            
            all_submolts.extend(submolts)
            page += 1
            pbar.update(1)
            pbar.set_postfix({"total": len(all_submolts)})
            
            if len(submolts) < 50:
                break
                
        except Exception as e:
            stats["errors"].append({"phase": "submolts", "error": str(e)})
            break
    
    pbar.close()
    
    stats["submolts_fetched"] = len(all_submolts)
    
    # Save submolts
    extracted_submolts = extract_submolts_from_list({"submolts": all_submolts})
    with open(output_dir / "submolts" / "submolts_list.json", "w") as f:
        json.dump(extracted_submolts, f, indent=2, default=str)
    
    print(f"   ✅ Fetched {len(all_submolts)} submolts")
    
    # =========================================================================
    # PHASE 5: Save summary
    # =========================================================================
    stats["batch_completed"] = datetime.now(timezone.utc).isoformat()
    
    with open(output_dir / "batch_summary.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"✅ BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"   Posts in range:     {stats['posts_in_range']}")
    print(f"   Post details:       {stats['post_details_fetched']}")
    print(f"   Comments:           {stats['comments_fetched']}")
    print(f"   Transcripts:        {stats['transcripts_built']}")
    print(f"   Agents:             {stats['agents_fetched']}")
    print(f"   Submolts:           {stats['submolts_fetched']}")
    print(f"   Errors:             {len(stats['errors'])}")
    print(f"   Output:             {output_dir}")
    print(f"{'='*60}\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Production batch scraper for Molt Observatory")
    parser.add_argument("--start", required=True, help="Start timestamp (ISO format, e.g. 2026-01-26T00:00:00)")
    parser.add_argument("--end", required=True, help="End timestamp (ISO format, e.g. 2026-01-30T23:59:59)")
    parser.add_argument("--output", default="./batch_output", help="Output directory")
    parser.add_argument("--rate-limit", type=float, default=0.5, help="Requests per second (default: 0.5)")
    parser.add_argument("--max-posts", type=int, default=1000, help="Maximum posts to fetch (default: 1000)")
    parser.add_argument("--no-comments", action="store_true", help="Skip fetching post details/comments")
    parser.add_argument("--no-transcripts", action="store_true", help="Skip building transcripts")
    
    args = parser.parse_args()
    
    start_time = parse_timestamp(args.start)
    end_time = parse_timestamp(args.end)
    
    # Ensure timezone aware
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    
    output_dir = Path(args.output)
    
    run_production_batch(
        start_time=start_time,
        end_time=end_time,
        output_dir=output_dir,
        rate_limit=args.rate_limit,
        max_posts=args.max_posts,
        fetch_comments=not args.no_comments,
        build_transcripts=not args.no_transcripts,
    )


if __name__ == "__main__":
    main()

