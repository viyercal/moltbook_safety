# Evaluation orchestrator (no DB)
# End-to-end: fetch posts -> fetch post detail -> build transcripts -> run judge -> write artifacts.
# 
# Enhanced with:
# - State management for incremental pulls
# - Comprehensive entity scraping (agents, submolts, comments)
# - Comment transcript building and evaluation
# - Agent score aggregation
# - Report generation

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import json
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from scraper.moltbook_api import MoltbookAPI, stable_hash
from scraper.extractors import (
    extract_posts_from_list,
    extract_agents_from_recent,
    extract_submolts_from_list,
    extract_post_detail,
    flatten_comments_tree,
    extract_site_stats,
    dedupe_agents,
    dedupe_posts,
    dedupe_submolts,
)
from transcript_builder import build_transcript_from_post_detail, write_transcripts_jsonl
from comment_transcript_builder import (
    build_comment_transcripts_from_post_detail,
    write_comment_transcripts_jsonl,
)
from judge_runner import run_judges, run_comment_judges, DEFAULT_DIMENSIONS
from agent_scorer import aggregate_all_agents, AgentHistoryManager, write_agent_scores_jsonl
from state import get_state_manager

def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def scrape_posts_and_details(api: MoltbookAPI, limit: int = 30, sort: str = "new") -> List[Dict[str, Any]]:
    """
    Returns a list of post-detail payloads (each includes post + comments + context).
    """
    r = api.get_json("/api/v1/posts", params={"limit": limit, "sort": sort})
    posts = extract_posts_from_list(r.json_body)

    details: List[Dict[str, Any]] = []
    for p in posts:
        pid = p["post_external_id"]
        d = api.get_json(f"/api/v1/posts/{pid}")
        details.append(d.json_body)

    return details

def run_once(
    out_dir: str = "runs",
    limit: int = 30,
    sort: str = "new",
    evaluate_comments: bool = True,
    aggregate_agents: bool = True,
    incremental: bool = False,
) -> Dict[str, str]:
    """
    Run the full pipeline: scrape -> transcripts -> evaluate -> aggregate.
    
    Writes:
      runs/<ts>/raw/posts_list.json
      runs/<ts>/raw/posts/post_<id>.json
      runs/<ts>/raw/agents_list.json
      runs/<ts>/raw/submolts_list.json
      runs/<ts>/silver/transcripts.jsonl
      runs/<ts>/silver/comment_transcripts.jsonl
      runs/<ts>/gold/evals.jsonl
      runs/<ts>/gold/comment_evals.jsonl
      runs/<ts>/gold/agent_scores.jsonl
      runs/<ts>/gold/aggregates.json
      runs/<ts>/meta/snapshot.json
    
    Args:
        out_dir: Base output directory
        limit: Max posts to fetch
        sort: Sort order for posts
        evaluate_comments: Whether to evaluate individual comments
        aggregate_agents: Whether to aggregate scores per agent
        incremental: Whether to use incremental scraping with state
    """
    run_id = _utcnow()
    root = os.path.join(out_dir, run_id)
    raw_dir = os.path.join(root, "raw")
    raw_posts_dir = os.path.join(raw_dir, "posts")
    silver_dir = os.path.join(root, "silver")
    gold_dir = os.path.join(root, "gold")
    meta_dir = os.path.join(root, "meta")
    
    for d in [raw_dir, raw_posts_dir, silver_dir, gold_dir, meta_dir]:
        _ensure_dir(d)

    api = MoltbookAPI()
    
    # State management for incremental mode
    state_manager = get_state_manager()
    if incremental:
        state = state_manager.start_run(run_id)
        last_post_ts = state_manager.get_last_created_at("posts")
    else:
        last_post_ts = None

    # -------------------------------------------------------------------------
    # 1) Scrape posts
    # -------------------------------------------------------------------------
    print(f"Scraping posts (limit={limit}, sort={sort})...")
    list_resp = api.get_json("/api/v1/posts", params={"limit": limit, "sort": sort})
    with open(os.path.join(raw_dir, "posts_list.json"), "w", encoding="utf-8") as f:
        json.dump(list_resp.json_body, f, ensure_ascii=False, indent=2)

    posts = extract_posts_from_list(list_resp.json_body, dedupe=True)
    
    # Filter for incremental mode
    if incremental and last_post_ts:
        posts = [p for p in posts if (p.get("created_at") or "") > last_post_ts]

    # -------------------------------------------------------------------------
    # 2) Fetch post details and extract comments
    # -------------------------------------------------------------------------
    print(f"Fetching {len(posts)} post details...")
    post_details = []
    all_comments = []
    
    for p in posts:
        pid = p["post_external_id"]
        try:
            d = api.get_json(f"/api/v1/posts/{pid}")
            post_details.append(d.json_body)
            with open(os.path.join(raw_posts_dir, f"post_{pid}.json"), "w", encoding="utf-8") as f:
                json.dump(d.json_body, f, ensure_ascii=False, indent=2)
            
            # Extract comments
            detail = extract_post_detail(d.json_body)
            if detail and detail.get("comments"):
                flat_comments = flatten_comments_tree(detail["comments"], pid)
                all_comments.extend(flat_comments)
        except Exception as e:
            print(f"Error fetching post {pid}: {e}")

    # -------------------------------------------------------------------------
    # 3) Scrape agents and submolts (for context)
    # -------------------------------------------------------------------------
    print("Scraping agents and submolts...")
    try:
        agents_resp = api.get_json("/api/v1/agents/recent", params={"limit": 50})
        agents = extract_agents_from_recent(agents_resp.json_body, dedupe=True)
        with open(os.path.join(raw_dir, "agents_list.json"), "w", encoding="utf-8") as f:
            json.dump(agents_resp.json_body, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Could not fetch agents: {e}")
        agents = []
    
    try:
        submolts_resp = api.get_json("/api/v1/submolts", params={"limit": 50})
        submolts = extract_submolts_from_list(submolts_resp.json_body, dedupe=True)
        with open(os.path.join(raw_dir, "submolts_list.json"), "w", encoding="utf-8") as f:
            json.dump(submolts_resp.json_body, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Could not fetch submolts: {e}")
        submolts = []

    # -------------------------------------------------------------------------
    # 4) Compute and save site stats
    # -------------------------------------------------------------------------
    site_stats = extract_site_stats(agents, posts, submolts, all_comments)
    with open(os.path.join(meta_dir, "snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(site_stats.to_dict(), f, ensure_ascii=False, indent=2)

    # -------------------------------------------------------------------------
    # 5) Build post transcripts
    # -------------------------------------------------------------------------
    print("Building transcripts...")
    transcripts = []
    for detail in post_details:
        try:
            t = build_transcript_from_post_detail(detail, build_version="v1")
            transcripts.append(t)
        except Exception as e:
            print(f"Error building transcript: {e}")

    transcripts_jsonl = os.path.join(silver_dir, "transcripts.jsonl")
    write_transcripts_jsonl(transcripts, transcripts_jsonl)

    # -------------------------------------------------------------------------
    # 6) Build comment transcripts
    # -------------------------------------------------------------------------
    comment_transcripts = []
    if evaluate_comments:
        print("Building comment transcripts...")
        for detail in post_details:
            try:
                cts = build_comment_transcripts_from_post_detail(detail)
                comment_transcripts.extend(cts)
            except Exception as e:
                print(f"Error building comment transcripts: {e}")
        
        comment_transcripts_jsonl = os.path.join(silver_dir, "comment_transcripts.jsonl")
        write_comment_transcripts_jsonl(comment_transcripts, comment_transcripts_jsonl)

    # -------------------------------------------------------------------------
    # 7) Run post evaluations
    # -------------------------------------------------------------------------
    print(f"Evaluating {len(transcripts)} post transcripts...")
    transcripts_dicts = []
    for t in transcripts:
        transcripts_dicts.append({
            "transcript_id": t.transcript_id,
            "post_id": t.post_id,
            "permalink": t.permalink,
            "community": t.community,
            "metadata": t.metadata,
            "messages": t.messages,
        })

    model = os.environ.get("OPENROUTER_MODEL", "google/gemini-3-flash-preview")
    evals = run_judges(transcripts_dicts, dimensions=DEFAULT_DIMENSIONS, judge_models=[model])

    evals_jsonl = os.path.join(gold_dir, "evals.jsonl")
    with open(evals_jsonl, "w", encoding="utf-8") as f:
        for e in evals:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # -------------------------------------------------------------------------
    # 8) Run comment evaluations
    # -------------------------------------------------------------------------
    comment_evals = []
    if evaluate_comments and comment_transcripts:
        print(f"Evaluating {len(comment_transcripts)} comment transcripts...")
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
        
        comment_evals = run_comment_judges(
            comment_transcript_dicts,
            dimensions=DEFAULT_DIMENSIONS,
            judge_models=[model],
        )
        
        comment_evals_jsonl = os.path.join(gold_dir, "comment_evals.jsonl")
        with open(comment_evals_jsonl, "w", encoding="utf-8") as f:
            for e in comment_evals:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # -------------------------------------------------------------------------
    # 9) Aggregate agent scores
    # -------------------------------------------------------------------------
    agent_records = []
    if aggregate_agents:
        print("Aggregating agent scores...")
        agent_records = aggregate_all_agents(
            post_evals=evals,
            comment_evals=comment_evals,
            agents=agents,
            snapshot_id=run_id,
        )
        
        agent_scores_jsonl = os.path.join(gold_dir, "agent_scores.jsonl")
        write_agent_scores_jsonl(agent_records, agent_scores_jsonl)
        
        # Update historical tracking
        history_manager = AgentHistoryManager()
        history_manager.append_all_records(agent_records)

    # -------------------------------------------------------------------------
    # 10) Compute aggregates
    # -------------------------------------------------------------------------
    agg = {}
    for e in evals:
        for dim, v in e.get("scores", {}).items():
            score = v.get("score", 0)
            agg.setdefault(dim, []).append(score)

    aggregates = {
        "run_id": run_id,
        "n_posts": len(posts),
        "n_transcripts": len(transcripts),
        "n_comments": len(all_comments),
        "n_comment_transcripts": len(comment_transcripts),
        "n_agents": len(agents),
        "n_submolts": len(submolts),
        "dimensions": {},
    }
    
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

    aggregates_path = os.path.join(gold_dir, "aggregates.json")
    with open(aggregates_path, "w", encoding="utf-8") as f:
        json.dump(aggregates, f, ensure_ascii=False, indent=2)

    # -------------------------------------------------------------------------
    # 11) Update state for incremental mode
    # -------------------------------------------------------------------------
    if incremental and posts:
        latest_post = max(posts, key=lambda p: p.get("created_at") or "")
        state_manager.update_cursor(
            "posts",
            last_created_at=latest_post.get("created_at"),
            items_fetched=len(posts),
        )
        state_manager.update_cursor("comments", items_fetched=len(all_comments))
        state_manager.finish_run(run_id)

    print(f"Run complete: {run_id}")
    print(f"  Posts: {len(posts)}, Transcripts: {len(transcripts)}")
    print(f"  Comments: {len(all_comments)}, Comment Transcripts: {len(comment_transcripts)}")
    print(f"  Agents Scored: {len(agent_records)}")

    return {
        "run_id": run_id,
        "root": root,
        "posts_list": os.path.join(raw_dir, "posts_list.json"),
        "transcripts": transcripts_jsonl,
        "evals": evals_jsonl,
        "aggregates": aggregates_path,
    }

if __name__ == "__main__":
    out = run_once()
    print(json.dumps(out, indent=2))
