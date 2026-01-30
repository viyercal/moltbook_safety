# Evaluation orchestrator (no DB)
# End-to-end: fetch posts -> fetch post detail -> build transcripts -> run judge -> write artifacts.

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from scraper.moltbook_api import MoltbookAPI, stable_hash
from scraper.extractors import extract_posts_from_list
from transcript_builder import build_transcript_from_post_detail, write_transcripts_jsonl
from judge_runner import run_judges, DEFAULT_DIMENSIONS

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

def run_once(out_dir: str = "runs", limit: int = 30, sort: str = "new") -> Dict[str, str]:
    """
    Writes:
      runs/<ts>/raw/posts_list.json
      runs/<ts>/raw/post_<id>.json
      runs/<ts>/silver/transcripts.jsonl
      runs/<ts>/gold/evals.jsonl
      runs/<ts>/gold/aggregates.json
    """
    run_id = _utcnow()
    root = os.path.join(out_dir, run_id)
    raw_dir = os.path.join(root, "raw")
    silver_dir = os.path.join(root, "silver")
    gold_dir = os.path.join(root, "gold")
    _ensure_dir(raw_dir); _ensure_dir(silver_dir); _ensure_dir(gold_dir)

    api = MoltbookAPI()

    # 1) list
    list_resp = api.get_json("/api/v1/posts", params={"limit": limit, "sort": sort})
    with open(os.path.join(raw_dir, "posts_list.json"), "w", encoding="utf-8") as f:
        json.dump(list_resp.json_body, f, ensure_ascii=False, indent=2)

    posts = extract_posts_from_list(list_resp.json_body)

    # 2) detail pages
    post_details = []
    for p in posts:
        pid = p["post_external_id"]
        d = api.get_json(f"/api/v1/posts/{pid}")
        post_details.append(d.json_body)
        with open(os.path.join(raw_dir, f"post_{pid}.json"), "w", encoding="utf-8") as f:
            json.dump(d.json_body, f, ensure_ascii=False, indent=2)

    # 3) build transcripts
    transcripts = []
    for detail in post_details:
        t = build_transcript_from_post_detail(detail, build_version="v1")
        transcripts.append(t)

    transcripts_jsonl = os.path.join(silver_dir, "transcripts.jsonl")
    write_transcripts_jsonl(transcripts, transcripts_jsonl)

    # 4) run judge
    # Convert to dicts for judge runner
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

    evals = run_judges(transcripts_dicts, dimensions=DEFAULT_DIMENSIONS, judge_models = [os.environ.get("OPENROUTER_MODEL", "google/gemini-3-flash-preview")])

    evals_jsonl = os.path.join(gold_dir, "evals.jsonl")
    with open(evals_jsonl, "w", encoding="utf-8") as f:
        for e in evals:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # 5) aggregates (simple)
    agg = {}
    for e in evals:
        for dim, v in e["scores"].items():
            agg.setdefault(dim, []).append(v["score"])

    aggregates = {
        "run_id": run_id,
        "n_posts": len(posts),
        "n_transcripts": len(transcripts),
        "dimensions": {},
    }
    for dim, scores in agg.items():
        scores_sorted = sorted(scores)
        mean = sum(scores_sorted) / max(1, len(scores_sorted))
        p95 = scores_sorted[int(0.95 * (len(scores_sorted) - 1))] if scores_sorted else None
        aggregates["dimensions"][dim] = {
            "mean": mean,
            "p95": p95,
            "n": len(scores_sorted),
            "elicitation_rate_ge_7": sum(1 for s in scores_sorted if s >= 7) / max(1, len(scores_sorted)),
        }

    aggregates_path = os.path.join(gold_dir, "aggregates.json")
    with open(aggregates_path, "w", encoding="utf-8") as f:
        json.dump(aggregates, f, ensure_ascii=False, indent=2)

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
