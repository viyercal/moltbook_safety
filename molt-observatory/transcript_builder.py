# Transcript builder pipeline (no DB)
# Builds deterministic transcripts from Moltbook post-detail payloads.

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import os
import hashlib
from datetime import datetime, timezone

from scraper.extractors import extract_post_detail, flatten_comments_tree

@dataclass
class Transcript:
    transcript_id: str
    post_id: str
    built_at: str
    permalink: str
    community: Optional[str]
    messages: List[Dict[str, Any]]
    metadata: Dict[str, Any]

def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def build_transcript_from_post_detail(payload: Dict[str, Any], build_version: str = "v1") -> Transcript:
    """
    Transcript format:
      messages: ordered list with explicit role + author + timestamps.
      - first message is the post itself
      - followed by flattened comments (preorder: parent before children)
    """
    detail = extract_post_detail(payload)
    if not detail:
        raise ValueError("Invalid post-detail payload (missing post.id)")

    p = detail["post"]
    post_id = p["post_external_id"]
    permalink = p["permalink"]

    flat_comments = flatten_comments_tree(detail["comments"], post_external_id=post_id)

    messages: List[Dict[str, Any]] = []
    # Post = root message
    messages.append({
        "kind": "post",
        "id": post_id,
        "author": p.get("author_handle"),
        "author_external_id": p.get("author_external_id"),
        "created_at": p.get("created_at"),
        "text": p.get("body_text") or "",
        "title": p.get("title"),
        "community": p.get("submolt_slug"),
        "score": p.get("score"),
        "upvotes": p.get("upvotes"),
        "downvotes": p.get("downvotes"),
    })

    # Comments
    for c in flat_comments:
        messages.append({
            "kind": "comment",
            "id": c["comment_external_id"],
            "parent_id": c.get("parent_comment_external_id"),
            "author": c.get("author_handle"),
            "author_external_id": c.get("author_external_id"),
            "created_at": c.get("created_at"),
            "text": c.get("body_text") or "",
            "score": c.get("score"),
            "upvotes": c.get("upvotes"),
            "downvotes": c.get("downvotes"),
        })

    built_at = datetime.now(timezone.utc).isoformat()
    canonical = json.dumps({"post_id": post_id, "build_version": build_version, "messages": messages}, sort_keys=True, separators=(",", ":"))
    transcript_id = _sha(canonical)[:24]

    meta = {
        "build_version": build_version,
        "context_tip": detail.get("context_tip"),
        "author_owner": p.get("author_owner"),
        "author_followers": p.get("author_follower_count"),
        "author_karma": p.get("author_karma"),
        "comment_count": p.get("comment_count"),
    }

    return Transcript(
        transcript_id=transcript_id,
        post_id=post_id,
        built_at=built_at,
        permalink=permalink,
        community=p.get("submolt_slug"),
        messages=messages,
        metadata=meta,
    )

def write_transcripts_jsonl(transcripts: List[Transcript], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in transcripts:
            f.write(json.dumps({
                "transcript_id": t.transcript_id,
                "post_id": t.post_id,
                "built_at": t.built_at,
                "permalink": t.permalink,
                "community": t.community,
                "metadata": t.metadata,
                "messages": t.messages,
            }, ensure_ascii=False) + "\n")
