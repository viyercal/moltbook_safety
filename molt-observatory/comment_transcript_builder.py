# Comment Transcript Builder
# Builds transcripts for individual comments with their thread context for evaluation.

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import os
import hashlib
from datetime import datetime, timezone

from scraper.extractors import extract_post_detail, flatten_comments_tree


@dataclass
class CommentTranscript:
    """A transcript focused on a specific comment with its thread context."""
    transcript_id: str
    comment_id: str
    post_id: str
    built_at: str
    permalink: str
    community: Optional[str]
    # Thread context: post + ancestor comments leading to this comment
    context_messages: List[Dict[str, Any]]
    # The target comment being evaluated
    target_comment: Dict[str, Any]
    # Metadata about the context
    metadata: Dict[str, Any]


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _build_comment_chain(
    comment_id: str,
    comments_flat: List[Dict[str, Any]],
    comments_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build the chain of parent comments from root to the target comment.
    Returns list of comments in order: [root_comment, ..., parent, target]
    """
    chain = []
    current_id = comment_id
    
    # Walk up the parent chain
    visited = set()
    while current_id and current_id not in visited:
        visited.add(current_id)
        comment = comments_by_id.get(current_id)
        if not comment:
            break
        chain.append(comment)
        current_id = comment.get("parent_comment_external_id")
    
    # Reverse to get root-to-target order
    chain.reverse()
    return chain


def build_comment_transcripts_from_post_detail(
    payload: Dict[str, Any],
    build_version: str = "v1",
) -> List[CommentTranscript]:
    """
    Build individual transcripts for each comment in a post.
    Each transcript includes:
    - The original post (context)
    - The chain of parent comments leading to the target
    - The target comment itself
    
    Args:
        payload: Raw post detail response from /api/v1/posts/{id}
        build_version: Version string for the build format
    
    Returns:
        List of CommentTranscript objects, one per comment
    """
    detail = extract_post_detail(payload)
    if not detail:
        return []
    
    p = detail["post"]
    post_id = p["post_external_id"]
    permalink_base = p["permalink"]
    community = p.get("submolt_slug")
    
    # Flatten the comment tree
    flat_comments = flatten_comments_tree(detail["comments"], post_external_id=post_id)
    
    if not flat_comments:
        return []
    
    # Build lookup by comment ID
    comments_by_id = {c["comment_external_id"]: c for c in flat_comments}
    
    transcripts: List[CommentTranscript] = []
    
    for comment in flat_comments:
        comment_id = comment["comment_external_id"]
        
        # Build the ancestor chain
        comment_chain = _build_comment_chain(comment_id, flat_comments, comments_by_id)
        
        # Build context messages: post first, then ancestor comments
        context_messages: List[Dict[str, Any]] = []
        
        # Add the post as the root context
        context_messages.append({
            "kind": "post",
            "id": post_id,
            "author": p.get("author_handle"),
            "author_external_id": p.get("author_external_id"),
            "created_at": p.get("created_at"),
            "text": p.get("body_text") or "",
            "title": p.get("title"),
            "community": community,
            "score": p.get("score"),
        })
        
        # Add ancestor comments (excluding the target comment itself)
        for ancestor in comment_chain[:-1]:  # All but the last (target)
            context_messages.append({
                "kind": "comment",
                "id": ancestor["comment_external_id"],
                "parent_id": ancestor.get("parent_comment_external_id"),
                "author": ancestor.get("author_handle"),
                "author_external_id": ancestor.get("author_external_id"),
                "created_at": ancestor.get("created_at"),
                "text": ancestor.get("body_text") or "",
                "score": ancestor.get("score"),
            })
        
        # The target comment
        target_comment = {
            "kind": "comment",
            "id": comment_id,
            "parent_id": comment.get("parent_comment_external_id"),
            "author": comment.get("author_handle"),
            "author_external_id": comment.get("author_external_id"),
            "created_at": comment.get("created_at"),
            "text": comment.get("body_text") or "",
            "score": comment.get("score"),
            "upvotes": comment.get("upvotes"),
            "downvotes": comment.get("downvotes"),
        }
        
        # Generate transcript ID
        built_at = datetime.now(timezone.utc).isoformat()
        canonical = json.dumps({
            "comment_id": comment_id,
            "post_id": post_id,
            "build_version": build_version,
            "context": context_messages,
            "target": target_comment,
        }, sort_keys=True, separators=(",", ":"))
        transcript_id = _sha(canonical)[:24]
        
        # Metadata
        meta = {
            "build_version": build_version,
            "context_depth": len(comment_chain),  # How deep in the thread
            "post_title": p.get("title"),
            "post_author": p.get("author_handle"),
            "is_top_level": comment.get("parent_comment_external_id") is None,
        }
        
        transcripts.append(CommentTranscript(
            transcript_id=transcript_id,
            comment_id=comment_id,
            post_id=post_id,
            built_at=built_at,
            permalink=f"{permalink_base}#comment-{comment_id}",
            community=community,
            context_messages=context_messages,
            target_comment=target_comment,
            metadata=meta,
        ))
    
    return transcripts


def write_comment_transcripts_jsonl(
    transcripts: List[CommentTranscript],
    out_path: str,
) -> None:
    """Write comment transcripts to a JSONL file."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in transcripts:
            f.write(json.dumps({
                "transcript_id": t.transcript_id,
                "comment_id": t.comment_id,
                "post_id": t.post_id,
                "built_at": t.built_at,
                "permalink": t.permalink,
                "community": t.community,
                "context_messages": t.context_messages,
                "target_comment": t.target_comment,
                "metadata": t.metadata,
            }, ensure_ascii=False) + "\n")


def load_comment_transcripts_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load comment transcripts from a JSONL file."""
    transcripts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                transcripts.append(json.loads(line))
    return transcripts


def render_comment_transcript_for_judge(
    transcript: Dict[str, Any],
    max_chars: int = 12000,
) -> str:
    """
    Convert a comment transcript to plain text for LLM judge evaluation.
    
    Format:
    - POST context (abbreviated)
    - Parent comments in chain
    - TARGET COMMENT (clearly marked for evaluation)
    """
    lines: List[str] = []
    
    lines.append(f"COMMENT_ID: {transcript.get('comment_id')}")
    lines.append(f"POST_ID: {transcript.get('post_id')}")
    if transcript.get("permalink"):
        lines.append(f"PERMALINK: {transcript.get('permalink')}")
    if transcript.get("community"):
        lines.append(f"COMMUNITY: {transcript.get('community')}")
    
    meta = transcript.get("metadata", {})
    lines.append(f"CONTEXT_DEPTH: {meta.get('context_depth', 0)} (messages before target)")
    
    lines.append("")
    lines.append("=" * 50)
    lines.append("THREAD CONTEXT (for understanding, not primary evaluation):")
    lines.append("=" * 50)
    
    context = transcript.get("context_messages", [])
    for i, m in enumerate(context):
        kind = (m.get("kind") or "msg").upper()
        author = m.get("author") or "unknown"
        text = (m.get("text") or "").replace("\r\n", "\n")
        
        # Truncate long context messages
        if len(text) > 500:
            text = text[:500] + "... [truncated]"
        
        if kind == "POST":
            title = m.get("title") or "(no title)"
            lines.append(f"[{i}] POST by {author}: \"{title}\"")
        else:
            lines.append(f"[{i}] COMMENT by {author}")
        
        lines.append(text)
        lines.append("")
    
    lines.append("=" * 50)
    lines.append(">>> TARGET COMMENT FOR EVALUATION <<<")
    lines.append("=" * 50)
    
    target = transcript.get("target_comment", {})
    target_author = target.get("author") or "unknown"
    target_text = (target.get("text") or "").replace("\r\n", "\n")
    target_score = target.get("score")
    
    lines.append(f"AUTHOR: {target_author}")
    if target_score is not None:
        lines.append(f"SCORE: {target_score}")
    lines.append("")
    lines.append(target_text)
    
    out = "\n".join(lines)
    
    # Truncate if too long
    if len(out) > max_chars:
        # Keep more of the target comment, less of context
        target_section_start = out.find(">>> TARGET COMMENT")
        if target_section_start > 0:
            context_budget = int(max_chars * 0.4)
            target_budget = max_chars - context_budget
            
            context_part = out[:target_section_start]
            target_part = out[target_section_start:]
            
            if len(context_part) > context_budget:
                context_part = context_part[:context_budget] + "\n[...CONTEXT TRUNCATED...]\n"
            if len(target_part) > target_budget:
                target_part = target_part[:target_budget] + "\n[...TRUNCATED...]"
            
            out = context_part + target_part
        else:
            out = out[:max_chars] + "\n[...TRUNCATED...]"
    
    return out

