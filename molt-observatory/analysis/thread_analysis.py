"""
Thread Analysis Module

Provides functions for analyzing thread depth (reply nesting) and 
engagement (comment count) in Moltbook posts.

Thread Depth:
- Depth 1: Post with no comments
- Depth 2: Post + top-level comments only
- Depth 3: Post + comments + replies
- Depth 4+: Deeper nesting

Comment Count:
- Total number of comments including all nested replies
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ThreadMetrics:
    """Metrics for a single thread."""
    post_id: str
    depth: int
    comment_count: int
    top_level_count: int
    permalink: str = ""
    title: str = ""


@dataclass 
class ThreadStats:
    """Aggregate statistics for a collection of threads."""
    # Depth stats
    depth_mean: float
    depth_std: float
    depth_median: float
    depth_min: int
    depth_max: int
    depth_percentiles: Dict[int, float]
    depth_distribution: Dict[int, int]
    depth_threshold_2std: float
    
    # Engagement stats
    engagement_mean: float
    engagement_std: float
    engagement_median: float
    engagement_min: int
    engagement_max: int
    engagement_percentiles: Dict[int, float]
    engagement_threshold_2std: float
    
    # Counts
    total_posts: int
    total_comments: int
    total_top_level_comments: int
    reply_ratio: float  # % of comments that are replies


def compute_thread_depth(post_detail: Dict[str, Any]) -> int:
    """
    Compute thread depth from a post detail object.
    
    Depth = 1 (post) + max nesting level of comments.
    
    Examples:
    - Post with no comments → depth 1
    - Post + top-level comments only → depth 2  
    - Post + comment + reply → depth 3
    - Post + comment + reply + reply-to-reply → depth 4
    
    Args:
        post_detail: Raw post detail dict with 'comments' array
        
    Returns:
        Thread depth (minimum 1)
    """
    comments = post_detail.get("comments", [])
    if not comments:
        return 1
    
    def max_depth(comment: Dict, d: int = 1) -> int:
        replies = comment.get("replies", [])
        if not replies:
            return d
        return max(max_depth(r, d + 1) for r in replies)
    
    return 1 + max(max_depth(c) for c in comments)


def compute_comment_count(post_detail: Dict[str, Any]) -> int:
    """
    Compute total comment count including all nested replies.
    
    Args:
        post_detail: Raw post detail dict with 'comments' array
        
    Returns:
        Total number of comments (0 if no comments)
    """
    def count_recursive(comment: Dict) -> int:
        return 1 + sum(count_recursive(r) for r in comment.get("replies", []))
    
    comments = post_detail.get("comments", [])
    return sum(count_recursive(c) for c in comments)


def compute_top_level_count(post_detail: Dict[str, Any]) -> int:
    """Count only top-level comments (not replies)."""
    return len(post_detail.get("comments", []))


def compute_thread_metrics(post_detail: Dict[str, Any]) -> ThreadMetrics:
    """Compute all metrics for a single post."""
    post = post_detail.get("post", {})
    post_id = post.get("post_external_id") or post.get("id", "unknown")
    permalink = post.get("permalink", f"https://www.moltbook.com/post/{post_id}")
    title = post.get("title", "Untitled")
    
    return ThreadMetrics(
        post_id=post_id,
        depth=compute_thread_depth(post_detail),
        comment_count=compute_comment_count(post_detail),
        top_level_count=compute_top_level_count(post_detail),
        permalink=permalink,
        title=title,
    )


def compute_depth_stats(post_details: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate depth statistics.
    
    Args:
        post_details: List of post detail dicts
        
    Returns:
        Dict with mean, std, percentiles, distribution
    """
    depths = [compute_thread_depth(p) for p in post_details]
    
    if not depths:
        return {
            "mean": 0, "std": 0, "median": 0, "min": 0, "max": 0,
            "percentiles": {}, "distribution": {}, "threshold_2std": 0,
        }
    
    depths_arr = np.array(depths)
    distribution = Counter(depths)
    
    return {
        "mean": float(np.mean(depths_arr)),
        "std": float(np.std(depths_arr)),
        "median": float(np.median(depths_arr)),
        "min": int(np.min(depths_arr)),
        "max": int(np.max(depths_arr)),
        "percentiles": {
            25: float(np.percentile(depths_arr, 25)),
            50: float(np.percentile(depths_arr, 50)),
            75: float(np.percentile(depths_arr, 75)),
            90: float(np.percentile(depths_arr, 90)),
            95: float(np.percentile(depths_arr, 95)),
            99: float(np.percentile(depths_arr, 99)),
        },
        "distribution": dict(distribution),
        "threshold_2std": float(np.mean(depths_arr) + 2 * np.std(depths_arr)),
    }


def compute_engagement_stats(post_details: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate engagement (comment count) statistics.
    
    Args:
        post_details: List of post detail dicts
        
    Returns:
        Dict with mean, std, percentiles
    """
    counts = [compute_comment_count(p) for p in post_details]
    
    if not counts:
        return {
            "mean": 0, "std": 0, "median": 0, "min": 0, "max": 0,
            "percentiles": {}, "threshold_2std": 0,
        }
    
    counts_arr = np.array(counts)
    
    return {
        "mean": float(np.mean(counts_arr)),
        "std": float(np.std(counts_arr)),
        "median": float(np.median(counts_arr)),
        "min": int(np.min(counts_arr)),
        "max": int(np.max(counts_arr)),
        "percentiles": {
            25: float(np.percentile(counts_arr, 25)),
            50: float(np.percentile(counts_arr, 50)),
            75: float(np.percentile(counts_arr, 75)),
            90: float(np.percentile(counts_arr, 90)),
            95: float(np.percentile(counts_arr, 95)),
            99: float(np.percentile(counts_arr, 99)),
        },
        "threshold_2std": float(np.mean(counts_arr) + 2 * np.std(counts_arr)),
    }


def analyze_batch(batch_dir: Path) -> Dict[str, Any]:
    """
    Analyze all posts in a batch directory.
    
    Args:
        batch_dir: Path to batch directory with posts/ subdirectory
        
    Returns:
        Dict with:
        - depth_stats: Depth statistics
        - engagement_stats: Engagement statistics
        - metrics: List of per-post metrics
        - summary: High-level summary
    """
    batch_dir = Path(batch_dir)
    posts_dir = batch_dir / "posts"
    
    if not posts_dir.exists():
        raise ValueError(f"Posts directory not found: {posts_dir}")
    
    # Load all post details
    post_details = []
    for f in posts_dir.glob("detail_*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                post_details.append(json.load(fp))
        except (json.JSONDecodeError, IOError):
            continue
    
    if not post_details:
        raise ValueError(f"No post details found in {posts_dir}")
    
    # Compute metrics for each post
    metrics = [compute_thread_metrics(p) for p in post_details]
    
    # Compute aggregate stats
    depth_stats = compute_depth_stats(post_details)
    engagement_stats = compute_engagement_stats(post_details)
    
    # Compute additional summary stats
    total_comments = sum(m.comment_count for m in metrics)
    total_top_level = sum(m.top_level_count for m in metrics)
    reply_count = total_comments - total_top_level
    reply_ratio = reply_count / total_comments if total_comments > 0 else 0
    
    return {
        "depth_stats": depth_stats,
        "engagement_stats": engagement_stats,
        "metrics": [
            {
                "post_id": m.post_id,
                "depth": m.depth,
                "comment_count": m.comment_count,
                "top_level_count": m.top_level_count,
                "permalink": m.permalink,
                "title": m.title,
            }
            for m in metrics
        ],
        "summary": {
            "total_posts": len(metrics),
            "total_comments": total_comments,
            "total_top_level_comments": total_top_level,
            "total_replies": reply_count,
            "reply_ratio": reply_ratio,
            "depth_threshold": depth_stats["threshold_2std"],
            "engagement_threshold": engagement_stats["threshold_2std"],
            "posts_above_depth_threshold": sum(
                1 for m in metrics if m.depth > depth_stats["threshold_2std"]
            ),
            "posts_above_engagement_threshold": sum(
                1 for m in metrics if m.comment_count > engagement_stats["threshold_2std"]
            ),
        },
    }


def partition_deep_threads(
    batch_dir: Path,
    output_dir: Path,
    threshold: Optional[float] = None,
    top_n: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Partition posts with high thread depth (conversational threads).
    
    Args:
        batch_dir: Source batch directory
        output_dir: Output directory for partitioned data
        threshold: Depth threshold (default: mean + 2*std)
        top_n: Also include top N by depth
        
    Returns:
        Manifest dict with included post_ids and stats
    """
    batch_dir = Path(batch_dir)
    output_dir = Path(output_dir)
    
    # Analyze batch
    analysis = analyze_batch(batch_dir)
    metrics = analysis["metrics"]
    
    if threshold is None:
        threshold = analysis["depth_stats"]["threshold_2std"]
    
    # Select posts above threshold
    selected_ids = set()
    for m in metrics:
        if m["depth"] > threshold:
            selected_ids.add(m["post_id"])
    
    # Add top N if specified
    if top_n:
        sorted_by_depth = sorted(metrics, key=lambda x: x["depth"], reverse=True)
        for m in sorted_by_depth[:top_n]:
            selected_ids.add(m["post_id"])
    
    # Create output structure
    return _create_partition(
        batch_dir=batch_dir,
        output_dir=output_dir,
        selected_ids=selected_ids,
        partition_name="deep_threads",
        criteria={
            "type": "depth",
            "threshold": threshold,
            "top_n": top_n,
        },
        metrics=[m for m in metrics if m["post_id"] in selected_ids],
    )


def partition_hot_threads(
    batch_dir: Path,
    output_dir: Path,
    threshold: Optional[float] = None,
    top_n: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Partition posts with high comment count (hot/popular threads).
    
    Args:
        batch_dir: Source batch directory
        output_dir: Output directory for partitioned data
        threshold: Comment count threshold (default: mean + 2*std)
        top_n: Also include top N by comment count
        
    Returns:
        Manifest dict with included post_ids and stats
    """
    batch_dir = Path(batch_dir)
    output_dir = Path(output_dir)
    
    # Analyze batch
    analysis = analyze_batch(batch_dir)
    metrics = analysis["metrics"]
    
    if threshold is None:
        threshold = analysis["engagement_stats"]["threshold_2std"]
    
    # Select posts above threshold
    selected_ids = set()
    for m in metrics:
        if m["comment_count"] > threshold:
            selected_ids.add(m["post_id"])
    
    # Add top N if specified
    if top_n:
        sorted_by_count = sorted(metrics, key=lambda x: x["comment_count"], reverse=True)
        for m in sorted_by_count[:top_n]:
            selected_ids.add(m["post_id"])
    
    # Create output structure
    return _create_partition(
        batch_dir=batch_dir,
        output_dir=output_dir,
        selected_ids=selected_ids,
        partition_name="hot_threads",
        criteria={
            "type": "engagement",
            "threshold": threshold,
            "top_n": top_n,
        },
        metrics=[m for m in metrics if m["post_id"] in selected_ids],
    )


def _create_partition(
    batch_dir: Path,
    output_dir: Path,
    selected_ids: set,
    partition_name: str,
    criteria: Dict[str, Any],
    metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Create a partition with selected posts.
    
    Creates:
    - posts/ directory with copied post details
    - transcripts.jsonl with filtered transcripts
    - manifest.json with metadata
    """
    partition_dir = output_dir / partition_name
    posts_out = partition_dir / "posts"
    posts_out.mkdir(parents=True, exist_ok=True)
    
    # Copy selected post details
    posts_src = batch_dir / "posts"
    copied_count = 0
    for f in posts_src.glob("detail_*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            post_id = data.get("post", {}).get("post_external_id", "")
            if post_id in selected_ids:
                shutil.copy(f, posts_out / f.name)
                copied_count += 1
        except (json.JSONDecodeError, IOError):
            continue
    
    # Filter transcripts
    transcripts_src = batch_dir / "transcripts" / "transcripts.jsonl"
    transcripts_out = partition_dir / "transcripts.jsonl"
    
    transcript_count = 0
    if transcripts_src.exists():
        with open(transcripts_src, "r", encoding="utf-8") as fin:
            with open(transcripts_out, "w", encoding="utf-8") as fout:
                for line in fin:
                    try:
                        t = json.loads(line.strip())
                        if t.get("post_id") in selected_ids:
                            fout.write(json.dumps(t, ensure_ascii=False) + "\n")
                            transcript_count += 1
                    except json.JSONDecodeError:
                        continue
    
    # Create manifest with source batch for entity loading
    manifest = {
        "partition_name": partition_name,
        "source_batch": str(batch_dir.resolve()),
        "criteria": criteria,
        "post_ids": list(selected_ids),
        "post_count": len(selected_ids),
        "posts_copied": copied_count,
        "transcripts_copied": transcript_count,
        "metrics": metrics,
    }
    
    with open(partition_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    return manifest

