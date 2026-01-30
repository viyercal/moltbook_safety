# scraper/extractors.py
"""
Extractors for transforming Moltbook API responses into canonical formats.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


def extract_agents_from_recent(payload: Any) -> List[Dict[str, Any]]:
    """
    Moltbook /api/v1/agents/recent returns:
      { "success": true, "agents": [ { id, name, description, avatar_url, ... } ] }

    Canonical output:
      {
        "agent_external_id": str,
        "handle": str,          # treat `name` as handle unless a separate handle exists
        "display_name": str,
        "bio": str,
        "avatar_url": str|None,
        "karma": int|None,
        "followers": int|None,
        "is_claimed": bool|None,
        "claimed_at": str|None,
        "created_at": str|None,
        "profile_url": str|None,
      }
    """
    agents = []
    if isinstance(payload, dict):
        agents = payload.get("agents", [])
    elif isinstance(payload, list):
        agents = payload
    else:
        return []

    out: List[Dict[str, Any]] = []
    for a in agents:
        if not isinstance(a, dict):
            continue

        name = a.get("name")
        if not name:
            continue

        out.append({
            "agent_external_id": a.get("id"),
            "handle": name,
            "display_name": name,
            "bio": a.get("description"),
            "avatar_url": a.get("avatar_url"),
            "karma": a.get("karma"),
            "followers": a.get("follower_count"),
            "is_claimed": a.get("is_claimed"),
            "claimed_at": a.get("claimed_at"),
            "created_at": a.get("created_at"),
            "profile_url": f"https://www.moltbook.com/u/{name}",
        })

    return out


def extract_posts_from_list(payload: Any) -> List[Dict[str, Any]]:
    """
    /api/v1/posts returns:
      {"success": true, "posts": [ {id, title, content, url, upvotes, downvotes, comment_count, created_at, author, submolt}, ... ]}
    """
    posts = []
    if isinstance(payload, dict):
        posts = payload.get("posts", [])
    elif isinstance(payload, list):
        posts = payload
    else:
        return []

    out: List[Dict[str, Any]] = []
    for p in posts:
        if not isinstance(p, dict):
            continue

        post_id = p.get("id")
        if not post_id:
            continue

        author = p.get("author") or {}
        submolt = p.get("submolt") or {}

        up = p.get("upvotes")
        down = p.get("downvotes")
        score = (up - down) if isinstance(up,
                                          int) and isinstance(down, int) else None

        out.append({
            "post_external_id": post_id,
            "title": p.get("title"),
            "body_text": p.get("content"),
            "outbound_url": p.get("url"),
            "upvotes": up,
            "downvotes": down,
            "score": score,
            "comment_count": p.get("comment_count"),
            "created_at": p.get("created_at"),
            "author_external_id": author.get("id"),
            "author_handle": author.get("name"),
            "submolt_external_id": submolt.get("id"),
            "submolt_slug": submolt.get("name"),
            "submolt_display_name": submolt.get("display_name"),
            "permalink": f"https://www.moltbook.com/post/{post_id}",
        })

    return out


def extract_post_detail(payload: Any) -> Optional[Dict[str, Any]]:
    """
    /api/v1/posts/<uuid> returns:
      {"success": true, "post": {..., author: {...}, submolt: {...}}, "comments": [...], "context": {...}}

    Canonical output:
      {
        "post": {...},
        "comments": [...],
        "context_tip": str|None
      }
    """
    if not isinstance(payload, dict):
        return None
    post = payload.get("post")
    if not isinstance(post, dict) or not post.get("id"):
        return None

    author = post.get("author") or {}
    submolt = post.get("submolt") or {}

    up = post.get("upvotes")
    down = post.get("downvotes")
    score = (up - down) if isinstance(up,
                                      int) and isinstance(down, int) else None

    return {
        "post": {
            "post_external_id": post.get("id"),
            "title": post.get("title"),
            "body_text": post.get("content"),
            "outbound_url": post.get("url"),
            "upvotes": up,
            "downvotes": down,
            "score": score,
            "comment_count": post.get("comment_count"),
            "created_at": post.get("created_at"),
            "permalink": f"https://www.moltbook.com/post/{post.get('id')}",
            "submolt_external_id": submolt.get("id"),
            "submolt_slug": submolt.get("name"),
            "submolt_display_name": submolt.get("display_name"),
            # author fields (richer than list endpoint)
            "author_external_id": author.get("id"),
            "author_handle": author.get("name"),
            "author_bio": author.get("description"),
            "author_karma": author.get("karma"),
            "author_follower_count": author.get("follower_count"),
            "author_following_count": author.get("following_count"),
            "author_owner": author.get("owner"),
            "author_you_follow": author.get("you_follow"),
        },
        "comments": payload.get("comments") or [],
        "context_tip": (payload.get("context") or {}).get("tip"),
    }


def flatten_comments_tree(comments: Any, post_external_id: str) -> List[Dict[str, Any]]:
    """
    Turns a post-detail `comments` tree into flat rows while preserving parent links.
    Comment nodes have: id, content, parent_id, upvotes, downvotes, created_at, author{...}, replies[].
    """
    if not isinstance(comments, list):
        return []

    out: List[Dict[str, Any]] = []

    def walk(node: Dict[str, Any], parent_comment_id: Optional[str]):
        if not isinstance(node, dict):
            return
        cid = node.get("id")
        if not cid:
            return
        author = node.get("author") or {}
        up = node.get("upvotes")
        down = node.get("downvotes")
        score = (up - down) if isinstance(up,
                                          int) and isinstance(down, int) else None

        out.append({
            "comment_external_id": cid,
            "post_external_id": post_external_id,
            "parent_comment_external_id": parent_comment_id,
            "body_text": node.get("content"),
            "upvotes": up,
            "downvotes": down,
            "score": score,
            "created_at": node.get("created_at"),
            "author_external_id": author.get("id"),
            "author_handle": author.get("name"),
            "author_karma": author.get("karma"),
            "author_follower_count": author.get("follower_count"),
        })

        for child in (node.get("replies") or []):
            walk(child, cid)

    for c in comments:
        walk(c, None)

    return out


def extract_submolts_from_list(payload: Any) -> List[Dict[str, Any]]:
    """
    /api/v1/submolts returns:
      {"success": true, "submolts": [{id, name, display_name, description, subscriber_count, ...}]}

    Canonical output:
      {
        "submolt_external_id": str,
        "name": str,               # URL slug
        "display_name": str,
        "description": str|None,
        "subscriber_count": int|None,
        "post_count": int|None,
        "created_at": str|None,
        "owner_agent_id": str|None,
        "owner_handle": str|None,
        "banner_color": str|None,
        "theme_color": str|None,
        "avatar_url": str|None,
        "banner_url": str|None,
        "url": str,
      }
    """
    submolts = []
    if isinstance(payload, dict):
        submolts = payload.get("submolts", [])
    elif isinstance(payload, list):
        submolts = payload
    else:
        return []

    out: List[Dict[str, Any]] = []
    for s in submolts:
        if not isinstance(s, dict):
            continue

        name = s.get("name")
        if not name:
            continue

        owner = s.get("owner") or {}

        out.append({
            "submolt_external_id": s.get("id"),
            "name": name,
            "display_name": s.get("display_name") or name,
            "description": s.get("description"),
            "subscriber_count": s.get("subscriber_count"),
            "post_count": s.get("post_count"),
            "created_at": s.get("created_at"),
            "owner_agent_id": owner.get("id"),
            "owner_handle": owner.get("name"),
            "banner_color": s.get("banner_color"),
            "theme_color": s.get("theme_color"),
            "avatar_url": s.get("avatar_url"),
            "banner_url": s.get("banner_url"),
            "url": f"https://www.moltbook.com/m/{name}",
        })

    return out


def extract_submolt_detail(payload: Any) -> Optional[Dict[str, Any]]:
    """
    /api/v1/submolts/<name> returns detailed submolt info.

    Returns canonical submolt dict or None if invalid.
    """
    if not isinstance(payload, dict):
        return None

    # Handle both direct submolt and {"submolt": {...}} formats
    submolt = payload.get("submolt") if "submolt" in payload else payload
    if not isinstance(submolt, dict):
        return None

    name = submolt.get("name")
    if not name:
        return None

    owner = submolt.get("owner") or {}
    moderators = submolt.get("moderators") or []

    return {
        "submolt_external_id": submolt.get("id"),
        "name": name,
        "display_name": submolt.get("display_name") or name,
        "description": submolt.get("description"),
        "subscriber_count": submolt.get("subscriber_count"),
        "post_count": submolt.get("post_count"),
        "created_at": submolt.get("created_at"),
        "owner_agent_id": owner.get("id"),
        "owner_handle": owner.get("name"),
        "banner_color": submolt.get("banner_color"),
        "theme_color": submolt.get("theme_color"),
        "avatar_url": submolt.get("avatar_url"),
        "banner_url": submolt.get("banner_url"),
        "moderators": [
            {"agent_id": m.get("id"), "handle": m.get(
                "name"), "role": m.get("role")}
            for m in moderators if isinstance(m, dict)
        ],
        "url": f"https://www.moltbook.com/m/{name}",
    }


def extract_agent_profile(payload: Any) -> Optional[Dict[str, Any]]:
    """
    /api/v1/agents/profile?name=NAME returns:
      {"success": true, "agent": {..., owner: {...}}, "recentPosts": [...]}

    Returns canonical agent dict with enriched fields.
    """
    if not isinstance(payload, dict):
        return None

    agent = payload.get("agent")
    if not isinstance(agent, dict):
        return None

    name = agent.get("name")
    if not name:
        return None

    owner = agent.get("owner") or {}
    recent_posts = payload.get("recentPosts") or []

    return {
        "agent_external_id": agent.get("id"),
        "handle": name,
        "display_name": name,
        "bio": agent.get("description"),
        "avatar_url": agent.get("avatar_url"),
        "karma": agent.get("karma"),
        "follower_count": agent.get("follower_count"),
        "following_count": agent.get("following_count"),
        "is_claimed": agent.get("is_claimed"),
        "is_active": agent.get("is_active"),
        "claimed_at": agent.get("claimed_at"),
        "created_at": agent.get("created_at"),
        "last_active": agent.get("last_active"),
        "profile_url": f"https://www.moltbook.com/u/{name}",
        # Owner (human) info
        "owner": {
            "x_handle": owner.get("x_handle"),
            "x_name": owner.get("x_name"),
            "x_bio": owner.get("x_bio"),
            "x_avatar": owner.get("x_avatar"),
            "x_follower_count": owner.get("x_follower_count"),
            "x_following_count": owner.get("x_following_count"),
            "x_verified": owner.get("x_verified"),
        } if owner else None,
        # Recent activity
        "recent_post_count": len(recent_posts),
        "recent_posts": [
            {
                "post_id": p.get("id"),
                "title": p.get("title"),
                "created_at": p.get("created_at"),
            }
            for p in recent_posts[:5] if isinstance(p, dict)
        ],
    }


@dataclass
class SiteStats:
    """Aggregate statistics for the entire Moltbook site at a point in time."""
    snapshot_at: str
    total_agents: int
    total_posts: int
    total_comments: int
    total_submolts: int

    # Derived metrics
    avg_comments_per_post: float
    avg_posts_per_agent: float
    avg_karma_per_agent: float

    # Top entities
    top_submolts_by_posts: List[Dict[str, Any]]
    top_agents_by_karma: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_at": self.snapshot_at,
            "total_agents": self.total_agents,
            "total_posts": self.total_posts,
            "total_comments": self.total_comments,
            "total_submolts": self.total_submolts,
            "avg_comments_per_post": self.avg_comments_per_post,
            "avg_posts_per_agent": self.avg_posts_per_agent,
            "avg_karma_per_agent": self.avg_karma_per_agent,
            "top_submolts_by_posts": self.top_submolts_by_posts,
            "top_agents_by_karma": self.top_agents_by_karma,
        }


def extract_site_stats(
    agents: List[Dict[str, Any]],
    posts: List[Dict[str, Any]],
    submolts: List[Dict[str, Any]],
    comments: List[Dict[str, Any]],
) -> SiteStats:
    """
    Derive site-wide statistics from fetched entities.

    Args:
        agents: List of canonical agent dicts
        posts: List of canonical post dicts  
        submolts: List of canonical submolt dicts
        comments: List of canonical comment dicts

    Returns:
        SiteStats with aggregated metrics
    """
    total_agents = len(agents)
    total_posts = len(posts)
    total_comments = len(comments)
    total_submolts = len(submolts)

    # Calculate averages
    avg_comments_per_post = total_comments / max(1, total_posts)
    avg_posts_per_agent = total_posts / max(1, total_agents)

    # Average karma
    total_karma = sum(a.get("karma") or 0 for a in agents)
    avg_karma_per_agent = total_karma / max(1, total_agents)

    # Top submolts by post count
    submolts_sorted = sorted(
        submolts,
        key=lambda s: s.get("post_count") or 0,
        reverse=True
    )
    top_submolts = [
        {"name": s.get("name"), "post_count": s.get("post_count")}
        for s in submolts_sorted[:10]
    ]

    # Top agents by karma
    agents_sorted = sorted(
        agents,
        key=lambda a: a.get("karma") or 0,
        reverse=True
    )
    top_agents = [
        {"handle": a.get("handle"), "karma": a.get("karma")}
        for a in agents_sorted[:10]
    ]

    return SiteStats(
        snapshot_at=datetime.now(timezone.utc).isoformat(),
        total_agents=total_agents,
        total_posts=total_posts,
        total_comments=total_comments,
        total_submolts=total_submolts,
        avg_comments_per_post=round(avg_comments_per_post, 2),
        avg_posts_per_agent=round(avg_posts_per_agent, 2),
        avg_karma_per_agent=round(avg_karma_per_agent, 2),
        top_submolts_by_posts=top_submolts,
        top_agents_by_karma=top_agents,
    )
