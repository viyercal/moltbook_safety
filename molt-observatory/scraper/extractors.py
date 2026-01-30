# scraper/extractors.py
from typing import Any, Dict, List, Optional

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
        score = (up - down) if isinstance(up, int) and isinstance(down, int) else None

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
    score = (up - down) if isinstance(up, int) and isinstance(down, int) else None

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
        score = (up - down) if isinstance(up, int) and isinstance(down, int) else None

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
