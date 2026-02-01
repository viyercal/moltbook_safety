# scraper/moltbook_api.py
"""
Moltbook API client with pagination, rate limiting, and comprehensive entity fetching.
API docs: https://www.moltbook.com/skill.md

Authentication:
    Set MOLTBOOK_API_KEY in your .env file to authenticate requests.
    All requests will include Authorization: Bearer {API_KEY} header.
"""
import json
import os
import time
import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class ApiResponse:
    url: str
    status: int
    elapsed_ms: int
    json_body: Any
    headers: Dict[str, str]
    content_type: str


class TokenBucket:
    def __init__(self, rate_per_sec: float, burst: int):
        self.rate = rate_per_sec
        self.capacity = burst
        self.tokens = burst
        self.last = time.time()

    def take(self, n: int = 1):
        while True:
            now = time.time()
            self.tokens = min(self.capacity, self.tokens +
                              (now - self.last) * self.rate)
            self.last = now
            if self.tokens >= n:
                self.tokens -= n
                return
            time.sleep(0.05)


def stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _parse_json_lenient(text: str) -> Any:
    """
    Parse JSON from the beginning of `text`, ignoring trailing junk.
    Useful if an edge layer accidentally appends HTML after valid JSON.
    """
    text = text.lstrip()
    dec = json.JSONDecoder()
    obj, _ = dec.raw_decode(text)
    return obj


class MoltbookAPI:
    """
    Moltbook API client with authentication, rate limiting, and pagination.

    Requires MOLTBOOK_API_KEY environment variable for authenticated requests.
    Without an API key, some endpoints may return limited data (e.g., author: null).
    """

    def __init__(self, base_url: str = "https://www.moltbook.com", user_agent: str = None,
                 rate_per_sec: float = 0.5, burst: int = 3, api_key: str = None):
        self.base = base_url.rstrip("/")
        self.sess = requests.Session()

        # Set default headers
        self.sess.headers.update({
            "User-Agent": user_agent or "molt-observatory-bot/0.1 (contact: security-research; purpose: public-safety-auditing)",
            "Accept": "application/json",
        })

        # Add Authorization header if API key is available
        # Priority: explicit api_key param > environment variable
        self.api_key = api_key or os.environ.get("MOLTBOOK_API_KEY")
        if self.api_key:
            self.sess.headers["Authorization"] = f"Bearer {self.api_key}"

        # ~1 req per 2s, burst 3 by default
        self.rl = TokenBucket(rate_per_sec=rate_per_sec, burst=burst)

    def get_json(self, path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 160) -> ApiResponse:
        url = f"{self.base}{path}"
        self.rl.take(1)

        attempt = 0
        while True:
            attempt += 1
            t0 = time.time()
            resp = self.sess.get(url, params=params, timeout=timeout)
            elapsed_ms = int((time.time() - t0) * 1000)

            if resp.status_code in (429,) or 500 <= resp.status_code < 600:
                if attempt >= 5:
                    resp.raise_for_status()
                sleep_s = min(60, (2 ** (attempt - 1)) + random.random())
                time.sleep(sleep_s)
                continue

            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")

            # Prefer strict JSON parsing for clean application/json responses.
            try:
                body = resp.json()
            except Exception:
                # Fallback if something strange happens (rare, but low-cost insurance).
                body = _parse_json_lenient(resp.text)

            return ApiResponse(
                url=resp.url,
                status=resp.status_code,
                elapsed_ms=elapsed_ms,
                json_body=body,
                headers=dict(resp.headers),
                content_type=ctype,
            )

    def paginate_all(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        response_key: str = "posts",
        limit: int = 50,
        max_pages: int = 100,
        stop_at_timestamp: Optional[str] = None,
        stop_at_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Paginate through all results until exhausted or stop condition met.

        Args:
            path: API endpoint path
            params: Base query parameters
            response_key: Key in response containing items (e.g., "posts", "agents")
            limit: Items per page
            max_pages: Maximum pages to fetch
            stop_at_timestamp: Stop when created_at <= this timestamp
            stop_at_id: Stop when this ID is encountered

        Returns:
            List of all fetched items
        """
        all_items: List[Dict[str, Any]] = []
        seen_ids: set = set()

        params = dict(params or {})
        params["limit"] = limit

        for page in range(max_pages):
            params["offset"] = page * limit

            resp = self.get_json(path, params=params)
            body = resp.json_body

            # Handle both {"key": [...]} and direct list responses
            if isinstance(body, dict):
                items = body.get(response_key, [])
            elif isinstance(body, list):
                items = body
            else:
                items = []

            if not items:
                break

            for item in items:
                item_id = item.get("id")

                # Check stop conditions
                if stop_at_id and item_id == stop_at_id:
                    return all_items

                if stop_at_timestamp:
                    created_at = item.get("created_at")
                    if created_at and created_at <= stop_at_timestamp:
                        return all_items

                # Deduplicate
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    all_items.append(item)

            # If we got fewer items than limit, we've reached the end
            if len(items) < limit:
                break

        return all_items

    # =========================================================================
    # Entity-specific fetching methods
    # =========================================================================

    def get_all_posts(
        self,
        sort: str = "new",
        limit: int = 50,
        max_pages: int = 100,
        stop_at_timestamp: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch all posts with pagination."""
        return self.paginate_all(
            "/api/v1/posts",
            params={"sort": sort},
            response_key="posts",
            limit=limit,
            max_pages=max_pages,
            stop_at_timestamp=stop_at_timestamp,
        )

    def get_post_detail(self, post_id: str) -> ApiResponse:
        """Fetch a single post with comments."""
        return self.get_json(f"/api/v1/posts/{post_id}")

    def get_all_agents(
        self,
        limit: int = 50,
        max_pages: int = 100,
        stop_at_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch all agents with pagination."""
        return self.paginate_all(
            "/api/v1/agents/recent",
            params={},
            response_key="agents",
            limit=limit,
            max_pages=max_pages,
            stop_at_id=stop_at_id,
        )

    def get_agent_profile(self, agent_name: str) -> ApiResponse:
        """Fetch detailed profile for a specific agent."""
        return self.get_json("/api/v1/agents/profile", params={"name": agent_name})

    def get_all_submolts(
        self,
        limit: int = 50,
        max_pages: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch all submolts with pagination."""
        return self.paginate_all(
            "/api/v1/submolts",
            params={},
            response_key="submolts",
            limit=limit,
            max_pages=max_pages,
        )

    def get_submolt_detail(self, submolt_name: str) -> ApiResponse:
        """Fetch detailed info for a specific submolt."""
        return self.get_json(f"/api/v1/submolts/{submolt_name}")

    def get_submolt_feed(
        self,
        submolt_name: str,
        sort: str = "new",
        limit: int = 50,
        max_pages: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch all posts from a specific submolt."""
        return self.paginate_all(
            f"/api/v1/submolts/{submolt_name}/feed",
            params={"sort": sort},
            response_key="posts",
            limit=limit,
            max_pages=max_pages,
        )

    def search(
        self,
        query: str,
        limit: int = 25,
    ) -> ApiResponse:
        """Search posts, agents, and submolts."""
        return self.get_json("/api/v1/search", params={"q": query, "limit": limit})

    def get_post_comments(self, post_id: str, sort: str = "top") -> ApiResponse:
        """Fetch comments for a specific post."""
        return self.get_json(f"/api/v1/posts/{post_id}/comments", params={"sort": sort})
