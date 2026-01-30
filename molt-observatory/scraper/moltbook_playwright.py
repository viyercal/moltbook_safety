# scraper/moltbook_api.py
import json
import time
import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional
import requests

@dataclass
class ApiResponse:
    url: str
    status: int
    elapsed_ms: int
    json_body: Any
    headers: Dict[str, str]

class TokenBucket:
    def __init__(self, rate_per_sec: float, burst: int):
        self.rate = rate_per_sec
        self.capacity = burst
        self.tokens = burst
        self.last = time.time()

    def take(self, n: int = 1):
        while True:
            now = time.time()
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens >= n:
                self.tokens -= n
                return
            time.sleep(0.05)

def stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

class MoltbookAPI:
    def __init__(self, base_url: str = "https://www.moltbook.com", user_agent: str = None):
        self.base = base_url.rstrip("/")
        self.sess = requests.Session()
        self.sess.headers.update({
            "User-Agent": user_agent or "molt-observatory-bot/0.1 (contact: security-research; purpose: public-safety-auditing)",
            "Accept": "application/json",
        })
        self.rl = TokenBucket(rate_per_sec=0.5, burst=3)  # ~1 req per 2s, burst 3

    def get_json(self, path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 20) -> ApiResponse:
        url = f"{self.base}{path}"
        # token bucket rate limiting
        self.rl.take(1)

        # retries with backoff
        attempt = 0
        while True:
            attempt += 1
            t0 = time.time()
            resp = self.sess.get(url, params=params, timeout=timeout)
            elapsed_ms = int((time.time() - t0) * 1000)

            # backoff on 429/5xx
            if resp.status_code in (429,) or 500 <= resp.status_code < 600:
                if attempt >= 5:
                    resp.raise_for_status()
                sleep_s = min(60, (2 ** (attempt - 1)) + random.random())
                time.sleep(sleep_s)
                continue

            resp.raise_for_status()
            return ApiResponse(
                url=resp.url,
                status=resp.status_code,
                elapsed_ms=elapsed_ms,
                json_body=resp.json(),
                headers=dict(resp.headers),
            )
