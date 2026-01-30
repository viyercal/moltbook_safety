import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class ORResponse:
    status_code: int
    text: str
    json: Optional[Dict[str, Any]]
    elapsed_ms: int


class OpenRouterClient:
    """
    Minimal OpenRouter chat-completions client.
    Endpoint + headers are OpenAI-compatible as per OpenRouter docs.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        referer: Optional[str] = None,
        title: Optional[str] = None,
        timeout_s: int = 60,
        max_retries: int = 5,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY")

        self.base_url = base_url
        self.referer = referer or os.environ.get("OPENROUTER_REFERER")
        self.title = title or os.environ.get("OPENROUTER_TITLE")
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def _headers(self) -> Dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Optional per OpenRouter quickstart
        if self.referer:
            h["HTTP-Referer"] = self.referer
        if self.title:
            h["X-Title"] = self.title
        return h

    def chat(self, payload: Dict[str, Any]) -> ORResponse:
        """
        Retries with exponential backoff on typical transient errors (429/502/503/408).
        """
        backoff_s = 1.0
        last_exc = None

        for attempt in range(1, self.max_retries + 1):
            t0 = time.time()
            try:
                r = requests.post(
                    self.base_url,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self.timeout_s,
                )
                elapsed_ms = int((time.time() - t0) * 1000)
                j = None
                try:
                    j = r.json()
                except Exception:
                    j = None

                # Success
                if 200 <= r.status_code < 300:
                    return ORResponse(r.status_code, r.text, j, elapsed_ms)

                # Retry on transient
                if r.status_code in (408, 429, 502, 503):
                    time.sleep(backoff_s)
                    backoff_s = min(backoff_s * 2.0, 30.0)
                    continue

                # Non-retryable
                return ORResponse(r.status_code, r.text, j, elapsed_ms)

            except Exception as e:
                last_exc = e
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2.0, 30.0)

        raise RuntimeError(f"OpenRouter request failed after retries: {last_exc}")
