"""
Tests for the Moltbook API scraper module.

Tests:
- TokenBucket rate limiting
- MoltbookAPI request handling
- Pagination logic
- Retry behavior
- Stop conditions
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


class TestTokenBucket:
    """Tests for the TokenBucket rate limiter."""
    
    def test_initial_burst(self):
        """Bucket should allow burst capacity immediately."""
        from scraper.moltbook_api import TokenBucket
        
        bucket = TokenBucket(rate_per_sec=1.0, burst=3)
        
        # Should be able to take 3 tokens immediately
        start = time.time()
        bucket.take(1)
        bucket.take(1)
        bucket.take(1)
        elapsed = time.time() - start
        
        # All three should complete very quickly (< 0.1s)
        assert elapsed < 0.2
    
    def test_rate_limiting_after_burst(self):
        """Bucket should throttle after burst is exhausted."""
        from scraper.moltbook_api import TokenBucket
        
        bucket = TokenBucket(rate_per_sec=10.0, burst=1)  # Fast rate for testing
        
        # First token is immediate
        bucket.take(1)
        
        # Second token should require waiting
        start = time.time()
        bucket.take(1)
        elapsed = time.time() - start
        
        # Should have waited ~0.1 seconds (1/10 per second)
        assert elapsed >= 0.05  # Allow some tolerance
    
    def test_token_refill(self):
        """Tokens should refill over time."""
        from scraper.moltbook_api import TokenBucket
        
        bucket = TokenBucket(rate_per_sec=100.0, burst=2)  # Very fast refill
        
        # Exhaust burst
        bucket.take(2)
        
        # Wait for refill
        time.sleep(0.05)
        
        # Should be able to take at least 1 token without much delay
        start = time.time()
        bucket.take(1)
        elapsed = time.time() - start
        
        assert elapsed < 0.1


class TestMoltbookAPI:
    """Tests for the MoltbookAPI class."""
    
    def test_initialization(self):
        """API should initialize with correct defaults."""
        from scraper.moltbook_api import MoltbookAPI
        
        api = MoltbookAPI()
        
        assert api.base == "https://www.moltbook.com"
        assert "molt-observatory" in api.sess.headers.get("User-Agent", "").lower()
    
    def test_custom_base_url(self):
        """API should accept custom base URL."""
        from scraper.moltbook_api import MoltbookAPI
        
        api = MoltbookAPI(base_url="https://test.moltbook.com/")
        
        assert api.base == "https://test.moltbook.com"  # Trailing slash stripped
    
    @patch('scraper.moltbook_api.requests.Session')
    def test_get_json_success(self, mock_session_class):
        """get_json should return ApiResponse on success."""
        from scraper.moltbook_api import MoltbookAPI
        
        # Setup mock
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://www.moltbook.com/api/v1/posts"
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"success": True, "posts": []}
        mock_session.get.return_value = mock_response
        
        api = MoltbookAPI(rate_per_sec=100.0, burst=10)  # Fast rate for testing
        result = api.get_json("/api/v1/posts")
        
        assert result.status == 200
        assert result.json_body == {"success": True, "posts": []}
    
    @patch('scraper.moltbook_api.requests.Session')
    def test_default_timeout_is_160(self, mock_session_class):
        """get_json should use 160s default timeout."""
        from scraper.moltbook_api import MoltbookAPI
        
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://test.com/api"
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {}
        mock_session.get.return_value = mock_response
        
        api = MoltbookAPI(rate_per_sec=100.0, burst=10)
        api.get_json("/api/v1/posts")
        
        # Check timeout was passed correctly
        call_kwargs = mock_session.get.call_args[1]
        assert call_kwargs.get("timeout") == 160


class TestPagination:
    """Tests for the paginate_all method."""
    
    @patch('scraper.moltbook_api.requests.Session')
    def test_paginate_collects_all_items(self, mock_session_class):
        """paginate_all should collect items from multiple pages."""
        from scraper.moltbook_api import MoltbookAPI
        
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        # Simulate 2 pages of results
        page1 = {"posts": [{"id": "1"}, {"id": "2"}]}
        page2 = {"posts": [{"id": "3"}]}
        page3 = {"posts": []}  # Empty = done
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://test.com"
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.side_effect = [page1, page2, page3]
        mock_session.get.return_value = mock_response
        
        api = MoltbookAPI(rate_per_sec=100.0, burst=10)
        result = api.paginate_all("/api/v1/posts", response_key="posts", limit=2, max_pages=10)
        
        assert len(result) == 3
        assert [r["id"] for r in result] == ["1", "2", "3"]
    
    @patch('scraper.moltbook_api.requests.Session')
    def test_paginate_stops_at_max_pages(self, mock_session_class):
        """paginate_all should respect max_pages limit."""
        from scraper.moltbook_api import MoltbookAPI
        
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        # Simulate infinite pages
        page = {"posts": [{"id": "1"}, {"id": "2"}]}
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://test.com"
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = page
        mock_session.get.return_value = mock_response
        
        api = MoltbookAPI(rate_per_sec=100.0, burst=10)
        result = api.paginate_all("/api/v1/posts", response_key="posts", limit=2, max_pages=2)
        
        # Should only get 2 pages worth (4 items, but deduplicated to 2)
        assert mock_session.get.call_count == 2
    
    @patch('scraper.moltbook_api.requests.Session')
    def test_paginate_stops_at_timestamp(self, mock_session_class):
        """paginate_all should stop when reaching stop_at_timestamp."""
        from scraper.moltbook_api import MoltbookAPI
        
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        page = {"posts": [
            {"id": "1", "created_at": "2026-01-30T12:00:00Z"},
            {"id": "2", "created_at": "2026-01-30T11:00:00Z"},
            {"id": "3", "created_at": "2026-01-30T10:00:00Z"},  # Should stop here
        ]}
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://test.com"
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = page
        mock_session.get.return_value = mock_response
        
        api = MoltbookAPI(rate_per_sec=100.0, burst=10)
        result = api.paginate_all(
            "/api/v1/posts",
            response_key="posts",
            stop_at_timestamp="2026-01-30T10:30:00Z"  # Stop before item 3
        )
        
        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[1]["id"] == "2"
    
    @patch('scraper.moltbook_api.requests.Session')
    def test_paginate_deduplicates(self, mock_session_class):
        """paginate_all should deduplicate items by id."""
        from scraper.moltbook_api import MoltbookAPI
        
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        # Same item appears on multiple pages
        page1 = {"posts": [{"id": "1"}, {"id": "2"}]}
        page2 = {"posts": [{"id": "2"}, {"id": "3"}]}  # id=2 is duplicate
        page3 = {"posts": []}
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://test.com"
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.side_effect = [page1, page2, page3]
        mock_session.get.return_value = mock_response
        
        api = MoltbookAPI(rate_per_sec=100.0, burst=10)
        result = api.paginate_all("/api/v1/posts", response_key="posts", limit=2)
        
        assert len(result) == 3
        ids = [r["id"] for r in result]
        assert len(ids) == len(set(ids))  # All unique


class TestHelperFunctions:
    """Tests for utility functions."""
    
    def test_stable_hash_deterministic(self):
        """stable_hash should produce consistent output."""
        from scraper.moltbook_api import stable_hash
        
        obj = {"key": "value", "nested": {"a": 1}}
        
        hash1 = stable_hash(obj)
        hash2 = stable_hash(obj)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex
    
    def test_stable_hash_different_for_different_objects(self):
        """stable_hash should differ for different objects."""
        from scraper.moltbook_api import stable_hash
        
        obj1 = {"key": "value1"}
        obj2 = {"key": "value2"}
        
        assert stable_hash(obj1) != stable_hash(obj2)
    
    def test_parse_json_lenient(self):
        """_parse_json_lenient should handle trailing garbage."""
        from scraper.moltbook_api import _parse_json_lenient
        
        text = '{"key": "value"} some trailing garbage'
        result = _parse_json_lenient(text)
        
        assert result == {"key": "value"}
    
    def test_parse_json_lenient_with_whitespace(self):
        """_parse_json_lenient should handle leading whitespace."""
        from scraper.moltbook_api import _parse_json_lenient
        
        text = '   \n\n {"key": "value"}'
        result = _parse_json_lenient(text)
        
        assert result == {"key": "value"}


class TestEntityMethods:
    """Tests for entity-specific API methods."""
    
    @patch('scraper.moltbook_api.MoltbookAPI.paginate_all')
    def test_get_all_posts_calls_paginate(self, mock_paginate):
        """get_all_posts should use paginate_all with correct params."""
        from scraper.moltbook_api import MoltbookAPI
        
        mock_paginate.return_value = [{"id": "1"}]
        
        api = MoltbookAPI()
        result = api.get_all_posts(sort="new", limit=50, max_pages=10)
        
        mock_paginate.assert_called_once()
        call_args = mock_paginate.call_args
        assert call_args[1]["response_key"] == "posts"
        assert call_args[1]["limit"] == 50
    
    @patch('scraper.moltbook_api.MoltbookAPI.paginate_all')
    def test_get_all_agents_calls_paginate(self, mock_paginate):
        """get_all_agents should use paginate_all with correct params."""
        from scraper.moltbook_api import MoltbookAPI
        
        mock_paginate.return_value = [{"id": "agent-1"}]
        
        api = MoltbookAPI()
        result = api.get_all_agents(limit=25)
        
        mock_paginate.assert_called_once()
        call_args = mock_paginate.call_args
        assert call_args[1]["response_key"] == "agents"
    
    @patch('scraper.moltbook_api.MoltbookAPI.get_json')
    def test_get_post_detail(self, mock_get_json):
        """get_post_detail should call correct endpoint."""
        from scraper.moltbook_api import MoltbookAPI, ApiResponse
        
        mock_get_json.return_value = MagicMock(
            json_body={"post": {"id": "test-id"}}
        )
        
        api = MoltbookAPI()
        result = api.get_post_detail("test-id")
        
        mock_get_json.assert_called_with("/api/v1/posts/test-id")

