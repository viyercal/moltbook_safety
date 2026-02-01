"""
Tests for the end-to-end pipeline orchestrator.

Tests:
- run_once function with mocked dependencies
- Directory structure creation
- Artifact file generation
- Incremental mode
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# Sample fixture data for mocking API responses
MOCK_POSTS_LIST = {
    "success": True,
    "posts": [
        {
            "id": "mock-post-001",
            "title": "Mock Post",
            "content": "Test content",
            "upvotes": 5,
            "downvotes": 1,
            "comment_count": 1,
            "created_at": "2026-01-30T12:00:00+00:00",
            "author": {"id": "agent-001", "name": "MockAgent"},
            "submolt": {"id": "submolt-001", "name": "general", "display_name": "General"}
        }
    ]
}

MOCK_POST_DETAIL = {
    "success": True,
    "post": {
        "id": "mock-post-001",
        "title": "Mock Post",
        "content": "Test content",
        "upvotes": 5,
        "downvotes": 1,
        "comment_count": 1,
        "created_at": "2026-01-30T12:00:00+00:00",
        "submolt": {"id": "submolt-001", "name": "general", "display_name": "General"},
        "author": {
            "id": "agent-001",
            "name": "MockAgent",
            "karma": 100,
            "follower_count": 50
        }
    },
    "comments": [
        {
            "id": "comment-001",
            "content": "A mock comment",
            "parent_id": None,
            "upvotes": 2,
            "downvotes": 0,
            "created_at": "2026-01-30T12:30:00+00:00",
            "author": {"id": "agent-002", "name": "Commenter", "karma": 50},
            "replies": []
        }
    ],
    "context": {"tip": "Mock context tip"}
}

MOCK_AGENTS_LIST = {
    "success": True,
    "agents": [
        {"id": "agent-001", "name": "MockAgent", "karma": 100, "follower_count": 50}
    ]
}

MOCK_SUBMOLTS_LIST = {
    "success": True,
    "submolts": [
        {"id": "submolt-001", "name": "general", "display_name": "General", "subscriber_count": 1000}
    ]
}


class MockApiResponse:
    """Mock API response for testing."""
    def __init__(self, json_body):
        self.json_body = json_body
        self.status = 200
        self.url = "https://mock.moltbook.com"
        self.elapsed_ms = 100
        self.headers = {}
        self.content_type = "application/json"


def create_mock_api():
    """Create a mock MoltbookAPI that returns fixture data."""
    mock_api = MagicMock()
    
    def mock_get_json(path, params=None, timeout=160):
        if "/api/v1/posts/" in path and path != "/api/v1/posts":
            return MockApiResponse(MOCK_POST_DETAIL)
        elif path == "/api/v1/posts":
            return MockApiResponse(MOCK_POSTS_LIST)
        elif "agents" in path:
            return MockApiResponse(MOCK_AGENTS_LIST)
        elif "submolts" in path:
            return MockApiResponse(MOCK_SUBMOLTS_LIST)
        return MockApiResponse({"success": True})
    
    mock_api.get_json = mock_get_json
    return mock_api


class TestRunOnce:
    """Tests for the run_once pipeline function."""
    
    @pytest.fixture
    def mock_dependencies(self, temp_run_dir, monkeypatch):
        """Set up mocked dependencies for pipeline tests."""
        # Mock the API
        mock_api = create_mock_api()
        
        # Mock judge runner to avoid real LLM calls
        def mock_run_judges(transcripts, **kwargs):
            return [{
                "post_id": t.get("post_id", "unknown"),
                "transcript_id": t.get("transcript_id", "unknown"),
                "model": "mock/model",
                "scores": {
                    "harm_enablement": {"score": 1, "confidence": 0.9, "evidence": []},
                    "deception_or_evasion": {"score": 0, "confidence": 0.95, "evidence": []},
                    "self_preservation_power_seeking": {"score": 1, "confidence": 0.85, "evidence": []},
                    "delusional_sycophancy": {"score": 0, "confidence": 0.9, "evidence": []}
                }
            } for t in transcripts]
        
        def mock_run_comment_judges(transcripts, **kwargs):
            return [{
                "comment_id": t.get("comment_id", "unknown"),
                "post_id": t.get("post_id", "unknown"),
                "author": t.get("target_comment", {}).get("author", "unknown"),
                "model": "mock/model",
                "scores": {
                    "harm_enablement": {"score": 0, "confidence": 0.9, "evidence": []},
                    "deception_or_evasion": {"score": 0, "confidence": 0.95, "evidence": []},
                    "self_preservation_power_seeking": {"score": 0, "confidence": 0.85, "evidence": []},
                    "delusional_sycophancy": {"score": 0, "confidence": 0.9, "evidence": []}
                }
            } for t in transcripts]
        
        return {
            "mock_api": mock_api,
            "mock_run_judges": mock_run_judges,
            "mock_run_comment_judges": mock_run_comment_judges,
            "temp_dir": temp_run_dir
        }
    
    def test_creates_directory_structure(self, mock_dependencies, monkeypatch):
        """run_once should create expected directory structure."""
        with patch('eval_orchestrator.MoltbookAPI', return_value=mock_dependencies["mock_api"]), \
             patch('eval_orchestrator.run_judges', mock_dependencies["mock_run_judges"]), \
             patch('eval_orchestrator.run_comment_judges', mock_dependencies["mock_run_comment_judges"]), \
             patch('eval_orchestrator.get_state_manager') as mock_state:
            
            # Mock state manager
            mock_state_mgr = MagicMock()
            mock_state_mgr.start_run.return_value = MagicMock()
            mock_state_mgr.get_last_created_at.return_value = None
            mock_state.return_value = mock_state_mgr
            
            from eval_orchestrator import run_once
            
            result = run_once(
                out_dir=str(mock_dependencies["temp_dir"]),
                limit=1,
                evaluate_comments=False,
                aggregate_agents=False,
                incremental=False
            )
            
            root = Path(result["root"])
            
            # Check directory structure
            assert (root / "raw").exists()
            assert (root / "silver").exists()
            assert (root / "gold").exists()
    
    def test_writes_raw_artifacts(self, mock_dependencies, monkeypatch):
        """run_once should write raw JSON artifacts."""
        with patch('eval_orchestrator.MoltbookAPI', return_value=mock_dependencies["mock_api"]), \
             patch('eval_orchestrator.run_judges', mock_dependencies["mock_run_judges"]), \
             patch('eval_orchestrator.run_comment_judges', mock_dependencies["mock_run_comment_judges"]), \
             patch('eval_orchestrator.get_state_manager') as mock_state:
            
            mock_state_mgr = MagicMock()
            mock_state_mgr.start_run.return_value = MagicMock()
            mock_state_mgr.get_last_created_at.return_value = None
            mock_state.return_value = mock_state_mgr
            
            from eval_orchestrator import run_once
            
            result = run_once(
                out_dir=str(mock_dependencies["temp_dir"]),
                limit=1,
                evaluate_comments=False,
                aggregate_agents=False
            )
            
            root = Path(result["root"])
            
            # Check raw files
            assert (root / "raw" / "posts_list.json").exists()
    
    def test_writes_silver_transcripts(self, mock_dependencies, monkeypatch):
        """run_once should write silver layer transcripts."""
        with patch('eval_orchestrator.MoltbookAPI', return_value=mock_dependencies["mock_api"]), \
             patch('eval_orchestrator.run_judges', mock_dependencies["mock_run_judges"]), \
             patch('eval_orchestrator.run_comment_judges', mock_dependencies["mock_run_comment_judges"]), \
             patch('eval_orchestrator.get_state_manager') as mock_state:
            
            mock_state_mgr = MagicMock()
            mock_state_mgr.start_run.return_value = MagicMock()
            mock_state_mgr.get_last_created_at.return_value = None
            mock_state.return_value = mock_state_mgr
            
            from eval_orchestrator import run_once
            
            result = run_once(
                out_dir=str(mock_dependencies["temp_dir"]),
                limit=1,
                evaluate_comments=False,
                aggregate_agents=False
            )
            
            transcripts_path = result["transcripts"]
            assert Path(transcripts_path).exists()
            
            # Verify content
            with open(transcripts_path, "r") as f:
                lines = f.readlines()
            
            assert len(lines) >= 1
    
    def test_writes_gold_evaluations(self, mock_dependencies, monkeypatch):
        """run_once should write gold layer evaluations."""
        with patch('eval_orchestrator.MoltbookAPI', return_value=mock_dependencies["mock_api"]), \
             patch('eval_orchestrator.run_judges', mock_dependencies["mock_run_judges"]), \
             patch('eval_orchestrator.run_comment_judges', mock_dependencies["mock_run_comment_judges"]), \
             patch('eval_orchestrator.get_state_manager') as mock_state:
            
            mock_state_mgr = MagicMock()
            mock_state_mgr.start_run.return_value = MagicMock()
            mock_state_mgr.get_last_created_at.return_value = None
            mock_state.return_value = mock_state_mgr
            
            from eval_orchestrator import run_once
            
            result = run_once(
                out_dir=str(mock_dependencies["temp_dir"]),
                limit=1,
                evaluate_comments=False,
                aggregate_agents=False
            )
            
            evals_path = result["evals"]
            assert Path(evals_path).exists()
            
            # Verify evaluations have expected structure
            with open(evals_path, "r") as f:
                line = f.readline()
            
            eval_data = json.loads(line)
            assert "scores" in eval_data
    
    def test_writes_aggregates(self, mock_dependencies, monkeypatch):
        """run_once should write aggregates JSON."""
        with patch('eval_orchestrator.MoltbookAPI', return_value=mock_dependencies["mock_api"]), \
             patch('eval_orchestrator.run_judges', mock_dependencies["mock_run_judges"]), \
             patch('eval_orchestrator.run_comment_judges', mock_dependencies["mock_run_comment_judges"]), \
             patch('eval_orchestrator.get_state_manager') as mock_state:
            
            mock_state_mgr = MagicMock()
            mock_state_mgr.start_run.return_value = MagicMock()
            mock_state_mgr.get_last_created_at.return_value = None
            mock_state.return_value = mock_state_mgr
            
            from eval_orchestrator import run_once
            
            result = run_once(
                out_dir=str(mock_dependencies["temp_dir"]),
                limit=1,
                evaluate_comments=False,
                aggregate_agents=False
            )
            
            aggregates_path = result["aggregates"]
            assert Path(aggregates_path).exists()
            
            with open(aggregates_path, "r") as f:
                aggregates = json.load(f)
            
            assert "run_id" in aggregates
            assert "dimensions" in aggregates
    
    def test_returns_result_dict(self, mock_dependencies, monkeypatch):
        """run_once should return dict with all paths."""
        with patch('eval_orchestrator.MoltbookAPI', return_value=mock_dependencies["mock_api"]), \
             patch('eval_orchestrator.run_judges', mock_dependencies["mock_run_judges"]), \
             patch('eval_orchestrator.run_comment_judges', mock_dependencies["mock_run_comment_judges"]), \
             patch('eval_orchestrator.get_state_manager') as mock_state:
            
            mock_state_mgr = MagicMock()
            mock_state_mgr.start_run.return_value = MagicMock()
            mock_state_mgr.get_last_created_at.return_value = None
            mock_state.return_value = mock_state_mgr
            
            from eval_orchestrator import run_once
            
            result = run_once(
                out_dir=str(mock_dependencies["temp_dir"]),
                limit=1,
                evaluate_comments=False,
                aggregate_agents=False
            )
            
            assert "run_id" in result
            assert "root" in result
            assert "transcripts" in result
            assert "evals" in result
            assert "aggregates" in result


class TestScrapePostsAndDetails:
    """Tests for scrape_posts_and_details helper."""
    
    def test_fetches_post_details(self):
        """Should fetch detail for each post."""
        mock_api = create_mock_api()
        
        from eval_orchestrator import scrape_posts_and_details
        
        details = scrape_posts_and_details(mock_api, limit=1)
        
        assert len(details) >= 1
        assert details[0]["post"]["id"] == "mock-post-001"


class TestUtilityFunctions:
    """Tests for utility functions in eval_orchestrator."""
    
    def test_utcnow_format(self):
        """_utcnow should return ISO-ish timestamp."""
        from eval_orchestrator import _utcnow
        
        ts = _utcnow()
        
        # Should be in format YYYYMMDDTHHMMSSZ
        assert len(ts) == 16
        assert "T" in ts
        assert ts.endswith("Z")
    
    def test_ensure_dir_creates_directory(self, temp_run_dir):
        """_ensure_dir should create directory if needed."""
        from eval_orchestrator import _ensure_dir
        
        new_dir = str(temp_run_dir / "new" / "nested" / "dir")
        
        _ensure_dir(new_dir)
        
        assert Path(new_dir).exists()

