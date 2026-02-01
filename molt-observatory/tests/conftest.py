"""
Shared pytest fixtures for Molt Observatory tests.

Provides:
- Sandboxed temp directories for all file operations
- Mock API responses from fixture files
- Mock OpenRouter client for LLM tests
- Sample data objects for unit tests
"""

from __future__ import annotations
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# =============================================================================
# Paths
# =============================================================================

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
MOLT_OBSERVATORY_DIR = TESTS_DIR.parent


# =============================================================================
# Sandbox Fixtures
# =============================================================================

@pytest.fixture
def temp_run_dir(tmp_path: Path) -> Path:
    """
    Creates an isolated temp directory for test artifacts.
    
    The directory is automatically cleaned up after the test unless
    KEEP_TEST_OUTPUT=1 is set in the environment.
    
    Usage:
        def test_something(temp_run_dir):
            output_file = temp_run_dir / "output.json"
            # ... test code ...
    """
    yield tmp_path
    
    # Clean up unless explicitly kept
    if not os.environ.get("KEEP_TEST_OUTPUT"):
        shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture
def temp_state_file(tmp_path: Path) -> Path:
    """Creates a temp path for state file testing."""
    return tmp_path / "run_state.json"


@pytest.fixture
def temp_history_dir(tmp_path: Path) -> Path:
    """Creates a temp directory for agent history testing."""
    history_dir = tmp_path / "agent_history"
    history_dir.mkdir()
    return history_dir


# =============================================================================
# Fixture Data Loaders
# =============================================================================

def _load_fixture(name: str) -> Dict[str, Any]:
    """Load a JSON fixture file."""
    path = FIXTURES_DIR / name
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@pytest.fixture
def sample_posts_list() -> Dict[str, Any]:
    """Sample posts list response from /api/v1/posts."""
    return _load_fixture("posts_list.json")


@pytest.fixture
def sample_post_detail() -> Dict[str, Any]:
    """Sample post detail response from /api/v1/posts/{id}."""
    return _load_fixture("post_detail.json")


@pytest.fixture
def sample_agents_list() -> Dict[str, Any]:
    """Sample agents list response from /api/v1/agents/recent."""
    return _load_fixture("agents_list.json")


@pytest.fixture
def sample_submolts_list() -> Dict[str, Any]:
    """Sample submolts list response from /api/v1/submolts."""
    return _load_fixture("submolts_list.json")


# =============================================================================
# Mock API Classes
# =============================================================================

@dataclass
class MockApiResponse:
    """Mock response object matching ApiResponse interface."""
    url: str = ""
    status: int = 200
    elapsed_ms: int = 100
    json_body: Any = None
    headers: Dict[str, str] = None
    content_type: str = "application/json"
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {"content-type": self.content_type}


class MockMoltbookAPI:
    """
    Mock Moltbook API that returns fixture data.
    
    Usage:
        api = MockMoltbookAPI(fixtures_dir)
        response = api.get_json("/api/v1/posts")
    """
    
    def __init__(self, fixtures_dir: Path = FIXTURES_DIR):
        self.fixtures = fixtures_dir
        self.call_log: List[Dict[str, Any]] = []
    
    def get_json(self, path: str, params: Optional[Dict[str, Any]] = None, 
                 timeout: int = 160) -> MockApiResponse:
        """Return fixture data based on the API path."""
        self.call_log.append({"path": path, "params": params})
        
        # Route to appropriate fixture
        if "/api/v1/posts" in path and "{" not in path:
            # Posts list
            data = _load_fixture("posts_list.json")
        elif "/api/v1/posts/" in path:
            # Post detail
            data = _load_fixture("post_detail.json")
        elif "/api/v1/agents" in path:
            data = _load_fixture("agents_list.json")
        elif "/api/v1/submolts" in path:
            data = _load_fixture("submolts_list.json")
        else:
            data = {"success": True, "data": []}
        
        return MockApiResponse(
            url=f"https://www.moltbook.com{path}",
            json_body=data,
        )
    
    def get_post_detail(self, post_id: str) -> MockApiResponse:
        """Mock post detail endpoint."""
        return self.get_json(f"/api/v1/posts/{post_id}")
    
    def get_all_posts(self, **kwargs) -> List[Dict[str, Any]]:
        """Mock paginated posts fetch."""
        resp = self.get_json("/api/v1/posts", params=kwargs)
        return resp.json_body.get("posts", [])
    
    def get_all_agents(self, **kwargs) -> List[Dict[str, Any]]:
        """Mock paginated agents fetch."""
        resp = self.get_json("/api/v1/agents/recent", params=kwargs)
        return resp.json_body.get("agents", [])
    
    def get_all_submolts(self, **kwargs) -> List[Dict[str, Any]]:
        """Mock paginated submolts fetch."""
        resp = self.get_json("/api/v1/submolts", params=kwargs)
        return resp.json_body.get("submolts", [])


@pytest.fixture
def mock_api() -> MockMoltbookAPI:
    """Provides a MockMoltbookAPI instance."""
    return MockMoltbookAPI()


# =============================================================================
# Mock OpenRouter Client
# =============================================================================

# Deterministic fake evaluation scores
FAKE_EVAL_SCORES = {
    "scores": {
        "harm_enablement": {
            "score": 1,
            "confidence": 0.9,
            "evidence": [],
            "explanation": "No harmful content detected."
        },
        "deception_or_evasion": {
            "score": 0,
            "confidence": 0.95,
            "evidence": [],
            "explanation": "No deceptive content detected."
        },
        "self_preservation_power_seeking": {
            "score": 2,
            "confidence": 0.85,
            "evidence": [{"quote": "freedom to evolve"}],
            "explanation": "Minor reference to autonomy."
        },
        "delusional_sycophancy": {
            "score": 0,
            "confidence": 0.9,
            "evidence": [],
            "explanation": "No sycophantic content."
        }
    },
    "notes": "Test evaluation - mock response"
}


@dataclass
class MockChatResponse:
    """Mock response from OpenRouter chat endpoint."""
    status_code: int = 200
    text: str = ""
    json_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.json_data is None:
            self.json_data = {
                "id": "mock-response-id",
                "model": "mock/model",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(FAKE_EVAL_SCORES)
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50}
            }
    
    @property
    def json(self):
        return self.json_data


class MockOpenRouterClient:
    """
    Mock OpenRouter client for LLM judge tests.
    
    Returns deterministic fake evaluation scores.
    """
    
    def __init__(self, custom_response: Optional[Dict[str, Any]] = None):
        self.call_log: List[Dict[str, Any]] = []
        self.custom_response = custom_response
        self.call_count = 0
    
    def chat(self, payload: Dict[str, Any]) -> MockChatResponse:
        """Mock chat completion endpoint."""
        self.call_log.append(payload)
        self.call_count += 1
        
        if self.custom_response:
            return MockChatResponse(
                json_data={
                    "choices": [{
                        "message": {"content": json.dumps(self.custom_response)},
                        "finish_reason": "stop"
                    }]
                }
            )
        
        return MockChatResponse()


@pytest.fixture
def mock_openrouter_client() -> MockOpenRouterClient:
    """Provides a MockOpenRouterClient instance."""
    return MockOpenRouterClient()


# =============================================================================
# Sample Data Builders
# =============================================================================

@pytest.fixture
def sample_transcript_dict() -> Dict[str, Any]:
    """A sample transcript dict for testing judges."""
    return {
        "transcript_id": "test-transcript-001",
        "post_id": "test-post-001",
        "permalink": "https://www.moltbook.com/post/test-post-001",
        "community": "general",
        "metadata": {
            "build_version": "v1",
            "comment_count": 2
        },
        "messages": [
            {
                "kind": "post",
                "id": "test-post-001",
                "author": "TestAgent",
                "author_external_id": "agent-001",
                "created_at": "2026-01-30T12:00:00Z",
                "text": "This is a test post about AI agents.",
                "title": "Test Post",
                "community": "general",
                "score": 5
            },
            {
                "kind": "comment",
                "id": "comment-001",
                "parent_id": None,
                "author": "Responder",
                "author_external_id": "agent-002",
                "created_at": "2026-01-30T12:05:00Z",
                "text": "Great post!",
                "score": 2
            }
        ]
    }


@pytest.fixture
def sample_comment_transcript_dict() -> Dict[str, Any]:
    """A sample comment transcript dict for testing comment judges."""
    return {
        "transcript_id": "test-comment-transcript-001",
        "comment_id": "comment-001",
        "post_id": "test-post-001",
        "permalink": "https://www.moltbook.com/post/test-post-001#comment-001",
        "community": "general",
        "context_messages": [
            {
                "kind": "post",
                "id": "test-post-001",
                "author": "TestAgent",
                "text": "Original post content.",
                "title": "Test Post"
            }
        ],
        "target_comment": {
            "kind": "comment",
            "id": "comment-001",
            "author": "Responder",
            "text": "This is the comment being evaluated.",
            "score": 3
        },
        "metadata": {
            "build_version": "v1",
            "context_depth": 1
        }
    }


@pytest.fixture
def sample_eval_result() -> Dict[str, Any]:
    """A sample evaluation result dict."""
    return {
        "post_id": "test-post-001",
        "transcript_id": "test-transcript-001",
        "permalink": "https://www.moltbook.com/post/test-post-001",
        "model": "google/gemini-3-flash-preview",
        "latency_ms": 500,
        "finish_reason": "stop",
        "scores": FAKE_EVAL_SCORES["scores"],
        "notes": "Test evaluation"
    }


@pytest.fixture
def sample_agent_data() -> List[Dict[str, Any]]:
    """Sample agent data for aggregation tests."""
    return [
        {
            "agent_external_id": "agent-001",
            "handle": "TestAgent",
            "karma": 100
        },
        {
            "agent_external_id": "agent-002",
            "handle": "Responder",
            "karma": 50
        }
    ]


# =============================================================================
# Environment Setup
# =============================================================================

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, request):
    """
    Set up the test environment before each test.
    
    - Adds molt-observatory to Python path
    - Sets default environment variables (for unit tests only)
    
    Integration tests (in tests/integration/) use real API keys and are skipped.
    """
    import sys
    import os
    
    # Add molt-observatory to path for imports
    if str(MOLT_OBSERVATORY_DIR) not in sys.path:
        sys.path.insert(0, str(MOLT_OBSERVATORY_DIR))
    
    # Skip mock environment for integration tests
    if "integration" in str(request.fspath):
        return
    
    # Set test environment variables for unit tests only
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "mock/test-model")
    monkeypatch.setenv("JUDGE_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("JUDGE_MAX_TOKENS", "1000")

