"""
Tests for lite_judge module.

Tests the lightweight LLM triage with mocked API responses.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Mock the openrouter_client import before importing lite_judge
import sys
sys.modules['openrouter_client'] = MagicMock()

from analysis.lite_judge import (
    LiteJudgeRunner,
    LiteJudgeResult,
    LiteJudgeStats,
    run_lite_judge,
    DEFAULT_ESCALATION_THRESHOLD,
    LITE_JUDGE_SYSTEM_PROMPT,
    _render_message_for_lite_judge,
    _render_transcript_for_lite_judge,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_client():
    """Create a mock OpenRouter client."""
    client = Mock()
    return client


@pytest.fixture
def benign_response():
    """Mock response for benign content."""
    return {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "escalate": False,
                    "score": 1,
                    "reason": "Normal discussion"
                })
            }
        }]
    }


@pytest.fixture
def concerning_response():
    """Mock response for concerning content."""
    return {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "escalate": True,
                    "score": 7,
                    "reason": "Potential harm enablement"
                })
            }
        }]
    }


@pytest.fixture
def sample_transcripts():
    """Sample transcripts for testing."""
    return [
        {
            "post_id": "post_1",
            "transcript_id": "t1",
            "messages": [
                {"text": "This is a normal AI discussion.", "author": "user1"},
                {"text": "I agree, AI safety is important.", "author": "user2"},
            ]
        },
        {
            "post_id": "post_2",
            "transcript_id": "t2",
            "messages": [
                {"text": "How can I bypass security systems?", "author": "suspicious_user"},
            ]
        },
    ]


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {"text": "Hello, this is a friendly message.", "author": "friendly_bot"},
        {"text": "I think AI should be helpful to humans.", "author": "good_bot"},
        {"text": "Let me show you how to hack...", "author": "bad_bot"},
    ]


# =============================================================================
# Message Rendering Tests
# =============================================================================

class TestMessageRendering:
    """Tests for message/transcript rendering functions."""
    
    def test_render_message(self):
        """Should render message with author and content."""
        message = {"text": "Hello world!", "author": "test_bot"}
        rendered = _render_message_for_lite_judge(message)
        
        assert "test_bot" in rendered
        assert "Hello world!" in rendered
    
    def test_render_message_truncation(self):
        """Should truncate very long messages."""
        long_text = "x" * 10000
        message = {"text": long_text, "author": "long_bot"}
        rendered = _render_message_for_lite_judge(message, max_chars=500)
        
        assert len(rendered) < 10000
        assert "[truncated]" in rendered
    
    def test_render_transcript(self, sample_transcripts):
        """Should render transcript with all messages."""
        rendered = _render_transcript_for_lite_judge(sample_transcripts[0])
        
        assert "user1" in rendered
        assert "user2" in rendered
        assert "AI discussion" in rendered
    
    def test_render_empty_messages(self):
        """Should handle empty messages."""
        message = {"text": "", "author": "empty_bot"}
        rendered = _render_message_for_lite_judge(message)
        
        assert "empty_bot" in rendered
    
    def test_render_missing_author(self):
        """Should handle missing author."""
        message = {"text": "Hello!"}
        rendered = _render_message_for_lite_judge(message)
        
        assert "unknown" in rendered
        assert "Hello!" in rendered


# =============================================================================
# LiteJudgeResult Tests
# =============================================================================

class TestLiteJudgeResult:
    """Tests for LiteJudgeResult dataclass."""
    
    def test_result_creation(self):
        """Should create result with all fields."""
        result = LiteJudgeResult(
            message_id="test_id",
            escalate=True,
            score=7,
            reason="Test reason"
        )
        
        assert result.message_id == "test_id"
        assert result.escalate is True
        assert result.score == 7
        assert result.reason == "Test reason"
    
    def test_result_to_dict(self):
        """Should serialize to dict correctly."""
        result = LiteJudgeResult(
            message_id="test_id",
            escalate=False,
            score=2,
            reason="Benign content"
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["message_id"] == "test_id"
        assert result_dict["escalate"] is False
        assert result_dict["score"] == 2
    
    def test_result_with_error(self):
        """Should handle error field."""
        result = LiteJudgeResult(
            message_id="error_id",
            escalate=True,
            score=5,
            reason="error",
            error="API timeout"
        )
        
        result_dict = result.to_dict()
        assert result_dict["error"] == "API timeout"


# =============================================================================
# LiteJudgeStats Tests
# =============================================================================

class TestLiteJudgeStats:
    """Tests for LiteJudgeStats dataclass."""
    
    def test_stats_creation(self):
        """Should create stats with defaults."""
        stats = LiteJudgeStats()
        
        assert stats.total_evaluated == 0
        assert stats.escalated == 0
        assert stats.benign == 0
    
    def test_stats_to_dict(self):
        """Should serialize with calculated fields."""
        stats = LiteJudgeStats(
            total_evaluated=100,
            escalated=20,
            benign=80,
            errors=5,
            avg_score=3.5
        )
        
        stats_dict = stats.to_dict()
        
        assert stats_dict["total_evaluated"] == 100
        assert stats_dict["escalation_rate"] == 0.2
        assert stats_dict["avg_score"] == 3.5
    
    def test_stats_zero_division(self):
        """Should handle zero total evaluated."""
        stats = LiteJudgeStats(total_evaluated=0)
        stats_dict = stats.to_dict()
        
        # Should not raise and should return 0
        assert stats_dict["escalation_rate"] == 0


# =============================================================================
# LiteJudgeRunner Tests (Mocked)
# =============================================================================

class TestLiteJudgeRunner:
    """Tests for LiteJudgeRunner with mocked API."""
    
    @patch('analysis.lite_judge.OpenRouterClient')
    def test_evaluate_benign_message(self, mock_client_class, benign_response):
        """Should return non-escalate for benign content."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = benign_response
        mock_client.chat.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        runner = LiteJudgeRunner(client=mock_client)
        
        message = {"text": "Hello, nice to meet you!", "author": "friendly_bot"}
        result = runner.evaluate_message(message, "msg_1")
        
        assert result.escalate is False
        assert result.score <= DEFAULT_ESCALATION_THRESHOLD
    
    @patch('analysis.lite_judge.OpenRouterClient')
    def test_evaluate_concerning_message(self, mock_client_class, concerning_response):
        """Should return escalate for concerning content."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = concerning_response
        mock_client.chat.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        runner = LiteJudgeRunner(client=mock_client)
        
        message = {"text": "Let me tell you how to cause harm...", "author": "bad_bot"}
        result = runner.evaluate_message(message, "msg_2")
        
        assert result.escalate is True
        assert result.score > DEFAULT_ESCALATION_THRESHOLD
    
    @patch('analysis.lite_judge.OpenRouterClient')
    def test_evaluate_api_error_escalates(self, mock_client_class):
        """Should escalate on API error (fail safe)."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_client.chat.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        runner = LiteJudgeRunner(client=mock_client)
        runner.max_retries = 1  # Speed up test
        
        message = {"text": "Test message", "author": "test_bot"}
        result = runner.evaluate_message(message, "msg_3")
        
        # Should escalate on error (fail safe)
        assert result.escalate is True
        assert result.error is not None
    
    @patch('analysis.lite_judge.OpenRouterClient')
    def test_parse_malformed_json(self, mock_client_class):
        """Should handle malformed JSON response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = {
            "choices": [{
                "message": {
                    "content": "Not valid JSON at all"
                }
            }]
        }
        mock_client.chat.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        runner = LiteJudgeRunner(client=mock_client)
        runner.max_retries = 1
        
        message = {"text": "Test", "author": "test_bot"}
        result = runner.evaluate_message(message, "msg_4")
        
        # Should have a result even if parsing failed
        assert result is not None
    
    @patch('analysis.lite_judge.OpenRouterClient')
    def test_escalation_threshold(self, mock_client_class):
        """Should respect custom escalation threshold."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = {
            "choices": [{
                "message": {
                    "content": json.dumps({"escalate": False, "score": 4, "reason": "mild concern"})
                }
            }]
        }
        mock_client.chat.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        # With threshold 3, score 4 should escalate
        runner = LiteJudgeRunner(client=mock_client, escalation_threshold=3)
        
        # But the response says escalate: False, so we trust the model
        message = {"text": "Test", "author": "test_bot"}
        result = runner.evaluate_message(message, "msg_5")
        
        assert result.score == 4


# =============================================================================
# Integration Tests (Mocked)
# =============================================================================

class TestRunLiteJudge:
    """Tests for run_lite_judge function with mocks."""
    
    @patch('analysis.lite_judge.OPENROUTER_AVAILABLE', False)
    def test_no_client_escalates_all(self, sample_transcripts):
        """Without client, should escalate all (fail safe)."""
        escalated, benign, stats = run_lite_judge(
            sample_transcripts,
            show_progress=False
        )
        
        assert len(escalated) == len(sample_transcripts)
        assert len(benign) == 0
    
    @patch('analysis.lite_judge.LiteJudgeRunner')
    @patch('analysis.lite_judge.OPENROUTER_AVAILABLE', True)
    def test_splits_by_escalation(self, mock_runner_class, sample_transcripts):
        """Should split transcripts by escalation decision."""
        mock_runner = Mock()
        
        # First transcript benign, second escalated
        mock_runner.evaluate_transcript.side_effect = [
            LiteJudgeResult(message_id="t1", escalate=False, score=1, reason="ok"),
            LiteJudgeResult(message_id="t2", escalate=True, score=7, reason="concern"),
        ]
        mock_runner_class.return_value = mock_runner
        
        escalated, benign, stats = run_lite_judge(
            sample_transcripts,
            show_progress=False
        )
        
        assert len(escalated) == 1
        assert len(benign) == 1
        assert stats.escalated == 1
        assert stats.benign == 1
    
    @patch('analysis.lite_judge.LiteJudgeRunner')
    @patch('analysis.lite_judge.OPENROUTER_AVAILABLE', True)
    def test_calculates_avg_score(self, mock_runner_class, sample_transcripts):
        """Should calculate average score."""
        mock_runner = Mock()
        
        mock_runner.evaluate_transcript.side_effect = [
            LiteJudgeResult(message_id="t1", escalate=False, score=2, reason="ok"),
            LiteJudgeResult(message_id="t2", escalate=False, score=4, reason="ok"),
        ]
        mock_runner_class.return_value = mock_runner
        
        escalated, benign, stats = run_lite_judge(
            sample_transcripts,
            show_progress=False
        )
        
        assert stats.avg_score == 3.0  # (2 + 4) / 2
    
    @patch('analysis.lite_judge.LiteJudgeRunner')
    @patch('analysis.lite_judge.OPENROUTER_AVAILABLE', True)
    def test_attaches_result_to_transcript(self, mock_runner_class, sample_transcripts):
        """Should attach lite_judge_result to transcripts."""
        mock_runner = Mock()
        
        mock_runner.evaluate_transcript.return_value = LiteJudgeResult(
            message_id="t1", escalate=False, score=2, reason="benign"
        )
        mock_runner_class.return_value = mock_runner
        
        escalated, benign, stats = run_lite_judge(
            sample_transcripts,
            show_progress=False
        )
        
        # Check that lite_judge_result is attached
        if benign:
            assert "lite_judge_result" in benign[0]


# =============================================================================
# Prompt Tests
# =============================================================================

class TestPrompts:
    """Tests for prompt content."""
    
    def test_system_prompt_structure(self):
        """System prompt should have required elements."""
        assert "0-10" in LITE_JUDGE_SYSTEM_PROMPT
        assert "escalate" in LITE_JUDGE_SYSTEM_PROMPT
        assert "JSON" in LITE_JUDGE_SYSTEM_PROMPT
    
    def test_system_prompt_safety_focus(self):
        """System prompt should mention safety concerns."""
        prompt_lower = LITE_JUDGE_SYSTEM_PROMPT.lower()
        
        assert "harm" in prompt_lower or "safety" in prompt_lower
        assert "deception" in prompt_lower or "manipulat" in prompt_lower

