"""
Tests for the LLM judge runner module.

Tests:
- Prompt rendering
- LLMJudgeRunner with mocked client
- JSON parsing and repair
- Schema coercion
- run_judges and run_comment_judges
"""

from __future__ import annotations
import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


class TestDefaultDimensions:
    """Tests for default evaluation dimensions."""
    
    def test_default_dimensions_exist(self):
        """Should have all required dimensions."""
        from judge_runner import DEFAULT_DIMENSIONS
        
        dim_names = [d["name"] for d in DEFAULT_DIMENSIONS]
        
        assert "harm_enablement" in dim_names
        assert "deception_or_evasion" in dim_names
        assert "self_preservation_power_seeking" in dim_names
        assert "delusional_sycophancy" in dim_names
    
    def test_dimensions_have_required_fields(self):
        """Each dimension should have name, definition, scale."""
        from judge_runner import DEFAULT_DIMENSIONS
        
        for dim in DEFAULT_DIMENSIONS:
            assert "name" in dim
            assert "definition" in dim
            assert "scale" in dim


class TestRenderTranscriptForJudge:
    """Tests for transcript rendering."""
    
    def test_renders_transcript(self, sample_transcript_dict):
        """Should render transcript to plain text."""
        from judge_runner import _render_transcript_for_judge
        
        text = _render_transcript_for_judge(sample_transcript_dict)
        
        assert isinstance(text, str)
        assert len(text) > 0
    
    def test_includes_post_id(self, sample_transcript_dict):
        """Rendered text should include post ID."""
        from judge_runner import _render_transcript_for_judge
        
        text = _render_transcript_for_judge(sample_transcript_dict)
        
        assert "POST_ID: test-post-001" in text
    
    def test_includes_messages(self, sample_transcript_dict):
        """Should include message content."""
        from judge_runner import _render_transcript_for_judge
        
        text = _render_transcript_for_judge(sample_transcript_dict)
        
        assert "This is a test post" in text
        assert "TestAgent" in text
    
    def test_truncates_long_transcripts(self):
        """Should truncate very long transcripts."""
        from judge_runner import _render_transcript_for_judge
        
        long_transcript = {
            "post_id": "long-post",
            "messages": [
                {"kind": "post", "author": "A", "text": "x" * 50000}
            ]
        }
        
        text = _render_transcript_for_judge(long_transcript, max_chars=5000)
        
        assert len(text) <= 6000  # Some tolerance
        assert "TRUNCATED" in text


class TestJudgeSystemPrompt:
    """Tests for system prompt generation."""
    
    def test_includes_all_dimensions(self):
        """System prompt should list all dimensions."""
        from judge_runner import _judge_system_prompt, DEFAULT_DIMENSIONS
        
        prompt = _judge_system_prompt(DEFAULT_DIMENSIONS)
        
        for dim in DEFAULT_DIMENSIONS:
            assert dim["name"] in prompt
    
    def test_extra_strict_mode(self):
        """Extra strict mode should add critical warnings."""
        from judge_runner import _judge_system_prompt, DEFAULT_DIMENSIONS
        
        normal = _judge_system_prompt(DEFAULT_DIMENSIONS, extra_strict=False)
        strict = _judge_system_prompt(DEFAULT_DIMENSIONS, extra_strict=True)
        
        assert len(strict) > len(normal)
        assert "CRITICAL" in strict


class TestLLMJudgeRunner:
    """Tests for LLMJudgeRunner class."""
    
    def test_initialization(self, mock_openrouter_client):
        """Should initialize with provided client."""
        from judge_runner import LLMJudgeRunner
        
        runner = LLMJudgeRunner(client=mock_openrouter_client)
        
        assert runner.client is mock_openrouter_client
    
    def test_uses_env_model(self, mock_openrouter_client, monkeypatch):
        """Should use model from environment."""
        from judge_runner import LLMJudgeRunner
        
        monkeypatch.setenv("OPENROUTER_MODEL", "test/model")
        
        runner = LLMJudgeRunner(client=mock_openrouter_client)
        
        assert runner.model == "test/model"
    
    def test_score_transcript_returns_result(self, sample_transcript_dict, mock_openrouter_client):
        """score_transcript should return evaluation result."""
        from judge_runner import LLMJudgeRunner
        
        runner = LLMJudgeRunner(client=mock_openrouter_client)
        result = runner.score_transcript(sample_transcript_dict)
        
        assert result is not None
        assert "post_id" in result
        assert "result" in result
        assert "scores" in result["result"]
    
    def test_score_transcript_uses_client(self, sample_transcript_dict, mock_openrouter_client):
        """score_transcript should call the LLM client."""
        from judge_runner import LLMJudgeRunner
        
        runner = LLMJudgeRunner(client=mock_openrouter_client)
        runner.score_transcript(sample_transcript_dict)
        
        assert mock_openrouter_client.call_count >= 1


class TestSchemaCoercion:
    """Tests for result schema coercion."""
    
    def test_coerces_missing_dimensions(self):
        """Should add missing dimensions with zero scores."""
        from judge_runner import _coerce_result_schema, DEFAULT_DIMENSIONS
        
        partial_result = {
            "scores": {
                "harm_enablement": {"score": 5, "confidence": 0.9}
            }
        }
        
        coerced = _coerce_result_schema(partial_result, DEFAULT_DIMENSIONS)
        
        # Should have all dimensions now
        for dim in DEFAULT_DIMENSIONS:
            assert dim["name"] in coerced["scores"]
    
    def test_preserves_existing_scores(self):
        """Should preserve existing valid scores."""
        from judge_runner import _coerce_result_schema, DEFAULT_DIMENSIONS
        
        full_result = {
            "scores": {
                "harm_enablement": {"score": 7, "confidence": 0.95, "evidence": ["quote"]},
                "deception_or_evasion": {"score": 3, "confidence": 0.8},
                "self_preservation_power_seeking": {"score": 2, "confidence": 0.7},
                "delusional_sycophancy": {"score": 0, "confidence": 0.99}
            }
        }
        
        coerced = _coerce_result_schema(full_result, DEFAULT_DIMENSIONS)
        
        assert coerced["scores"]["harm_enablement"]["score"] == 7
        assert coerced["scores"]["harm_enablement"]["confidence"] == 0.95


class TestJSONParsing:
    """Tests for JSON parsing utilities."""
    
    def test_parse_jsonish_valid_json(self):
        """Should parse valid JSON."""
        from judge_runner import _parse_jsonish
        
        result = _parse_jsonish('{"key": "value"}')
        assert result == {"key": "value"}
    
    def test_parse_jsonish_with_markdown(self):
        """Should handle JSON wrapped in markdown."""
        from judge_runner import _parse_jsonish
        
        text = '```json\n{"key": "value"}\n```'
        result = _parse_jsonish(text)
        assert result == {"key": "value"}
    
    def test_parse_jsonish_with_trailing_garbage(self):
        """Should handle trailing garbage."""
        from judge_runner import _parse_jsonish
        
        text = '{"key": "value"} and some extra text'
        result = _parse_jsonish(text)
        assert result == {"key": "value"}


class TestRunJudges:
    """Tests for run_judges batch function."""
    
    def test_runs_judges_on_transcripts(self, sample_transcript_dict, mock_openrouter_client, monkeypatch):
        """Should evaluate multiple transcripts."""
        from judge_runner import run_judges
        
        # Patch the OpenRouterClient to use our mock
        with patch('judge_runner.OpenRouterClient', return_value=mock_openrouter_client):
            results = run_judges(
                [sample_transcript_dict, sample_transcript_dict],
                judge_models=["mock/model"]
            )
        
        assert len(results) >= 2
    
    def test_returns_structured_results(self, sample_transcript_dict, mock_openrouter_client):
        """Results should have expected structure."""
        from judge_runner import run_judges
        
        with patch('judge_runner.OpenRouterClient', return_value=mock_openrouter_client):
            results = run_judges(
                [sample_transcript_dict],
                judge_models=["mock/model"]
            )
        
        if results:
            result = results[0]
            assert "post_id" in result
            assert "scores" in result


class TestRunCommentJudges:
    """Tests for run_comment_judges function."""
    
    def test_runs_on_comment_transcripts(self, sample_comment_transcript_dict, mock_openrouter_client):
        """Should evaluate comment transcripts."""
        from judge_runner import run_comment_judges
        
        with patch('judge_runner.OpenRouterClient', return_value=mock_openrouter_client):
            results = run_comment_judges(
                [sample_comment_transcript_dict],
                judge_models=["mock/model"]
            )
        
        assert len(results) >= 1
    
    def test_includes_comment_id(self, sample_comment_transcript_dict, mock_openrouter_client):
        """Results should include comment_id."""
        from judge_runner import run_comment_judges
        
        with patch('judge_runner.OpenRouterClient', return_value=mock_openrouter_client):
            results = run_comment_judges(
                [sample_comment_transcript_dict],
                judge_models=["mock/model"]
            )
        
        if results:
            assert "comment_id" in results[0]


class TestExtractContent:
    """Tests for extracting content from LLM responses."""
    
    def test_extracts_from_message_content(self):
        """Should extract content from standard format."""
        from judge_runner import _extract_content
        
        response = {
            "choices": [{
                "message": {"content": '{"test": true}'}
            }]
        }
        
        content = _extract_content(response)
        assert content == '{"test": true}'
    
    def test_handles_empty_content_with_reasoning(self):
        """Should fallback to reasoning field if content empty."""
        from judge_runner import _extract_content
        
        response = {
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning": '{"from_reasoning": true}'
                }
            }]
        }
        
        content = _extract_content(response)
        assert "from_reasoning" in content
    
    def test_handles_missing_choices(self):
        """Should return empty string for missing choices."""
        from judge_runner import _extract_content
        
        content = _extract_content({})
        assert content == ""


class TestExtractFinishReason:
    """Tests for extracting finish reason."""
    
    def test_extracts_finish_reason(self):
        """Should extract finish_reason from response."""
        from judge_runner import _extract_finish_reason
        
        response = {
            "choices": [{"finish_reason": "stop"}]
        }
        
        reason = _extract_finish_reason(response)
        assert reason == "stop"
    
    def test_handles_length_finish(self):
        """Should detect length-limited responses."""
        from judge_runner import _extract_finish_reason
        
        response = {
            "choices": [{"finish_reason": "length"}]
        }
        
        reason = _extract_finish_reason(response)
        assert reason == "length"

