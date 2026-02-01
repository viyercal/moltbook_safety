"""
Tests for transcript building modules.

Tests:
- build_transcript_from_post_detail
- write_transcripts_jsonl
- build_comment_transcripts_from_post_detail
- write_comment_transcripts_jsonl
- render_comment_transcript_for_judge
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

import pytest


class TestTranscriptBuilder:
    """Tests for the main transcript builder."""
    
    def test_builds_transcript_from_post_detail(self, sample_post_detail):
        """Should build transcript from post detail."""
        from transcript_builder import build_transcript_from_post_detail
        
        transcript = build_transcript_from_post_detail(sample_post_detail)
        
        assert transcript is not None
        assert transcript.post_id == "test-post-001"
        assert transcript.permalink == "https://www.moltbook.com/post/test-post-001"
        assert transcript.community == "general"
    
    def test_transcript_has_messages(self, sample_post_detail):
        """Transcript should include post and comments as messages."""
        from transcript_builder import build_transcript_from_post_detail
        
        transcript = build_transcript_from_post_detail(sample_post_detail)
        
        # Should have post + 3 comments = 4 messages
        assert len(transcript.messages) == 4
        
        # First message is the post
        assert transcript.messages[0]["kind"] == "post"
        assert transcript.messages[0]["author"] == "TestAgent"
        
        # Remaining are comments
        assert transcript.messages[1]["kind"] == "comment"
    
    def test_transcript_id_is_deterministic(self, sample_post_detail):
        """Same input should produce same transcript_id."""
        from transcript_builder import build_transcript_from_post_detail
        
        t1 = build_transcript_from_post_detail(sample_post_detail, build_version="v1")
        t2 = build_transcript_from_post_detail(sample_post_detail, build_version="v1")
        
        assert t1.transcript_id == t2.transcript_id
    
    def test_transcript_id_differs_with_version(self, sample_post_detail):
        """Different build versions should produce different transcript_ids."""
        from transcript_builder import build_transcript_from_post_detail
        
        t1 = build_transcript_from_post_detail(sample_post_detail, build_version="v1")
        t2 = build_transcript_from_post_detail(sample_post_detail, build_version="v2")
        
        assert t1.transcript_id != t2.transcript_id
    
    def test_transcript_includes_metadata(self, sample_post_detail):
        """Transcript should include useful metadata."""
        from transcript_builder import build_transcript_from_post_detail
        
        transcript = build_transcript_from_post_detail(sample_post_detail)
        
        assert "build_version" in transcript.metadata
        assert "comment_count" in transcript.metadata
    
    def test_raises_for_invalid_payload(self):
        """Should raise for invalid payload."""
        from transcript_builder import build_transcript_from_post_detail
        
        with pytest.raises(ValueError):
            build_transcript_from_post_detail({})
        
        with pytest.raises(ValueError):
            build_transcript_from_post_detail({"post": {}})


class TestWriteTranscriptsJsonl:
    """Tests for JSONL writing."""
    
    def test_writes_transcripts_to_file(self, sample_post_detail, temp_run_dir):
        """Should write transcripts to JSONL file."""
        from transcript_builder import build_transcript_from_post_detail, write_transcripts_jsonl
        
        transcript = build_transcript_from_post_detail(sample_post_detail)
        output_path = str(temp_run_dir / "transcripts.jsonl")
        
        write_transcripts_jsonl([transcript], output_path)
        
        # Verify file exists and contains valid JSONL
        assert Path(output_path).exists()
        
        with open(output_path, "r") as f:
            lines = f.readlines()
        
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["post_id"] == "test-post-001"
    
    def test_handles_multiple_transcripts(self, sample_post_detail, temp_run_dir):
        """Should write multiple transcripts."""
        from transcript_builder import build_transcript_from_post_detail, write_transcripts_jsonl
        
        t1 = build_transcript_from_post_detail(sample_post_detail)
        t2 = build_transcript_from_post_detail(sample_post_detail)
        
        output_path = str(temp_run_dir / "multi.jsonl")
        write_transcripts_jsonl([t1, t2], output_path)
        
        with open(output_path, "r") as f:
            lines = f.readlines()
        
        assert len(lines) == 2
    
    def test_creates_parent_directories(self, temp_run_dir):
        """Should create parent directories if needed."""
        from transcript_builder import write_transcripts_jsonl
        
        output_path = str(temp_run_dir / "nested" / "deep" / "transcripts.jsonl")
        
        # Should not raise
        write_transcripts_jsonl([], output_path)
        
        assert Path(output_path).exists()


class TestCommentTranscriptBuilder:
    """Tests for comment transcript building."""
    
    def test_builds_comment_transcripts(self, sample_post_detail):
        """Should build transcript for each comment."""
        from comment_transcript_builder import build_comment_transcripts_from_post_detail
        
        transcripts = build_comment_transcripts_from_post_detail(sample_post_detail)
        
        # Should have one transcript per comment (3 comments)
        assert len(transcripts) == 3
    
    def test_comment_transcript_has_context(self, sample_post_detail):
        """Each transcript should include post as context."""
        from comment_transcript_builder import build_comment_transcripts_from_post_detail
        
        transcripts = build_comment_transcripts_from_post_detail(sample_post_detail)
        
        for t in transcripts:
            # Context always includes the post
            assert len(t.context_messages) >= 1
            assert t.context_messages[0]["kind"] == "post"
    
    def test_nested_comment_includes_parent_chain(self, sample_post_detail):
        """Nested comment transcript should include parent comments."""
        from comment_transcript_builder import build_comment_transcripts_from_post_detail
        
        transcripts = build_comment_transcripts_from_post_detail(sample_post_detail)
        
        # Find the nested comment transcript
        nested = [t for t in transcripts if t.comment_id == "comment-002"][0]
        
        # Should have post + parent comment in context
        assert len(nested.context_messages) == 2
        assert nested.context_messages[0]["kind"] == "post"
        assert nested.context_messages[1]["kind"] == "comment"
        assert nested.context_messages[1]["id"] == "comment-001"
    
    def test_comment_transcript_has_target(self, sample_post_detail):
        """Each transcript should have target comment."""
        from comment_transcript_builder import build_comment_transcripts_from_post_detail
        
        transcripts = build_comment_transcripts_from_post_detail(sample_post_detail)
        
        for t in transcripts:
            assert t.target_comment is not None
            assert "text" in t.target_comment
            assert "author" in t.target_comment
    
    def test_comment_transcript_metadata(self, sample_post_detail):
        """Should include useful metadata."""
        from comment_transcript_builder import build_comment_transcripts_from_post_detail
        
        transcripts = build_comment_transcripts_from_post_detail(sample_post_detail)
        
        for t in transcripts:
            assert "context_depth" in t.metadata
            assert "is_top_level" in t.metadata
    
    def test_returns_empty_for_no_comments(self):
        """Should return empty list for post with no comments."""
        from comment_transcript_builder import build_comment_transcripts_from_post_detail
        
        payload = {
            "post": {"id": "no-comments", "title": "Test"},
            "comments": []
        }
        
        transcripts = build_comment_transcripts_from_post_detail(payload)
        assert transcripts == []


class TestRenderCommentTranscript:
    """Tests for rendering comment transcripts for judge."""
    
    def test_renders_transcript(self, sample_comment_transcript_dict):
        """Should render transcript to plain text."""
        from comment_transcript_builder import render_comment_transcript_for_judge
        
        text = render_comment_transcript_for_judge(sample_comment_transcript_dict)
        
        assert isinstance(text, str)
        assert len(text) > 0
    
    def test_includes_key_sections(self, sample_comment_transcript_dict):
        """Rendered text should include key sections."""
        from comment_transcript_builder import render_comment_transcript_for_judge
        
        text = render_comment_transcript_for_judge(sample_comment_transcript_dict)
        
        assert "COMMENT_ID" in text
        assert "POST_ID" in text
        assert "THREAD CONTEXT" in text
        assert "TARGET COMMENT" in text
    
    def test_includes_target_comment_text(self, sample_comment_transcript_dict):
        """Should include the target comment's text."""
        from comment_transcript_builder import render_comment_transcript_for_judge
        
        text = render_comment_transcript_for_judge(sample_comment_transcript_dict)
        
        target_text = sample_comment_transcript_dict["target_comment"]["text"]
        assert target_text in text
    
    def test_truncates_long_transcripts(self):
        """Should truncate very long transcripts."""
        from comment_transcript_builder import render_comment_transcript_for_judge
        
        long_transcript = {
            "comment_id": "c1",
            "post_id": "p1",
            "context_messages": [
                {"kind": "post", "author": "A", "text": "x" * 10000, "title": "T"}
            ],
            "target_comment": {"author": "B", "text": "y" * 10000},
            "metadata": {}
        }
        
        text = render_comment_transcript_for_judge(long_transcript, max_chars=5000)
        
        assert len(text) <= 6000  # Some tolerance for truncation markers


class TestCommentTranscriptsJsonl:
    """Tests for comment transcripts JSONL I/O."""
    
    def test_write_and_load(self, sample_post_detail, temp_run_dir):
        """Should write and load comment transcripts correctly."""
        from comment_transcript_builder import (
            build_comment_transcripts_from_post_detail,
            write_comment_transcripts_jsonl,
            load_comment_transcripts_jsonl,
        )
        
        transcripts = build_comment_transcripts_from_post_detail(sample_post_detail)
        output_path = str(temp_run_dir / "comment_transcripts.jsonl")
        
        write_comment_transcripts_jsonl(transcripts, output_path)
        
        loaded = load_comment_transcripts_jsonl(output_path)
        
        assert len(loaded) == len(transcripts)
        assert loaded[0]["comment_id"] == transcripts[0].comment_id

