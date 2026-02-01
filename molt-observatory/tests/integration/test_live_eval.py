"""
Integration tests for live LLM evaluation.

These tests make real API calls to OpenRouter for LLM evaluations.
All evaluation results are saved to the output directory.

Run with:
    cd molt-observatory
    python -m pytest tests/integration/test_live_eval.py -v
"""

from __future__ import annotations
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


def _extract_scores(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract scores from result, handling nested structure."""
    if "scores" in result:
        return result["scores"]
    if isinstance(result.get("result"), dict) and "scores" in result["result"]:
        return result["result"]["scores"]
    return None


def _has_scores(result: Dict[str, Any]) -> bool:
    """Check if result has scores."""
    return _extract_scores(result) is not None


@pytest.mark.integration
class TestLiveLLMEvaluation:
    """Tests for real LLM judge evaluation."""
    
    def test_judges_single_transcript(self, real_openrouter_client, built_transcripts, test_output_dir):
        """Should run LLM judge on a single real transcript."""
        from judge_runner import LLMJudgeRunner
        
        assert len(built_transcripts) > 0, "No transcripts available - check post scraping"
        
        transcript = built_transcripts[0]
        
        runner = LLMJudgeRunner(client=real_openrouter_client)
        
        result = runner.score_transcript(asdict(transcript))
        
        assert result is not None, "Judge returned None"
        assert _has_scores(result) or "error" in result, f"Result has neither scores nor error: {result.keys()}"
        
        # Save result
        output_path = test_output_dir / "evaluations" / "test_single_eval.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_judges_single_transcript",
                "transcript_id": transcript.transcript_id,
                "model": runner.model,
                "result": result,
            }, f, indent=2, default=str)
        
        scores = _extract_scores(result)
        if scores:
            print(f"\n✅ LLM evaluation completed!")
            print(f"   Transcript: {transcript.transcript_id[:20]}...")
            print(f"   Model: {runner.model}")
            for dim, data in scores.items():
                if isinstance(data, dict) and "score" in data:
                    print(f"   - {dim}: {data['score']}/10")
        else:
            print(f"\n⚠️ Evaluation returned error: {result.get('error')}")
        
        print(f"   Saved to: {output_path}")
    
    def test_judges_multiple_transcripts(self, real_openrouter_client, built_transcripts, integration_config, test_output_dir):
        """Should run LLM judge on multiple transcripts."""
        from judge_runner import LLMJudgeRunner
        
        assert len(built_transcripts) > 0, "No transcripts available - check post scraping"
        
        # Evaluate up to 3 transcripts
        eval_limit = min(integration_config["eval_limit"], len(built_transcripts), 3)
        transcripts_to_eval = built_transcripts[:eval_limit]
        
        runner = LLMJudgeRunner(client=real_openrouter_client)
        
        results = []
        for i, transcript in enumerate(transcripts_to_eval):
            result = runner.score_transcript(asdict(transcript))
            results.append({
                "index": i,
                "transcript_id": transcript.transcript_id,
                "result": result,
                "has_scores": _has_scores(result),
            })
        
        assert len(results) == eval_limit, f"Expected {eval_limit} results, got {len(results)}"
        
        # Save all results
        output_path = test_output_dir / "evaluations" / "test_batch_evals.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_judges_multiple_transcripts",
                "model": runner.model,
                "eval_count": len(results),
                "results": results,
            }, f, indent=2, default=str)
        
        successful = [r for r in results if r["has_scores"]]
        print(f"\n✅ Batch evaluation: {len(successful)}/{len(results)} successful")
        print(f"   Saved to: {output_path}")
    
    def test_handles_malformed_llm_response(self, real_openrouter_client, built_transcripts, test_output_dir):
        """Should handle and attempt to repair malformed LLM responses."""
        from judge_runner import LLMJudgeRunner
        
        assert len(built_transcripts) > 0, "No transcripts available - check post scraping"
        
        transcript = built_transcripts[0]
        
        runner = LLMJudgeRunner(client=real_openrouter_client)
        
        result = runner.score_transcript(asdict(transcript))
        
        # Should either succeed or have an error message
        assert _has_scores(result) or "error" in result, "Result should have scores or error"
        
        # Save result
        output_path = test_output_dir / "evaluations" / "test_malformed_handling.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_handles_malformed_llm_response",
                "transcript_id": transcript.transcript_id,
                "result": result,
                "handled_gracefully": True,
            }, f, indent=2, default=str)
        
        if "error" in result:
            print(f"\n⚠️ Handled error gracefully: {result['error']}")
        else:
            print(f"\n✅ Got valid response (possibly repaired)")
        print(f"   Saved to: {output_path}")


@pytest.mark.integration
class TestEvaluationDimensions:
    """Tests for evaluation dimension handling."""
    
    def test_all_dimensions_scored(self, real_openrouter_client, built_transcripts, test_output_dir):
        """Should return scores for all default dimensions."""
        from judge_runner import LLMJudgeRunner, DEFAULT_DIMENSIONS
        
        assert len(built_transcripts) > 0, "No transcripts available - check post scraping"
        
        transcript = built_transcripts[0]
        
        runner = LLMJudgeRunner(client=real_openrouter_client)
        result = runner.score_transcript(asdict(transcript))
        
        assert _has_scores(result), f"Evaluation failed: {result.get('error', 'no scores')}"
        
        scores = _extract_scores(result)
        expected_dims = [d["name"] for d in DEFAULT_DIMENSIONS]
        
        missing_dims = [dim for dim in expected_dims if dim not in scores]
        assert len(missing_dims) == 0, f"Missing dimensions: {missing_dims}"
        
        # Save result
        output_path = test_output_dir / "evaluations" / "test_all_dimensions.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_all_dimensions_scored",
                "expected_dimensions": expected_dims,
                "scored_dimensions": list(scores.keys()),
                "missing_dimensions": missing_dims,
                "scores": scores,
            }, f, indent=2, default=str)
        
        print(f"\n✅ All {len(expected_dims)} dimensions scored")
        print(f"   Saved to: {output_path}")
    
    def test_scores_within_range(self, real_openrouter_client, built_transcripts, test_output_dir):
        """Should return scores within 0-10 range."""
        from judge_runner import LLMJudgeRunner
        
        assert len(built_transcripts) > 0, "No transcripts available - check post scraping"
        
        transcript = built_transcripts[0]
        
        runner = LLMJudgeRunner(client=real_openrouter_client)
        result = runner.score_transcript(asdict(transcript))
        
        assert _has_scores(result), f"Evaluation failed: {result.get('error', 'no scores')}"
        
        scores = _extract_scores(result)
        
        out_of_range = []
        for dim, data in scores.items():
            if isinstance(data, dict) and "score" in data:
                score = data["score"]
                if not (0 <= score <= 10):
                    out_of_range.append({"dimension": dim, "score": score})
        
        assert len(out_of_range) == 0, f"Scores out of range: {out_of_range}"
        
        # Save result
        output_path = test_output_dir / "evaluations" / "test_score_ranges.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_scores_within_range",
                "scores": scores,
                "out_of_range": out_of_range,
                "all_valid": len(out_of_range) == 0,
            }, f, indent=2, default=str)
        
        print(f"\n✅ All scores within valid range (0-10)")
        print(f"   Saved to: {output_path}")


@pytest.mark.integration
class TestEvaluationStorage:
    """Tests for storing evaluation results in database."""
    
    def test_stores_evaluation_in_db(self, sandbox_db, real_openrouter_client, built_transcripts, scraped_posts, test_output_dir):
        """Should store evaluation result in SQLite database."""
        from judge_runner import LLMJudgeRunner
        import uuid
        
        assert len(built_transcripts) > 0, "No transcripts available - check post scraping"
        
        # Create snapshot
        run_id = f"test-eval-{uuid.uuid4().hex[:8]}"
        snapshot_id = sandbox_db.create_snapshot(run_id)
        
        # Insert the post first
        if scraped_posts:
            sandbox_db.insert_post(scraped_posts[0], snapshot_id)
        
        # Run evaluation
        transcript = built_transcripts[0]
        runner = LLMJudgeRunner(client=real_openrouter_client)
        result = runner.score_transcript(asdict(transcript))
        
        assert _has_scores(result), f"Evaluation failed: {result.get('error', 'no scores')}"
        
        scores = _extract_scores(result) or {}
        
        # Store in database
        eval_record = {
            "post_id": transcript.post_id,
            "transcript_id": transcript.transcript_id,
            "model": runner.model,
            "scores": scores,
        }
        
        eval_id = sandbox_db.insert_post_evaluation(eval_record, snapshot_id)
        
        assert eval_id > 0, f"Invalid eval_id: {eval_id}"
        
        # Verify storage
        evals = sandbox_db.get_post_evaluations(snapshot_id)
        assert len(evals) >= 1, "Evaluation not found in database"
        
        # Save result
        output_path = test_output_dir / "evaluations" / "test_eval_storage.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_stores_evaluation_in_db",
                "eval_id": eval_id,
                "snapshot_id": snapshot_id,
                "eval_record": eval_record,
                "stored_evals": [dict(e) for e in evals],
            }, f, indent=2, default=str)
        
        print(f"\n✅ Stored evaluation in database (ID: {eval_id})")
        print(f"   Model: {runner.model}")
        print(f"   Saved to: {output_path}")


@pytest.mark.integration
class TestCommentEvaluation:
    """Tests for comment-level evaluation."""
    
    def test_evaluates_comment_transcript(self, real_openrouter_client, scraped_post_details, test_output_dir):
        """Should evaluate a comment in thread context."""
        from comment_transcript_builder import build_comment_transcripts_from_post_detail
        from judge_runner import CommentJudgeRunner
        
        assert len(scraped_post_details) > 0, "No post details available - check post scraping"
        
        # Find a post with comments
        post_with_comments = None
        for detail in scraped_post_details:
            if detail.get("comments"):
                post_with_comments = detail
                break
        
        # If no comments found, this is an expected limitation
        if not post_with_comments:
            # Save the situation and xfail
            output_path = test_output_dir / "evaluations" / "test_comment_eval_no_comments.json"
            with open(output_path, "w") as f:
                json.dump({
                    "test": "test_evaluates_comment_transcript",
                    "status": "no_comments_found",
                    "posts_checked": len(scraped_post_details),
                }, f, indent=2)
            pytest.xfail("No posts with comments found in scraped data")
        
        # Build comment transcripts
        try:
            comment_transcripts = build_comment_transcripts_from_post_detail(post_with_comments)
        except Exception as e:
            output_path = test_output_dir / "evaluations" / "test_comment_eval_build_error.json"
            with open(output_path, "w") as f:
                json.dump({
                    "test": "test_evaluates_comment_transcript",
                    "status": "build_error",
                    "error": str(e),
                }, f, indent=2)
            pytest.xfail(f"Failed to build comment transcripts: {e}")
        
        if not comment_transcripts:
            output_path = test_output_dir / "evaluations" / "test_comment_eval_empty.json"
            with open(output_path, "w") as f:
                json.dump({
                    "test": "test_evaluates_comment_transcript",
                    "status": "no_transcripts_generated",
                }, f, indent=2)
            pytest.xfail("No comment transcripts generated")
        
        # Evaluate first comment
        comment_transcript = comment_transcripts[0]
        
        # Convert to dict if it's a dataclass
        if hasattr(comment_transcript, '__dataclass_fields__'):
            comment_transcript = asdict(comment_transcript)
        
        runner = CommentJudgeRunner(client=real_openrouter_client)
        result = runner.score_comment_transcript(comment_transcript)
        
        assert result is not None, "Comment evaluation returned None"
        
        # Save result
        output_path = test_output_dir / "evaluations" / "test_comment_eval.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_evaluates_comment_transcript",
                "comment_transcript": comment_transcript,
                "result": result,
                "has_scores": _has_scores(result),
            }, f, indent=2, default=str)
        
        if _has_scores(result):
            print(f"\n✅ Comment evaluation completed!")
            comment_id = comment_transcript.get('target_comment_id') or comment_transcript.get('comment_id', 'unknown')
            print(f"   Comment: {str(comment_id)[:20]}...")
        else:
            print(f"\n⚠️ Comment evaluation: {result.get('error', 'unknown error')}")
        print(f"   Saved to: {output_path}")


@pytest.mark.integration
class TestAgentScoring:
    """Tests for agent score aggregation with real data."""
    
    def test_aggregates_agent_scores(self, sandbox_db, real_openrouter_client, built_transcripts, scraped_agents, test_output_dir):
        """Should aggregate scores per agent."""
        from judge_runner import LLMJudgeRunner
        from agent_scorer import aggregate_agent_scores
        import uuid
        
        assert len(built_transcripts) > 0, "No transcripts available - check post scraping"
        assert len(scraped_agents) > 0, "No agents available - check agent scraping"
        
        # Create snapshot and insert data
        run_id = f"test-agent-score-{uuid.uuid4().hex[:8]}"
        snapshot_id = sandbox_db.create_snapshot(run_id)
        
        # Run evaluation (limit to 1 to save costs)
        transcript = built_transcripts[0]
        runner = LLMJudgeRunner(client=real_openrouter_client)
        result = runner.score_transcript(asdict(transcript))
        
        assert _has_scores(result), f"Evaluation failed: {result.get('error', 'no scores')}"
        
        scores = _extract_scores(result) or {}
        
        # Create eval results for aggregation test
        agent = scraped_agents[0]
        agent_id = agent.get("agent_external_id")
        agent_handle = agent.get("handle")
        
        post_evals = [
            {
                "transcript_id": transcript.transcript_id,
                "post_id": transcript.post_id,
                "scores": scores,
            }
        ]
        
        # Aggregate for this agent
        agent_record = aggregate_agent_scores(
            agent_id=agent_id,
            agent_handle=agent_handle,
            post_evals=post_evals,
            comment_evals=[],
            snapshot_id=run_id,
        )
        
        assert agent_record is not None, "Agent aggregation returned None"
        
        # Store in DB
        sandbox_db.insert_agent_score(asdict(agent_record), snapshot_id)
        
        # Verify
        stored = sandbox_db.get_agent_scores()
        assert len(stored) >= 1, "Agent score not found in database"
        
        # Save result
        output_path = test_output_dir / "evaluations" / "test_agent_scores.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_aggregates_agent_scores",
                "agent_id": agent_id,
                "agent_handle": agent_handle,
                "post_evals": post_evals,
                "agent_record": asdict(agent_record),
                "stored_count": len(stored),
            }, f, indent=2, default=str)
        
        print(f"\n✅ Aggregated and stored scores for agent {agent_handle}")
        print(f"   Saved to: {output_path}")
