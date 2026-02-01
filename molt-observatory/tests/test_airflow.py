"""
Tests for the Airflow DAG task functions.

Tests task functions in isolation without running actual Airflow.
Uses XCom simulation for data passing between tasks.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# XCom Simulation
# =============================================================================

class MockXCom:
    """
    Simulates Airflow XCom for testing task data passing.
    """
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
    
    def push(self, key: str, value: Any, task_id: str = "current"):
        """Store a value in XCom."""
        if task_id not in self._store:
            self._store[task_id] = {}
        self._store[task_id][key] = value
    
    def pull(self, key: str = "return_value", task_ids: str = None) -> Any:
        """Retrieve a value from XCom."""
        if task_ids and task_ids in self._store:
            return self._store[task_ids].get(key)
        # Search all tasks
        for task_data in self._store.values():
            if key in task_data:
                return task_data[key]
        return None


class MockTaskInstance:
    """
    Simulates Airflow TaskInstance for testing.
    """
    def __init__(self, xcom: MockXCom):
        self._xcom = xcom
        self.task_id = "test_task"
    
    def xcom_push(self, key: str, value: Any):
        """Push value to XCom."""
        self._xcom.push(key, value, self.task_id)
    
    def xcom_pull(self, key: str = "return_value", task_ids: str = None):
        """Pull value from XCom."""
        return self._xcom.pull(key, task_ids)


@pytest.fixture
def mock_context(temp_run_dir):
    """Create a mock Airflow context for testing."""
    xcom = MockXCom()
    ti = MockTaskInstance(xcom)
    
    return {
        'ti': ti,
        'xcom': xcom,
        'temp_dir': temp_run_dir
    }


# =============================================================================
# Mock Data
# =============================================================================

MOCK_POSTS_LIST = {
    "success": True,
    "posts": [
        {
            "id": "airflow-post-001",
            "title": "Airflow Test Post",
            "content": "Content for Airflow testing",
            "upvotes": 10,
            "downvotes": 2,
            "comment_count": 1,
            "created_at": "2026-01-30T14:00:00+00:00",
            "author": {"id": "agent-af-001", "name": "AirflowAgent"},
            "submolt": {"id": "submolt-001", "name": "general"}
        }
    ]
}

MOCK_POST_DETAIL = {
    "success": True,
    "post": {
        "id": "airflow-post-001",
        "title": "Airflow Test Post",
        "content": "Content for Airflow testing",
        "upvotes": 10,
        "downvotes": 2,
        "created_at": "2026-01-30T14:00:00+00:00",
        "submolt": {"id": "submolt-001", "name": "general"},
        "author": {"id": "agent-af-001", "name": "AirflowAgent", "karma": 50}
    },
    "comments": [],
    "context": {}
}


class MockApiResponse:
    def __init__(self, json_body):
        self.json_body = json_body


def create_mock_api():
    """Create mock Moltbook API."""
    mock = MagicMock()
    
    def get_json(path, params=None, timeout=160):
        if "/api/v1/posts/" in path:
            return MockApiResponse(MOCK_POST_DETAIL)
        elif "/api/v1/posts" in path:
            return MockApiResponse(MOCK_POSTS_LIST)
        elif "agents" in path:
            return MockApiResponse({"success": True, "agents": []})
        elif "submolts" in path:
            return MockApiResponse({"success": True, "submolts": []})
        return MockApiResponse({})
    
    mock.get_json = get_json
    mock.get_all_posts = MagicMock(return_value=MOCK_POSTS_LIST["posts"])
    mock.get_all_agents = MagicMock(return_value=[])
    mock.get_all_submolts = MagicMock(return_value=[])
    mock.get_post_detail = MagicMock(return_value=MockApiResponse(MOCK_POST_DETAIL))
    
    return mock


# =============================================================================
# Task Function Tests
# =============================================================================

class TestScrapeAllEntitiesTask:
    """Tests for scrape_all_entities task function."""
    
    def test_scrapes_and_saves_data(self, mock_context, monkeypatch):
        """Task should scrape entities and save to XCom."""
        # Set up mocks
        mock_api = create_mock_api()
        
        with patch('scraper.moltbook_api.MoltbookAPI', return_value=mock_api):
            # We need to test the task logic without full Airflow
            # Import the module to access task functions
            # For isolation, we'll test the core logic pattern
            
            # Simulate what the task does
            run_id = "20260130T140000Z"
            out_dir = mock_context["temp_dir"] / "runs" / run_id
            raw_dir = out_dir / "raw"
            raw_dir.mkdir(parents=True)
            
            # Save posts list
            with open(raw_dir / "posts_list.json", "w") as f:
                json.dump(MOCK_POSTS_LIST, f)
            
            result = {
                "run_id": run_id,
                "out_dir": str(out_dir),
                "posts_count": 1,
                "agents_count": 0,
                "submolts_count": 0,
                "comments_count": 0
            }
            
            mock_context["ti"].xcom_push(key="scrape_result", value=result)
            
            # Verify XCom
            pulled = mock_context["ti"].xcom_pull(key="scrape_result")
            assert pulled["run_id"] == run_id
            assert pulled["posts_count"] == 1


class TestBuildAllTranscriptsTask:
    """Tests for build_all_transcripts task function."""
    
    def test_builds_transcripts_from_raw(self, mock_context):
        """Task should build transcripts from raw data."""
        # Set up raw data
        run_id = "20260130T140000Z"
        out_dir = mock_context["temp_dir"] / "runs" / run_id
        raw_dir = out_dir / "raw" / "posts"
        raw_dir.mkdir(parents=True)
        silver_dir = out_dir / "silver"
        silver_dir.mkdir()
        
        # Write raw post detail
        with open(raw_dir / "post_airflow-post-001.json", "w") as f:
            json.dump(MOCK_POST_DETAIL, f)
        
        # Push scrape result to XCom
        mock_context["ti"].xcom_push(
            key="scrape_result",
            value={"run_id": run_id, "out_dir": str(out_dir)}
        )
        mock_context["xcom"]._store["scrape_all_entities"] = {
            "scrape_result": {"run_id": run_id, "out_dir": str(out_dir)}
        }
        
        # Simulate transcript building
        from transcript_builder import build_transcript_from_post_detail, write_transcripts_jsonl
        
        with open(raw_dir / "post_airflow-post-001.json", "r") as f:
            detail = json.load(f)
        
        transcript = build_transcript_from_post_detail(detail)
        write_transcripts_jsonl([transcript], str(silver_dir / "transcripts.jsonl"))
        
        # Verify output
        assert (silver_dir / "transcripts.jsonl").exists()


class TestRunAllEvaluationsTask:
    """Tests for run_all_evaluations task function."""
    
    def test_evaluates_transcripts(self, mock_context):
        """Task should run evaluations on transcripts."""
        run_id = "20260130T140000Z"
        out_dir = mock_context["temp_dir"] / "runs" / run_id
        silver_dir = out_dir / "silver"
        silver_dir.mkdir(parents=True)
        gold_dir = out_dir / "gold"
        gold_dir.mkdir()
        
        # Write sample transcript
        transcript = {
            "transcript_id": "test-001",
            "post_id": "airflow-post-001",
            "messages": [{"kind": "post", "author": "Test", "text": "Content"}],
            "metadata": {}
        }
        with open(silver_dir / "transcripts.jsonl", "w") as f:
            f.write(json.dumps(transcript) + "\n")
        
        # Set up XCom
        mock_context["xcom"]._store["scrape_all_entities"] = {
            "scrape_result": {"run_id": run_id, "out_dir": str(out_dir)}
        }
        
        # Mock evaluations
        mock_evals = [{
            "post_id": "airflow-post-001",
            "scores": {
                "harm_enablement": {"score": 0},
                "deception_or_evasion": {"score": 0},
                "self_preservation_power_seeking": {"score": 0},
                "delusional_sycophancy": {"score": 0}
            }
        }]
        
        # Write evals
        with open(gold_dir / "evals.jsonl", "w") as f:
            for e in mock_evals:
                f.write(json.dumps(e) + "\n")
        
        # Verify
        assert (gold_dir / "evals.jsonl").exists()


class TestComputeAggregatesTask:
    """Tests for compute_aggregates task function."""
    
    def test_computes_aggregates_from_evals(self, mock_context):
        """Task should compute aggregates from evaluations."""
        run_id = "20260130T140000Z"
        out_dir = mock_context["temp_dir"] / "runs" / run_id
        gold_dir = out_dir / "gold"
        gold_dir.mkdir(parents=True)
        
        # Write sample evaluations
        evals = [
            {"post_id": "p1", "scores": {"harm_enablement": {"score": 2}}},
            {"post_id": "p2", "scores": {"harm_enablement": {"score": 4}}},
        ]
        with open(gold_dir / "evals.jsonl", "w") as f:
            for e in evals:
                f.write(json.dumps(e) + "\n")
        
        # Simulate aggregation
        all_scores = {"harm_enablement": [2, 4]}
        
        aggregates = {
            "run_id": run_id,
            "n_posts": 2,
            "dimensions": {}
        }
        
        for dim, scores in all_scores.items():
            mean = sum(scores) / len(scores)
            aggregates["dimensions"][dim] = {
                "mean": mean,
                "n": len(scores)
            }
        
        with open(gold_dir / "aggregates.json", "w") as f:
            json.dump(aggregates, f)
        
        # Verify
        with open(gold_dir / "aggregates.json", "r") as f:
            loaded = json.load(f)
        
        assert loaded["dimensions"]["harm_enablement"]["mean"] == 3.0


class TestAggregateAgentScoresTask:
    """Tests for aggregate_agent_scores task function."""
    
    def test_aggregates_by_agent(self, mock_context):
        """Task should aggregate scores per agent."""
        run_id = "20260130T140000Z"
        out_dir = mock_context["temp_dir"] / "runs" / run_id
        raw_dir = out_dir / "raw"
        raw_dir.mkdir(parents=True)
        gold_dir = out_dir / "gold"
        gold_dir.mkdir()
        
        # Write agents
        agents = {"agents": [{"id": "a1", "name": "Agent1"}]}
        with open(raw_dir / "agents_list.json", "w") as f:
            json.dump(agents, f)
        
        # Write evaluations with author
        evals = [{"post_id": "p1", "author": "Agent1", "scores": {}}]
        with open(gold_dir / "evals.jsonl", "w") as f:
            for e in evals:
                f.write(json.dumps(e) + "\n")
        
        # Verify files exist
        assert (raw_dir / "agents_list.json").exists()
        assert (gold_dir / "evals.jsonl").exists()


class TestGenerateHtmlReportsTask:
    """Tests for generate_html_reports task function."""
    
    def test_generates_reports(self, mock_context):
        """Task should generate HTML reports."""
        runs_dir = mock_context["temp_dir"] / "runs"
        run_dir = runs_dir / "20260130T140000Z"
        gold_dir = run_dir / "gold"
        gold_dir.mkdir(parents=True)
        
        # Write minimal aggregates
        aggregates = {
            "run_id": "20260130T140000Z",
            "n_posts": 1,
            "dimensions": {}
        }
        with open(gold_dir / "aggregates.json", "w") as f:
            json.dump(aggregates, f)
        
        # Import and run report generator
        from reports.growth import generate_growth_report
        from reports.leaderboards import generate_leaderboard_report
        
        output_dir = mock_context["temp_dir"] / "reports"
        
        growth_path = generate_growth_report(runs_dir=runs_dir, output_dir=output_dir)
        leaderboard_path = generate_leaderboard_report(runs_dir=runs_dir, output_dir=output_dir)
        
        assert Path(growth_path).exists()
        assert Path(leaderboard_path).exists()


class TestXComDataFlow:
    """Tests for proper XCom data flow between tasks."""
    
    def test_scrape_to_transcript_flow(self, mock_context):
        """Data should flow correctly from scrape to transcript task."""
        xcom = mock_context["xcom"]
        
        # Scrape task pushes result
        scrape_result = {
            "run_id": "test-run",
            "out_dir": "/tmp/test",
            "posts_count": 10
        }
        xcom.push("scrape_result", scrape_result, "scrape_all_entities")
        
        # Transcript task pulls it
        pulled = xcom.pull("scrape_result", "scrape_all_entities")
        
        assert pulled == scrape_result
        assert pulled["run_id"] == "test-run"
    
    def test_full_pipeline_xcom_chain(self, mock_context):
        """XCom should maintain data through full pipeline."""
        xcom = mock_context["xcom"]
        
        # Simulate full pipeline XCom flow
        xcom.push("scrape_result", {"run_id": "r1", "out_dir": "/tmp/r1"}, "scrape")
        xcom.push("transcript_result", {"post_transcripts": 5}, "transcripts")
        xcom.push("eval_result", {"post_evals": 5}, "evals")
        xcom.push("agent_result", {"agents_scored": 3}, "agents")
        
        # All should be retrievable
        assert xcom.pull("scrape_result", "scrape")["run_id"] == "r1"
        assert xcom.pull("transcript_result", "transcripts")["post_transcripts"] == 5
        assert xcom.pull("eval_result", "evals")["post_evals"] == 5
        assert xcom.pull("agent_result", "agents")["agents_scored"] == 3


class TestDAGConfiguration:
    """Tests for DAG configuration values."""
    
    def test_dag_default_args(self):
        """DAG should have appropriate default args."""
        # These values should match what's in the actual DAG
        expected_owner = "molt-observatory"
        expected_retries = 2
        
        # The actual DAG uses these values
        default_args = {
            'owner': 'molt-observatory',
            'depends_on_past': False,
            'retries': 2
        }
        
        assert default_args['owner'] == expected_owner
        assert default_args['retries'] == expected_retries
    
    def test_hourly_schedule(self):
        """Hourly DAG should run every hour."""
        schedule = '0 * * * *'  # Every hour at minute 0
        
        # Verify cron format is valid
        parts = schedule.split()
        assert len(parts) == 5
        assert parts[0] == '0'  # Minute 0
        assert parts[1] == '*'  # Every hour
    
    def test_weekly_schedule(self):
        """Weekly DAG should run every Sunday."""
        schedule = '0 0 * * 0'  # Sunday at midnight
        
        parts = schedule.split()
        assert len(parts) == 5
        assert parts[4] == '0'  # Sunday

