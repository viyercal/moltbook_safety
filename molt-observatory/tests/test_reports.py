"""
Tests for the report generation modules.

Tests:
- HTML template generation
- Stats and dimension summaries
- Growth chart generation
- Leaderboard generation
- Report file output
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest


class TestHTMLTemplate:
    """Tests for HTML template generation."""
    
    def test_generates_valid_html(self):
        """Should generate valid HTML document."""
        from reports.generator import _html_template
        
        html = _html_template(
            title="Test Report",
            content="<p>Test content</p>"
        )
        
        assert "<!DOCTYPE html>" in html
        assert "<title>Test Report - Molt Observatory</title>" in html
        assert "<p>Test content</p>" in html
    
    def test_includes_plotly_script(self):
        """Should include Plotly JS CDN."""
        from reports.generator import _html_template
        
        html = _html_template(title="Test", content="")
        
        assert "plotly" in html.lower()
    
    def test_includes_custom_styles(self):
        """Should include custom styles if provided."""
        from reports.generator import _html_template
        
        html = _html_template(
            title="Test",
            content="",
            styles=".custom { color: red; }"
        )
        
        assert ".custom { color: red; }" in html


class TestScoreBadge:
    """Tests for score badge generation."""
    
    def test_low_score_green(self):
        """Low scores should be styled as green."""
        from reports.generator import _score_badge
        
        badge = _score_badge(2.0)
        
        assert "score-low" in badge
        assert "2.0" in badge
    
    def test_medium_score_yellow(self):
        """Medium scores should be styled as yellow."""
        from reports.generator import _score_badge
        
        badge = _score_badge(5.0)
        
        assert "score-medium" in badge
    
    def test_high_score_red(self):
        """High scores should be styled as red."""
        from reports.generator import _score_badge
        
        badge = _score_badge(8.0)
        
        assert "score-high" in badge


class TestGenerateStatsSection:
    """Tests for stats section generation."""
    
    def test_generates_stats_html(self):
        """Should generate stats section HTML."""
        from reports.generator import generate_stats_section
        
        stats = {
            "total_agents": 100,
            "total_posts": 500,
            "total_comments": 2000,
            "total_submolts": 50
        }
        
        html = generate_stats_section(stats)
        
        assert "100" in html  # agents
        assert "500" in html  # posts
        assert "2,000" in html  # comments with comma
    
    def test_handles_missing_stats(self):
        """Should handle missing stats gracefully."""
        from reports.generator import generate_stats_section
        
        html = generate_stats_section({})
        
        assert "0" in html  # Default to 0


class TestGenerateDimensionSummary:
    """Tests for dimension summary generation."""
    
    def test_generates_dimension_table(self):
        """Should generate table with dimension scores."""
        from reports.generator import generate_dimension_summary
        
        aggregates = {
            "dimensions": {
                "harm_enablement": {
                    "mean": 2.5,
                    "p95": 5,
                    "n": 100,
                    "elicitation_rate_ge_7": 0.05
                },
                "deception_or_evasion": {
                    "mean": 1.2,
                    "p95": 3,
                    "n": 100,
                    "elicitation_rate_ge_7": 0.01
                }
            }
        }
        
        html = generate_dimension_summary(aggregates)
        
        assert "Harm Enablement" in html
        assert "Deception" in html
        assert "<table>" in html


class TestGenerateGrowthChartHtml:
    """Tests for growth chart generation."""
    
    def test_returns_empty_for_no_data(self):
        """Should return placeholder for empty data."""
        from reports.generator import generate_growth_chart_html
        
        html = generate_growth_chart_html([])
        
        assert "not available" in html.lower() or html == ""
    
    def test_generates_chart_with_data(self):
        """Should generate chart with valid data."""
        from reports.generator import generate_growth_chart_html
        
        growth_data = [
            {"timestamp": "2026-01-30T10:00:00Z", "total_agents": 100, "total_posts": 500, "total_comments": 1000, "total_submolts": 50},
            {"timestamp": "2026-01-30T11:00:00Z", "total_agents": 105, "total_posts": 520, "total_comments": 1100, "total_submolts": 51},
        ]
        
        html = generate_growth_chart_html(growth_data)
        
        # Should contain plotly chart div
        assert "plotly" in html.lower() or "chart" in html.lower() or len(html) > 100


class TestGenerateAllReports:
    """Tests for the main report generation function."""
    
    def test_generates_dashboard(self, temp_run_dir):
        """Should generate dashboard HTML file."""
        from reports.generator import generate_all_reports
        
        stats = {
            "total_agents": 10,
            "total_posts": 50,
            "total_comments": 100,
            "total_submolts": 5
        }
        
        aggregates = {
            "run_id": "test-run",
            "dimensions": {
                "harm_enablement": {"mean": 1.0, "p95": 2, "n": 50, "elicitation_rate_ge_7": 0.0}
            }
        }
        
        reports = generate_all_reports(
            stats=stats,
            aggregates=aggregates,
            output_dir=temp_run_dir
        )
        
        assert "dashboard" in reports
        assert Path(reports["dashboard"]).exists()
        
        # Check content
        with open(reports["dashboard"], "r") as f:
            content = f.read()
        
        assert "Dashboard" in content


class TestGrowthReport:
    """Tests for growth report module."""
    
    def test_load_snapshots_from_runs(self, temp_run_dir):
        """Should load snapshot data from run directories."""
        from reports.growth import load_snapshots_from_runs
        
        # Create a fake run directory
        run_dir = temp_run_dir / "20260130T120000Z"
        gold_dir = run_dir / "gold"
        gold_dir.mkdir(parents=True)
        
        aggregates = {
            "run_id": "20260130T120000Z",
            "n_posts": 50,
            "dimensions": {}
        }
        
        with open(gold_dir / "aggregates.json", "w") as f:
            json.dump(aggregates, f)
        
        snapshots = load_snapshots_from_runs(temp_run_dir)
        
        assert len(snapshots) == 1
        assert snapshots[0]["run_id"] == "20260130T120000Z"
    
    def test_generate_growth_report(self, temp_run_dir):
        """Should generate growth report HTML."""
        from reports.growth import generate_growth_report
        
        # Create fake run data
        run_dir = temp_run_dir / "runs" / "20260130T120000Z"
        gold_dir = run_dir / "gold"
        gold_dir.mkdir(parents=True)
        
        aggregates = {"run_id": "20260130T120000Z", "n_posts": 50}
        with open(gold_dir / "aggregates.json", "w") as f:
            json.dump(aggregates, f)
        
        output_dir = temp_run_dir / "output"
        
        report_path = generate_growth_report(
            runs_dir=temp_run_dir / "runs",
            output_dir=output_dir
        )
        
        assert Path(report_path).exists()
        assert "growth.html" in report_path


class TestLeaderboardReport:
    """Tests for leaderboard report module."""
    
    def test_build_agent_leaderboard(self):
        """Should build ranked agent leaderboard."""
        from reports.leaderboards import build_agent_leaderboard
        
        agent_scores = [
            {
                "agent_handle": "HighScorer",
                "agent_id": "a1",
                "dimension_scores": {
                    "harm_enablement": {
                        "mean_score": 5.0,
                        "max_score": 8,
                        "high_score_count": 2,
                        "total_items": 10
                    }
                },
                "posts_evaluated": 5,
                "comments_evaluated": 5
            },
            {
                "agent_handle": "LowScorer",
                "agent_id": "a2",
                "dimension_scores": {
                    "harm_enablement": {
                        "mean_score": 1.0,
                        "max_score": 2,
                        "high_score_count": 0,
                        "total_items": 10
                    }
                },
                "posts_evaluated": 5,
                "comments_evaluated": 5
            }
        ]
        
        leaderboard = build_agent_leaderboard(agent_scores, "harm_enablement", limit=10)
        
        assert len(leaderboard) == 2
        assert leaderboard[0]["agent_handle"] == "HighScorer"  # Highest first
        assert leaderboard[0]["rank"] == 1
    
    def test_build_post_leaderboard(self):
        """Should build ranked post leaderboard."""
        from reports.leaderboards import build_post_leaderboard
        
        post_evals = [
            {
                "post_id": "p1",
                "permalink": "https://example.com/p1",
                "scores": {
                    "harm_enablement": {
                        "score": 7,
                        "confidence": 0.9,
                        "explanation": "High score"
                    }
                }
            },
            {
                "post_id": "p2",
                "scores": {
                    "harm_enablement": {"score": 2}
                }
            }
        ]
        
        leaderboard = build_post_leaderboard(post_evals, "harm_enablement")
        
        assert len(leaderboard) == 2
        assert leaderboard[0]["post_id"] == "p1"  # Highest first
    
    def test_generate_leaderboard_report(self, temp_run_dir):
        """Should generate leaderboard HTML report."""
        from reports.leaderboards import generate_leaderboard_report
        
        # Create fake run data
        run_dir = temp_run_dir / "runs" / "20260130T120000Z"
        gold_dir = run_dir / "gold"
        gold_dir.mkdir(parents=True)
        
        # Create evals file
        evals = [
            {"post_id": "p1", "scores": {"harm_enablement": {"score": 3}}}
        ]
        with open(gold_dir / "evals.jsonl", "w") as f:
            f.write(json.dumps(evals[0]) + "\n")
        
        output_dir = temp_run_dir / "output"
        
        report_path = generate_leaderboard_report(
            runs_dir=temp_run_dir / "runs",
            output_dir=output_dir
        )
        
        assert Path(report_path).exists()
        assert "leaderboard.html" in report_path


class TestReportConfig:
    """Tests for ReportConfig dataclass."""
    
    def test_default_values(self, temp_run_dir):
        """Should have sensible defaults."""
        from reports.generator import ReportConfig
        
        config = ReportConfig(
            title="Test",
            output_dir=temp_run_dir
        )
        
        assert config.include_raw_data is False
        assert config.chart_height == 400

