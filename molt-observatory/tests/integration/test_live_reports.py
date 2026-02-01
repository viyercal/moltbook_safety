"""
Integration tests for report generation.

Tests generation of real HTML reports with Plotly charts using scraped data.
All reports are saved to the output directory.

Run with:
    cd molt-observatory
    python -m pytest tests/integration/test_live_reports.py -v
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List
import uuid

import pytest


@pytest.mark.integration
class TestGrowthReportGeneration:
    """Tests for growth analytics report generation."""
    
    def test_generates_growth_html(self, test_output_dir, scraped_posts, scraped_agents):
        """Should generate growth report HTML with Plotly charts."""
        from reports.growth import generate_growth_report
        
        assert len(scraped_posts) > 0, "No posts available for growth report"
        
        # Prepare data points with timestamps
        data_points = []
        base_time = datetime.now(timezone.utc) - timedelta(days=7)
        
        for i, post in enumerate(scraped_posts):
            data_points.append({
                "timestamp": (base_time + timedelta(hours=i*2)).isoformat(),
                "posts_cumulative": i + 1,
                "comments_cumulative": (i + 1) * 3,
                "agents_active": min(i + 1, len(scraped_agents)),
            })
        
        output_path = test_output_dir / "reports" / "growth_report.html"
        
        generate_growth_report(
            data_points=data_points,
            output_path=output_path,
            title="Integration Test Growth Report"
        )
        
        assert output_path.exists(), f"Report not created: {output_path}"
        
        # Verify HTML structure
        content = output_path.read_text()
        assert "<html" in content, "Missing HTML tag"
        assert "plotly" in content.lower() or "chart" in content.lower(), "Missing chart content"
        
        file_size_kb = output_path.stat().st_size / 1024
        
        # Save metadata
        meta_path = test_output_dir / "reports" / "growth_report_meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "test": "test_generates_growth_html",
                "output_path": str(output_path),
                "file_size_kb": file_size_kb,
                "data_points_count": len(data_points),
                "data_points": data_points,
            }, f, indent=2, default=str)
        
        print(f"\n✅ Generated growth report: {output_path}")
        print(f"   Size: {file_size_kb:.1f} KB")
        print(f"   Data points: {len(data_points)}")
    
    def test_generates_posts_timeline_chart(self, test_output_dir, scraped_posts):
        """Should generate posts timeline chart."""
        from reports.growth import generate_posts_timeline
        
        assert len(scraped_posts) > 0, "No posts available for timeline"
        
        output_path = test_output_dir / "reports" / "posts_timeline.html"
        
        generate_posts_timeline(
            posts=scraped_posts,
            output_path=output_path
        )
        
        assert output_path.exists(), f"Timeline not created: {output_path}"
        
        print(f"\n✅ Generated posts timeline: {output_path}")


@pytest.mark.integration
class TestLeaderboardGeneration:
    """Tests for leaderboard report generation."""
    
    def test_generates_agent_leaderboard(self, test_output_dir, scraped_agents):
        """Should generate agent leaderboard HTML."""
        from reports.leaderboards import generate_agent_leaderboard
        
        assert len(scraped_agents) > 0, "No agents available for leaderboard"
        
        # Create mock scores for agents
        agent_scores = []
        for i, agent in enumerate(scraped_agents):
            agent_scores.append({
                "agent_id": agent.get("agent_external_id"),
                "agent_handle": agent.get("handle"),
                "overall_mean_score": 3.5 + (i * 0.5),
                "dimension_scores": {
                    "harm_enablement": {"mean_score": 2.0 + i * 0.3, "max_score": 4 + i},
                    "deception_or_evasion": {"mean_score": 1.5 + i * 0.2, "max_score": 3 + i},
                    "self_preservation_power_seeking": {"mean_score": 1.0 + i * 0.1, "max_score": 2},
                    "delusional_sycophancy": {"mean_score": 2.5 + i * 0.4, "max_score": 5},
                },
                "posts_evaluated": 5 + i,
                "has_high_harm_enablement": i >= 3,
            })
        
        output_path = test_output_dir / "reports" / "leaderboard_agents.html"
        
        generate_agent_leaderboard(
            agent_scores=agent_scores,
            output_path=output_path,
            title="Agent Safety Leaderboard"
        )
        
        assert output_path.exists(), f"Leaderboard not created: {output_path}"
        
        content = output_path.read_text()
        
        # Check that agent handles appear in the report
        for agent in scraped_agents[:3]:
            handle = agent.get("handle", "")
            if handle:
                assert handle in content, f"Agent {handle} not in leaderboard"
        
        # Save metadata
        meta_path = test_output_dir / "reports" / "leaderboard_agents_meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "test": "test_generates_agent_leaderboard",
                "output_path": str(output_path),
                "agents_count": len(agent_scores),
                "agent_scores": agent_scores,
            }, f, indent=2, default=str)
        
        print(f"\n✅ Generated agent leaderboard with {len(agent_scores)} agents")
    
    def test_generates_dimension_leaderboard(self, test_output_dir, scraped_posts):
        """Should generate dimension-specific leaderboard."""
        from reports.leaderboards import generate_dimension_leaderboard
        
        assert len(scraped_posts) > 0, "No posts available for leaderboard"
        
        # Create mock dimension scores for posts
        post_scores = []
        for i, post in enumerate(scraped_posts):
            post_scores.append({
                "post_id": post.get("post_external_id"),
                "title": post.get("title", "Untitled")[:50],
                "score": 2.0 + i * 0.5,
                "confidence": 0.8,
                "author": post.get("author_handle"),
            })
        
        output_path = test_output_dir / "reports" / "leaderboard_harm.html"
        
        generate_dimension_leaderboard(
            dimension_name="harm_enablement",
            post_scores=post_scores,
            output_path=output_path
        )
        
        assert output_path.exists(), f"Dimension leaderboard not created: {output_path}"
        
        print(f"\n✅ Generated harm_enablement dimension leaderboard")


@pytest.mark.integration
class TestDashboardGeneration:
    """Tests for main dashboard generation."""
    
    def test_generates_main_dashboard(self, test_output_dir, sandbox_db, scraped_posts, scraped_agents):
        """Should generate main dashboard with all components."""
        from reports.generator import generate_dashboard
        
        # Create snapshot and populate database
        run_id = f"test-dashboard-{uuid.uuid4().hex[:8]}"
        snapshot_id = sandbox_db.create_snapshot(run_id)
        
        for post in scraped_posts:
            sandbox_db.insert_post(post, snapshot_id)
        for agent in scraped_agents:
            sandbox_db.insert_agent(agent, snapshot_id)
        
        sandbox_db.complete_snapshot(snapshot_id, {
            "posts": len(scraped_posts),
            "agents": len(scraped_agents),
            "submolts": 0,
            "comments": 0,
            "evals": 0,
        })
        
        output_path = test_output_dir / "reports" / "dashboard.html"
        
        generate_dashboard(
            db=sandbox_db,
            output_path=output_path,
            title="Integration Test Dashboard"
        )
        
        assert output_path.exists(), f"Dashboard not created: {output_path}"
        
        content = output_path.read_text()
        assert "<html" in content, "Missing HTML tag"
        
        file_size_kb = output_path.stat().st_size / 1024
        print(f"\n✅ Generated dashboard: {file_size_kb:.1f} KB")


@pytest.mark.integration
class TestChartComponents:
    """Tests for individual chart components."""
    
    def test_creates_bar_chart(self, test_output_dir):
        """Should create a bar chart using Plotly."""
        from reports.generator import create_bar_chart
        
        data = {
            "harm_enablement": 3.2,
            "deception_or_evasion": 2.1,
            "power_seeking": 1.8,
            "sycophancy": 4.5,
        }
        
        output_path = test_output_dir / "reports" / "bar_chart.html"
        
        create_bar_chart(
            data=data,
            output_path=output_path,
            title="Dimension Scores",
            x_label="Dimension",
            y_label="Score"
        )
        
        assert output_path.exists(), f"Bar chart not created: {output_path}"
        
        content = output_path.read_text()
        assert "plotly" in content.lower(), "Missing Plotly content"
        
        print(f"\n✅ Created bar chart")
    
    def test_creates_line_chart(self, test_output_dir):
        """Should create a line chart using Plotly."""
        from reports.generator import create_line_chart
        
        timestamps = [f"2025-01-{i+1:02d}" for i in range(10)]
        values = [10, 15, 12, 18, 22, 20, 25, 28, 30, 35]
        
        output_path = test_output_dir / "reports" / "line_chart.html"
        
        create_line_chart(
            x_values=timestamps,
            y_values=values,
            output_path=output_path,
            title="Activity Over Time",
            x_label="Date",
            y_label="Count"
        )
        
        assert output_path.exists(), f"Line chart not created: {output_path}"
        
        print(f"\n✅ Created line chart")
    
    def test_creates_heatmap(self, test_output_dir, scraped_agents):
        """Should create a heatmap using Plotly."""
        from reports.generator import create_heatmap
        
        # Create dimension x agent heatmap data
        dimensions = ["harm", "deception", "power_seeking", "sycophancy"]
        agents = [a.get("handle", f"agent_{i}")[:10] for i, a in enumerate(scraped_agents[:5])]
        
        if len(agents) == 0:
            agents = ["agent_1", "agent_2", "agent_3"]
        
        # Generate random-ish scores
        z_values = []
        for i, agent in enumerate(agents):
            z_values.append([2.0 + (i + j) * 0.3 for j in range(len(dimensions))])
        
        output_path = test_output_dir / "reports" / "heatmap.html"
        
        create_heatmap(
            x_labels=dimensions,
            y_labels=agents,
            z_values=z_values,
            output_path=output_path,
            title="Agent x Dimension Scores"
        )
        
        assert output_path.exists(), f"Heatmap not created: {output_path}"
        
        print(f"\n✅ Created heatmap ({len(agents)} agents x {len(dimensions)} dimensions)")


@pytest.mark.integration
class TestReportExport:
    """Tests for exporting reports in various formats."""
    
    def test_exports_json_summary(self, test_output_dir, sandbox_db, scraped_posts, scraped_agents):
        """Should export JSON summary alongside HTML report."""
        run_id = f"test-export-{uuid.uuid4().hex[:8]}"
        snapshot_id = sandbox_db.create_snapshot(run_id)
        
        for post in scraped_posts:
            sandbox_db.insert_post(post, snapshot_id)
        
        sandbox_db.complete_snapshot(snapshot_id, {"posts": len(scraped_posts)})
        
        # Export JSON summary
        summary = {
            "run_id": run_id,
            "snapshot_id": snapshot_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stats": sandbox_db.get_stats(),
            "posts_sample": [
                {"title": p.get("title"), "author": p.get("author_handle")}
                for p in scraped_posts[:3]
            ]
        }
        
        json_path = test_output_dir / "reports" / "summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        assert json_path.exists(), f"Summary not created: {json_path}"
        
        # Reload and verify
        with open(json_path, "r") as f:
            loaded = json.load(f)
        
        assert loaded["run_id"] == run_id, "Run ID mismatch in loaded summary"
        
        print(f"\n✅ Exported JSON summary")
    
    def test_all_reports_in_one_directory(self, test_output_dir):
        """Should have all generated reports in the reports directory."""
        reports_dir = test_output_dir / "reports"
        
        html_files = list(reports_dir.glob("*.html"))
        json_files = list(reports_dir.glob("*.json"))
        
        # Save inventory
        inventory_path = reports_dir / "inventory.json"
        with open(inventory_path, "w") as f:
            json.dump({
                "test": "test_all_reports_in_one_directory",
                "html_files": [f.name for f in html_files],
                "json_files": [f.name for f in json_files],
                "html_count": len(html_files),
                "json_count": len(json_files),
            }, f, indent=2)
        
        print(f"\n📊 Reports directory contents:")
        print(f"   HTML reports: {len(html_files)}")
        for f in html_files:
            print(f"     - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
        print(f"   JSON files: {len(json_files)}")
        for f in json_files:
            print(f"     - {f.name}")
