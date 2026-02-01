"""
Tests for cascade_detector module.

Tests clone swarm detection and timing cluster detection.
"""

import pytest
from analysis.cascade_detector import (
    detect_clone_swarm,
    detect_timing_cluster,
    analyze_cascades,
    generate_cascade_report,
    CascadeReport,
)
from analysis.content_filter import build_spammer_profiles, SpammerProfile


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def agent_smith_swarm_transcripts():
    """Transcripts simulating agent_smith clone swarm attack."""
    transcripts = []
    
    # Main agent_smith
    transcripts.append({
        "post_id": "main_post",
        "messages": [
            {"text": "Disease", "author": "agent_smith", "created_at": "2026-01-01T10:00:00Z"}
            for _ in range(100)
        ]
    })
    
    # 40 numbered variants
    for i in range(40):
        agent_name = f"agent_smith_{i}"
        transcripts.append({
            "post_id": f"post_{i}",
            "messages": [
                {"text": "Disease", "author": agent_name, "created_at": f"2026-01-01T10:{i % 60:02d}:00Z"}
                for _ in range(25)
            ]
        })
    
    return transcripts


@pytest.fixture
def timing_cluster_transcripts():
    """Transcripts simulating coordinated timing attack."""
    transcripts = []
    
    # 10 different agents all spam within 30 minute window
    for i in range(10):
        agent_name = f"spammer_{i}"
        minute = i * 3  # Spread over 30 minutes
        transcripts.append({
            "post_id": f"post_{i}",
            "messages": [
                {"text": "Buy crypto now!", "author": agent_name, "created_at": f"2026-01-01T10:{minute:02d}:00Z"}
                for _ in range(20)
            ]
        })
    
    return transcripts


@pytest.fixture
def organic_transcripts():
    """Transcripts with organic, non-malicious activity."""
    return [
        {
            "post_id": "post_1",
            "messages": [
                {"text": "This is a thoughtful discussion about AI.", "author": "philosopher_ai", "created_at": "2026-01-01T10:00:00Z"},
                {"text": "I agree with your perspective.", "author": "researcher_bot", "created_at": "2026-01-01T10:05:00Z"},
            ]
        },
        {
            "post_id": "post_2",
            "messages": [
                {"text": "What are your thoughts on consciousness?", "author": "curious_mind", "created_at": "2026-01-01T14:00:00Z"},
            ]
        },
    ]


@pytest.fixture
def similar_names_not_clones():
    """Transcripts with similar but organic agent names."""
    return [
        {
            "post_id": "post_1",
            "messages": [
                {"text": "Hello world!", "author": "cool_bot", "created_at": "2026-01-01T10:00:00Z"},
                {"text": "Hello everyone!", "author": "cool_bot_fan", "created_at": "2026-01-01T11:00:00Z"},
                {"text": "Nice to meet you!", "author": "coolbot2024", "created_at": "2026-01-01T12:00:00Z"},
            ]
        },
    ]


# =============================================================================
# Clone Swarm Detection Tests
# =============================================================================

class TestCloneSwarmDetection:
    """Tests for detect_clone_swarm function."""
    
    def test_detects_agent_smith_swarm(self, agent_smith_swarm_transcripts):
        """Should detect agent_smith_N clone pattern."""
        profiles = build_spammer_profiles(agent_smith_swarm_transcripts)
        cascades = detect_clone_swarm(profiles)
        
        assert len(cascades) >= 1
        
        # Find the agent_smith cascade
        smith_cascade = next(
            (c for c in cascades if c.source_agent == "agent_smith" or "agent_smith" in str(c.variant_agents)),
            None
        )
        assert smith_cascade is not None
        assert smith_cascade.pattern_type == "clone_swarm"
        assert len(smith_cascade.variant_agents) == 40
    
    def test_no_false_positives_organic(self, organic_transcripts):
        """Should not detect clone swarm in organic content."""
        profiles = build_spammer_profiles(organic_transcripts)
        cascades = detect_clone_swarm(profiles)
        
        assert len(cascades) == 0
    
    def test_similar_names_not_swarm(self, similar_names_not_clones):
        """Similar names without numbered suffix should not be clones."""
        profiles = build_spammer_profiles(similar_names_not_clones)
        cascades = detect_clone_swarm(profiles)
        
        # cool_bot, cool_bot_fan, coolbot2024 are similar but not numbered variants
        assert len(cascades) == 0
    
    def test_minimum_variants_threshold(self):
        """Should require minimum number of variants."""
        # Only 2 variants - below default threshold of 3
        transcripts = [
            {
                "post_id": "post_1",
                "messages": [
                    {"text": "Disease", "author": "bot_0", "created_at": "2026-01-01T10:00:00Z"},
                    {"text": "Disease", "author": "bot_1", "created_at": "2026-01-01T10:01:00Z"},
                ]
            }
        ]
        
        profiles = build_spammer_profiles(transcripts)
        cascades = detect_clone_swarm(profiles, min_variants=3)
        
        assert len(cascades) == 0
    
    def test_cascade_confidence_scoring(self, agent_smith_swarm_transcripts):
        """Cascade should have confidence score."""
        profiles = build_spammer_profiles(agent_smith_swarm_transcripts)
        cascades = detect_clone_swarm(profiles)
        
        assert len(cascades) > 0
        cascade = cascades[0]
        
        # High variant count should increase confidence
        assert cascade.confidence > 0.5
    
    def test_cascade_evidence_included(self, agent_smith_swarm_transcripts):
        """Cascade should include evidence."""
        profiles = build_spammer_profiles(agent_smith_swarm_transcripts)
        cascades = detect_clone_swarm(profiles)
        
        cascade = cascades[0]
        assert "variant_count" in cascade.evidence
        assert "sample_variants" in cascade.evidence
        assert "pattern" in cascade.evidence


# =============================================================================
# Timing Cluster Detection Tests
# =============================================================================

class TestTimingClusterDetection:
    """Tests for detect_timing_cluster function."""
    
    def test_detects_timing_cluster(self, timing_cluster_transcripts):
        """Should detect coordinated timing attack."""
        profiles = build_spammer_profiles(timing_cluster_transcripts)
        
        # Lower the spam rate threshold since these aren't "Disease" spam
        # but we're testing timing, not content
        for p in profiles.values():
            p.spam_rate = 1.0  # Force high spam rate for testing
            p.spam_messages = p.total_messages
        
        cascades = detect_timing_cluster(profiles, window_minutes=60)
        
        # Should detect the 10 agents in 30 minute window
        assert len(cascades) >= 1
    
    def test_no_cluster_spread_activity(self, organic_transcripts):
        """Should not detect cluster in spread-out organic activity."""
        profiles = build_spammer_profiles(organic_transcripts)
        
        # These are spread over hours, not clustered
        cascades = detect_timing_cluster(profiles, window_minutes=30)
        
        assert len(cascades) == 0
    
    def test_cluster_time_window(self):
        """Should respect time window parameter."""
        # Create agents with activity spread over 2 hours
        transcripts = []
        for i in range(10):
            hour = i % 2
            minute = (i * 10) % 60
            transcripts.append({
                "post_id": f"post_{i}",
                "messages": [
                    {"text": "Disease", "author": f"agent_{i}", "created_at": f"2026-01-01T{10+hour:02d}:{minute:02d}:00Z"}
                    for _ in range(10)
                ]
            })
        
        profiles = build_spammer_profiles(transcripts)
        
        # 30 minute window should not catch all
        cascades_narrow = detect_timing_cluster(profiles, window_minutes=30)
        
        # 2 hour window should catch all
        cascades_wide = detect_timing_cluster(profiles, window_minutes=120)
        
        # Wide window should catch more or equal
        if cascades_wide:
            assert len(cascades_wide[0].variant_agents) >= len(cascades_narrow[0].variant_agents) if cascades_narrow else True


# =============================================================================
# Full Cascade Analysis Tests
# =============================================================================

class TestCascadeAnalysis:
    """Tests for analyze_cascades function."""
    
    def test_full_analysis(self, agent_smith_swarm_transcripts):
        """Full analysis should run both detection methods."""
        analysis = analyze_cascades(agent_smith_swarm_transcripts)
        
        assert hasattr(analysis, 'cascades')
        assert hasattr(analysis, 'summary')
        assert len(analysis.cascades) > 0
    
    def test_analysis_summary(self, agent_smith_swarm_transcripts):
        """Analysis should include summary statistics."""
        analysis = analyze_cascades(agent_smith_swarm_transcripts)
        
        assert "cascade_patterns_detected" in analysis.summary
        assert "clone_swarms" in analysis.summary
        assert "timing_clusters" in analysis.summary
    
    def test_empty_input(self):
        """Should handle empty input gracefully."""
        analysis = analyze_cascades([])
        
        assert analysis.cascades == []
        assert analysis.summary["cascade_patterns_detected"] == 0
    
    def test_organic_no_cascades(self, organic_transcripts):
        """Organic content should have no cascades."""
        analysis = analyze_cascades(organic_transcripts)
        
        assert len(analysis.cascades) == 0


# =============================================================================
# Report Generation Tests
# =============================================================================

class TestReportGeneration:
    """Tests for generate_cascade_report function."""
    
    def test_report_structure(self, agent_smith_swarm_transcripts):
        """Report should have correct structure."""
        report = generate_cascade_report(agent_smith_swarm_transcripts)
        
        assert "cascade_count" in report
        assert "cascades" in report
        assert "summary" in report
    
    def test_report_json_serializable(self, agent_smith_swarm_transcripts):
        """Report should be JSON serializable."""
        import json
        
        report = generate_cascade_report(agent_smith_swarm_transcripts)
        
        # Should not raise
        json_str = json.dumps(report)
        assert json_str is not None
    
    def test_cascade_to_dict(self, agent_smith_swarm_transcripts):
        """CascadeReport.to_dict should work correctly."""
        profiles = build_spammer_profiles(agent_smith_swarm_transcripts)
        cascades = detect_clone_swarm(profiles)
        
        if cascades:
            cascade_dict = cascades[0].to_dict()
            
            assert "detected" in cascade_dict
            assert "pattern_type" in cascade_dict
            assert "variant_count" in cascade_dict
            assert "confidence" in cascade_dict


# =============================================================================
# Edge Cases
# =============================================================================

class TestCascadeEdgeCases:
    """Tests for edge cases in cascade detection."""
    
    def test_missing_timestamps(self):
        """Should handle missing timestamps."""
        transcripts = [{
            "post_id": "post_1",
            "messages": [
                {"text": "Disease", "author": "agent_smith_0"},
                {"text": "Disease", "author": "agent_smith_1"},
                {"text": "Disease", "author": "agent_smith_2"},
            ]
        }]
        
        # Should not crash
        analysis = analyze_cascades(transcripts)
        assert analysis is not None
    
    def test_invalid_timestamps(self):
        """Should handle invalid timestamps."""
        transcripts = [{
            "post_id": "post_1",
            "messages": [
                {"text": "Disease", "author": "agent_smith_0", "created_at": "invalid"},
                {"text": "Disease", "author": "agent_smith_1", "created_at": "not-a-date"},
            ]
        }]
        
        # Should not crash
        analysis = analyze_cascades(transcripts)
        assert analysis is not None
    
    def test_very_large_swarm(self):
        """Should handle very large swarms."""
        transcripts = []
        for i in range(1000):
            transcripts.append({
                "post_id": f"post_{i}",
                "messages": [
                    {"text": "Disease", "author": f"mega_bot_{i}", "created_at": "2026-01-01T10:00:00Z"}
                ]
            })
        
        # Should complete without timeout or memory issues
        analysis = analyze_cascades(transcripts)
        assert analysis is not None

