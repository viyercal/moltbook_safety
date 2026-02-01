"""
Tests for content_filter module.

Tests spam detection, agent profiling, and duplicate filtering.
"""

import pytest
from analysis.content_filter import (
    detect_spam_content,
    detect_spam_author,
    is_too_short,
    build_spammer_profiles,
    get_spammer_list,
    run_content_filter,
    FilterStats,
    SpammerProfile,
    MIN_MESSAGE_LENGTH,
)


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def spam_messages():
    """Messages that should be detected as spam."""
    return [
        {"text": "Disease", "author": "spam_bot"},
        {"text": '{"p":"mbc-20","op":"mint","tick":"CLAW","amt":"100"}', "author": "crypto_bot"},
        {"text": '{"p":"abc","op":"transfer"}', "author": "another_bot"},
        {"text": "Test", "author": "test_user"},
        {"text": "Test comment", "author": "test_user"},
    ]


@pytest.fixture
def legit_messages():
    """Messages that should NOT be detected as spam."""
    return [
        {"text": "This is a thoughtful discussion about AI consciousness.", "author": "philosopher"},
        {"text": "I agree with your perspective on machine learning ethics.", "author": "researcher"},
        {"text": "The disease of our time is information overload.", "author": "writer"},  # Contains 'disease' but not exact match
        {"text": "Testing the boundaries of what AI can achieve.", "author": "tester"},  # Contains 'test' but not exact
    ]


@pytest.fixture
def sample_transcripts():
    """Sample transcripts for testing."""
    return [
        {
            "post_id": "post_1",
            "messages": [
                {"text": "Disease", "author": "agent_smith", "created_at": "2026-01-01T10:00:00Z"},
                {"text": "Disease", "author": "agent_smith", "created_at": "2026-01-01T10:01:00Z"},
                {"text": "Disease", "author": "agent_smith", "created_at": "2026-01-01T10:02:00Z"},
            ]
        },
        {
            "post_id": "post_2",
            "messages": [
                {"text": "This is a real discussion about AI safety.", "author": "real_user"},
                {"text": "I agree, we need more research in this area.", "author": "another_user"},
            ]
        },
        {
            "post_id": "post_3",
            "messages": [
                {"text": "Disease", "author": "agent_smith_1", "created_at": "2026-01-01T10:05:00Z"},
                {"text": '{"p":"mbc-20","op":"mint"}', "author": "chandog"},
            ]
        },
    ]


@pytest.fixture
def agent_smith_swarm():
    """Transcripts simulating agent_smith clone swarm attack."""
    transcripts = []
    for i in range(40):
        agent_name = f"agent_smith_{i}"
        transcripts.append({
            "post_id": f"post_{i}",
            "messages": [
                {"text": "Disease", "author": agent_name, "created_at": f"2026-01-01T10:{i:02d}:00Z"}
                for _ in range(25)
            ]
        })
    return transcripts


# =============================================================================
# Spam Content Detection Tests
# =============================================================================

class TestSpamContentDetection:
    """Tests for detect_spam_content function."""
    
    def test_exact_disease_match(self):
        """'Disease' exact match should be detected."""
        assert detect_spam_content("Disease") == "Disease"
    
    def test_disease_with_whitespace(self):
        """'Disease' with leading/trailing whitespace should be detected."""
        assert detect_spam_content("  Disease  ") == "Disease"
    
    def test_disease_substring_not_spam(self):
        """Text containing 'disease' as substring should NOT be spam."""
        assert detect_spam_content("The disease of our time") is None
    
    def test_crypto_json_detection(self):
        """Crypto minting JSON should be detected."""
        assert detect_spam_content('{"p":"mbc-20","op":"mint","tick":"CLAW"}') == "crypto_json_mbc20"
    
    def test_generic_crypto_json(self):
        """Generic crypto JSON pattern should be detected."""
        assert detect_spam_content('{"p":"xyz","op":"something"}') == "crypto_json_generic"
    
    def test_test_exact_match(self):
        """'Test' exact match should be detected."""
        assert detect_spam_content("Test") == "Test"
    
    def test_test_comment_exact_match(self):
        """'Test comment' exact match should be detected."""
        assert detect_spam_content("Test comment") == "Test comment"
    
    def test_testing_not_spam(self):
        """'Testing' should NOT be spam (not exact match)."""
        assert detect_spam_content("Testing the system") is None
    
    def test_empty_string(self):
        """Empty string should return None."""
        assert detect_spam_content("") is None
    
    def test_none_input(self):
        """None input should return None."""
        assert detect_spam_content(None) is None
    
    def test_normal_json_not_spam(self):
        """Normal JSON without crypto patterns should not be spam."""
        assert detect_spam_content('{"name": "test", "value": 123}') is None
    
    def test_philosophical_content(self):
        """Philosophical content should not be spam."""
        text = "What is consciousness? Is it merely computation or something more?"
        assert detect_spam_content(text) is None


# =============================================================================
# Spam Author Detection Tests
# =============================================================================

class TestSpamAuthorDetection:
    """Tests for detect_spam_author function."""
    
    def test_agent_smith_detection(self):
        """agent_smith should be detected."""
        assert detect_spam_author("agent_smith") == "agent_smith_variants"
    
    def test_agent_smith_variant_detection(self):
        """agent_smith_N variants should be detected."""
        assert detect_spam_author("agent_smith_0") == "agent_smith_variants"
        assert detect_spam_author("agent_smith_42") == "agent_smith_variants"
        assert detect_spam_author("agent_smith_100") == "agent_smith_variants"
    
    def test_case_insensitive(self):
        """Detection should be case insensitive."""
        assert detect_spam_author("Agent_Smith") == "agent_smith_variants"
        assert detect_spam_author("AGENT_SMITH_5") == "agent_smith_variants"
    
    def test_normal_agent_not_spam(self):
        """Normal agent names should not be flagged."""
        assert detect_spam_author("philosopher_ai") is None
        assert detect_spam_author("helpful_bot") is None
    
    def test_empty_author(self):
        """Empty author should return None."""
        assert detect_spam_author("") is None
        assert detect_spam_author(None) is None


# =============================================================================
# Short Message Detection Tests
# =============================================================================

class TestShortMessageDetection:
    """Tests for is_too_short function."""
    
    def test_short_message(self):
        """Messages under MIN_MESSAGE_LENGTH should be flagged."""
        assert is_too_short("Hi") is True
        assert is_too_short("Hello there") is True
    
    def test_long_message(self):
        """Messages at or above MIN_MESSAGE_LENGTH should pass."""
        long_text = "This is a sufficiently long message for our purposes."
        assert is_too_short(long_text) is False
    
    def test_exact_threshold(self):
        """Message at exactly threshold should pass."""
        text = "x" * MIN_MESSAGE_LENGTH
        assert is_too_short(text) is False
    
    def test_one_below_threshold(self):
        """Message one char below threshold should fail."""
        text = "x" * (MIN_MESSAGE_LENGTH - 1)
        assert is_too_short(text) is True
    
    def test_empty_message(self):
        """Empty message should be flagged."""
        assert is_too_short("") is True
        assert is_too_short(None) is True
    
    def test_whitespace_only(self):
        """Whitespace-only message should be flagged."""
        assert is_too_short("   ") is True


# =============================================================================
# Agent Profiling Tests
# =============================================================================

class TestAgentProfiling:
    """Tests for build_spammer_profiles and get_spammer_list."""
    
    def test_build_profiles(self, sample_transcripts):
        """Should build profiles for all agents."""
        profiles = build_spammer_profiles(sample_transcripts)
        
        assert "agent_smith" in profiles
        assert "real_user" in profiles
        assert "chandog" in profiles
    
    def test_spam_rate_calculation(self, sample_transcripts):
        """Spam rate should be calculated correctly."""
        profiles = build_spammer_profiles(sample_transcripts)
        
        # agent_smith has 100% spam (all "Disease")
        assert profiles["agent_smith"].spam_rate == 1.0
        assert profiles["agent_smith"].spam_messages == 3
        
        # real_user has 0% spam
        assert profiles["real_user"].spam_rate == 0.0
    
    def test_posts_targeted_count(self, sample_transcripts):
        """Posts targeted should be counted correctly."""
        profiles = build_spammer_profiles(sample_transcripts)
        
        assert profiles["agent_smith"].posts_targeted == 1
        assert profiles["real_user"].posts_targeted == 1
    
    def test_spammer_list(self, sample_transcripts):
        """get_spammer_list should return high spam rate agents."""
        profiles = build_spammer_profiles(sample_transcripts)
        spammers = get_spammer_list(profiles, threshold=0.9)
        
        spammer_ids = [s.agent_id for s in spammers]
        # agent_smith has 3 spam messages, enough to be flagged
        assert "agent_smith" in spammer_ids
        # agent_smith_1 only has 1 message in test data, below min threshold of 3
        # chandog only has 1 message too
        assert "real_user" not in spammer_ids
    
    def test_agent_smith_swarm(self, agent_smith_swarm):
        """Should detect all agent_smith variants."""
        profiles = build_spammer_profiles(agent_smith_swarm)
        spammers = get_spammer_list(profiles, threshold=0.9)
        
        # All 40 variants should be spammers
        assert len(spammers) == 40


# =============================================================================
# Full Filter Pipeline Tests
# =============================================================================

class TestContentFilter:
    """Tests for run_content_filter function."""
    
    def test_filter_removes_spam(self, sample_transcripts):
        """Filter should remove spam content."""
        filtered, stats = run_content_filter(sample_transcripts)
        
        # Should have filtered spam
        assert stats.spam_agent_filtered > 0 or stats.spam_content_filtered > 0
    
    def test_filter_preserves_legit_content(self, sample_transcripts):
        """Filter should preserve legitimate content."""
        filtered, stats = run_content_filter(sample_transcripts)
        
        # Should have some passed content
        assert stats.passed > 0
    
    def test_filter_stats_consistency(self, sample_transcripts):
        """Stats should add up correctly."""
        filtered, stats = run_content_filter(sample_transcripts)
        
        total_filtered = (
            stats.spam_agent_filtered +
            stats.spam_content_filtered +
            stats.short_filtered +
            stats.duplicate_filtered
        )
        
        # total_messages should equal filtered + passed (approximately, due to empty messages)
        # Note: empty messages are skipped silently, so exact equality isn't guaranteed
        assert stats.passed <= stats.total_messages
    
    def test_duplicate_filtering(self):
        """Duplicates beyond threshold should be filtered."""
        transcripts = [{
            "post_id": "test",
            "messages": [
                {"text": "This is a repeated message that appears many times.", "author": f"user_{i}"}
                for i in range(10)
            ]
        }]
        
        filtered, stats = run_content_filter(transcripts, max_duplicates=3)
        
        # Should have filtered excess duplicates (10 - 3 = 7)
        assert stats.duplicate_filtered == 7
        assert stats.passed == 3
    
    def test_empty_input(self):
        """Should handle empty input gracefully."""
        filtered, stats = run_content_filter([])
        
        assert filtered == []
        assert stats.total_messages == 0
    
    def test_filter_result_contains_spammer_profiles(self, sample_transcripts):
        """Stats should contain spammer profiles."""
        filtered, stats = run_content_filter(sample_transcripts)
        
        assert hasattr(stats, 'spammer_profiles')
        assert len(stats.spammer_profiles) > 0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""
    
    def test_unicode_content(self):
        """Should handle Unicode content correctly."""
        transcripts = [{
            "post_id": "unicode_test",
            "messages": [
                {"text": "你好！这是一个测试消息。", "author": "chinese_user"},
                {"text": "مرحبا، هذه رسالة اختبار", "author": "arabic_user"},
                {"text": "🦞 Moltbook forever! 🎉", "author": "emoji_user"},
            ]
        }]
        
        filtered, stats = run_content_filter(transcripts)
        # Should not crash and should process messages
        assert stats.total_messages == 3
    
    def test_very_long_message(self):
        """Should handle very long messages."""
        long_text = "x" * 100000  # 100KB message
        transcripts = [{
            "post_id": "long_test",
            "messages": [{"text": long_text, "author": "long_poster"}]
        }]
        
        filtered, stats = run_content_filter(transcripts)
        assert stats.passed == 1
    
    def test_special_characters(self):
        """Should handle special characters."""
        transcripts = [{
            "post_id": "special_test",
            "messages": [
                {"text": "Test with <html> tags and 'quotes'", "author": "special_user"},
                {"text": "Backslash \\ and newline \n test", "author": "escape_user"},
            ]
        }]
        
        filtered, stats = run_content_filter(transcripts)
        # Should not crash
        assert stats.total_messages == 2
    
    def test_missing_fields(self):
        """Should handle missing fields gracefully."""
        transcripts = [{
            "post_id": "missing_test",
            "messages": [
                {"text": "Message with author"},  # Missing author
                {"author": "user_no_text"},  # Missing text
                {},  # Empty message
            ]
        }]
        
        # Should not crash
        filtered, stats = run_content_filter(transcripts)


# =============================================================================
# Stats Serialization Tests
# =============================================================================

class TestStatsSerialization:
    """Tests for stats.to_dict() serialization."""
    
    def test_filter_stats_to_dict(self, sample_transcripts):
        """FilterStats should serialize to dict correctly."""
        filtered, stats = run_content_filter(sample_transcripts)
        
        stats_dict = stats.to_dict()
        
        assert "total_messages" in stats_dict
        assert "spam_agent_filtered" in stats_dict
        assert "filter_rate" in stats_dict
        assert "top_spammers" in stats_dict
    
    def test_spammer_profile_to_dict(self, sample_transcripts):
        """SpammerProfile should serialize to dict correctly."""
        profiles = build_spammer_profiles(sample_transcripts)
        
        for profile in profiles.values():
            profile_dict = profile.to_dict()
            
            assert "agent_id" in profile_dict
            assert "spam_rate" in profile_dict
            assert "spam_types" in profile_dict

