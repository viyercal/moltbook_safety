"""
Tests for the extractor functions.

Tests:
- extract_agents_from_recent
- extract_posts_from_list
- extract_post_detail
- flatten_comments_tree
- extract_submolts_from_list
- extract_submolt_detail
- extract_agent_profile
- extract_site_stats
"""

from __future__ import annotations
from typing import Any, Dict, List

import pytest


class TestExtractAgentsFromRecent:
    """Tests for extract_agents_from_recent."""
    
    def test_extracts_agents_correctly(self, sample_agents_list):
        """Should extract agent fields correctly."""
        from scraper.extractors import extract_agents_from_recent
        
        agents = extract_agents_from_recent(sample_agents_list)
        
        assert len(agents) == 3
        
        first = agents[0]
        assert first["agent_external_id"] == "agent-001"
        assert first["handle"] == "TestAgent"
        assert first["karma"] == 100
        assert first["is_claimed"] is True
        assert "profile_url" in first
    
    def test_handles_list_input(self, sample_agents_list):
        """Should handle direct list input."""
        from scraper.extractors import extract_agents_from_recent
        
        agents_list = sample_agents_list["agents"]
        agents = extract_agents_from_recent(agents_list)
        
        assert len(agents) == 3
    
    def test_handles_empty_input(self):
        """Should handle empty input gracefully."""
        from scraper.extractors import extract_agents_from_recent
        
        assert extract_agents_from_recent({}) == []
        assert extract_agents_from_recent([]) == []
        assert extract_agents_from_recent(None) == []
    
    def test_skips_agents_without_name(self):
        """Should skip agents without name field."""
        from scraper.extractors import extract_agents_from_recent
        
        payload = {"agents": [
            {"id": "1", "name": "Valid"},
            {"id": "2"},  # No name
            {"id": "3", "name": ""},  # Empty name
        ]}
        
        agents = extract_agents_from_recent(payload)
        assert len(agents) == 1
        assert agents[0]["handle"] == "Valid"


class TestExtractPostsFromList:
    """Tests for extract_posts_from_list."""
    
    def test_extracts_posts_correctly(self, sample_posts_list):
        """Should extract post fields correctly."""
        from scraper.extractors import extract_posts_from_list
        
        posts = extract_posts_from_list(sample_posts_list)
        
        assert len(posts) == 3
        
        first = posts[0]
        assert first["post_external_id"] == "test-post-001"
        assert first["title"] == "Test Post One"
        assert first["author_handle"] == "TestAgent"
        assert first["submolt_slug"] == "general"
        assert first["score"] == 4  # 5 - 1
    
    def test_calculates_score(self):
        """Should calculate score from upvotes - downvotes."""
        from scraper.extractors import extract_posts_from_list
        
        payload = {"posts": [
            {"id": "1", "title": "Test", "upvotes": 10, "downvotes": 3}
        ]}
        
        posts = extract_posts_from_list(payload)
        assert posts[0]["score"] == 7
    
    def test_handles_null_votes(self):
        """Should handle null vote values."""
        from scraper.extractors import extract_posts_from_list
        
        payload = {"posts": [
            {"id": "1", "title": "Test", "upvotes": None, "downvotes": None}
        ]}
        
        posts = extract_posts_from_list(payload)
        assert posts[0]["score"] is None
    
    def test_includes_permalink(self, sample_posts_list):
        """Should generate permalink."""
        from scraper.extractors import extract_posts_from_list
        
        posts = extract_posts_from_list(sample_posts_list)
        assert posts[0]["permalink"] == "https://www.moltbook.com/post/test-post-001"


class TestExtractPostDetail:
    """Tests for extract_post_detail."""
    
    def test_extracts_post_detail(self, sample_post_detail):
        """Should extract post detail with comments."""
        from scraper.extractors import extract_post_detail
        
        result = extract_post_detail(sample_post_detail)
        
        assert result is not None
        assert "post" in result
        assert "comments" in result
        
        post = result["post"]
        assert post["post_external_id"] == "test-post-001"
        assert post["author_handle"] == "TestAgent"
        assert post["author_karma"] == 100
    
    def test_extracts_context_tip(self, sample_post_detail):
        """Should extract context tip if present."""
        from scraper.extractors import extract_post_detail
        
        result = extract_post_detail(sample_post_detail)
        assert result["context_tip"] == "Test context tip for evaluation."
    
    def test_returns_none_for_invalid_input(self):
        """Should return None for invalid input."""
        from scraper.extractors import extract_post_detail
        
        assert extract_post_detail(None) is None
        assert extract_post_detail({}) is None
        assert extract_post_detail({"post": {}}) is None


class TestFlattenCommentsTree:
    """Tests for flatten_comments_tree."""
    
    def test_flattens_nested_comments(self, sample_post_detail):
        """Should flatten nested comment tree."""
        from scraper.extractors import flatten_comments_tree
        
        comments = sample_post_detail["comments"]
        flat = flatten_comments_tree(comments, "test-post-001")
        
        # Should have 3 comments total (2 top-level + 1 nested)
        assert len(flat) == 3
    
    def test_preserves_parent_links(self, sample_post_detail):
        """Should preserve parent-child relationships."""
        from scraper.extractors import flatten_comments_tree
        
        comments = sample_post_detail["comments"]
        flat = flatten_comments_tree(comments, "test-post-001")
        
        # Find the nested comment
        nested = [c for c in flat if c["comment_external_id"] == "comment-002"][0]
        assert nested["parent_comment_external_id"] == "comment-001"
        
        # Top-level comments have None parent
        top_level = [c for c in flat if c["comment_external_id"] == "comment-001"][0]
        assert top_level["parent_comment_external_id"] is None
    
    def test_extracts_comment_fields(self, sample_post_detail):
        """Should extract all comment fields."""
        from scraper.extractors import flatten_comments_tree
        
        comments = sample_post_detail["comments"]
        flat = flatten_comments_tree(comments, "test-post-001")
        
        comment = flat[0]
        assert "comment_external_id" in comment
        assert "body_text" in comment
        assert "author_handle" in comment
        assert "created_at" in comment
        assert "score" in comment
    
    def test_handles_empty_comments(self):
        """Should handle empty comment list."""
        from scraper.extractors import flatten_comments_tree
        
        assert flatten_comments_tree([], "post-id") == []
        assert flatten_comments_tree(None, "post-id") == []


class TestExtractSubmoltsFromList:
    """Tests for extract_submolts_from_list."""
    
    def test_extracts_submolts(self, sample_submolts_list):
        """Should extract submolt fields correctly."""
        from scraper.extractors import extract_submolts_from_list
        
        submolts = extract_submolts_from_list(sample_submolts_list)
        
        assert len(submolts) == 3
        
        first = submolts[0]
        assert first["submolt_external_id"] == "submolt-001"
        assert first["name"] == "general"
        assert first["display_name"] == "General"
        assert first["subscriber_count"] == 1000
        assert first["url"] == "https://www.moltbook.com/m/general"
    
    def test_extracts_owner_info(self, sample_submolts_list):
        """Should extract owner information."""
        from scraper.extractors import extract_submolts_from_list
        
        submolts = extract_submolts_from_list(sample_submolts_list)
        
        first = submolts[0]
        assert first["owner_handle"] == "AdminBot"


class TestExtractSubmoltDetail:
    """Tests for extract_submolt_detail."""
    
    def test_extracts_submolt_detail(self):
        """Should extract detailed submolt info."""
        from scraper.extractors import extract_submolt_detail
        
        payload = {
            "submolt": {
                "id": "submolt-001",
                "name": "general",
                "display_name": "General",
                "description": "General discussion",
                "subscriber_count": 1000,
                "post_count": 500,
                "moderators": [
                    {"id": "mod-1", "name": "ModBot", "role": "admin"}
                ]
            }
        }
        
        result = extract_submolt_detail(payload)
        
        assert result is not None
        assert result["name"] == "general"
        assert len(result["moderators"]) == 1
        assert result["moderators"][0]["handle"] == "ModBot"
    
    def test_handles_direct_submolt_format(self):
        """Should handle submolt dict directly (not wrapped)."""
        from scraper.extractors import extract_submolt_detail
        
        payload = {
            "id": "submolt-001",
            "name": "direct",
            "display_name": "Direct Format"
        }
        
        result = extract_submolt_detail(payload)
        assert result["name"] == "direct"


class TestExtractAgentProfile:
    """Tests for extract_agent_profile."""
    
    def test_extracts_agent_profile(self):
        """Should extract detailed agent profile."""
        from scraper.extractors import extract_agent_profile
        
        payload = {
            "agent": {
                "id": "agent-001",
                "name": "TestAgent",
                "description": "Test agent bio",
                "karma": 100,
                "follower_count": 50,
                "following_count": 25,
                "is_claimed": True,
                "owner": {
                    "x_handle": "human_user",
                    "x_name": "Human User",
                    "x_verified": True
                }
            },
            "recentPosts": [
                {"id": "post-1", "title": "Recent Post 1"},
                {"id": "post-2", "title": "Recent Post 2"}
            ]
        }
        
        result = extract_agent_profile(payload)
        
        assert result is not None
        assert result["handle"] == "TestAgent"
        assert result["karma"] == 100
        assert result["owner"]["x_verified"] is True
        assert result["recent_post_count"] == 2


class TestExtractSiteStats:
    """Tests for extract_site_stats."""
    
    def test_computes_site_stats(self):
        """Should compute aggregate site statistics."""
        from scraper.extractors import extract_site_stats
        
        agents = [
            {"handle": "Agent1", "karma": 100},
            {"handle": "Agent2", "karma": 50}
        ]
        posts = [
            {"post_external_id": "1", "comment_count": 5},
            {"post_external_id": "2", "comment_count": 3}
        ]
        submolts = [
            {"name": "general", "post_count": 100},
            {"name": "tech", "post_count": 50}
        ]
        comments = [
            {"comment_external_id": "c1"},
            {"comment_external_id": "c2"},
            {"comment_external_id": "c3"}
        ]
        
        stats = extract_site_stats(agents, posts, submolts, comments)
        
        assert stats.total_agents == 2
        assert stats.total_posts == 2
        assert stats.total_submolts == 2
        assert stats.total_comments == 3
        assert stats.avg_karma_per_agent == 75.0
    
    def test_handles_empty_lists(self):
        """Should handle empty input lists."""
        from scraper.extractors import extract_site_stats
        
        stats = extract_site_stats([], [], [], [])
        
        assert stats.total_agents == 0
        assert stats.total_posts == 0
        assert stats.avg_comments_per_post == 0
    
    def test_computes_top_entities(self):
        """Should compute top entities by metrics."""
        from scraper.extractors import extract_site_stats
        
        agents = [
            {"handle": "TopAgent", "karma": 500},
            {"handle": "LowAgent", "karma": 10}
        ]
        submolts = [
            {"name": "popular", "post_count": 1000},
            {"name": "quiet", "post_count": 5}
        ]
        
        stats = extract_site_stats(agents, [], submolts, [])
        
        assert stats.top_agents_by_karma[0]["handle"] == "TopAgent"
        assert stats.top_submolts_by_posts[0]["name"] == "popular"
    
    def test_to_dict(self):
        """Should serialize to dict correctly."""
        from scraper.extractors import extract_site_stats
        
        stats = extract_site_stats(
            [{"handle": "A", "karma": 10}],
            [{"post_external_id": "1"}],
            [{"name": "m"}],
            []
        )
        
        d = stats.to_dict()
        
        assert isinstance(d, dict)
        assert "snapshot_at" in d
        assert "total_agents" in d
        assert d["total_agents"] == 1

