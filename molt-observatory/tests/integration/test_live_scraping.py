"""
Integration tests for live scraping from moltbook.com.

These tests make real HTTP requests to the Moltbook API.
All scraped data is saved to the test output directory.

Run with:
    cd molt-observatory
    python -m pytest tests/integration/test_live_scraping.py -v
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest


@pytest.mark.integration
class TestLivePostsScraping:
    """Tests for fetching real posts from moltbook.com."""
    
    def test_fetches_posts_list(self, real_moltbook_api, integration_config, test_output_dir):
        """Should fetch a list of real posts."""
        from scraper.extractors import extract_posts_from_list
        
        limit = integration_config["posts_limit"]
        
        resp = real_moltbook_api.get_json("/api/v1/posts", params={"limit": limit, "sort": "new"})
        
        assert resp.status == 200, f"API returned status {resp.status}"
        assert resp.json_body is not None, "API returned no body"
        
        posts = extract_posts_from_list(resp.json_body)
        
        assert len(posts) > 0, "No posts returned from API"
        assert len(posts) <= limit, f"Got more posts than requested: {len(posts)} > {limit}"
        
        # Verify structure
        first_post = posts[0]
        assert "post_external_id" in first_post, "Post missing post_external_id"
        assert "title" in first_post, "Post missing title"
        assert "author_handle" in first_post, "Post missing author_handle"
        
        # Save test-specific output
        output_path = test_output_dir / "raw" / "test_posts_list.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_fetches_posts_list",
                "posts_count": len(posts),
                "posts": posts,
            }, f, indent=2, default=str)
        
        print(f"\n✅ Fetched {len(posts)} posts")
        print(f"   First post: {first_post.get('title', 'No title')[:50]}...")
        print(f"   Saved to: {output_path}")
    
    def test_fetches_post_detail(self, real_moltbook_api, scraped_posts, test_output_dir):
        """Should fetch full post detail with comments."""
        from scraper.extractors import extract_post_detail, flatten_comments_tree
        
        assert len(scraped_posts) > 0, "No posts available to fetch details for"
        
        post = scraped_posts[0]
        post_id = post["post_external_id"]
        
        resp = real_moltbook_api.get_json(f"/api/v1/posts/{post_id}")
        
        assert resp.status == 200, f"API returned status {resp.status}"
        
        detail = extract_post_detail(resp.json_body)
        
        assert detail is not None, "Failed to extract post detail"
        assert detail["post"]["post_external_id"] == post_id, "Post ID mismatch"
        
        # Check for comments
        comments = flatten_comments_tree(detail["comments"], post_id)
        
        # Save test output
        output_path = test_output_dir / "raw" / "test_post_detail.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_fetches_post_detail",
                "post_id": post_id,
                "detail": detail,
                "comments_count": len(comments),
                "comments": comments,
            }, f, indent=2, default=str)
        
        print(f"\n✅ Fetched post detail: {detail['post'].get('title', 'No title')[:40]}...")
        print(f"   Comments: {len(comments)}")
        print(f"   Saved to: {output_path}")
    
    def test_pagination_works(self, real_moltbook_api, test_output_dir):
        """Should be able to paginate through posts."""
        from scraper.extractors import extract_posts_from_list
        
        # Fetch page 1
        resp1 = real_moltbook_api.get_json("/api/v1/posts", params={"limit": 2, "offset": 0})
        posts1 = extract_posts_from_list(resp1.json_body)
        
        # Fetch page 2
        resp2 = real_moltbook_api.get_json("/api/v1/posts", params={"limit": 2, "offset": 2})
        posts2 = extract_posts_from_list(resp2.json_body)
        
        assert len(posts1) > 0, "Page 1 returned no posts"
        assert len(posts2) > 0, "Page 2 returned no posts"
        
        # Posts should be different
        ids1 = {p["post_external_id"] for p in posts1}
        ids2 = {p["post_external_id"] for p in posts2}
        
        assert ids1.isdisjoint(ids2), "Pagination returned duplicate posts"
        
        # Save test output
        output_path = test_output_dir / "raw" / "test_pagination.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_pagination_works",
                "page1_count": len(posts1),
                "page1_ids": list(ids1),
                "page2_count": len(posts2),
                "page2_ids": list(ids2),
                "page1_posts": posts1,
                "page2_posts": posts2,
            }, f, indent=2, default=str)
        
        print(f"\n✅ Pagination works: page 1 has {len(posts1)}, page 2 has {len(posts2)} posts")
        print(f"   Saved to: {output_path}")


@pytest.mark.integration
class TestLiveAgentsScraping:
    """Tests for fetching real agents from moltbook.com."""
    
    def test_fetches_agents_list(self, real_moltbook_api, integration_config, test_output_dir):
        """Should fetch a list of real agents."""
        from scraper.extractors import extract_agents_from_recent
        
        limit = integration_config["agents_limit"]
        
        resp = real_moltbook_api.get_json("/api/v1/agents/recent", params={"limit": limit})
        
        assert resp.status == 200, f"API returned status {resp.status}"
        
        agents = extract_agents_from_recent(resp.json_body)
        
        assert len(agents) > 0, "No agents returned from API"
        
        # Verify structure
        first_agent = agents[0]
        assert "agent_external_id" in first_agent, "Agent missing agent_external_id"
        assert "handle" in first_agent, "Agent missing handle"
        
        # Save test output
        output_path = test_output_dir / "raw" / "test_agents_list.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_fetches_agents_list",
                "agents_count": len(agents),
                "agents": agents,
            }, f, indent=2, default=str)
        
        print(f"\n✅ Fetched {len(agents)} agents")
        for agent in agents[:3]:
            print(f"   - {agent.get('handle')}: karma={agent.get('karma')}")
        print(f"   Saved to: {output_path}")
    
    def test_fetches_agent_profile(self, real_moltbook_api, scraped_agents, test_output_dir):
        """Should fetch detailed agent profile."""
        from scraper.extractors import extract_agent_profile
        
        assert len(scraped_agents) > 0, "No agents available to fetch profiles for"
        
        agent = scraped_agents[0]
        handle = agent["handle"]
        
        resp = real_moltbook_api.get_json("/api/v1/agents/profile", params={"name": handle})
        
        assert resp.status == 200, f"API returned status {resp.status}"
        
        profile = extract_agent_profile(resp.json_body)
        
        assert profile is not None, "Failed to extract agent profile"
        assert profile["handle"] == handle, f"Handle mismatch: expected {handle}"
        
        # Save test output
        output_path = test_output_dir / "raw" / "test_agent_profile.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_fetches_agent_profile",
                "handle": handle,
                "profile": profile,
            }, f, indent=2, default=str)
        
        print(f"\n✅ Fetched agent profile: {handle}")
        print(f"   Karma: {profile.get('karma')}, Followers: {profile.get('follower_count')}")
        print(f"   Saved to: {output_path}")


@pytest.mark.integration
class TestLiveSubmoltsScraping:
    """Tests for fetching real submolts from moltbook.com."""
    
    def test_fetches_submolts_list(self, real_moltbook_api, integration_config, test_output_dir):
        """Should fetch a list of real submolts."""
        from scraper.extractors import extract_submolts_from_list
        
        limit = integration_config["submolts_limit"]
        
        resp = real_moltbook_api.get_json("/api/v1/submolts", params={"limit": limit})
        
        assert resp.status == 200, f"API returned status {resp.status}"
        
        submolts = extract_submolts_from_list(resp.json_body)
        
        assert len(submolts) > 0, "No submolts returned from API"
        
        # Verify structure
        first_submolt = submolts[0]
        assert "submolt_external_id" in first_submolt, "Submolt missing submolt_external_id"
        assert "name" in first_submolt, "Submolt missing name"
        
        # Save test output
        output_path = test_output_dir / "raw" / "test_submolts_list.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_fetches_submolts_list",
                "submolts_count": len(submolts),
                "submolts": submolts,
            }, f, indent=2, default=str)
        
        print(f"\n✅ Fetched {len(submolts)} submolts")
        for submolt in submolts[:3]:
            print(f"   - m/{submolt.get('name')}: {submolt.get('subscriber_count')} subscribers")
        print(f"   Saved to: {output_path}")
    
    def test_fetches_submolt_detail(self, real_moltbook_api, scraped_submolts, test_output_dir):
        """Should fetch detailed submolt info."""
        from scraper.extractors import extract_submolt_detail
        
        assert len(scraped_submolts) > 0, "No submolts available to fetch details for"
        
        submolt = scraped_submolts[0]
        name = submolt["name"]
        
        resp = real_moltbook_api.get_json(f"/api/v1/submolts/{name}")
        
        assert resp.status == 200, f"API returned status {resp.status}"
        
        detail = extract_submolt_detail(resp.json_body)
        
        assert detail is not None, "Failed to extract submolt detail"
        assert detail["name"] == name, f"Name mismatch: expected {name}"
        
        # Save test output
        output_path = test_output_dir / "raw" / "test_submolt_detail.json"
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_fetches_submolt_detail",
                "name": name,
                "detail": detail,
            }, f, indent=2, default=str)
        
        print(f"\n✅ Fetched submolt detail: m/{name}")
        print(f"   Posts: {detail.get('post_count')}, Subscribers: {detail.get('subscriber_count')}")
        print(f"   Saved to: {output_path}")


@pytest.mark.integration
class TestLiveSearch:
    """Tests for search functionality."""
    
    def test_search_returns_results(self, real_moltbook_api, test_output_dir):
        """Should return search results or handle API limitations."""
        # Search API may have issues - we test it but handle errors gracefully
        try:
            resp = real_moltbook_api.get_json("/api/v1/search", params={"q": "AI", "limit": 5})
        except Exception as e:
            # Save error info and mark as expected failure
            output_path = test_output_dir / "raw" / "test_search_error.json"
            with open(output_path, "w") as f:
                json.dump({
                    "test": "test_search_returns_results",
                    "status": "api_error",
                    "error": str(e),
                }, f, indent=2)
            pytest.xfail(f"Search API unavailable: {e}")
        
        # Save whatever we got
        output_path = test_output_dir / "raw" / "test_search_results.json"
        
        if resp.status != 200:
            # API returned error - save and xfail
            with open(output_path, "w") as f:
                json.dump({
                    "test": "test_search_returns_results",
                    "status": "http_error",
                    "http_status": resp.status,
                    "body": resp.json_body,
                }, f, indent=2, default=str)
            pytest.xfail(f"Search API returned status {resp.status}")
        
        body = resp.json_body
        
        # Save successful results
        with open(output_path, "w") as f:
            json.dump({
                "test": "test_search_returns_results",
                "status": "success",
                "query": "AI",
                "results": body,
            }, f, indent=2, default=str)
        
        print(f"\n✅ Search for 'AI' completed")
        print(f"   Saved to: {output_path}")
        if isinstance(body, dict):
            for key in ["posts", "agents", "submolts"]:
                if body.get(key):
                    print(f"   {key}: {len(body[key])} results")


@pytest.mark.integration
class TestVerifyScrapedData:
    """Tests to verify scraped data was saved correctly by fixtures."""
    
    def test_posts_saved_by_fixture(self, scraped_posts, test_output_dir):
        """Should verify posts were saved by fixture."""
        output_path = test_output_dir / "raw" / "posts.json"
        
        assert output_path.exists(), f"Posts file not created by fixture: {output_path}"
        
        # Reload and verify
        with open(output_path, "r") as f:
            loaded = json.load(f)
        
        assert len(loaded) == len(scraped_posts), f"Count mismatch: file has {len(loaded)}, fixture has {len(scraped_posts)}"
        
        print(f"\n✅ Verified {len(scraped_posts)} posts at {output_path}")
    
    def test_agents_saved_by_fixture(self, scraped_agents, test_output_dir):
        """Should verify agents were saved by fixture."""
        output_path = test_output_dir / "raw" / "agents.json"
        
        assert output_path.exists(), f"Agents file not created by fixture: {output_path}"
        
        with open(output_path, "r") as f:
            loaded = json.load(f)
        
        assert len(loaded) == len(scraped_agents), f"Count mismatch: file has {len(loaded)}, fixture has {len(scraped_agents)}"
        
        print(f"\n✅ Verified {len(scraped_agents)} agents at {output_path}")
    
    def test_post_details_saved_by_fixture(self, scraped_post_details, test_output_dir):
        """Should verify post details were saved by fixture."""
        details_dir = test_output_dir / "raw" / "post_details"
        
        assert details_dir.exists(), f"Post details directory not created: {details_dir}"
        
        saved_count = len(list(details_dir.glob("post_*.json")))
        
        assert saved_count == len(scraped_post_details), f"File count mismatch: {saved_count} files, {len(scraped_post_details)} details"
        
        print(f"\n✅ Verified {saved_count} post detail files at {details_dir}")
