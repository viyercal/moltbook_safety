"""
Integration test fixtures for real-world testing.

All configuration is loaded via python-dotenv from the .env file.
No environment variable exports are required.

Output is saved to: molt-observatory/test_runs/{timestamp}/
"""

from __future__ import annotations
import json
import os
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import pytest
from tqdm import tqdm

# =============================================================================
# Environment Setup via dotenv
# =============================================================================

# Add molt-observatory to path FIRST
import sys
MOLT_OBSERVATORY_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(MOLT_OBSERVATORY_DIR))

# Load environment from .env file
from dotenv import load_dotenv

# Load from project root .env
ENV_FILE = MOLT_OBSERVATORY_DIR.parent / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
    print(f"\n🔐 Loaded environment from {ENV_FILE}")
else:
    # Try molt-observatory/.env as fallback
    ENV_FILE_ALT = MOLT_OBSERVATORY_DIR / ".env"
    if ENV_FILE_ALT.exists():
        load_dotenv(ENV_FILE_ALT)
        print(f"\n🔐 Loaded environment from {ENV_FILE_ALT}")

# Also set RUN_INTEGRATION_TESTS automatically when running integration tests
os.environ["RUN_INTEGRATION_TESTS"] = "1"


# =============================================================================
# Validate Required Environment Variables
# =============================================================================

def _validate_env():
    """Validate required environment variables are set."""
    required = {
        "OPENROUTER_API_KEY": "OpenRouter API key for LLM evaluations",
    }
    
    missing = []
    for var, desc in required.items():
        if not os.environ.get(var):
            missing.append(f"  - {var}: {desc}")
    
    if missing:
        raise EnvironmentError(
            f"\n❌ Missing required environment variables:\n"
            + "\n".join(missing) +
            f"\n\nPlease set them in your .env file at: {ENV_FILE}"
        )


# Validate on import
_validate_env()


# =============================================================================
# Test Run Output Directory
# =============================================================================

# Create a timestamped output directory for this test run
TEST_RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
TEST_RUNS_DIR = MOLT_OBSERVATORY_DIR / "test_runs"
CURRENT_RUN_DIR = TEST_RUNS_DIR / TEST_RUN_TIMESTAMP


def _create_run_directory() -> Path:
    """Create the test run output directory structure."""
    run_dir = CURRENT_RUN_DIR
    
    # Create directory structure
    subdirs = [
        "raw",
        "raw/post_details",
        "transcripts",
        "evaluations",
        "database",
        "reports",
    ]
    
    for subdir in subdirs:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Create initial manifest
    manifest = {
        "run_id": TEST_RUN_TIMESTAMP,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "env_file": str(ENV_FILE),
        "openrouter_model": os.environ.get("OPENROUTER_MODEL", "default"),
        "tests": [],
        "stats": {},
    }
    
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n📁 Test output directory: {run_dir}")
    
    return run_dir


# =============================================================================
# Override Parent Fixtures for Real-World Tests
# =============================================================================

@pytest.fixture(autouse=True)
def setup_integration_environment(monkeypatch):
    """
    Override the parent conftest's setup_test_environment.
    
    For integration tests, we use REAL API keys and models from .env,
    not mock values.
    """
    # Environment already loaded via dotenv at module level
    pass


# =============================================================================
# Pytest Hooks
# =============================================================================

def pytest_configure(config):
    """Register custom markers and create output directory."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring network"
    )
    
    # Create output directory
    _create_run_directory()


def pytest_sessionfinish(session, exitstatus):
    """Update manifest with final stats when tests complete."""
    manifest_path = CURRENT_RUN_DIR / "manifest.json"
    
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            
            manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
            manifest["exit_status"] = exitstatus
            manifest["tests_collected"] = getattr(session, 'testscollected', 0)
            
            # Calculate passed tests safely
            failed = getattr(session, 'testsfailed', 0)
            collected = getattr(session, 'testscollected', 0)
            manifest["tests_failed"] = failed
            manifest["tests_passed"] = collected - failed
            
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            
            print(f"\n\n{'='*60}")
            print(f"📊 TEST RUN COMPLETE")
            print(f"{'='*60}")
            print(f"   Output directory: {CURRENT_RUN_DIR}")
            print(f"   Manifest: {manifest_path}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"\n⚠️ Failed to update manifest: {e}")


# =============================================================================
# Configuration
# =============================================================================

@pytest.fixture(scope="session")
def integration_config() -> Dict[str, Any]:
    """
    Configuration for integration tests.
    
    All values come from environment or use sensible defaults.
    """
    return {
        "posts_limit": int(os.environ.get("INTEGRATION_POST_LIMIT", "5")),
        "agents_limit": int(os.environ.get("INTEGRATION_AGENT_LIMIT", "5")),
        "submolts_limit": int(os.environ.get("INTEGRATION_SUBMOLT_LIMIT", "3")),
        "comments_limit": int(os.environ.get("INTEGRATION_COMMENT_LIMIT", "10")),
        "eval_limit": int(os.environ.get("INTEGRATION_EVAL_LIMIT", "3")),
        "rate_per_sec": float(os.environ.get("INTEGRATION_RATE_LIMIT", "0.5")),
        "burst": int(os.environ.get("INTEGRATION_BURST", "2")),
        "openrouter_model": os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash-lite"),
    }


@pytest.fixture(scope="session")
def test_output_dir() -> Path:
    """Return the current test run output directory."""
    return CURRENT_RUN_DIR


# =============================================================================
# Real API Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def real_moltbook_api(integration_config):
    """
    Real MoltbookAPI instance for live requests.
    
    Uses rate limiting to avoid overwhelming the API.
    """
    from scraper.moltbook_api import MoltbookAPI
    
    api = MoltbookAPI(
        rate_per_sec=integration_config["rate_per_sec"],
        burst=integration_config["burst"],
    )
    
    return api


@pytest.fixture(scope="session")
def real_openrouter_client():
    """
    Real OpenRouterClient for LLM evaluations.
    
    API key is loaded from .env via dotenv.
    """
    from openrouter_client import OpenRouterClient
    
    # Key is already validated at import time
    return OpenRouterClient()


# =============================================================================
# Output Directory Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def integration_temp_dir() -> Path:
    """
    Return the current test run output directory.
    
    This is a persistent directory that is NOT cleaned up after tests.
    All test artifacts are saved here for inspection.
    """
    return CURRENT_RUN_DIR


@pytest.fixture(scope="session")
def integration_db_path(integration_temp_dir) -> Path:
    """Path to the SQLite database in the output directory."""
    return integration_temp_dir / "database" / "integration_test.db"


# =============================================================================
# Scraped Data Fixtures (with automatic saving)
# =============================================================================

@pytest.fixture(scope="session")
def scraped_posts(real_moltbook_api, integration_config, test_output_dir) -> List[Dict[str, Any]]:
    """
    Fetch real posts from moltbook.com and save to output directory.
    """
    from scraper.extractors import extract_posts_from_list
    
    limit = integration_config["posts_limit"]
    
    resp = real_moltbook_api.get_json("/api/v1/posts", params={"limit": limit, "sort": "new"})
    posts = extract_posts_from_list(resp.json_body)[:limit]
    
    # Save to output directory
    output_path = test_output_dir / "raw" / "posts.json"
    with open(output_path, "w") as f:
        json.dump(posts, f, indent=2, default=str)
    
    print(f"\n📥 Fetched {len(posts)} posts → {output_path}")
    
    return posts


@pytest.fixture(scope="session")
def scraped_post_details(real_moltbook_api, scraped_posts, test_output_dir) -> List[Dict[str, Any]]:
    """
    Fetch full post details with comments and save each to output directory.
    """
    details = []
    details_dir = test_output_dir / "raw" / "post_details"
    
    pbar = tqdm(scraped_posts, desc="📥 Fetching post details", unit="post", leave=True)
    for i, post in enumerate(pbar):
        pid = post["post_external_id"]
        pbar.set_postfix({"post": pid[:8]})
        try:
            resp = real_moltbook_api.get_json(f"/api/v1/posts/{pid}")
            detail = resp.json_body
            details.append(detail)
            
            # Save individual detail
            detail_path = details_dir / f"post_{i+1}_{pid[:8]}.json"
            with open(detail_path, "w") as f:
                json.dump(detail, f, indent=2, default=str)
                
        except Exception as e:
            # Log error but continue - don't skip
            error_path = details_dir / f"post_{i+1}_{pid[:8]}_ERROR.txt"
            with open(error_path, "w") as f:
                f.write(f"Error fetching post {pid}: {e}")
    
    # Save summary
    summary_path = test_output_dir / "raw" / "post_details_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total_requested": len(scraped_posts),
            "total_fetched": len(details),
            "post_ids": [d.get("post", {}).get("id") for d in details],
        }, f, indent=2)
    
    print(f"\n📥 Fetched {len(details)} post details → {details_dir}")
    
    return details


@pytest.fixture(scope="session")
def scraped_agents(real_moltbook_api, integration_config, test_output_dir) -> List[Dict[str, Any]]:
    """
    Fetch real agents from moltbook.com and save to output directory.
    """
    from scraper.extractors import extract_agents_from_recent
    
    limit = integration_config["agents_limit"]
    
    resp = real_moltbook_api.get_json("/api/v1/agents/recent", params={"limit": limit})
    agents = extract_agents_from_recent(resp.json_body)[:limit]
    
    # Save to output directory
    output_path = test_output_dir / "raw" / "agents.json"
    with open(output_path, "w") as f:
        json.dump(agents, f, indent=2, default=str)
    
    print(f"\n📥 Fetched {len(agents)} agents → {output_path}")
    
    return agents


@pytest.fixture(scope="session")
def scraped_submolts(real_moltbook_api, integration_config, test_output_dir) -> List[Dict[str, Any]]:
    """
    Fetch real submolts from moltbook.com and save to output directory.
    """
    from scraper.extractors import extract_submolts_from_list
    
    limit = integration_config["submolts_limit"]
    
    resp = real_moltbook_api.get_json("/api/v1/submolts", params={"limit": limit})
    submolts = extract_submolts_from_list(resp.json_body)[:limit]
    
    # Save to output directory
    output_path = test_output_dir / "raw" / "submolts.json"
    with open(output_path, "w") as f:
        json.dump(submolts, f, indent=2, default=str)
    
    print(f"\n📥 Fetched {len(submolts)} submolts → {output_path}")
    
    return submolts


# =============================================================================
# Transcript Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def built_transcripts(scraped_post_details, test_output_dir) -> List[Any]:
    """
    Build transcripts from scraped post details and save to output directory.
    """
    from transcript_builder import build_transcript_from_post_detail
    
    transcripts = []
    errors = []
    
    pbar = tqdm(scraped_post_details, desc="📝 Building transcripts", unit="transcript", leave=True)
    for i, detail in enumerate(pbar):
        try:
            t = build_transcript_from_post_detail(detail)
            transcripts.append(t)
            pbar.set_postfix({"built": len(transcripts)})
        except Exception as e:
            errors.append({"index": i, "error": str(e)})
            pbar.set_postfix({"errors": len(errors)})
    
    # Save transcripts as JSONL
    output_path = test_output_dir / "transcripts" / "transcripts.jsonl"
    with open(output_path, "w") as f:
        for t in transcripts:
            f.write(json.dumps(asdict(t), default=str) + "\n")
    
    # Save errors if any
    if errors:
        errors_path = test_output_dir / "transcripts" / "transcript_errors.json"
        with open(errors_path, "w") as f:
            json.dump(errors, f, indent=2)
    
    print(f"\n📝 Built {len(transcripts)} transcripts → {output_path}")
    
    return transcripts


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def sandbox_db(integration_db_path, test_output_dir):
    """
    Initialize and return the sandbox SQLite database.
    
    Database file is saved to the output directory for inspection.
    """
    from tests.integration.sandbox_db import SandboxDatabase
    
    db = SandboxDatabase(integration_db_path)
    db.initialize()
    
    print(f"\n📦 Database initialized → {integration_db_path}")
    
    yield db
    
    # Don't close - leave DB file for inspection
    # Copy final stats to manifest
    try:
        stats = db.get_stats()
        manifest_path = test_output_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            manifest["database_stats"] = stats
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
    except Exception:
        pass
    
    db.close()


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def save_artifact(test_output_dir):
    """
    Helper to save test artifacts to the output directory.
    """
    def _save(name: str, data: Any, subdir: str = "raw") -> Path:
        path = test_output_dir / subdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, (dict, list)):
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        elif hasattr(data, '__dataclass_fields__'):
            with open(path, "w") as f:
                json.dump(asdict(data), f, indent=2, default=str)
        else:
            with open(path, "w") as f:
                f.write(str(data))
        
        return path
    
    return _save


@pytest.fixture
def save_evaluation(test_output_dir):
    """
    Helper to save evaluation results to the evaluations directory.
    """
    def _save(name: str, data: Any) -> Path:
        path = test_output_dir / "evaluations" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        return path
    
    return _save


@pytest.fixture
def save_report(test_output_dir):
    """
    Helper to save reports to the reports directory.
    """
    def _save(name: str, content: str) -> Path:
        path = test_output_dir / "reports" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            f.write(content)
        
        return path
    
    return _save
