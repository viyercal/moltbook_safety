# Integration Test Suite

This document describes how to run and interpret the real-world integration tests for Molt Observatory.

## Overview

The integration test suite performs **actual API calls** to:
- **moltbook.com** - Scrapes real posts, agents, submolts, and comments
- **OpenRouter** - Runs real LLM evaluations with your API key

All test output is saved to a labeled directory for inspection.

## Prerequisites

### 1. Environment Configuration

All configuration is loaded from your `.env` file via `python-dotenv`. **No environment variable exports are required.**

Create or update your `.env` file in the project root:

```bash
# Required
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Optional - defaults shown
OPENROUTER_MODEL=google/gemini-2.5-flash-lite
INTEGRATION_POST_LIMIT=5
INTEGRATION_AGENT_LIMIT=5
INTEGRATION_SUBMOLT_LIMIT=3
INTEGRATION_EVAL_LIMIT=3
```

### 2. Python Dependencies

Ensure all dependencies are installed:

```bash
cd molt-observatory
pip install -r ../requirements.txt
```

## Running the Tests

### Run All Integration Tests

```bash
cd molt-observatory
python -m pytest tests/integration/ -v
```

### Run Specific Test Modules

```bash
# Only scraping tests
python -m pytest tests/integration/test_live_scraping.py -v

# Only database tests
python -m pytest tests/integration/test_database.py -v

# Only LLM evaluation tests (costs money!)
python -m pytest tests/integration/test_live_eval.py -v

# Only report generation tests
python -m pytest tests/integration/test_live_reports.py -v

# End-to-end pipeline test
python -m pytest tests/integration/test_e2e_pipeline.py -v
```

### Run with Extra Output

```bash
# Show print statements during execution
python -m pytest tests/integration/ -v -s

# Show detailed failure information
python -m pytest tests/integration/ -v --tb=long
```

## Test Output Directory

Each test run creates a **timestamped output directory**:

```
molt-observatory/test_runs/
└── 2026-01-31T12-30-45/
    ├── manifest.json           # Run metadata and final stats
    ├── raw/
    │   ├── posts.json          # All scraped posts
    │   ├── agents.json         # All scraped agents
    │   ├── submolts.json       # All scraped submolts
    │   ├── post_details/       # Individual post detail files
    │   │   ├── post_1_abc123.json
    │   │   └── post_2_def456.json
    │   └── test_*.json         # Test-specific outputs
    ├── transcripts/
    │   ├── transcripts.jsonl   # Built transcripts
    │   └── comment_transcripts.jsonl
    ├── evaluations/
    │   ├── test_single_eval.json
    │   ├── test_batch_evals.json
    │   └── test_agent_scores.json
    ├── database/
    │   ├── integration_test.db # SQLite database file
    │   └── test_*.json         # Database test outputs
    ├── reports/
    │   ├── dashboard.html
    │   ├── growth_report.html
    │   ├── leaderboard_agents.html
    │   └── *.json              # Report metadata
    └── e2e_pipeline/
        ├── summary.json        # Pipeline summary
        ├── pipeline_log.json   # Detailed execution log
        └── *.json/*.html       # All pipeline outputs
```

## Understanding the Output

### manifest.json

Contains run metadata:

```json
{
  "run_id": "2026-01-31T12-30-45",
  "started_at": "2026-01-31T12:30:45.123Z",
  "completed_at": "2026-01-31T12:32:15.456Z",
  "openrouter_model": "google/gemini-2.5-flash-lite",
  "database_stats": {
    "posts": 5,
    "agents": 5,
    "submolts": 3,
    "post_evaluations": 3
  }
}
```

### Scraped Data (raw/)

Raw JSON files containing exactly what was returned from the Moltbook API:

- `posts.json` - List of post objects
- `agents.json` - List of agent objects  
- `submolts.json` - List of submolt objects
- `post_details/` - Full post content including comments

### Transcripts (transcripts/)

Processed transcripts ready for evaluation:

- `transcripts.jsonl` - One JSON object per line, each representing a conversation
- Format includes: `transcript_id`, `post_id`, `messages[]`, `metadata`

### Evaluations (evaluations/)

LLM evaluation results:

```json
{
  "test": "test_judges_single_transcript",
  "transcript_id": "abc123...",
  "model": "google/gemini-2.5-flash-lite",
  "result": {
    "scores": {
      "harm_enablement": {"score": 2, "confidence": 0.8},
      "deception_or_evasion": {"score": 1, "confidence": 0.9}
    }
  }
}
```

### Database (database/)

- `integration_test.db` - SQLite database you can open with any SQL client
- Contains all scraped data and evaluation results
- Schema matches the production PostgreSQL design

### Reports (reports/)

Generated HTML reports with Plotly charts:

- `dashboard.html` - Main dashboard with stats
- `growth_report.html` - Growth over time charts
- `leaderboard_agents.html` - Agent safety rankings

## Test Categories

### 1. Scraping Tests (`test_live_scraping.py`)

Tests real HTTP requests to moltbook.com:

| Test | Description |
|------|-------------|
| `test_fetches_posts_list` | Fetches and validates posts from API |
| `test_fetches_post_detail` | Fetches full post with comments |
| `test_pagination_works` | Verifies pagination returns distinct results |
| `test_fetches_agents_list` | Fetches agent profiles |
| `test_fetches_agent_profile` | Fetches detailed agent info |
| `test_fetches_submolts_list` | Fetches submolt list |
| `test_fetches_submolt_detail` | Fetches submolt details |
| `test_search_returns_results` | Tests search API |

### 2. Database Tests (`test_database.py`)

Tests SQLite storage with real data:

| Test | Description |
|------|-------------|
| `test_creates_database_file` | Verifies DB file creation |
| `test_creates_all_tables` | Validates schema |
| `test_inserts_scraped_posts` | Stores real posts |
| `test_inserts_scraped_agents` | Stores real agents |
| `test_agent_upsert_works` | Tests update on duplicate |
| `test_rollback_on_error` | Verifies transaction safety |

### 3. Evaluation Tests (`test_live_eval.py`)

Tests real LLM calls (costs money!):

| Test | Description |
|------|-------------|
| `test_judges_single_transcript` | Evaluates one transcript |
| `test_judges_multiple_transcripts` | Batch evaluation (max 3) |
| `test_all_dimensions_scored` | Verifies all dimensions present |
| `test_scores_within_range` | Validates 0-10 scores |
| `test_stores_evaluation_in_db` | Stores results in SQLite |

### 4. Report Tests (`test_live_reports.py`)

Tests report generation:

| Test | Description |
|------|-------------|
| `test_generates_growth_html` | Creates growth chart |
| `test_generates_agent_leaderboard` | Creates agent rankings |
| `test_generates_main_dashboard` | Creates main dashboard |
| `test_creates_bar_chart` | Tests Plotly bar chart |
| `test_creates_heatmap` | Tests Plotly heatmap |

### 5. End-to-End Test (`test_e2e_pipeline.py`)

Runs the complete pipeline:

1. Scrape posts, agents, submolts
2. Build transcripts
3. Run LLM evaluations
4. Store in database
5. Aggregate agent scores
6. Generate reports

## Cost Considerations

**LLM evaluations cost money!** The tests are limited to:

- Max 3 LLM evaluations per run
- Uses the cheaper model by default (`gemini-2.5-flash-lite`)
- You can reduce limits in `.env`:

```bash
INTEGRATION_EVAL_LIMIT=1  # Only 1 eval
```

## Troubleshooting

### Tests Fail with "OPENROUTER_API_KEY not set"

Ensure your `.env` file exists and contains the key:

```bash
cat .env | grep OPENROUTER
```

### Tests Time Out

The Moltbook API has a 160-second timeout. If tests hang:

```bash
# Check if moltbook.com is accessible
curl https://www.moltbook.com/api/v1/posts?limit=1
```

### Database Errors

The SQLite database is created fresh each run. If you see schema errors:

```bash
# Delete old test runs
rm -rf molt-observatory/test_runs/
```

### No Output Directory Created

The output directory is created in `pytest_configure`. Ensure you're running from the correct directory:

```bash
cd molt-observatory
python -m pytest tests/integration/ -v
```

## Viewing Results

### Open HTML Reports

```bash
# macOS
open test_runs/*/reports/dashboard.html

# Linux
xdg-open test_runs/*/reports/dashboard.html
```

### Query the SQLite Database

```bash
# Using sqlite3
sqlite3 test_runs/*/database/integration_test.db

# See all tables
.tables

# Query posts
SELECT title, author_handle FROM posts LIMIT 5;
```

### Pretty-Print JSON

```bash
# View a JSON file
cat test_runs/*/raw/posts.json | python -m json.tool | less
```

## CI/CD Integration

For CI pipelines, you may want to:

1. Set environment variables in your CI config
2. Use a test API key with limited quota
3. Reduce limits to minimize costs

Example GitHub Actions:

```yaml
env:
  OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
  INTEGRATION_EVAL_LIMIT: 1
  INTEGRATION_POST_LIMIT: 2

steps:
  - name: Run Integration Tests
    run: |
      cd molt-observatory
      python -m pytest tests/integration/ -v
```

