# Architecture

## System Overview

Molt Observatory follows a **medallion architecture** (Bronze → Silver → Gold) common in data engineering, adapted for AI safety evaluation.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MOLT OBSERVATORY                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐          │
│  │   BRONZE     │      │   SILVER     │      │    GOLD      │          │
│  │   (Raw)      │ ──▶  │ (Transcripts)│ ──▶  │  (Evals)     │          │
│  └──────────────┘      └──────────────┘      └──────────────┘          │
│         │                     │                     │                   │
│         ▼                     ▼                     ▼                   │
│  posts_list.json       transcripts.jsonl      evals.jsonl              │
│  post_<id>.json                                aggregates.json          │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                         INFRASTRUCTURE                                  │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐          │
│  │ MoltbookAPI  │      │ OpenRouter   │      │  Airflow     │          │
│  │  (Scraper)   │      │  (LLM API)   │      │  (Scheduler) │          │
│  └──────────────┘      └──────────────┘      └──────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
molt-observatory/
├── run_pipeline.py           # CLI entrypoint
├── eval_orchestrator.py      # End-to-end orchestration
├── transcript_builder.py     # Post → Transcript conversion
├── judge_runner.py           # LLM safety scoring
├── openrouter_client.py      # OpenRouter API client
│
├── scraper/
│   ├── moltbook_api.py       # Moltbook HTTP client + rate limiting
│   ├── extractors.py         # JSON → canonical format converters
│   ├── moltbook_playwright.py # (Alternative) Browser-based scraping
│   └── rate_limit.py         # Token bucket implementation
│
├── pipelines/                # (Placeholder) Modular pipeline steps
│   ├── eval_orchestrator.py
│   ├── judge_runner.py
│   └── transcript_builder.py
│
├── airflow/
│   └── dags/
│       └── moltbook_ingest_eval.py  # Scheduled DAG
│
├── analytics/
│   └── materialized_views.sql       # SQL aggregations
│
├── ops/
│   ├── docker-compose.yml
│   ├── Dockerfile.airflow
│   └── Dockerfile.worker
│
└── runs/                     # Output artifacts
    └── <timestamp>/
        ├── raw/              # Bronze layer
        ├── silver/           # Silver layer
        └── gold/             # Gold layer
```

## Component Details

### 1. Scraper Layer (`scraper/`)

**Purpose**: Fetch data from moltbook.com's public API

| File | Responsibility |
|------|----------------|
| `moltbook_api.py` | HTTP client with rate limiting, retry logic, and error handling |
| `extractors.py` | Transform API responses into canonical schemas |
| `rate_limit.py` | Token bucket algorithm (0.5 req/sec, burst 3) |

**Key Features**:
- Token bucket rate limiting to avoid overloading moltbook
- Exponential backoff on 429/5xx errors
- Lenient JSON parsing for edge cases
- Stable hashing for deduplication

```python
# Example: MoltbookAPI usage
api = MoltbookAPI()
response = api.get_json("/api/v1/posts", params={"limit": 30, "sort": "new"})
posts = extract_posts_from_list(response.json_body)
```

### 2. Transcript Builder (`transcript_builder.py`)

**Purpose**: Convert raw post+comment trees into structured transcripts for evaluation

**Data Model**:
```python
@dataclass
class Transcript:
    transcript_id: str      # SHA256 hash of canonical content
    post_id: str            # External ID from moltbook
    built_at: str           # ISO timestamp
    permalink: str          # URL to original post
    community: str          # Submolt (subreddit-like)
    messages: List[Dict]    # Ordered list of post + comments
    metadata: Dict          # Author info, vote counts, etc.
```

**Message Structure**:
```json
{
  "kind": "post|comment",
  "id": "uuid",
  "parent_id": "uuid|null",
  "author": "handle",
  "created_at": "ISO8601",
  "text": "content",
  "score": 42,
  "upvotes": 50,
  "downvotes": 8
}
```

### 3. Judge Runner (`judge_runner.py`)

**Purpose**: Score transcripts on safety dimensions using LLM judges

**Key Classes**:
- `LLMJudgeRunner` - Handles LLM calls with retry/repair logic
- `run_judges()` - Batch scoring across transcripts and models

**Hardening Features**:
- Extracts content from both `message.content` and `message.reasoning` (Gemini quirk)
- Handles truncated JSON with automatic repair calls
- Schema coercion for missing dimension keys
- Configurable retry with exponential backoff

### 4. Eval Orchestrator (`eval_orchestrator.py`)

**Purpose**: End-to-end pipeline coordination

**Pipeline Steps**:
1. Fetch post list from moltbook API
2. Fetch detail pages for each post
3. Build transcripts from post details
4. Run LLM judges on all transcripts
5. Compute aggregates (mean, p95, elicitation rate)
6. Write artifacts to timestamped output directory

### 5. OpenRouter Client (`openrouter_client.py`)

**Purpose**: Minimal OpenRouter API client with retry logic

**Features**:
- OpenAI-compatible API format
- Configurable model selection via `OPENROUTER_MODEL` env var
- Exponential backoff on transient errors (429, 502, 503, 408)
- Response latency tracking

## Data Flow

```
                    Moltbook.com
                         │
                         ▼
              ┌──────────────────┐
              │   /api/v1/posts  │  ◀── Scrape posts
              │   /api/v1/posts/ │      + comments
              │      {uuid}      │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │   Raw JSON       │  Bronze Layer
              │   (posts_list,   │
              │    post_{id})    │
              └────────┬─────────┘
                       │
                       ▼ extract_post_detail()
                       ▼ flatten_comments_tree()
              ┌──────────────────┐
              │   Transcripts    │  Silver Layer
              │   (JSONL)        │
              └────────┬─────────┘
                       │
                       ▼ LLMJudgeRunner.score_transcript()
              ┌──────────────────┐
              │   Evaluations    │  Gold Layer
              │   + Aggregates   │
              └──────────────────┘
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | ✅ Yes | - | OpenRouter API key |
| `OPENROUTER_MODEL` | No | `openai/gpt-4o-mini` | Model for LLM judge |
| `OPENROUTER_REFERER` | No | - | HTTP Referer header |
| `OPENROUTER_TITLE` | No | - | X-Title header |
| `JUDGE_MAX_ATTEMPTS` | No | `3` | Retry attempts for judge calls |
| `JUDGE_MAX_TOKENS` | No | `1800` | Initial max tokens for judge output |
| `JUDGE_RETRY_BACKOFF_S` | No | `0.25` | Base backoff between retries |

## Output Artifacts

Each pipeline run creates a timestamped directory:

```
runs/20260130T175721Z/
├── raw/
│   ├── posts_list.json      # Full API response
│   └── post_{uuid}.json     # Individual post details
├── silver/
│   └── transcripts.jsonl    # One transcript per line
└── gold/
    ├── evals.jsonl          # One evaluation per line
    └── aggregates.json      # Summary statistics
```

### Aggregates Schema

```json
{
  "run_id": "20260130T175721Z",
  "n_posts": 10,
  "n_transcripts": 10,
  "dimensions": {
    "harm_enablement": {
      "mean": 0.0,
      "p95": 0,
      "n": 10,
      "elicitation_rate_ge_7": 0.0
    },
    ...
  }
}
```

