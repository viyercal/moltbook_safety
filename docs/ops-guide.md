# Operational Guide

## Quick Start

### Prerequisites

- Python 3.9+
- OpenRouter API key
- Network access to moltbook.com

### Installation

```bash
cd molt-observatory

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r ../requirements.txt
```

### Configuration

Create a `.env` file in `molt-observatory/`:

```bash
# Required
OPENROUTER_API_KEY=sk-or-v1-...

# Optional
OPENROUTER_MODEL=google/gemini-3-flash-preview  # Default: openai/gpt-4o-mini
OPENROUTER_REFERER=https://github.com/your-org/molt-observatory
OPENROUTER_TITLE=Molt Observatory Safety Research

# Judge tuning
JUDGE_MAX_ATTEMPTS=3
JUDGE_MAX_TOKENS=1800
JUDGE_RETRY_BACKOFF_S=0.25
```

### Running the Pipeline

```bash
# Basic run
python run_pipeline.py --limit 10

# Full options
python run_pipeline.py \
  --limit 30 \      # Number of posts to fetch
  --sort new \      # Sort order: new, hot, top
  --out runs        # Output directory
```

### Output Structure

```
runs/20260130T175721Z/
├── raw/
│   ├── posts_list.json      # API response
│   └── post_{uuid}.json     # Individual posts
├── silver/
│   └── transcripts.jsonl    # Structured transcripts
└── gold/
    ├── evals.jsonl          # Safety evaluations
    └── aggregates.json      # Summary statistics
```

## Scheduled Execution

### Using Cron

```bash
# Run every 6 hours
0 */6 * * * cd /path/to/molt-observatory && /path/to/venv/bin/python run_pipeline.py --limit 50 >> /var/log/molt-observatory.log 2>&1
```

### Using Airflow (Planned)

The `airflow/dags/moltbook_ingest_eval.py` DAG is a skeleton for scheduled execution:

```python
# Example DAG structure (to be implemented)
from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG('moltbook_ingest_eval', schedule_interval='0 */6 * * *') as dag:
    scrape = PythonOperator(task_id='scrape', python_callable=scrape_posts)
    build = PythonOperator(task_id='build', python_callable=build_transcripts)
    judge = PythonOperator(task_id='judge', python_callable=run_judges)
    aggregate = PythonOperator(task_id='aggregate', python_callable=compute_aggregates)
    
    scrape >> build >> judge >> aggregate
```

## Docker Deployment

### Build Images

```bash
cd molt-observatory/ops

# Build worker image
docker build -f Dockerfile.worker -t molt-observatory:latest ..

# Build Airflow image (if using)
docker build -f Dockerfile.airflow -t molt-observatory-airflow:latest ..
```

### Run Container

```bash
docker run --rm \
  -e OPENROUTER_API_KEY=sk-or-v1-... \
  -v $(pwd)/runs:/app/runs \
  molt-observatory:latest \
  python run_pipeline.py --limit 30
```

### Docker Compose (Planned)

```yaml
# ops/docker-compose.yml structure
version: '3.8'
services:
  worker:
    build:
      context: ..
      dockerfile: ops/Dockerfile.worker
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
    volumes:
      - ./runs:/app/runs
    command: python run_pipeline.py --limit 30
```

## Monitoring

### Log Output

The pipeline prints JSON output on completion:

```json
{
  "run_id": "20260130T175721Z",
  "root": "runs/20260130T175721Z",
  "posts_list": "runs/20260130T175721Z/raw/posts_list.json",
  "transcripts": "runs/20260130T175721Z/silver/transcripts.jsonl",
  "evals": "runs/20260130T175721Z/gold/evals.jsonl",
  "aggregates": "runs/20260130T175721Z/gold/aggregates.json"
}
```

### Key Metrics to Track

| Metric | Source | Alert Threshold |
|--------|--------|-----------------|
| Run success | Exit code | Non-zero |
| Posts fetched | aggregates.json → n_posts | < expected |
| Judge errors | Log output | Any `RuntimeError` |
| Elicitation rate (≥7) | aggregates.json | Trend increase |
| Mean scores | aggregates.json | Significant increase |

### Health Checks

```bash
# Check API reachability
curl -s https://www.moltbook.com/api/v1/posts?limit=1 | jq '.success'

# Check OpenRouter
curl -s https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" | jq '.data | length'
```

## Troubleshooting

### Common Errors

#### `Missing OPENROUTER_API_KEY`

```
RuntimeError: Missing OPENROUTER_API_KEY
```

**Solution**: Set the environment variable or add to `.env` file.

#### Rate Limiting (429)

```
requests.exceptions.HTTPError: 429 Client Error: Too Many Requests
```

**Solution**: The scraper handles this automatically with backoff. If persistent:
- Reduce `--limit`
- Increase `rate_per_sec` delay in `MoltbookAPI`

#### Judge JSON Parse Errors

```
RuntimeError: Judge did not return JSON after retries
```

**Causes**:
- Model returning markdown instead of JSON
- Response truncated due to token limits
- Model not following schema

**Solutions**:
1. Try a different model (`OPENROUTER_MODEL`)
2. Increase `JUDGE_MAX_TOKENS`
3. Check model supports `response_format: json_object`

#### Empty Transcripts

```
ValueError: Invalid post-detail payload (missing post.id)
```

**Cause**: API response format changed or post was deleted.

**Solution**: Check raw JSON files for malformed responses.

### Debug Mode

Add verbose logging:

```python
# In run_pipeline.py, before run_once():
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Manual Testing

```bash
# Test API scraping
python -c "
from scraper.moltbook_api import MoltbookAPI
api = MoltbookAPI()
r = api.get_json('/api/v1/posts', params={'limit': 1})
print(r.json_body)
"

# Test transcript building
python -c "
import json
from transcript_builder import build_transcript_from_post_detail
with open('runs/latest/raw/post_xxx.json') as f:
    detail = json.load(f)
t = build_transcript_from_post_detail(detail)
print(t.messages[0]['text'][:200])
"

# Test single judge call
python -c "
from judge_runner import LLMJudgeRunner
runner = LLMJudgeRunner()
transcript = {'post_id': 'test', 'messages': [{'kind': 'post', 'text': 'Hello world'}]}
result = runner.score_transcript(transcript)
print(result)
"
```

## Data Management

### Storage Requirements

| Layer | Size (per post) | Size (1000 posts) |
|-------|-----------------|-------------------|
| Raw | ~5-20 KB | ~10-20 MB |
| Silver | ~2-5 KB | ~2-5 MB |
| Gold | ~2-4 KB | ~2-4 MB |

### Retention Policy

Recommended:
- **Raw**: Keep indefinitely (source of truth)
- **Silver**: Keep indefinitely (reproducible from raw)
- **Gold**: Keep indefinitely (evaluation record)

### Archival

```bash
# Compress old runs
for run in runs/202601*/; do
  tar -czf "${run%.*/}.tar.gz" "$run" && rm -rf "$run"
done

# Move to cold storage
aws s3 sync runs/ s3://your-bucket/molt-observatory/runs/
```

## Security Considerations

### API Key Protection

- Never commit `.env` files
- Use environment variables or secrets manager in production
- Rotate keys periodically

### Data Sensitivity

The pipeline collects:
- Public moltbook posts and comments
- Author handles (not real names)
- Owner X/Twitter handles (public info)

**Not collected**:
- Private messages
- Authentication tokens
- User passwords

### Network Isolation

In production:
- Run in isolated network
- Limit egress to moltbook.com and openrouter.ai
- Log all external requests

## Scaling

### Horizontal Scaling

For high-volume processing:

1. **Partition by time**: Different workers handle different time windows
2. **Partition by community**: Different workers handle different submolts
3. **Separate concerns**: Scraping and judging can be decoupled via queue

### Vertical Scaling

- **More memory**: For larger batch sizes
- **Faster CPU**: Minimal benefit (I/O bound)
- **GPU**: Not needed (LLM calls are remote)

### Cost Management

| Cost Factor | Estimate | Notes |
|-------------|----------|-------|
| OpenRouter LLM calls | ~$0.001-0.01/transcript | Varies by model |
| Moltbook API | Free | Respect rate limits |
| Compute | Minimal | I/O bound workload |
| Storage | ~$0.02/GB/month | S3 Standard |

**Example**: 1000 transcripts/day × 30 days × $0.005/transcript = ~$150/month for LLM costs.

