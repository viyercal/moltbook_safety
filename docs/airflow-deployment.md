# Airflow Deployment Guide

This guide covers deploying and operating the Molt Observatory Airflow DAGs for hourly automated scraping and evaluation.

## Overview

Molt Observatory uses Apache Airflow to orchestrate:
- **Hourly scrapes** of moltbook.com (posts, agents, submolts, comments)
- **Transcript building** for LLM evaluation
- **Safety evaluations** using OpenRouter LLM judges
- **Score aggregation** per agent with historical tracking
- **Report generation** with Plotly charts

## DAGs

### `moltbook_hourly_ingest_eval`

The primary DAG that runs every hour.

| Task | Description | Duration |
|------|-------------|----------|
| `scrape_all_entities` | Fetch posts, agents, submolts, comments | 5-15 min |
| `build_transcripts` | Create post and comment transcripts | 1-3 min |
| `run_evaluations` | LLM safety scoring via OpenRouter | 10-60 min |
| `compute_aggregates` | Calculate run statistics | <1 min |
| `aggregate_agent_scores` | Update per-agent score history | 1-2 min |
| `generate_reports` | Create HTML dashboards | <1 min |

```
start → scrape → build_transcripts → evaluate → [aggregates, agent_scores] → reports → end
```

### `moltbook_weekly_full_refresh`

Weekly full refresh that resets cursors and scrapes everything.

- **Schedule**: Sundays at midnight UTC
- **Purpose**: Catch any missed content, update all metrics
- **Duration**: 2-4 hours depending on site size

## Prerequisites

- Python 3.9+
- Apache Airflow 2.5+
- Docker (recommended for deployment)
- OpenRouter API key
- Network access to moltbook.com

## Installation

### Option 1: Docker Compose (Recommended)

```yaml
# docker-compose.yml
version: '3.8'

x-airflow-common: &airflow-common
  build:
    context: .
    dockerfile: ops/Dockerfile.airflow
  environment:
    AIRFLOW__CORE__EXECUTOR: LocalExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    AIRFLOW__CORE__FERNET_KEY: ${FERNET_KEY:-}
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'false'
    AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    AIRFLOW__API__AUTH_BACKENDS: 'airflow.api.auth.backend.basic_auth'
    # Molt Observatory settings
    OPENROUTER_API_KEY: ${OPENROUTER_API_KEY}
    OPENROUTER_MODEL: ${OPENROUTER_MODEL:-google/gemini-3-flash-preview}
  volumes:
    - ./molt-observatory:/opt/molt-observatory
    - ./molt-observatory/airflow/dags:/opt/airflow/dags
    - airflow-logs:/opt/airflow/logs
  depends_on:
    postgres:
      condition: service_healthy

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 10s
      retries: 5

  airflow-init:
    <<: *airflow-common
    entrypoint: /bin/bash
    command:
      - -c
      - |
        airflow db init
        airflow users create \
          --username admin \
          --password admin \
          --firstname Admin \
          --lastname User \
          --role Admin \
          --email admin@example.com
    restart: on-failure

  airflow-webserver:
    <<: *airflow-common
    command: webserver
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 5
    restart: always

  airflow-scheduler:
    <<: *airflow-common
    command: scheduler
    healthcheck:
      test: ["CMD-SHELL", "airflow jobs check --job-type SchedulerJob --hostname $(hostname)"]
      interval: 30s
      timeout: 10s
      retries: 5
    restart: always

volumes:
  postgres-data:
  airflow-logs:
```

### Option 2: Local Installation

```bash
# Create virtual environment
python -m venv airflow-venv
source airflow-venv/bin/activate

# Install Airflow
pip install apache-airflow==2.8.0

# Install Molt Observatory dependencies
pip install -r requirements.txt

# Initialize database
export AIRFLOW_HOME=~/airflow
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Link DAGs
ln -s /path/to/molt-observatory/airflow/dags ~/airflow/dags/moltbook

# Start services (in separate terminals)
airflow scheduler
airflow webserver --port 8080
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | - | OpenRouter API key for LLM calls |
| `OPENROUTER_MODEL` | No | `google/gemini-3-flash-preview` | Model for evaluations |
| `AIRFLOW__CORE__PARALLELISM` | No | 32 | Max parallel tasks |
| `JUDGE_MAX_ATTEMPTS` | No | 3 | Retries for LLM calls |
| `JUDGE_MAX_TOKENS` | No | 1800 | Max tokens for judge output |

### Airflow Variables

Set these in the Airflow UI (Admin → Variables):

```json
{
  "moltbook_max_posts_per_run": 500,
  "moltbook_max_agents_per_run": 200,
  "moltbook_eval_batch_size": 50,
  "moltbook_report_output_path": "/opt/molt-observatory/reports/output"
}
```

### Connections

Create an HTTP connection for OpenRouter (Admin → Connections):

- **Conn ID**: `openrouter`
- **Conn Type**: HTTP
- **Host**: `https://openrouter.ai`
- **Extra**: `{"Authorization": "Bearer ${OPENROUTER_API_KEY}"}`

## Dockerfile

```dockerfile
# ops/Dockerfile.airflow
FROM apache/airflow:2.8.0-python3.11

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install Molt Observatory as editable
COPY --chown=airflow:root molt-observatory /opt/molt-observatory
ENV PYTHONPATH="/opt/molt-observatory:${PYTHONPATH}"
```

## Operations

### Starting the DAG

1. Access Airflow UI at http://localhost:8080
2. Navigate to DAGs
3. Find `moltbook_hourly_ingest_eval`
4. Toggle the DAG to "On"

The DAG will run at the top of every hour.

### Manual Trigger

```bash
# Via CLI
airflow dags trigger moltbook_hourly_ingest_eval

# Via API
curl -X POST "http://localhost:8080/api/v1/dags/moltbook_hourly_ingest_eval/dagRuns" \
    -H "Content-Type: application/json" \
    -u admin:admin \
    -d '{}'
```

### Backfilling

To backfill historical data:

```bash
# Backfill the last 7 days
airflow dags backfill \
    --start-date 2026-01-23 \
    --end-date 2026-01-30 \
    moltbook_hourly_ingest_eval
```

Note: Due to incremental scraping, backfills may not fetch all historical posts. Use the weekly full refresh for complete coverage.

### Monitoring

#### Task Logs

Access logs via:
- Airflow UI → DAG → Task Instance → Logs
- CLI: `airflow tasks logs moltbook_hourly_ingest_eval scrape_all_entities 2026-01-30`

#### Metrics

Key metrics to monitor:
- `posts_count`: Posts scraped per run
- `eval_duration`: Time for LLM evaluations
- `high_risk_count`: Content scoring ≥7

#### Alerts

Configure email alerts in `default_args`:

```python
default_args = {
    'email': ['alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': True,
}
```

Or use Slack:

```python
from airflow.providers.slack.operators.slack import SlackAPIPostOperator

slack_alert = SlackAPIPostOperator(
    task_id='slack_alert',
    channel='#moltbook-alerts',
    text='Pipeline completed: {{ task_instance.xcom_pull("scrape_result") }}',
    trigger_rule='all_done',
)
```

### Failure Handling

#### Common Failures

| Error | Cause | Resolution |
|-------|-------|------------|
| `OpenRouter error 429` | Rate limited | Reduce batch size, increase delays |
| `Timeout` | LLM call too slow | Increase `execution_timeout` |
| `JSON parse error` | Malformed LLM output | Automatic retry handles this |
| `Connection refused` | moltbook.com down | Wait and retry |

#### Task Retries

Tasks are configured with 2 retries and 5-minute delays. Adjust in `default_args`:

```python
default_args = {
    'retries': 3,
    'retry_delay': timedelta(minutes=10),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(hours=1),
}
```

### Scaling

#### Horizontal Scaling

For high-volume deployments:

1. **Use CeleryExecutor**:
```python
AIRFLOW__CORE__EXECUTOR: CeleryExecutor
AIRFLOW__CELERY__BROKER_URL: redis://redis:6379/0
```

2. **Add workers**:
```bash
docker-compose scale airflow-worker=3
```

3. **Partition by submolt**:
Create separate DAGs for different communities.

#### Vertical Scaling

For LLM-heavy workloads:
- Increase worker memory to 4GB+
- Use faster models (`gpt-4o-mini` vs `claude-3-opus`)
- Batch evaluations

## Cost Management

### OpenRouter Costs

Approximate costs per run (500 posts + comments):

| Model | Cost/Run | Cost/Month (hourly) |
|-------|----------|---------------------|
| `gpt-4o-mini` | ~$0.50 | ~$360 |
| `gemini-3-flash-preview` | ~$0.20 | ~$144 |
| `claude-3-haiku` | ~$0.15 | ~$108 |

### Optimization Tips

1. **Skip unchanged content**: Check post hashes before re-evaluating
2. **Reduce frequency**: Run every 4 hours instead of hourly
3. **Sample evaluation**: Evaluate 20% of comments randomly
4. **Use cheaper models**: `gemini-flash` for routine, `claude-opus` for high-risk

## Maintenance

### Log Rotation

Airflow logs grow quickly. Configure rotation:

```python
# airflow.cfg
[logging]
base_log_folder = /opt/airflow/logs
dag_processor_manager_log_location = /opt/airflow/logs/dag_processor_manager.log
log_rotation = 7  # Keep 7 days
```

### Database Cleanup

Clean old DAG runs:

```bash
# Delete runs older than 30 days
airflow db clean --clean-before-timestamp $(date -d '30 days ago' +%Y-%m-%d)
```

### Upgrading

```bash
# Stop services
docker-compose down

# Pull new code
git pull

# Rebuild images
docker-compose build

# Migrate database
docker-compose run airflow-init

# Start services
docker-compose up -d
```

## Troubleshooting

### DAG Not Appearing

1. Check DAG file syntax:
```bash
python molt-observatory/airflow/dags/moltbook_ingest_eval.py
```

2. Check Airflow logs:
```bash
docker-compose logs airflow-scheduler | grep moltbook
```

3. Verify DAG folder:
```bash
ls -la ~/airflow/dags/
```

### Tasks Stuck in "Queued"

1. Check executor:
```bash
airflow config get-value core executor
```

2. Verify scheduler is running:
```bash
docker-compose ps airflow-scheduler
```

3. Clear stuck tasks:
```bash
airflow tasks clear moltbook_hourly_ingest_eval -s 2026-01-30 -e 2026-01-30
```

### Memory Issues

If workers run out of memory:

```yaml
# docker-compose.yml
airflow-scheduler:
  deploy:
    resources:
      limits:
        memory: 4G
```

## Security

### Secrets Management

Use Airflow's secrets backend:

```python
# airflow.cfg
[secrets]
backend = airflow.providers.hashicorp.secrets.vault.VaultBackend
backend_kwargs = {"url": "http://vault:8200", "token": "..."}
```

### Network Isolation

Run in isolated network with egress restricted to:
- moltbook.com (port 443)
- openrouter.ai (port 443)
- PostgreSQL (internal)

### Audit Logging

Enable task logging to external storage:

```python
[logging]
remote_logging = True
remote_log_conn_id = s3_logs
remote_base_log_folder = s3://molt-observatory-logs/airflow
```

## Related Documentation

- [Architecture](./architecture.md)
- [Database Schema](./database-schema.md)
- [Operational Guide](./ops-guide.md)

