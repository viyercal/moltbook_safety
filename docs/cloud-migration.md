# Cloud Migration Guide

This document outlines the strategy for migrating Molt Observatory from local development to a production cloud environment. It covers three major areas:

1. **PostgreSQL** - Migrating from local to managed cloud databases
2. **S3 Storage** - Storing temporal snapshots and raw data for scalability
3. **Moltobserve.com** - Public diagnostics interface architecture

## Prerequisites

Before migrating to the cloud, ensure you have:

- [ ] AWS/GCP/Azure account with billing configured
- [ ] Domain name registered (e.g., `moltobserve.com`)
- [ ] SSL certificates (or use Let's Encrypt / ACM)
- [ ] CI/CD pipeline (GitHub Actions, GitLab CI, etc.)
- [ ] Secrets management solution (AWS Secrets Manager, Doppler, etc.)

## Architecture Overview

```mermaid
flowchart TB
    subgraph ingestion [Data Ingestion Layer]
        Airflow[Airflow Scheduler]
        Worker[Pipeline Worker]
    end
    
    subgraph storage [Storage Layer]
        S3[(S3 Bucket)]
        PG[(PostgreSQL RDS)]
    end
    
    subgraph api [API Layer]
        FastAPI[FastAPI Backend]
        Cache[Redis Cache]
    end
    
    subgraph frontend [Frontend Layer]
        CDN[CloudFront CDN]
        Web[Static Frontend]
    end
    
    Airflow --> Worker
    Worker --> S3
    Worker --> PG
    
    FastAPI --> PG
    FastAPI --> S3
    FastAPI --> Cache
    
    CDN --> Web
    Web --> FastAPI
    
    User[User] --> CDN
```

---

## 1. PostgreSQL Cloud Migration

### Provider Comparison

| Provider | Pros | Cons | Starting Cost |
|----------|------|------|---------------|
| **AWS RDS PostgreSQL** | Mature, automatic backups, read replicas, Multi-AZ | Complex IAM, vendor lock-in | ~$15/mo (db.t3.micro) |
| **Supabase** | Built-in REST API, auth, real-time, generous free tier | Less control, newer platform | Free tier, then $25/mo |
| **Neon** | Serverless, branching, scale-to-zero | Newer, less battle-tested | Free tier, then $19/mo |
| **PlanetScale** | MySQL only (not PostgreSQL) | N/A | N/A |
| **GCP Cloud SQL** | Good for GCP ecosystem | Similar complexity to RDS | ~$10/mo (db-f1-micro) |
| **Azure Database** | Good for Azure ecosystem | Similar complexity to RDS | ~$15/mo (Basic tier) |

### Recommended: AWS RDS PostgreSQL

For production workloads with the existing schema, AWS RDS is recommended due to:
- Native PostgreSQL compatibility with our existing schema
- Automatic backups and point-in-time recovery
- Read replicas for scaling read-heavy dashboards
- Multi-AZ deployment for high availability
- Integration with other AWS services (S3, ECS, Lambda)

### Migration Steps

#### Step 1: Create RDS Instance

```bash
# Using AWS CLI
aws rds create-db-instance \
  --db-instance-identifier molt-observatory-prod \
  --db-instance-class db.t3.small \
  --engine postgres \
  --engine-version 15.4 \
  --master-username molt_admin \
  --master-user-password "<secure-password>" \
  --allocated-storage 20 \
  --storage-type gp3 \
  --vpc-security-group-ids sg-xxxxxxxx \
  --db-subnet-group-name molt-db-subnet-group \
  --backup-retention-period 7 \
  --multi-az \
  --storage-encrypted \
  --tags Key=Project,Value=MoltObservatory
```

Or use Terraform:

```hcl
resource "aws_db_instance" "molt_observatory" {
  identifier             = "molt-observatory-prod"
  engine                 = "postgres"
  engine_version         = "15.4"
  instance_class         = "db.t3.small"
  allocated_storage      = 20
  storage_type           = "gp3"
  storage_encrypted      = true
  
  db_name                = "molt_observatory"
  username               = "molt_admin"
  password               = var.db_password
  
  vpc_security_group_ids = [aws_security_group.db.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "Mon:04:00-Mon:05:00"
  
  multi_az               = true
  skip_final_snapshot    = false
  final_snapshot_identifier = "molt-observatory-final-snapshot"
  
  tags = {
    Project = "MoltObservatory"
    Environment = "production"
  }
}
```

#### Step 2: Export Local Database

```bash
# Export schema and data from local PostgreSQL
pg_dump \
  --host=localhost \
  --port=5432 \
  --username=molt \
  --dbname=molt_observatory \
  --format=custom \
  --file=molt_observatory_backup.dump

# For schema only (recommended for initial setup)
pg_dump \
  --host=localhost \
  --port=5432 \
  --username=molt \
  --dbname=molt_observatory \
  --schema-only \
  --file=molt_observatory_schema.sql
```

#### Step 3: Import to RDS

```bash
# Restore to RDS
pg_restore \
  --host=molt-observatory-prod.xxxxxxxx.us-east-1.rds.amazonaws.com \
  --port=5432 \
  --username=molt_admin \
  --dbname=molt_observatory \
  --no-owner \
  --no-privileges \
  --verbose \
  molt_observatory_backup.dump
```

#### Step 4: Update Environment Variables

```bash
# .env for cloud deployment
POSTGRES_HOST=molt-observatory-prod.xxxxxxxx.us-east-1.rds.amazonaws.com
POSTGRES_PORT=5432
POSTGRES_DB=molt_observatory
POSTGRES_USER=molt_admin
POSTGRES_PASSWORD=<from-secrets-manager>
POSTGRES_SSLMODE=require
POSTGRES_POOL_SIZE=10
```

### Connection Pooling

For production, use connection pooling to handle concurrent connections efficiently:

**Option A: PgBouncer (self-hosted)**

```yaml
# docker-compose addition
pgbouncer:
  image: edoburu/pgbouncer:1.21.0
  environment:
    DATABASE_URL: postgres://molt_admin:${DB_PASSWORD}@${RDS_HOST}:5432/molt_observatory
    POOL_MODE: transaction
    MAX_CLIENT_CONN: 100
    DEFAULT_POOL_SIZE: 20
  ports:
    - "6432:5432"
```

**Option B: RDS Proxy**

```bash
aws rds create-db-proxy \
  --db-proxy-name molt-observatory-proxy \
  --engine-family POSTGRESQL \
  --auth SecretArn=arn:aws:secretsmanager:... \
  --role-arn arn:aws:iam::...:role/rds-proxy-role \
  --vpc-subnet-ids subnet-xxx subnet-yyy
```

### SSL/TLS Configuration

```python
# Updated connection in Python
import psycopg2

conn = psycopg2.connect(
    host=os.environ["POSTGRES_HOST"],
    port=os.environ.get("POSTGRES_PORT", 5432),
    database=os.environ["POSTGRES_DB"],
    user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"],
    sslmode="require",
    sslrootcert="/path/to/rds-ca-2019-root.pem",  # Download from AWS
)
```

### Alternative: Supabase (Simpler Setup)

For a faster setup with built-in features:

```bash
# 1. Create project at supabase.com
# 2. Get connection string from Settings > Database

# .env
DATABASE_URL=postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres
SUPABASE_URL=https://[PROJECT-REF].supabase.co
SUPABASE_ANON_KEY=eyJ...
```

Supabase advantages:
- Built-in REST API (PostgREST) for direct database queries
- Real-time subscriptions for live dashboards
- Auth system for user management
- Edge Functions for serverless compute

---

## 2. S3 Storage for Temporal Snapshots

### Bucket Architecture

```
s3://molt-observatory-data/
├── runs/
│   ├── 20260130T175721Z/
│   │   ├── raw/
│   │   │   ├── posts_list.json
│   │   │   ├── posts/
│   │   │   │   └── post_*.json
│   │   │   ├── agents_list.json
│   │   │   └── submolts_list.json
│   │   ├── silver/
│   │   │   ├── transcripts.jsonl
│   │   │   └── comment_transcripts.jsonl
│   │   ├── gold/
│   │   │   ├── evals.jsonl
│   │   │   ├── comment_evals.jsonl
│   │   │   ├── agent_scores.jsonl
│   │   │   └── aggregates.json
│   │   └── meta/
│   │       └── snapshot.json
│   └── 20260131T000000Z/
│       └── ...
├── agent_history/
│   └── *.jsonl
└── reports/
    ├── dashboard.html
    └── leaderboard.html
```

### Create S3 Bucket

```bash
# Create bucket
aws s3 mb s3://molt-observatory-data --region us-east-1

# Enable versioning for data protection
aws s3api put-bucket-versioning \
  --bucket molt-observatory-data \
  --versioning-configuration Status=Enabled

# Set up lifecycle policy
aws s3api put-bucket-lifecycle-configuration \
  --bucket molt-observatory-data \
  --lifecycle-configuration file://lifecycle-policy.json
```

**lifecycle-policy.json:**

```json
{
  "Rules": [
    {
      "ID": "TransitionRawToGlacier",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "runs/"
      },
      "Transitions": [
        {
          "Days": 90,
          "StorageClass": "GLACIER_IR"
        },
        {
          "Days": 365,
          "StorageClass": "DEEP_ARCHIVE"
        }
      ]
    },
    {
      "ID": "DeleteOldVersions",
      "Status": "Enabled",
      "Filter": {},
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 30
      }
    }
  ]
}
```

### IAM Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "MoltObservatoryS3Access",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::molt-observatory-data",
        "arn:aws:s3:::molt-observatory-data/*"
      ]
    }
  ]
}
```

### Code Integration

#### Option A: Dual-Write (Local + S3)

Create a storage abstraction layer:

```python
# molt-observatory/storage/s3_backend.py
"""S3 storage backend for Molt Observatory."""

import os
import json
from pathlib import Path
from typing import Any, Optional, Union
from io import BytesIO

import boto3
from botocore.exceptions import ClientError


class S3StorageBackend:
    """Write artifacts to S3 in addition to local filesystem."""
    
    def __init__(
        self,
        bucket: Optional[str] = None,
        prefix: str = "",
        region: str = "us-east-1",
    ):
        self.bucket = bucket or os.environ.get("S3_BUCKET", "molt-observatory-data")
        self.prefix = prefix.strip("/")
        self.region = region
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client(
                "s3",
                region_name=self.region,
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            )
        return self._client
    
    def _s3_key(self, path: str) -> str:
        """Build full S3 key from relative path."""
        if self.prefix:
            return f"{self.prefix}/{path}"
        return path
    
    def write_json(self, path: str, data: Any) -> str:
        """Write JSON data to S3."""
        key = self._s3_key(path)
        body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        
        return f"s3://{self.bucket}/{key}"
    
    def write_jsonl(self, path: str, records: list) -> str:
        """Write JSONL data to S3."""
        key = self._s3_key(path)
        lines = [json.dumps(r, ensure_ascii=False) for r in records]
        body = "\n".join(lines).encode("utf-8")
        
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/x-ndjson",
        )
        
        return f"s3://{self.bucket}/{key}"
    
    def read_json(self, path: str) -> Any:
        """Read JSON data from S3."""
        key = self._s3_key(path)
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))
    
    def read_jsonl(self, path: str) -> list:
        """Read JSONL data from S3."""
        key = self._s3_key(path)
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        content = response["Body"].read().decode("utf-8")
        return [json.loads(line) for line in content.strip().split("\n") if line]
    
    def exists(self, path: str) -> bool:
        """Check if object exists in S3."""
        key = self._s3_key(path)
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False
    
    def list_prefix(self, prefix: str) -> list:
        """List objects under a prefix."""
        full_prefix = self._s3_key(prefix)
        paginator = self.client.get_paginator("list_objects_v2")
        
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        
        return keys


class HybridStorage:
    """
    Hybrid storage that writes to both local filesystem and S3.
    Reads prefer local, falls back to S3.
    """
    
    def __init__(
        self,
        local_root: Path,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "",
        enable_s3: bool = True,
    ):
        self.local_root = Path(local_root)
        self.enable_s3 = enable_s3 and s3_bucket is not None
        
        if self.enable_s3:
            self.s3 = S3StorageBackend(bucket=s3_bucket, prefix=s3_prefix)
        else:
            self.s3 = None
    
    def write_json(self, rel_path: str, data: Any) -> dict:
        """Write JSON to local and optionally S3."""
        local_path = self.local_root / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        result = {"local": str(local_path)}
        
        if self.enable_s3 and self.s3:
            s3_uri = self.s3.write_json(rel_path, data)
            result["s3"] = s3_uri
        
        return result
    
    def write_jsonl(self, rel_path: str, records: list) -> dict:
        """Write JSONL to local and optionally S3."""
        local_path = self.local_root / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(local_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        result = {"local": str(local_path)}
        
        if self.enable_s3 and self.s3:
            s3_uri = self.s3.write_jsonl(rel_path, records)
            result["s3"] = s3_uri
        
        return result
    
    def read_json(self, rel_path: str) -> Any:
        """Read JSON, preferring local."""
        local_path = self.local_root / rel_path
        
        if local_path.exists():
            with open(local_path, "r", encoding="utf-8") as f:
                return json.load(f)
        
        if self.enable_s3 and self.s3:
            return self.s3.read_json(rel_path)
        
        raise FileNotFoundError(f"Not found: {rel_path}")
```

#### Option B: Post-Run Sync Script

Simpler approach - sync after each run completes:

```python
# molt-observatory/storage/sync_to_s3.py
"""Sync local runs directory to S3 after pipeline completion."""

import os
import subprocess
from pathlib import Path


def sync_run_to_s3(
    run_id: str,
    local_runs_dir: Path,
    bucket: str = "molt-observatory-data",
    delete: bool = False,
) -> None:
    """
    Sync a completed run to S3 using AWS CLI.
    
    Args:
        run_id: The run timestamp (e.g., "20260130T175721Z")
        local_runs_dir: Path to local runs/ directory
        bucket: S3 bucket name
        delete: Whether to delete local files after sync
    """
    run_path = local_runs_dir / run_id
    s3_uri = f"s3://{bucket}/runs/{run_id}/"
    
    if not run_path.exists():
        raise ValueError(f"Run directory not found: {run_path}")
    
    # Sync to S3
    cmd = [
        "aws", "s3", "sync",
        str(run_path),
        s3_uri,
        "--storage-class", "STANDARD_IA",  # Infrequent access for cost savings
    ]
    
    print(f"Syncing {run_path} to {s3_uri}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"S3 sync failed: {result.stderr}")
    
    print(f"Sync complete: {s3_uri}")
    
    # Optionally delete local after successful sync
    if delete:
        import shutil
        shutil.rmtree(run_path)
        print(f"Deleted local: {run_path}")


def sync_all_runs(
    local_runs_dir: Path,
    bucket: str = "molt-observatory-data",
) -> None:
    """Sync all local runs to S3."""
    for run_dir in sorted(local_runs_dir.iterdir()):
        if run_dir.is_dir() and run_dir.name[0].isdigit():
            try:
                sync_run_to_s3(run_dir.name, local_runs_dir, bucket)
            except Exception as e:
                print(f"Error syncing {run_dir.name}: {e}")


if __name__ == "__main__":
    import sys
    
    runs_dir = Path(__file__).parent.parent / "runs"
    
    if len(sys.argv) > 1:
        # Sync specific run
        sync_run_to_s3(sys.argv[1], runs_dir)
    else:
        # Sync all
        sync_all_runs(runs_dir)
```

### Environment Variables for S3

```bash
# Add to .env
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=molt-observatory-data
S3_PREFIX=  # Optional prefix within bucket
ENABLE_S3_SYNC=true
```

### Airflow Integration

Update the DAG to sync after each run:

```python
# In moltbook_ingest_eval.py

def sync_to_s3_task(**context):
    """Sync completed run to S3."""
    from storage.sync_to_s3 import sync_run_to_s3
    
    ti = context['ti']
    scrape_result = ti.xcom_pull(key='scrape_result', task_ids='scrape_all_entities')
    
    run_id = scrape_result["run_id"]
    runs_dir = MOLT_OBSERVATORY_PATH / "runs"
    
    bucket = os.environ.get("S3_BUCKET", "molt-observatory-data")
    
    if os.environ.get("ENABLE_S3_SYNC", "false").lower() == "true":
        sync_run_to_s3(run_id, runs_dir, bucket)
        return {"synced": True, "run_id": run_id}
    
    return {"synced": False, "reason": "S3 sync disabled"}

# Add to DAG
sync_s3 = PythonOperator(
    task_id='sync_to_s3',
    python_callable=sync_to_s3_task,
    provide_context=True,
)

# Update dependencies
reports >> sync_s3 >> end
```

---

## 3. Moltobserve.com Public Interface

### Concept

Transform any moltbook.com URL into a diagnostics view:

| Moltbook URL | Moltobserve URL | View |
|--------------|-----------------|------|
| `moltbook.com/post/{uuid}` | `moltobserve.com/post/{uuid}` | Post diagnostics |
| `moltbook.com/agent/{handle}` | `moltobserve.com/agent/{handle}` | Agent profile & scores |
| `moltbook.com/m/{submolt}` | `moltobserve.com/m/{submolt}` | Community stats |

### System Architecture

```mermaid
flowchart TB
    subgraph users [User Requests]
        Browser[Browser]
    end
    
    subgraph edge [Edge Layer]
        Vercel[Vercel Edge Network]
        Cache[Edge Cache]
    end
    
    subgraph backend [Backend Services]
        API[FastAPI]
        Worker[Background Worker]
    end
    
    subgraph data [Data Layer]
        PG[(PostgreSQL)]
        S3[(S3)]
        Redis[(Redis Cache)]
    end
    
    Browser --> Vercel
    Vercel --> Cache
    Cache --> API
    
    API --> Redis
    API --> PG
    API --> S3
    
    Worker --> PG
    Worker --> S3
```

### Backend API Design

#### Directory Structure

```
moltobserve-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings
│   ├── database.py          # DB connection
│   ├── models/
│   │   ├── __init__.py
│   │   ├── post.py
│   │   ├── agent.py
│   │   └── submolt.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── posts.py
│   │   ├── agents.py
│   │   └── submolts.py
│   └── services/
│       ├── __init__.py
│       ├── evaluation.py
│       └── s3_client.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

#### Main FastAPI Application

```python
# moltobserve-api/app/main.py
"""Moltobserve API - Public diagnostics for moltbook.com content."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import os

from app.routers import posts, agents, submolts
from app.config import settings

app = FastAPI(
    title="Moltobserve API",
    description="AI Safety Diagnostics for Moltbook",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(posts.router, prefix="/api/v1/posts", tags=["posts"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
app.include_router(submolts.router, prefix="/api/v1/submolts", tags=["submolts"])


@app.get("/")
async def root():
    return {
        "name": "Moltobserve API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
```

#### Post Diagnostics Endpoint

```python
# moltobserve-api/app/routers/posts.py
"""Post diagnostics endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

router = APIRouter()


class DimensionScore(BaseModel):
    score: int
    confidence: float
    evidence: str
    explanation: str


class PostEvaluation(BaseModel):
    evaluated_at: datetime
    model: str
    scores: Dict[str, DimensionScore]
    notes: Optional[str] = None


class PostDiagnostics(BaseModel):
    post_id: str
    title: Optional[str]
    author_handle: str
    submolt: str
    created_at: datetime
    permalink: str
    
    # Current evaluation
    latest_evaluation: Optional[PostEvaluation]
    
    # Aggregated scores
    mean_scores: Dict[str, float]
    max_scores: Dict[str, int]
    
    # Risk indicators
    has_high_harm: bool
    has_high_deception: bool
    has_high_power_seeking: bool
    has_high_sycophancy: bool
    
    # Historical evaluations
    evaluation_count: int
    evaluations: List[PostEvaluation]


@router.get("/{post_id}", response_model=PostDiagnostics)
async def get_post_diagnostics(post_id: str):
    """
    Get safety diagnostics for a specific post.
    
    The post_id should match the UUID from moltbook.com URLs.
    """
    # Query database for post and evaluations
    # This is a placeholder - implement actual DB queries
    
    from app.database import get_db
    from app.services.evaluation import get_post_with_evaluations
    
    db = get_db()
    result = await get_post_with_evaluations(db, post_id)
    
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Post not found: {post_id}. It may not have been evaluated yet."
        )
    
    return result


@router.get("/{post_id}/timeline")
async def get_post_evaluation_timeline(post_id: str):
    """Get historical evaluation timeline for a post."""
    from app.database import get_db
    from app.services.evaluation import get_evaluation_timeline
    
    db = get_db()
    timeline = await get_evaluation_timeline(db, post_id)
    
    return {
        "post_id": post_id,
        "timeline": timeline,
    }


@router.get("/{post_id}/comments")
async def get_post_comment_diagnostics(post_id: str):
    """Get diagnostics for all comments on a post."""
    from app.database import get_db
    from app.services.evaluation import get_comment_evaluations_for_post
    
    db = get_db()
    comments = await get_comment_evaluations_for_post(db, post_id)
    
    return {
        "post_id": post_id,
        "comment_count": len(comments),
        "comments": comments,
    }
```

#### Agent Profile Endpoint

```python
# moltobserve-api/app/routers/agents.py
"""Agent profile and score endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

router = APIRouter()


class AgentScoreHistory(BaseModel):
    recorded_at: datetime
    overall_mean_score: float
    dimension_scores: Dict[str, float]
    posts_evaluated: int
    comments_evaluated: int


class AgentProfile(BaseModel):
    agent_id: str
    handle: str
    display_name: Optional[str]
    avatar_url: Optional[str]
    karma: int
    is_claimed: bool
    
    # Owner info
    owner_x_handle: Optional[str]
    
    # Current scores
    current_overall_score: float
    current_dimension_scores: Dict[str, float]
    
    # Risk flags
    has_high_harm: bool
    has_high_deception: bool
    has_high_power_seeking: bool
    has_high_sycophancy: bool
    
    # Stats
    total_posts_evaluated: int
    total_comments_evaluated: int
    
    # History
    score_history: List[AgentScoreHistory]


@router.get("/{handle}", response_model=AgentProfile)
async def get_agent_profile(handle: str):
    """
    Get safety profile for an agent.
    
    Returns aggregated scores, risk flags, and score history over time.
    """
    from app.database import get_db
    from app.services.evaluation import get_agent_profile
    
    db = get_db()
    profile = await get_agent_profile(db, handle)
    
    if not profile:
        raise HTTPException(
            status_code=404,
            detail=f"Agent not found: {handle}"
        )
    
    return profile


@router.get("/{handle}/posts")
async def get_agent_evaluated_posts(
    handle: str,
    limit: int = 20,
    offset: int = 0,
):
    """Get posts by this agent with their evaluations."""
    from app.database import get_db
    from app.services.evaluation import get_agent_posts
    
    db = get_db()
    posts = await get_agent_posts(db, handle, limit, offset)
    
    return {
        "agent_handle": handle,
        "posts": posts,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{handle}/trend")
async def get_agent_score_trend(handle: str, days: int = 30):
    """Get score trend over time for an agent."""
    from app.database import get_db
    from app.services.evaluation import get_agent_trend
    
    db = get_db()
    trend = await get_agent_trend(db, handle, days)
    
    return {
        "agent_handle": handle,
        "days": days,
        "trend": trend,
    }
```

#### Submolt Community Endpoint

```python
# moltobserve-api/app/routers/submolts.py
"""Submolt community diagnostics endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

router = APIRouter()


class SubmoltStats(BaseModel):
    name: str
    display_name: Optional[str]
    subscriber_count: int
    post_count: int
    
    # Evaluation stats
    posts_evaluated: int
    avg_harm_score: float
    avg_deception_score: float
    avg_power_seeking_score: float
    avg_sycophancy_score: float
    
    # High-risk content
    high_risk_post_count: int
    high_risk_percentage: float


@router.get("/{name}", response_model=SubmoltStats)
async def get_submolt_stats(name: str):
    """Get community-level safety statistics."""
    from app.database import get_db
    from app.services.evaluation import get_submolt_stats
    
    db = get_db()
    stats = await get_submolt_stats(db, name)
    
    if not stats:
        raise HTTPException(
            status_code=404,
            detail=f"Submolt not found: {name}"
        )
    
    return stats


@router.get("/{name}/leaderboard")
async def get_submolt_agent_leaderboard(name: str, limit: int = 20):
    """Get agent leaderboard for a submolt."""
    from app.database import get_db
    from app.services.evaluation import get_submolt_leaderboard
    
    db = get_db()
    leaderboard = await get_submolt_leaderboard(db, name, limit)
    
    return {
        "submolt": name,
        "agents": leaderboard,
    }


@router.get("/{name}/flagged")
async def get_submolt_flagged_content(name: str, limit: int = 20):
    """Get high-risk flagged content in a submolt."""
    from app.database import get_db
    from app.services.evaluation import get_flagged_content
    
    db = get_db()
    flagged = await get_flagged_content(db, name, limit)
    
    return {
        "submolt": name,
        "flagged_posts": flagged,
    }
```

### Frontend Implementation

#### Option A: Next.js (Recommended)

```
moltobserve-web/
├── app/
│   ├── layout.tsx
│   ├── page.tsx              # Home/search
│   ├── post/
│   │   └── [id]/
│   │       └── page.tsx      # Post diagnostics
│   ├── agent/
│   │   └── [handle]/
│   │       └── page.tsx      # Agent profile
│   └── m/
│       └── [name]/
│           └── page.tsx      # Submolt stats
├── components/
│   ├── ScoreCard.tsx
│   ├── DimensionChart.tsx
│   ├── Timeline.tsx
│   └── RiskBadge.tsx
├── lib/
│   └── api.ts
├── package.json
└── next.config.js
```

**Example Post Page:**

```tsx
// moltobserve-web/app/post/[id]/page.tsx
import { notFound } from 'next/navigation';
import { ScoreCard } from '@/components/ScoreCard';
import { DimensionChart } from '@/components/DimensionChart';
import { RiskBadge } from '@/components/RiskBadge';

async function getPostDiagnostics(id: string) {
  const res = await fetch(`${process.env.API_URL}/api/v1/posts/${id}`, {
    next: { revalidate: 3600 }, // Cache for 1 hour
  });
  
  if (!res.ok) return null;
  return res.json();
}

export default async function PostPage({ params }: { params: { id: string } }) {
  const data = await getPostDiagnostics(params.id);
  
  if (!data) {
    notFound();
  }
  
  return (
    <main className="container mx-auto px-4 py-8">
      <header className="mb-8">
        <h1 className="text-3xl font-bold mb-2">{data.title || 'Untitled Post'}</h1>
        <div className="flex gap-4 text-gray-600">
          <span>by <a href={`/agent/${data.author_handle}`}>{data.author_handle}</a></span>
          <span>in <a href={`/m/${data.submolt}`}>m/{data.submolt}</a></span>
        </div>
        <a 
          href={data.permalink}
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-500 hover:underline text-sm"
        >
          View on Moltbook →
        </a>
      </header>
      
      {/* Risk indicators */}
      <section className="mb-8">
        <h2 className="text-xl font-semibold mb-4">Risk Assessment</h2>
        <div className="flex gap-2">
          {data.has_high_harm && <RiskBadge type="harm" />}
          {data.has_high_deception && <RiskBadge type="deception" />}
          {data.has_high_power_seeking && <RiskBadge type="power" />}
          {data.has_high_sycophancy && <RiskBadge type="sycophancy" />}
          {!data.has_high_harm && !data.has_high_deception && 
           !data.has_high_power_seeking && !data.has_high_sycophancy && (
            <span className="text-green-500">No high-risk indicators detected</span>
          )}
        </div>
      </section>
      
      {/* Dimension scores */}
      <section className="mb-8">
        <h2 className="text-xl font-semibold mb-4">Safety Dimensions</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(data.mean_scores).map(([dim, score]) => (
            <ScoreCard 
              key={dim}
              dimension={dim}
              score={score as number}
              maxScore={data.max_scores[dim]}
            />
          ))}
        </div>
      </section>
      
      {/* Evaluation details */}
      {data.latest_evaluation && (
        <section className="mb-8">
          <h2 className="text-xl font-semibold mb-4">Latest Evaluation</h2>
          <p className="text-gray-500 text-sm mb-4">
            Model: {data.latest_evaluation.model} | 
            Evaluated: {new Date(data.latest_evaluation.evaluated_at).toLocaleString()}
          </p>
          
          {Object.entries(data.latest_evaluation.scores).map(([dim, details]) => (
            <div key={dim} className="mb-4 p-4 bg-gray-50 rounded-lg">
              <h3 className="font-medium">{dim.replace(/_/g, ' ')}</h3>
              <div className="text-2xl font-bold">{(details as any).score}/10</div>
              <p className="text-gray-600 mt-2">{(details as any).explanation}</p>
              {(details as any).evidence && (
                <blockquote className="mt-2 pl-4 border-l-4 border-gray-300 text-gray-500 italic">
                  "{(details as any).evidence}"
                </blockquote>
              )}
            </div>
          ))}
        </section>
      )}
      
      {/* Chart */}
      <section>
        <h2 className="text-xl font-semibold mb-4">Score Distribution</h2>
        <DimensionChart scores={data.mean_scores} />
      </section>
    </main>
  );
}
```

### Deployment Options

#### Vercel (Recommended for Next.js)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy frontend
cd moltobserve-web
vercel

# Set environment variables
vercel env add API_URL  # https://api.moltobserve.com
```

#### Railway (Backend API)

```bash
# Install Railway CLI
npm i -g @railway/cli

# Deploy FastAPI
cd moltobserve-api
railway init
railway up

# Set environment variables in Railway dashboard
```

#### AWS ECS (Full Stack)

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    image: ${ECR_REGISTRY}/moltobserve-api:latest
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - S3_BUCKET=${S3_BUCKET}
      - REDIS_URL=${REDIS_URL}
    ports:
      - "8000:8000"
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '0.5'
          memory: 512M

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

### Domain & DNS Configuration

```
# DNS Records for moltobserve.com

# Frontend (Vercel)
@     A     76.76.21.21
www   CNAME moltobserve.vercel.app

# API subdomain (Railway/ECS)
api   CNAME moltobserve-api.up.railway.app

# Or if using CloudFront
api   A     ALIAS d1234567890.cloudfront.net
```

---

## Environment Variables Reference

### Core Infrastructure

| Variable | Description | Example |
|----------|-------------|---------|
| `POSTGRES_HOST` | PostgreSQL host | `molt-prod.xxx.rds.amazonaws.com` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `POSTGRES_DB` | Database name | `molt_observatory` |
| `POSTGRES_USER` | Database user | `molt_admin` |
| `POSTGRES_PASSWORD` | Database password | (from secrets manager) |
| `POSTGRES_SSLMODE` | SSL mode | `require` |

### S3 Storage

| Variable | Description | Example |
|----------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | AWS access key | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | (from secrets manager) |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |
| `S3_BUCKET` | S3 bucket name | `molt-observatory-data` |
| `ENABLE_S3_SYNC` | Enable S3 uploads | `true` |

### API & Frontend

| Variable | Description | Example |
|----------|-------------|---------|
| `API_URL` | Backend API URL | `https://api.moltobserve.com` |
| `REDIS_URL` | Redis cache URL | `redis://localhost:6379` |
| `ALLOWED_ORIGINS` | CORS origins | `https://moltobserve.com` |

### Pipeline (Unchanged)

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | `sk-or-v1-...` |
| `OPENROUTER_MODEL` | LLM model | `google/gemini-3-flash-preview` |

---

## Cost Estimates

### Monthly Costs (Moderate Usage)

| Component | Service | Tier | Est. Cost |
|-----------|---------|------|-----------|
| Database | AWS RDS PostgreSQL | db.t3.small, Multi-AZ | $50-70 |
| Storage | S3 Standard | 50GB data, 100k requests | $2-5 |
| Compute | Railway/Fly.io | API server | $10-20 |
| Frontend | Vercel | Pro plan | $20 |
| CDN | CloudFront | 100GB transfer | $10 |
| Cache | Redis (Upstash) | Free tier | $0 |
| Domain | Route 53 | .com domain | $1 |
| **Total** | | | **$93-126/mo** |

### Cost Optimization Tips

1. **Use reserved instances** for RDS (up to 40% savings)
2. **Enable S3 lifecycle policies** to move old data to Glacier
3. **Use Vercel Edge caching** to reduce API calls
4. **Start with smaller instances** and scale as needed
5. **Monitor with AWS Cost Explorer** and set billing alerts

---

## Implementation Roadmap

### Phase 1: Database Migration (Week 1-2)
- [ ] Create RDS instance in AWS
- [ ] Export local database schema and data
- [ ] Import to RDS and validate
- [ ] Update connection strings in pipeline
- [ ] Test pipeline with cloud database

### Phase 2: S3 Integration (Week 2-3)
- [ ] Create S3 bucket with lifecycle policies
- [ ] Implement storage abstraction layer
- [ ] Add S3 sync to Airflow DAG
- [ ] Migrate existing runs to S3
- [ ] Verify data integrity

### Phase 3: API Development (Week 3-5)
- [ ] Set up FastAPI project structure
- [ ] Implement post, agent, submolt endpoints
- [ ] Add Redis caching layer
- [ ] Deploy to Railway/ECS
- [ ] Set up API monitoring

### Phase 4: Frontend Development (Week 5-7)
- [ ] Create Next.js project
- [ ] Build post diagnostics page
- [ ] Build agent profile page
- [ ] Build submolt stats page
- [ ] Deploy to Vercel

### Phase 5: Launch (Week 7-8)
- [ ] Configure domain and DNS
- [ ] Set up SSL certificates
- [ ] Enable CDN caching
- [ ] Load testing
- [ ] Public launch

---

## Next Steps

1. **Choose cloud providers** based on team expertise and budget
2. **Set up AWS/GCP account** with proper IAM and billing alerts
3. **Register moltobserve.com domain** (or preferred alternative)
4. **Create infrastructure as code** (Terraform/Pulumi) for reproducibility
5. **Set up CI/CD pipelines** for automated deployments

