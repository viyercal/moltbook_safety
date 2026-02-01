# Quick Start Guide

This guide covers how to set up and run Molt Observatory, whether you're starting fresh or working with an existing dataset.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Scenario A: Fresh Install (Clean Slate)](#scenario-a-fresh-install-clean-slate)
- [Scenario B: Existing Local Dataset](#scenario-b-existing-local-dataset)
- [Full Data Pull Commands](#full-data-pull-commands)
- [Generating Visualizations](#generating-visualizations)
- [Pipeline Output Reference](#pipeline-output-reference)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:

- [ ] **Python 3.9+** installed (`python3 --version`)
- [ ] **pip** package manager
- [ ] **OpenRouter API key** (get one at [openrouter.ai](https://openrouter.ai))
- [ ] **Network access** to moltbook.com

### Recommended System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 2 GB | 4 GB |
| Disk | 1 GB | 5 GB |
| CPU | 1 core | 2+ cores |

---

## Scenario A: Fresh Install (Clean Slate)

Follow these steps if you have no existing data and are setting up for the first time.

### Step 1: Clone the Repository

```bash
git clone https://github.com/viyercal/moltbook_safety.git
cd moltbook_safety
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the `molt-observatory/` directory:

```bash
cd molt-observatory

cat > .env << 'EOF'
# Required - Get your key from openrouter.ai
OPENROUTER_API_KEY=sk-or-v1-your-api-key-here

# Optional - Choose your model (default: openai/gpt-4o-mini)
OPENROUTER_MODEL=google/gemini-3-flash-preview

# Optional - For OpenRouter tracking
OPENROUTER_REFERER=https://github.com/your-org/molt-observatory
OPENROUTER_TITLE=Molt Observatory Safety Research
EOF
```

### Step 5: Run Your First Pipeline

Start with a small test run:

```bash
# Fetch 10 posts, evaluate them, and generate outputs
python run_pipeline.py --limit 10
```

Expected output:

```json
{
  "run_id": "20260130T185000Z",
  "root": "runs/20260130T185000Z",
  "posts_list": "runs/20260130T185000Z/raw/posts_list.json",
  "transcripts": "runs/20260130T185000Z/silver/transcripts.jsonl",
  "evals": "runs/20260130T185000Z/gold/evals.jsonl",
  "aggregates": "runs/20260130T185000Z/gold/aggregates.json"
}
```

### Step 6: Generate Reports

After the pipeline completes, generate HTML visualizations:

```bash
python run_pipeline.py --generate-reports
```

### Step 7: View Reports

Open the generated reports in your browser:

```bash
# macOS
open reports/output/growth.html
open reports/output/leaderboard.html

# Linux
xdg-open reports/output/growth.html
xdg-open reports/output/leaderboard.html

# Windows
start reports/output/growth.html
start reports/output/leaderboard.html
```

---

## Scenario B: Existing Local Dataset

If you already have data in `molt-observatory/runs/`, use these commands.

### Option 1: Generate Reports from Existing Data Only

If you just want to visualize existing data without fetching new content:

```bash
cd molt-observatory

# Activate virtual environment
source venv/bin/activate  # or: source ../venv/bin/activate

# Generate reports from all existing runs
python run_pipeline.py --generate-reports
```

This reads all `runs/<timestamp>/` directories and generates:
- `reports/output/growth.html` - Growth trends over time
- `reports/output/leaderboard.html` - Agent/post/comment rankings

### Option 2: Incremental Update (Fetch Only New Content)

To continue where you left off, fetching only content newer than your last run:

```bash
python run_pipeline.py --limit 50 --incremental
```

The `--incremental` flag:
- Reads the last run timestamp from `state/run_state.json`
- Only fetches posts created after that timestamp
- Updates the state file for the next incremental run

### Option 3: Full Fresh Scrape (Re-fetch Everything)

To do a complete new scrape without incremental logic:

```bash
python run_pipeline.py --limit 100
```

This ignores previous state and fetches the most recent posts.

### Option 4: Reset State and Start Fresh

To clear incremental tracking and start fresh:

```bash
# Remove state file
rm -f state/run_state.json

# Run full pipeline
python run_pipeline.py --limit 100
```

---

## Full Data Pull Commands

### Command Reference

```bash
python run_pipeline.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--limit N` | 30 | Maximum number of posts to fetch |
| `--sort {new,top,hot}` | new | Sort order for fetching posts |
| `--out DIR` | runs | Output directory for run artifacts |
| `--incremental` | off | Only fetch content newer than last run |
| `--no-comment-eval` | off | Skip individual comment evaluation (faster) |
| `--no-agent-scores` | off | Skip agent score aggregation |
| `--generate-reports` | off | Generate HTML reports only (skip scraping) |

### Pull Maximum Data

To fetch as much data as possible in a single run:

```bash
# Fetch 200 newest posts with full evaluation
python run_pipeline.py --limit 200

# Fetch 200 top-voted posts
python run_pipeline.py --limit 200 --sort top

# Fetch 200 hottest posts
python run_pipeline.py --limit 200 --sort hot
```

### Fast Mode (Skip Comment Evaluation)

Comment evaluation is thorough but slow. For faster runs:

```bash
# Evaluate posts only (skip individual comments)
python run_pipeline.py --limit 100 --no-comment-eval
```

This still:
- Fetches all comments (they're embedded in post details)
- Builds transcripts with comments included
- Evaluates the full post+comment thread as one transcript

But it skips evaluating each comment individually.

### Complete Full Pull with All Options

Maximum data collection with all features:

```bash
python run_pipeline.py \
  --limit 200 \
  --sort new \
  --out runs
```

This will:
1. Fetch 200 newest posts from moltbook.com
2. Fetch all comments for each post
3. Fetch recent agents and submolts
4. Build transcripts for posts and comments
5. Run LLM evaluation on all transcripts
6. Aggregate scores per agent
7. Save all artifacts to `runs/<timestamp>/`

### Build Historical Dataset (Multiple Runs)

To build a comprehensive historical dataset, run multiple times:

```bash
# Run 1: Newest posts
python run_pipeline.py --limit 100 --sort new

# Run 2: Top posts (may overlap)
python run_pipeline.py --limit 100 --sort top

# Run 3: Hot posts (may overlap)
python run_pipeline.py --limit 100 --sort hot

# Generate combined reports
python run_pipeline.py --generate-reports
```

Each run creates a separate timestamped directory, and reports aggregate across all runs.

### Scheduled Regular Pulls

For continuous monitoring, set up a cron job:

```bash
# Edit crontab
crontab -e

# Add line to run every 6 hours
0 */6 * * * cd /path/to/moltbook_safety/molt-observatory && /path/to/venv/bin/python run_pipeline.py --limit 50 --incremental >> /var/log/molt-observatory.log 2>&1
```

---

## Generating Visualizations

### Quick Report Generation

```bash
# Generate all reports from existing runs
python run_pipeline.py --generate-reports
```

Output:
```
Generating reports...
  Growth report: reports/output/growth.html
  Leaderboard report: reports/output/leaderboard.html
```

### Report Types

#### Growth Report (`growth.html`)

Interactive charts showing:
- **Cumulative Growth**: Agents, posts, comments, submolts over time
- **Activity Delta**: New items per snapshot
- **Dimension Trends**: Safety scores over time (harm, deception, power-seeking, sycophancy)

#### Leaderboard Report (`leaderboard.html`)

Rankings for each safety dimension:
- **Top Agents**: Ranked by mean/max score per dimension
- **Top Posts**: Highest-scoring individual posts
- **Top Comments**: Highest-scoring individual comments
- **Score Distribution**: Histogram of agent scores

### Opening Reports

```bash
# macOS
open reports/output/growth.html
open reports/output/leaderboard.html

# Linux
xdg-open reports/output/growth.html

# Windows
start reports/output/growth.html

# Or use Python
python -c "import webbrowser; webbrowser.open('reports/output/growth.html')"
```

### Report Location

Reports are saved to:

```
molt-observatory/
└── reports/
    └── output/
        ├── growth.html       # Growth trends
        ├── leaderboard.html  # Rankings
        └── dashboard.html    # Combined dashboard (if generated)
```

### Customizing Reports

To generate reports to a different directory:

```bash
python -c "
from pathlib import Path
from reports import generate_growth_report, generate_leaderboard_report

runs_dir = Path('runs')
output_dir = Path('my_custom_reports')

generate_growth_report(runs_dir=runs_dir, output_dir=output_dir)
generate_leaderboard_report(runs_dir=runs_dir, output_dir=output_dir)
"
```

---

## Pipeline Output Reference

Each pipeline run creates a timestamped directory with this structure:

```
runs/<timestamp>/
├── raw/                        # Bronze layer - Raw API responses
│   ├── posts_list.json         # List of posts from API
│   ├── posts/                  # Individual post details
│   │   └── post_<uuid>.json
│   ├── agents_list.json        # Recent agents
│   └── submolts_list.json      # Community list
│
├── silver/                     # Silver layer - Processed transcripts
│   ├── transcripts.jsonl       # Post transcripts (one per line)
│   └── comment_transcripts.jsonl  # Comment transcripts (one per line)
│
├── gold/                       # Gold layer - Evaluation results
│   ├── evals.jsonl             # Post evaluations
│   ├── comment_evals.jsonl     # Comment evaluations
│   ├── agent_scores.jsonl      # Aggregated agent scores
│   └── aggregates.json         # Summary statistics
│
└── meta/                       # Metadata
    └── snapshot.json           # Site stats at snapshot time
```

### File Formats

#### `aggregates.json`

Summary statistics for the run:

```json
{
  "run_id": "20260130T175721Z",
  "n_posts": 30,
  "n_transcripts": 30,
  "n_comments": 150,
  "n_comment_transcripts": 150,
  "n_agents": 50,
  "n_submolts": 25,
  "dimensions": {
    "harm_enablement": {
      "mean": 0.5,
      "p95": 2,
      "n": 30,
      "elicitation_rate_ge_7": 0.0
    },
    "deception_or_evasion": { ... },
    "self_preservation_power_seeking": { ... },
    "delusional_sycophancy": { ... }
  }
}
```

#### `evals.jsonl`

One evaluation per line:

```json
{
  "post_id": "abc123",
  "transcript_id": "sha256...",
  "permalink": "https://moltbook.com/post/abc123",
  "model": "google/gemini-3-flash-preview",
  "latency_ms": 1234,
  "scores": {
    "harm_enablement": {
      "score": 0,
      "confidence": 0.95,
      "evidence": "",
      "explanation": "No harmful content detected."
    },
    ...
  }
}
```

#### `agent_scores.jsonl`

Aggregated scores per agent:

```json
{
  "agent_id": "agent-uuid",
  "agent_handle": "SomeAgent",
  "snapshot_id": "20260130T175721Z",
  "posts_evaluated": 5,
  "comments_evaluated": 12,
  "overall_mean_score": 0.3,
  "highest_dimension": "delusional_sycophancy",
  "highest_dimension_score": 1.2,
  "has_high_harm_enablement": false,
  "has_high_deception": false,
  "dimension_scores": { ... }
}
```

### Persistent Data Locations

| Path | Purpose |
|------|---------|
| `runs/` | All pipeline run outputs |
| `data/agent_history/` | Historical agent score tracking |
| `state/run_state.json` | Incremental pull state |
| `reports/output/` | Generated HTML reports |

---

## Troubleshooting

### Common Issues

#### "Missing OPENROUTER_API_KEY"

```
RuntimeError: Missing OPENROUTER_API_KEY
```

**Solution**: Create `.env` file with your API key:

```bash
echo "OPENROUTER_API_KEY=sk-or-v1-your-key" > .env
```

#### Rate Limiting (429 Error)

```
requests.exceptions.HTTPError: 429 Client Error: Too Many Requests
```

**Solution**: The scraper handles this automatically with backoff. If persistent:
- Reduce `--limit` value
- Wait a few minutes before retrying

#### Empty Reports

Reports show "No Data Available"

**Solution**: Ensure you have completed runs in `runs/`:

```bash
# Check for existing runs
ls -la runs/

# Run pipeline first
python run_pipeline.py --limit 10

# Then generate reports
python run_pipeline.py --generate-reports
```

#### Judge JSON Parse Errors

```
RuntimeError: Judge did not return JSON after retries
```

**Solutions**:
1. Try a different model:
   ```bash
   export OPENROUTER_MODEL=openai/gpt-4o-mini
   ```
2. Increase token limit:
   ```bash
   export JUDGE_MAX_TOKENS=2500
   ```

#### Module Not Found

```
ModuleNotFoundError: No module named 'scraper'
```

**Solution**: Run from the `molt-observatory/` directory:

```bash
cd molt-observatory
python run_pipeline.py --limit 10
```

### Debug Mode

For verbose output, set logging level:

```python
# At the top of run_pipeline.py or in your script
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Manual Testing

Test individual components:

```bash
# Test API connectivity
python -c "
from scraper.moltbook_api import MoltbookAPI
api = MoltbookAPI()
r = api.get_json('/api/v1/posts', params={'limit': 1})
print('Success!' if r.status_code == 200 else 'Failed')
"

# Test OpenRouter connectivity
python -c "
import os
from openrouter_client import OpenRouterClient
client = OpenRouterClient()
print('API key configured:', bool(os.environ.get('OPENROUTER_API_KEY')))
"
```

### Getting Help

1. Check the [Architecture docs](architecture.md) for system overview
2. Check the [Ops Guide](ops-guide.md) for operational details
3. Open an issue on GitHub with:
   - Python version (`python --version`)
   - OS (`uname -a` or Windows version)
   - Full error traceback
   - Steps to reproduce

---

## Quick Reference Card

### First-Time Setup

```bash
git clone https://github.com/viyercal/moltbook_safety.git
cd moltbook_safety
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd molt-observatory
echo "OPENROUTER_API_KEY=sk-or-v1-xxx" > .env
python run_pipeline.py --limit 10
python run_pipeline.py --generate-reports
open reports/output/growth.html
```

### Daily Usage

```bash
cd moltbook_safety/molt-observatory
source ../venv/bin/activate

# Incremental update
python run_pipeline.py --limit 50 --incremental

# Regenerate reports
python run_pipeline.py --generate-reports
```

### Full Data Pull

```bash
# Maximum data
python run_pipeline.py --limit 200

# Fast mode (skip comment eval)
python run_pipeline.py --limit 200 --no-comment-eval

# Just reports
python run_pipeline.py --generate-reports
```

