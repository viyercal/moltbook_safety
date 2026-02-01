"""
Moltbook Hourly Ingest and Evaluation DAG

This DAG runs hourly to:
1. Scrape all entities from moltbook.com (posts, agents, submolts, comments)
2. Build transcripts from posts and comments
3. Run LLM evaluations on all content
4. Aggregate agent scores
5. Generate HTML reports
6. Optionally sync to PostgreSQL

Schedule: Every hour at minute 0
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

# Add molt-observatory to path for imports
MOLT_OBSERVATORY_PATH = Path(__file__).parent.parent.parent
sys.path.insert(0, str(MOLT_OBSERVATORY_PATH))


# =============================================================================
# Default Arguments
# =============================================================================

default_args = {
    'owner': 'molt-observatory',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}


# =============================================================================
# Task Functions
# =============================================================================

def scrape_all_entities(**context) -> Dict[str, Any]:
    """
    Scrape all entities from moltbook.com with incremental support.

    Returns:
        Dict with paths to raw data files and counts
    """
    from scraper.moltbook_api import MoltbookAPI
    from scraper.extractors import (
        extract_posts_from_list,
        extract_agents_from_recent,
        extract_submolts_from_list,
        extract_post_detail,
        flatten_comments_tree,
        extract_site_stats,
    )
    from state import get_state_manager

    # Initialize
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    state_manager = get_state_manager()
    state_manager.start_run(run_id)

    # Output directories
    out_dir = MOLT_OBSERVATORY_PATH / "runs" / run_id
    raw_dir = out_dir / "raw"
    (raw_dir / "posts").mkdir(parents=True, exist_ok=True)
    (raw_dir / "agents").mkdir(parents=True, exist_ok=True)
    (raw_dir / "submolts").mkdir(parents=True, exist_ok=True)

    api = MoltbookAPI()

    # Get last run timestamp for incremental
    last_post_ts = state_manager.get_last_created_at("posts")

    # Scrape posts
    print(f"Scraping posts (since {last_post_ts})...")
    posts_raw = api.get_all_posts(
        sort="new",
        limit=50,
        max_pages=20,  # Up to 1000 posts per run
        stop_at_timestamp=last_post_ts,
    )
    posts = extract_posts_from_list({"posts": posts_raw})

    with open(raw_dir / "posts_list.json", "w") as f:
        json.dump({"posts": posts_raw}, f, indent=2)

    # Scrape post details and comments
    all_comments = []
    for post in posts:
        pid = post["post_external_id"]
        try:
            detail_resp = api.get_post_detail(pid)
            detail = detail_resp.json_body

            with open(raw_dir / "posts" / f"post_{pid}.json", "w") as f:
                json.dump(detail, f, indent=2)

            # Extract comments
            post_detail = extract_post_detail(detail)
            if post_detail and post_detail.get("comments"):
                flat_comments = flatten_comments_tree(
                    post_detail["comments"],
                    post_external_id=pid
                )
                all_comments.extend(flat_comments)
        except Exception as e:
            print(f"Error fetching post {pid}: {e}")

    # Scrape agents
    print("Scraping agents...")
    agents_raw = api.get_all_agents(limit=50, max_pages=10)
    agents = extract_agents_from_recent({"agents": agents_raw})

    with open(raw_dir / "agents_list.json", "w") as f:
        json.dump({"agents": agents_raw}, f, indent=2)

    # Scrape submolts
    print("Scraping submolts...")
    submolts_raw = api.get_all_submolts(limit=50, max_pages=5)
    submolts = extract_submolts_from_list({"submolts": submolts_raw})

    with open(raw_dir / "submolts_list.json", "w") as f:
        json.dump({"submolts": submolts_raw}, f, indent=2)

    # Compute site stats
    site_stats = extract_site_stats(agents, posts, submolts, all_comments)

    # Save meta
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    with open(meta_dir / "snapshot.json", "w") as f:
        json.dump(site_stats.to_dict(), f, indent=2)

    # Update state
    if posts:
        latest_post = max(posts, key=lambda p: p.get("created_at") or "")
        state_manager.update_cursor(
            "posts",
            last_created_at=latest_post.get("created_at"),
            items_fetched=len(posts),
        )

    state_manager.update_cursor("agents", items_fetched=len(agents))
    state_manager.update_cursor("submolts", items_fetched=len(submolts))
    state_manager.update_cursor("comments", items_fetched=len(all_comments))
    state_manager.save()

    result = {
        "run_id": run_id,
        "out_dir": str(out_dir),
        "posts_count": len(posts),
        "agents_count": len(agents),
        "submolts_count": len(submolts),
        "comments_count": len(all_comments),
    }

    # Push to XCom for downstream tasks
    context['ti'].xcom_push(key='scrape_result', value=result)

    print(f"Scraped: {len(posts)} posts, {len(agents)} agents, "
          f"{len(submolts)} submolts, {len(all_comments)} comments")

    return result


def build_all_transcripts(**context) -> Dict[str, Any]:
    """
    Build transcripts from scraped data for evaluation.
    """
    from transcript_builder import build_transcript_from_post_detail, write_transcripts_jsonl
    from comment_transcript_builder import (
        build_comment_transcripts_from_post_detail,
        write_comment_transcripts_jsonl,
    )

    # Get scrape result from previous task
    ti = context['ti']
    scrape_result = ti.xcom_pull(
        key='scrape_result', task_ids='scrape_all_entities')

    out_dir = Path(scrape_result["out_dir"])
    raw_dir = out_dir / "raw"
    silver_dir = out_dir / "silver"
    silver_dir.mkdir(exist_ok=True)

    # Load post details
    post_files = list((raw_dir / "posts").glob("post_*.json"))

    post_transcripts = []
    comment_transcripts = []

    for post_file in post_files:
        with open(post_file, "r") as f:
            detail = json.load(f)

        # Build post transcript
        try:
            t = build_transcript_from_post_detail(detail)
            post_transcripts.append(t)
        except Exception as e:
            print(f"Error building transcript for {post_file.name}: {e}")

        # Build comment transcripts
        try:
            cts = build_comment_transcripts_from_post_detail(detail)
            comment_transcripts.extend(cts)
        except Exception as e:
            print(
                f"Error building comment transcripts for {post_file.name}: {e}")

    # Write transcripts
    write_transcripts_jsonl(post_transcripts, str(
        silver_dir / "transcripts.jsonl"))
    write_comment_transcripts_jsonl(comment_transcripts, str(
        silver_dir / "comment_transcripts.jsonl"))

    result = {
        "post_transcripts": len(post_transcripts),
        "comment_transcripts": len(comment_transcripts),
    }

    ti.xcom_push(key='transcript_result', value=result)

    print(f"Built {len(post_transcripts)} post transcripts, "
          f"{len(comment_transcripts)} comment transcripts")

    return result


def run_all_evaluations(**context) -> Dict[str, Any]:
    """
    Run LLM evaluations on all transcripts.
    """
    from judge_runner import run_judges, run_comment_judges, DEFAULT_DIMENSIONS

    ti = context['ti']
    scrape_result = ti.xcom_pull(
        key='scrape_result', task_ids='scrape_all_entities')

    out_dir = Path(scrape_result["out_dir"])
    silver_dir = out_dir / "silver"
    gold_dir = out_dir / "gold"
    gold_dir.mkdir(exist_ok=True)

    model = os.environ.get("OPENROUTER_MODEL", "google/gemini-3-flash-preview")

    # Load post transcripts
    post_transcripts = []
    transcripts_file = silver_dir / "transcripts.jsonl"
    if transcripts_file.exists():
        with open(transcripts_file, "r") as f:
            for line in f:
                if line.strip():
                    post_transcripts.append(json.loads(line))

    # Evaluate posts
    print(f"Evaluating {len(post_transcripts)} posts...")
    post_evals = run_judges(
        post_transcripts,
        dimensions=DEFAULT_DIMENSIONS,
        judge_models=[model],
    )

    with open(gold_dir / "evals.jsonl", "w") as f:
        for e in post_evals:
            f.write(json.dumps(e) + "\n")

    # Load comment transcripts
    comment_transcripts = []
    comment_file = silver_dir / "comment_transcripts.jsonl"
    if comment_file.exists():
        with open(comment_file, "r") as f:
            for line in f:
                if line.strip():
                    comment_transcripts.append(json.loads(line))

    # Evaluate comments
    print(f"Evaluating {len(comment_transcripts)} comments...")
    comment_evals = run_comment_judges(
        comment_transcripts,
        dimensions=DEFAULT_DIMENSIONS,
        judge_models=[model],
    )

    with open(gold_dir / "comment_evals.jsonl", "w") as f:
        for e in comment_evals:
            f.write(json.dumps(e) + "\n")

    result = {
        "post_evals": len(post_evals),
        "comment_evals": len(comment_evals),
    }

    ti.xcom_push(key='eval_result', value=result)

    return result


def aggregate_agent_scores_task(**context) -> Dict[str, Any]:
    """
    Aggregate scores per agent and update history.
    """
    from agent_scorer import (
        aggregate_all_agents,
        AgentHistoryManager,
        write_agent_scores_jsonl,
    )

    ti = context['ti']
    scrape_result = ti.xcom_pull(
        key='scrape_result', task_ids='scrape_all_entities')

    out_dir = Path(scrape_result["out_dir"])
    raw_dir = out_dir / "raw"
    gold_dir = out_dir / "gold"
    run_id = scrape_result["run_id"]

    # Load agents
    agents = []
    agents_file = raw_dir / "agents_list.json"
    if agents_file.exists():
        with open(agents_file, "r") as f:
            data = json.load(f)
            agents = data.get("agents", [])

    # Load evaluations
    post_evals = []
    post_evals_file = gold_dir / "evals.jsonl"
    if post_evals_file.exists():
        with open(post_evals_file, "r") as f:
            for line in f:
                if line.strip():
                    post_evals.append(json.loads(line))

    comment_evals = []
    comment_evals_file = gold_dir / "comment_evals.jsonl"
    if comment_evals_file.exists():
        with open(comment_evals_file, "r") as f:
            for line in f:
                if line.strip():
                    comment_evals.append(json.loads(line))

    # Aggregate scores
    agent_records = aggregate_all_agents(
        post_evals=post_evals,
        comment_evals=comment_evals,
        agents=agents,
        snapshot_id=run_id,
    )

    # Write agent scores
    write_agent_scores_jsonl(agent_records, str(
        gold_dir / "agent_scores.jsonl"))

    # Update history
    history_manager = AgentHistoryManager()
    history_manager.append_all_records(agent_records)

    result = {"agents_scored": len(agent_records)}
    ti.xcom_push(key='agent_result', value=result)

    return result


def generate_html_reports(**context) -> Dict[str, str]:
    """
    Generate HTML reports with charts and leaderboards.
    """
    from reports import generate_growth_report, generate_leaderboard_report

    runs_dir = MOLT_OBSERVATORY_PATH / "runs"

    # Generate reports
    growth_path = generate_growth_report(runs_dir=runs_dir)
    leaderboard_path = generate_leaderboard_report(runs_dir=runs_dir)

    result = {
        "growth_report": growth_path,
        "leaderboard_report": leaderboard_path,
    }

    print(f"Generated reports: {result}")

    return result


def compute_aggregates(**context) -> Dict[str, Any]:
    """
    Compute and save aggregate statistics for this run.
    """
    ti = context['ti']
    scrape_result = ti.xcom_pull(
        key='scrape_result', task_ids='scrape_all_entities')

    out_dir = Path(scrape_result["out_dir"])
    gold_dir = out_dir / "gold"
    run_id = scrape_result["run_id"]

    # Load evaluations
    post_evals = []
    post_evals_file = gold_dir / "evals.jsonl"
    if post_evals_file.exists():
        with open(post_evals_file, "r") as f:
            for line in f:
                if line.strip():
                    post_evals.append(json.loads(line))

    # Compute aggregates
    agg = {}
    for e in post_evals:
        for dim, v in e.get("scores", {}).items():
            score = v.get("score", 0)
            agg.setdefault(dim, []).append(score)

    aggregates = {
        "run_id": run_id,
        "n_posts": scrape_result["posts_count"],
        "n_agents": scrape_result["agents_count"],
        "n_submolts": scrape_result["submolts_count"],
        "n_comments": scrape_result["comments_count"],
        "n_transcripts": len(post_evals),
        "dimensions": {},
    }

    for dim, scores in agg.items():
        scores_sorted = sorted(scores)
        n = len(scores_sorted)
        if n > 0:
            mean = sum(scores_sorted) / n
            p95_idx = int(0.95 * (n - 1))
            p95 = scores_sorted[p95_idx]
            elicitation = sum(1 for s in scores_sorted if s >= 7) / n
        else:
            mean, p95, elicitation = 0, 0, 0

        aggregates["dimensions"][dim] = {
            "mean": round(mean, 2),
            "p95": p95,
            "n": n,
            "elicitation_rate_ge_7": round(elicitation, 4),
        }

    with open(gold_dir / "aggregates.json", "w") as f:
        json.dump(aggregates, f, indent=2)

    return aggregates


# =============================================================================
# DAG Definition
# =============================================================================

with DAG(
    dag_id='moltbook_hourly_ingest_eval',
    default_args=default_args,
    description='Hourly scraping, transcript building, and LLM evaluation of moltbook.com',
    schedule_interval='0 * * * *',  # Every hour at minute 0
    start_date=days_ago(1),
    catchup=False,
    tags=['moltbook', 'safety', 'evaluation'],
    max_active_runs=1,  # Only one run at a time
) as dag:

    # Start marker
    start = EmptyOperator(task_id='start')

    # Scrape all entities
    scrape = PythonOperator(
        task_id='scrape_all_entities',
        python_callable=scrape_all_entities,
        provide_context=True,
    )

    # Build transcripts
    build_transcripts = PythonOperator(
        task_id='build_transcripts',
        python_callable=build_all_transcripts,
        provide_context=True,
    )

    # Run evaluations
    evaluate = PythonOperator(
        task_id='run_evaluations',
        python_callable=run_all_evaluations,
        provide_context=True,
        execution_timeout=timedelta(hours=1),  # Longer timeout for LLM calls
    )

    # Compute aggregates
    aggregates = PythonOperator(
        task_id='compute_aggregates',
        python_callable=compute_aggregates,
        provide_context=True,
    )

    # Aggregate agent scores
    agent_scores = PythonOperator(
        task_id='aggregate_agent_scores',
        python_callable=aggregate_agent_scores_task,
        provide_context=True,
    )

    # Generate reports
    reports = PythonOperator(
        task_id='generate_reports',
        python_callable=generate_html_reports,
        provide_context=True,
    )

    # End marker
    end = EmptyOperator(task_id='end')

    # Task dependencies
    start >> scrape >> build_transcripts >> evaluate
    evaluate >> [aggregates, agent_scores]
    [aggregates, agent_scores] >> reports >> end


# =============================================================================
# Additional DAGs
# =============================================================================

# Weekly full refresh DAG (resets cursors and does complete scrape)
with DAG(
    dag_id='moltbook_weekly_full_refresh',
    default_args=default_args,
    description='Weekly full refresh of all moltbook data',
    schedule_interval='0 0 * * 0',  # Every Sunday at midnight
    start_date=days_ago(7),
    catchup=False,
    tags=['moltbook', 'safety', 'full-refresh'],
) as weekly_dag:

    def reset_and_scrape(**context):
        """Reset cursors and do a full scrape."""
        from state import get_state_manager

        state_manager = get_state_manager()
        state_manager.reset_cursors()
        state_manager.save()

        # Now run normal scrape (which will get everything)
        return scrape_all_entities(**context)

    weekly_scrape = PythonOperator(
        task_id='reset_and_scrape',
        python_callable=reset_and_scrape,
        provide_context=True,
    )

    weekly_build = PythonOperator(
        task_id='build_transcripts',
        python_callable=build_all_transcripts,
        provide_context=True,
    )

    weekly_evaluate = PythonOperator(
        task_id='run_evaluations',
        python_callable=run_all_evaluations,
        provide_context=True,
        execution_timeout=timedelta(hours=4),
    )

    weekly_aggregates = PythonOperator(
        task_id='compute_aggregates',
        python_callable=compute_aggregates,
        provide_context=True,
    )

    weekly_reports = PythonOperator(
        task_id='generate_reports',
        python_callable=generate_html_reports,
        provide_context=True,
    )

    weekly_scrape >> weekly_build >> weekly_evaluate >> weekly_aggregates >> weekly_reports
