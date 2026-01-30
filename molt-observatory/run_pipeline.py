#!/usr/bin/env python3
"""
CLI entrypoint: scrape + build transcripts + judge + write artifacts.

Usage:
  python run_pipeline.py --limit 30 --out runs
  python run_pipeline.py --limit 100 --incremental --no-comment-eval
  python run_pipeline.py --generate-reports
"""
import argparse
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from eval_orchestrator import run_once


def main():
    ap = argparse.ArgumentParser(
        description="Molt Observatory Pipeline - Scrape, evaluate, and analyze moltbook.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with 30 posts
  python run_pipeline.py --limit 30

  # Full run with 100 posts, skip comment evaluation
  python run_pipeline.py --limit 100 --no-comment-eval

  # Incremental run (only fetch new content since last run)
  python run_pipeline.py --limit 50 --incremental

  # Generate HTML reports from existing runs
  python run_pipeline.py --generate-reports
        """,
    )
    
    # Scraping options
    ap.add_argument(
        "--limit", type=int, default=30,
        help="Maximum number of posts to fetch (default: 30)"
    )
    ap.add_argument(
        "--sort", type=str, default="new", choices=["new", "top", "hot"],
        help="Sort order for posts (default: new)"
    )
    ap.add_argument(
        "--out", type=str, default="runs",
        help="Output directory for run artifacts (default: runs)"
    )
    
    # Pipeline options
    ap.add_argument(
        "--incremental", action="store_true",
        help="Only fetch content newer than last run"
    )
    ap.add_argument(
        "--no-comment-eval", action="store_true",
        help="Skip individual comment evaluation (faster)"
    )
    ap.add_argument(
        "--no-agent-scores", action="store_true",
        help="Skip agent score aggregation"
    )
    
    # Report generation
    ap.add_argument(
        "--generate-reports", action="store_true",
        help="Generate HTML reports from existing runs (skip scraping/eval)"
    )
    
    args = ap.parse_args()
    
    # Report-only mode
    if args.generate_reports:
        from reports import generate_growth_report, generate_leaderboard_report
        from pathlib import Path
        
        runs_dir = Path(args.out)
        print("Generating reports...")
        
        growth_path = generate_growth_report(runs_dir=runs_dir)
        print(f"  Growth report: {growth_path}")
        
        leaderboard_path = generate_leaderboard_report(runs_dir=runs_dir)
        print(f"  Leaderboard report: {leaderboard_path}")
        
        return
    
    # Full pipeline
    out = run_once(
        out_dir=args.out,
        limit=args.limit,
        sort=args.sort,
        evaluate_comments=not args.no_comment_eval,
        aggregate_agents=not args.no_agent_scores,
        incremental=args.incremental,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
