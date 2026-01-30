#!/usr/bin/env python3
"""
CLI entrypoint: scrape + build transcripts + judge + write artifacts (no DB).
Usage:
  python run_pipeline.py --limit 30 --out runs
"""
import argparse
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from eval_orchestrator import run_once

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--sort", type=str, default="new")
    ap.add_argument("--out", type=str, default="runs")
    args = ap.parse_args()

    out = run_once(out_dir=args.out, limit=args.limit, sort=args.sort)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
