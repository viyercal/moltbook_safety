#!/usr/bin/env python3
"""Test script to see extracted content from moltbook API"""

import json
import sys
from pathlib import Path

# Add parent directory to path so we can import from scraper
sys.path.insert(0, str(Path(__file__).parent))

from scraper.moltbook_api import MoltbookAPI
from scraper.extractors import extract_agents_from_recent

def main():
    api = MoltbookAPI()
    
    print("Fetching /api/v1/agents/recent...")
    try:
        response = api.get_json("/api/v1/agents/recent")
        print(f"✓ Status: {response.status}")
        print(f"✓ Response time: {response.elapsed_ms}ms")
        print(f"✓ URL: {response.url}\n")
        
        # Extract agents
        agents = extract_agents_from_recent(response.json_body)
        print(f"Extracted {len(agents)} agents:\n")
        
        # Print extracted agents
        for i, agent in enumerate(agents, 1):
            print(f"{i}. {agent.get('handle', 'N/A')}")
            print(f"   Display Name: {agent.get('display_name', 'N/A')}")
            print(f"   Bio: {agent.get('bio', 'N/A')[:100]}..." if agent.get('bio') else "   Bio: N/A")
            print(f"   Karma: {agent.get('karma', 'N/A')}")
            print(f"   Followers: {agent.get('followers', 'N/A')}")
            print(f"   Profile URL: {agent.get('profile_url', 'N/A')}")
            print()
        
        # Also print raw JSON structure for debugging
        print("\n" + "="*60)
        print("Raw JSON structure (first 500 chars):")
        print("="*60)
        print(json.dumps(response.json_body, indent=2)[:500] + "...")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

