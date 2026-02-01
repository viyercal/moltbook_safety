# Molt Observatory Integration Tests
"""
Real-world integration tests that make actual API calls.

These tests are skipped by default. To run them:
    RUN_INTEGRATION_TESTS=1 python -m pytest tests/integration/ -v

Requirements:
    - Network access to moltbook.com
    - OPENROUTER_API_KEY environment variable set
    - All data stored in temporary SQLite database
"""

