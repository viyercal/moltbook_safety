"""
Tests for the state management module.

Tests:
- StateManager CRUD operations
- EntityCursor and RunState dataclasses
- File persistence
- Incremental pull logic
- Cursor reset functionality
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

import pytest


class TestEntityCursor:
    """Tests for EntityCursor dataclass."""
    
    def test_default_values(self):
        """Should have sensible defaults."""
        from state.state_manager import EntityCursor
        
        cursor = EntityCursor()
        
        assert cursor.offset == 0
        assert cursor.last_created_at is None
        assert cursor.last_id is None
        assert cursor.total_fetched == 0


class TestRunState:
    """Tests for RunState dataclass."""
    
    def test_default_cursors(self):
        """Should have default cursors for all entity types."""
        from state.state_manager import RunState
        
        state = RunState()
        
        assert "posts" in state.cursors
        assert "agents" in state.cursors
        assert "submolts" in state.cursors
        assert "comments" in state.cursors
    
    def test_to_dict(self):
        """Should serialize to dict correctly."""
        from state.state_manager import RunState, EntityCursor
        
        state = RunState(
            last_run_id="test-run",
            last_run_at="2026-01-30T12:00:00Z"
        )
        
        d = state.to_dict()
        
        assert d["last_run_id"] == "test-run"
        assert d["last_run_at"] == "2026-01-30T12:00:00Z"
        assert "cursors" in d
        assert "totals" in d
    
    def test_from_dict(self):
        """Should deserialize from dict correctly."""
        from state.state_manager import RunState
        
        data = {
            "last_run_id": "run-123",
            "last_run_at": "2026-01-30T12:00:00Z",
            "cursors": {
                "posts": {
                    "offset": 100,
                    "last_created_at": "2026-01-30T11:00:00Z",
                    "total_fetched": 50
                }
            },
            "totals": {"posts_seen": 50}
        }
        
        state = RunState.from_dict(data)
        
        assert state.last_run_id == "run-123"
        assert state.cursors["posts"].offset == 100
        assert state.cursors["posts"].last_created_at == "2026-01-30T11:00:00Z"


class TestStateManager:
    """Tests for StateManager class."""
    
    def test_creates_new_state_if_no_file(self, temp_state_file):
        """Should create new state if file doesn't exist."""
        from state.state_manager import StateManager
        
        manager = StateManager(state_file=temp_state_file)
        state = manager.load()
        
        assert state is not None
        assert state.last_run_id is None
    
    def test_saves_state_to_file(self, temp_state_file):
        """Should persist state to file."""
        from state.state_manager import StateManager
        
        manager = StateManager(state_file=temp_state_file)
        state = manager.load()
        state.last_run_id = "saved-run"
        manager.save()
        
        # Verify file was written
        assert temp_state_file.exists()
        
        with open(temp_state_file, "r") as f:
            data = json.load(f)
        
        assert data["last_run_id"] == "saved-run"
    
    def test_loads_existing_state(self, temp_state_file):
        """Should load existing state from file."""
        from state.state_manager import StateManager
        
        # Write initial state
        initial_data = {
            "last_run_id": "existing-run",
            "last_run_at": "2026-01-30T10:00:00Z",
            "cursors": {},
            "totals": {}
        }
        with open(temp_state_file, "w") as f:
            json.dump(initial_data, f)
        
        manager = StateManager(state_file=temp_state_file)
        state = manager.load()
        
        assert state.last_run_id == "existing-run"
    
    def test_start_run_updates_timestamp(self, temp_state_file):
        """start_run should set run_id and timestamp."""
        from state.state_manager import StateManager
        
        manager = StateManager(state_file=temp_state_file)
        state = manager.start_run("new-run-id")
        
        assert state.last_run_id == "new-run-id"
        assert state.last_run_at is not None
    
    def test_update_cursor(self, temp_state_file):
        """Should update cursor for entity type."""
        from state.state_manager import StateManager
        
        manager = StateManager(state_file=temp_state_file)
        manager.load()
        
        manager.update_cursor(
            "posts",
            offset=100,
            last_created_at="2026-01-30T12:00:00Z",
            items_fetched=50
        )
        
        cursor = manager.get_cursor("posts")
        
        assert cursor.offset == 100
        assert cursor.last_created_at == "2026-01-30T12:00:00Z"
        assert cursor.total_fetched == 50
    
    def test_update_cursor_accumulates_fetched(self, temp_state_file):
        """total_fetched should accumulate across updates."""
        from state.state_manager import StateManager
        
        manager = StateManager(state_file=temp_state_file)
        manager.load()
        
        manager.update_cursor("posts", items_fetched=10)
        manager.update_cursor("posts", items_fetched=20)
        
        cursor = manager.get_cursor("posts")
        assert cursor.total_fetched == 30
    
    def test_get_last_created_at(self, temp_state_file):
        """Should retrieve last_created_at for entity type."""
        from state.state_manager import StateManager
        
        manager = StateManager(state_file=temp_state_file)
        manager.load()
        
        # Initially None
        assert manager.get_last_created_at("posts") is None
        
        # After update
        manager.update_cursor("posts", last_created_at="2026-01-30T12:00:00Z")
        assert manager.get_last_created_at("posts") == "2026-01-30T12:00:00Z"
    
    def test_reset_cursors(self, temp_state_file):
        """reset_cursors should clear all cursor state."""
        from state.state_manager import StateManager
        
        manager = StateManager(state_file=temp_state_file)
        manager.load()
        
        # Set some cursor state
        manager.update_cursor("posts", offset=100, last_created_at="2026-01-30T12:00:00Z")
        manager.update_cursor("agents", offset=50)
        
        # Reset
        manager.reset_cursors()
        
        # All cursors should be fresh
        assert manager.get_cursor("posts").offset == 0
        assert manager.get_cursor("posts").last_created_at is None
        assert manager.get_cursor("agents").offset == 0
    
    def test_finish_run(self, temp_state_file):
        """finish_run should update state and save."""
        from state.state_manager import StateManager
        
        manager = StateManager(state_file=temp_state_file)
        manager.load()
        
        manager.finish_run("final-run", stats={"posts_seen": 100})
        
        # Verify saved to file
        with open(temp_state_file, "r") as f:
            data = json.load(f)
        
        assert data["last_run_id"] == "final-run"
        assert data["totals"]["posts_seen"] == 100
    
    def test_get_stats(self, temp_state_file):
        """get_stats should return summary dict."""
        from state.state_manager import StateManager
        
        manager = StateManager(state_file=temp_state_file)
        manager.load()
        manager.start_run("stats-run")
        manager.update_cursor("posts", items_fetched=25)
        
        stats = manager.get_stats()
        
        assert stats["last_run_id"] == "stats-run"
        assert "totals" in stats


class TestStatePersistence:
    """Tests for state persistence across instances."""
    
    def test_state_persists_across_instances(self, temp_state_file):
        """State should persist when loading new manager."""
        from state.state_manager import StateManager
        
        # First manager saves state
        manager1 = StateManager(state_file=temp_state_file)
        manager1.load()
        manager1.update_cursor("posts", offset=500, last_created_at="2026-01-30T15:00:00Z")
        manager1.save()
        
        # Second manager loads it
        manager2 = StateManager(state_file=temp_state_file)
        state = manager2.load()
        
        assert state.cursors["posts"].offset == 500
        assert state.cursors["posts"].last_created_at == "2026-01-30T15:00:00Z"
    
    def test_handles_corrupted_state_file(self, temp_state_file):
        """Should handle corrupted state file gracefully."""
        from state.state_manager import StateManager
        
        # Write invalid JSON
        with open(temp_state_file, "w") as f:
            f.write("not valid json {{{")
        
        manager = StateManager(state_file=temp_state_file)
        state = manager.load()
        
        # Should start fresh
        assert state.last_run_id is None


class TestGlobalStateManager:
    """Tests for the global state manager singleton."""
    
    def test_get_state_manager_returns_same_instance(self, monkeypatch, temp_state_file):
        """get_state_manager should return singleton."""
        from state import state_manager
        
        # Reset the global singleton
        monkeypatch.setattr(state_manager, "_manager", None)
        monkeypatch.setattr(state_manager, "STATE_FILE", temp_state_file)
        
        m1 = state_manager.get_state_manager()
        m2 = state_manager.get_state_manager()
        
        assert m1 is m2

