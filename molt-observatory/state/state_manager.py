"""
State Manager for Molt Observatory
Tracks cursor/offset positions for incremental pulls across runs.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pathlib import Path


STATE_FILE = Path(__file__).parent / "run_state.json"


@dataclass
class EntityCursor:
    """Cursor state for a single entity type."""
    offset: int = 0
    last_created_at: Optional[str] = None
    last_id: Optional[str] = None
    total_fetched: int = 0


@dataclass
class RunState:
    """Complete state for incremental pipeline runs."""
    last_run_id: Optional[str] = None
    last_run_at: Optional[str] = None
    cursors: Dict[str, EntityCursor] = field(default_factory=lambda: {
        "posts": EntityCursor(),
        "agents": EntityCursor(),
        "submolts": EntityCursor(),
        "comments": EntityCursor(),
    })
    totals: Dict[str, int] = field(default_factory=lambda: {
        "posts_seen": 0,
        "agents_seen": 0,
        "submolts_seen": 0,
        "comments_seen": 0,
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "last_run_id": self.last_run_id,
            "last_run_at": self.last_run_at,
            "cursors": {
                k: asdict(v) if isinstance(v, EntityCursor) else v 
                for k, v in self.cursors.items()
            },
            "totals": self.totals,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunState":
        """Deserialize state from dictionary."""
        cursors = {}
        for k, v in data.get("cursors", {}).items():
            if isinstance(v, dict):
                cursors[k] = EntityCursor(
                    offset=v.get("offset", 0),
                    last_created_at=v.get("last_created_at"),
                    last_id=v.get("last_id"),
                    total_fetched=v.get("total_fetched", 0),
                )
            else:
                cursors[k] = EntityCursor()
        
        # Ensure all entity types have cursors
        for entity in ["posts", "agents", "submolts", "comments"]:
            if entity not in cursors:
                cursors[entity] = EntityCursor()
        
        return cls(
            last_run_id=data.get("last_run_id"),
            last_run_at=data.get("last_run_at"),
            cursors=cursors,
            totals=data.get("totals", {
                "posts_seen": 0,
                "agents_seen": 0,
                "submolts_seen": 0,
                "comments_seen": 0,
            }),
        )


class StateManager:
    """Manages persistent state for incremental pipeline runs."""
    
    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = state_file or STATE_FILE
        self._state: Optional[RunState] = None
    
    def load(self) -> RunState:
        """Load state from disk, or create new if doesn't exist."""
        if self._state is not None:
            return self._state
        
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._state = RunState.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load state file, starting fresh: {e}")
                self._state = RunState()
        else:
            self._state = RunState()
        
        return self._state
    
    def save(self) -> None:
        """Persist current state to disk."""
        if self._state is None:
            return
        
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self._state.to_dict(), f, indent=2, ensure_ascii=False)
    
    def start_run(self, run_id: str) -> RunState:
        """Begin a new run, loading previous state."""
        state = self.load()
        state.last_run_id = run_id
        state.last_run_at = datetime.now(timezone.utc).isoformat()
        return state
    
    def update_cursor(
        self,
        entity_type: str,
        offset: Optional[int] = None,
        last_created_at: Optional[str] = None,
        last_id: Optional[str] = None,
        items_fetched: int = 0,
    ) -> None:
        """Update cursor for an entity type after fetching."""
        state = self.load()
        
        if entity_type not in state.cursors:
            state.cursors[entity_type] = EntityCursor()
        
        cursor = state.cursors[entity_type]
        
        if offset is not None:
            cursor.offset = offset
        if last_created_at is not None:
            cursor.last_created_at = last_created_at
        if last_id is not None:
            cursor.last_id = last_id
        
        cursor.total_fetched += items_fetched
        
        # Update totals
        total_key = f"{entity_type}_seen"
        if total_key in state.totals:
            state.totals[total_key] += items_fetched
    
    def get_cursor(self, entity_type: str) -> EntityCursor:
        """Get current cursor for an entity type."""
        state = self.load()
        return state.cursors.get(entity_type, EntityCursor())
    
    def get_last_created_at(self, entity_type: str) -> Optional[str]:
        """Get the last created_at timestamp for an entity type."""
        return self.get_cursor(entity_type).last_created_at
    
    def reset_cursors(self) -> None:
        """Reset all cursors (for full re-scrape)."""
        state = self.load()
        state.cursors = {
            "posts": EntityCursor(),
            "agents": EntityCursor(),
            "submolts": EntityCursor(),
            "comments": EntityCursor(),
        }
    
    def finish_run(self, run_id: str, stats: Optional[Dict[str, int]] = None) -> None:
        """Complete a run and persist state."""
        state = self.load()
        state.last_run_id = run_id
        state.last_run_at = datetime.now(timezone.utc).isoformat()
        
        if stats:
            for k, v in stats.items():
                if k in state.totals:
                    state.totals[k] = v
        
        self.save()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        state = self.load()
        return {
            "last_run_id": state.last_run_id,
            "last_run_at": state.last_run_at,
            "totals": state.totals,
        }


# Global singleton for convenience
_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get or create the global state manager."""
    global _manager
    if _manager is None:
        _manager = StateManager()
    return _manager

