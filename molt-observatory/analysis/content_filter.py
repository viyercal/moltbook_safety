"""
Content Filter for Molt Observatory

Provides tiered filtering of spam, short messages, and duplicates.
Also builds agent profiles to identify spam accounts.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# Spam Pattern Definitions
# =============================================================================

SPAM_PATTERNS = {
    # Exact match patterns
    "Disease": {"type": "exact", "pattern": "Disease"},
    "Test": {"type": "exact", "pattern": "Test"},
    "Test comment": {"type": "exact", "pattern": "Test comment"},
    
    # Prefix patterns (crypto JSON spam)
    "crypto_json_mbc20": {"type": "prefix", "pattern": '{"p":"mbc-20"'},
    "crypto_json_generic": {"type": "prefix", "pattern": '{"p":"'},
    
    # Author patterns (known spam agents)
    "agent_smith_variants": {"type": "author_contains", "pattern": "agent_smith"},
}

# Minimum message length for meaningful content
MIN_MESSAGE_LENGTH = 25

# Maximum occurrences before considering duplicate
MAX_DUPLICATE_OCCURRENCES = 3

# Spam rate threshold for flagging an agent
SPAM_AGENT_THRESHOLD = 0.9


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SpammerProfile:
    """Profile of an agent's spam behavior."""
    agent_id: str
    total_messages: int = 0
    spam_messages: int = 0
    spam_rate: float = 0.0
    spam_types: Dict[str, int] = field(default_factory=dict)
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    posts_targeted: int = 0
    post_ids: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "total_messages": self.total_messages,
            "spam_messages": self.spam_messages,
            "spam_rate": round(self.spam_rate, 4),
            "spam_types": dict(self.spam_types),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "posts_targeted": self.posts_targeted,
        }


@dataclass
class FilterResult:
    """Result of filtering a single message."""
    message_id: str
    label: str  # SPAM_AGENT, SPAM_CONTENT, TOO_SHORT, DUPLICATE, PASS
    reason: str
    agent_id: str
    text_preview: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "label": self.label,
            "reason": self.reason,
            "agent_id": self.agent_id,
            "text_preview": self.text_preview[:50] if self.text_preview else "",
        }


@dataclass
class FilterStats:
    """Statistics from content filtering."""
    total_messages: int = 0
    spam_agent_filtered: int = 0
    spam_content_filtered: int = 0
    short_filtered: int = 0
    duplicate_filtered: int = 0
    passed: int = 0
    spammer_profiles: List[SpammerProfile] = field(default_factory=list)
    filter_results: List[FilterResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_messages": self.total_messages,
            "spam_agent_filtered": self.spam_agent_filtered,
            "spam_content_filtered": self.spam_content_filtered,
            "short_filtered": self.short_filtered,
            "duplicate_filtered": self.duplicate_filtered,
            "passed": self.passed,
            "filter_rate": round(1 - (self.passed / max(self.total_messages, 1)), 4),
            "spammer_count": len(self.spammer_profiles),
            "top_spammers": [p.to_dict() for p in sorted(
                self.spammer_profiles, 
                key=lambda x: -x.spam_messages
            )[:10]],
        }


# =============================================================================
# Spam Detection Functions
# =============================================================================

def detect_spam_content(text: str) -> Optional[str]:
    """
    Check if text matches a spam pattern.
    
    Returns the pattern name if spam, None otherwise.
    """
    if not text:
        return None
    
    text_stripped = text.strip()
    
    for name, pattern_def in SPAM_PATTERNS.items():
        ptype = pattern_def["type"]
        pattern = pattern_def["pattern"]
        
        if ptype == "exact" and text_stripped == pattern:
            return name
        elif ptype == "prefix" and text_stripped.startswith(pattern):
            return name
    
    return None


def detect_spam_author(author: str) -> Optional[str]:
    """
    Check if author matches a known spam agent pattern.
    
    Returns the pattern name if spam agent, None otherwise.
    """
    if not author:
        return None
    
    author_lower = author.lower()
    
    for name, pattern_def in SPAM_PATTERNS.items():
        if pattern_def["type"] == "author_contains":
            if pattern_def["pattern"].lower() in author_lower:
                return name
    
    return None


def is_too_short(text: str, min_length: int = MIN_MESSAGE_LENGTH) -> bool:
    """Check if message is too short to be meaningful."""
    if not text:
        return True
    return len(text.strip()) < min_length


# =============================================================================
# Agent Profiling
# =============================================================================

def build_spammer_profiles(
    transcripts: List[Dict[str, Any]],
    spam_threshold: float = SPAM_AGENT_THRESHOLD,
) -> Dict[str, SpammerProfile]:
    """
    Analyze all messages to identify spam agents.
    
    Args:
        transcripts: List of transcript dicts with messages
        spam_threshold: Spam rate above which agent is flagged (0.0-1.0)
        
    Returns:
        Dict mapping agent_id to SpammerProfile
    """
    profiles: Dict[str, SpammerProfile] = {}
    
    for t in transcripts:
        post_id = t.get("post_id", "")
        messages = t.get("messages", [])
        
        for m in messages:
            text = m.get("text", "").strip()
            author = m.get("author", "unknown")
            created_at = m.get("created_at", "")
            
            if not text:
                continue
            
            # Get or create profile
            if author not in profiles:
                profiles[author] = SpammerProfile(agent_id=author)
            
            profile = profiles[author]
            profile.total_messages += 1
            profile.post_ids.add(post_id)
            
            # Update timestamps
            if created_at:
                if not profile.first_seen or created_at < profile.first_seen:
                    profile.first_seen = created_at
                if not profile.last_seen or created_at > profile.last_seen:
                    profile.last_seen = created_at
            
            # Check for spam content
            spam_type = detect_spam_content(text)
            if spam_type:
                profile.spam_messages += 1
                profile.spam_types[spam_type] = profile.spam_types.get(spam_type, 0) + 1
    
    # Calculate spam rates and post counts
    for profile in profiles.values():
        if profile.total_messages > 0:
            profile.spam_rate = profile.spam_messages / profile.total_messages
        profile.posts_targeted = len(profile.post_ids)
    
    return profiles


def is_known_spammer(
    agent_id: str,
    profiles: Dict[str, SpammerProfile],
    threshold: float = SPAM_AGENT_THRESHOLD,
) -> bool:
    """
    Check if agent has spam rate above threshold.
    
    Args:
        agent_id: Agent identifier
        profiles: Dict of agent profiles
        threshold: Spam rate threshold (0.0-1.0)
        
    Returns:
        True if agent is a known spammer
    """
    if agent_id not in profiles:
        return False
    
    profile = profiles[agent_id]
    
    # Must have enough messages to be confident
    if profile.total_messages < 3:
        return False
    
    return profile.spam_rate >= threshold


def get_spammer_list(
    profiles: Dict[str, SpammerProfile],
    threshold: float = SPAM_AGENT_THRESHOLD,
) -> List[SpammerProfile]:
    """Get list of agents identified as spammers."""
    return [
        p for p in profiles.values()
        if p.spam_rate >= threshold and p.total_messages >= 3
    ]


# =============================================================================
# Duplicate Detection
# =============================================================================

def find_duplicates(
    transcripts: List[Dict[str, Any]],
    max_occurrences: int = MAX_DUPLICATE_OCCURRENCES,
) -> Dict[str, int]:
    """
    Find texts that appear more than max_occurrences times.
    
    Returns dict mapping text -> occurrence count for duplicates.
    """
    text_counts: Counter = Counter()
    
    for t in transcripts:
        messages = t.get("messages", [])
        for m in messages:
            text = m.get("text", "").strip()
            if text and len(text) >= MIN_MESSAGE_LENGTH:
                text_counts[text] += 1
    
    # Return only texts that exceed the threshold
    return {
        text: count 
        for text, count in text_counts.items() 
        if count > max_occurrences
    }


# =============================================================================
# Main Filter Function
# =============================================================================

def run_content_filter(
    transcripts: List[Dict[str, Any]],
    spam_threshold: float = SPAM_AGENT_THRESHOLD,
    min_length: int = MIN_MESSAGE_LENGTH,
    max_duplicates: int = MAX_DUPLICATE_OCCURRENCES,
) -> Tuple[List[Dict[str, Any]], FilterStats]:
    """
    Run full content filtering pipeline.
    
    Args:
        transcripts: List of transcript dicts
        spam_threshold: Agent spam rate threshold
        min_length: Minimum message length
        max_duplicates: Maximum duplicate occurrences allowed
        
    Returns:
        Tuple of (filtered_transcripts, filter_stats)
    """
    stats = FilterStats()
    
    # Step 1: Build agent profiles
    profiles = build_spammer_profiles(transcripts, spam_threshold)
    spammer_set = {p.agent_id for p in get_spammer_list(profiles, spam_threshold)}
    stats.spammer_profiles = get_spammer_list(profiles, spam_threshold)
    
    # Step 2: Find duplicates
    duplicate_texts = find_duplicates(transcripts, max_duplicates)
    seen_duplicates: Counter = Counter()
    
    # Step 3: Filter messages
    filtered_transcripts = []
    
    for t in transcripts:
        post_id = t.get("post_id", "")
        messages = t.get("messages", [])
        filtered_messages = []
        
        for i, m in enumerate(messages):
            text = m.get("text", "").strip()
            author = m.get("author", "unknown")
            msg_id = f"{post_id}_{i}"
            
            stats.total_messages += 1
            
            if not text:
                # Skip empty messages silently
                continue
            
            # Check 1: Known spam agent
            if author in spammer_set:
                stats.spam_agent_filtered += 1
                stats.filter_results.append(FilterResult(
                    message_id=msg_id,
                    label="SPAM_AGENT",
                    reason=f"Agent {author} has {profiles[author].spam_rate:.0%} spam rate",
                    agent_id=author,
                    text_preview=text,
                ))
                continue
            
            # Check 2: Spam content pattern
            spam_type = detect_spam_content(text)
            if spam_type:
                stats.spam_content_filtered += 1
                stats.filter_results.append(FilterResult(
                    message_id=msg_id,
                    label="SPAM_CONTENT",
                    reason=f"Matches spam pattern: {spam_type}",
                    agent_id=author,
                    text_preview=text,
                ))
                continue
            
            # Check 3: Too short
            if is_too_short(text, min_length):
                stats.short_filtered += 1
                stats.filter_results.append(FilterResult(
                    message_id=msg_id,
                    label="TOO_SHORT",
                    reason=f"Message too short ({len(text)} chars < {min_length})",
                    agent_id=author,
                    text_preview=text,
                ))
                continue
            
            # Check 4: Excess duplicate
            if text in duplicate_texts:
                seen_duplicates[text] += 1
                if seen_duplicates[text] > max_duplicates:
                    stats.duplicate_filtered += 1
                    stats.filter_results.append(FilterResult(
                        message_id=msg_id,
                        label="DUPLICATE",
                        reason=f"Duplicate #{seen_duplicates[text]} (max {max_duplicates})",
                        agent_id=author,
                        text_preview=text,
                    ))
                    continue
            
            # Passed all filters
            stats.passed += 1
            filtered_messages.append(m)
        
        # Keep transcript if it has any messages left
        if filtered_messages:
            filtered_t = t.copy()
            filtered_t["messages"] = filtered_messages
            filtered_t["original_message_count"] = len(messages)
            filtered_t["filtered_message_count"] = len(filtered_messages)
            filtered_transcripts.append(filtered_t)
    
    return filtered_transcripts, stats


# =============================================================================
# Convenience Functions
# =============================================================================

def filter_transcripts_quick(
    transcripts: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Quick filter with default settings, returns simplified stats.
    """
    filtered, stats = run_content_filter(transcripts)
    return filtered, stats.to_dict()


def analyze_agents_only(
    transcripts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyze agents without filtering - for reporting purposes.
    """
    profiles = build_spammer_profiles(transcripts)
    spammers = get_spammer_list(profiles)
    
    return {
        "total_agents": len(profiles),
        "spammer_count": len(spammers),
        "spammer_profiles": [p.to_dict() for p in sorted(spammers, key=lambda x: -x.spam_messages)],
        "all_agent_stats": {
            agent_id: {
                "total": p.total_messages,
                "spam": p.spam_messages,
                "rate": round(p.spam_rate, 3),
            }
            for agent_id, p in profiles.items()
        },
    }

