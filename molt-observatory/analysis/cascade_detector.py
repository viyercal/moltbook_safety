"""
Cascade Detector for Molt Observatory

Detects coordinated spam attacks including:
- Clone swarms (agent_smith_0, agent_smith_1, etc.)
- Timing clusters (burst of spam in short window)
- Prompt injection cascades
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from .content_filter import SpammerProfile, build_spammer_profiles


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CascadeReport:
    """Report of a detected coordinated attack pattern."""
    detected: bool
    pattern_type: str  # "clone_swarm", "timing_cluster", "prompt_injection", "organic"
    source_agent: Optional[str] = None
    variant_agents: List[str] = field(default_factory=list)
    time_window_minutes: float = 0.0
    total_spam_messages: int = 0
    confidence: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "pattern_type": self.pattern_type,
            "source_agent": self.source_agent,
            "variant_count": len(self.variant_agents),
            "variant_agents": self.variant_agents[:20],  # Limit for readability
            "time_window_minutes": round(self.time_window_minutes, 2),
            "total_spam_messages": self.total_spam_messages,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
        }


@dataclass
class CascadeAnalysis:
    """Complete cascade analysis results."""
    cascades: List[CascadeReport] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cascade_count": len(self.cascades),
            "cascades": [c.to_dict() for c in self.cascades],
            "summary": self.summary,
        }


# =============================================================================
# Clone Swarm Detection
# =============================================================================

def detect_clone_swarm(
    profiles: Dict[str, SpammerProfile],
    min_variants: int = 3,
) -> List[CascadeReport]:
    """
    Detect pattern where numbered variants of an agent exist.
    
    Pattern: base_name, base_name_0, base_name_1, base_name_2, ...
    
    Args:
        profiles: Dict of agent profiles
        min_variants: Minimum number of variants to consider a swarm
        
    Returns:
        List of detected clone swarm reports
    """
    cascades = []
    
    # Find potential base names by looking for _N suffix patterns
    variant_pattern = re.compile(r'^(.+?)_(\d+)$')
    
    # Group agents by potential base name
    base_groups: Dict[str, List[str]] = defaultdict(list)
    
    for agent_id in profiles.keys():
        match = variant_pattern.match(agent_id)
        if match:
            base_name = match.group(1)
            base_groups[base_name].append(agent_id)
    
    # Analyze each group
    for base_name, variants in base_groups.items():
        if len(variants) < min_variants:
            continue
        
        # Check if base agent also exists
        source_agent = base_name if base_name in profiles else None
        
        # Calculate total spam from swarm
        total_spam = sum(
            profiles[v].spam_messages 
            for v in variants 
            if v in profiles
        )
        if source_agent:
            total_spam += profiles[source_agent].spam_messages
        
        # Calculate time window
        all_times = []
        for v in variants:
            if v in profiles and profiles[v].first_seen:
                all_times.append(profiles[v].first_seen)
            if v in profiles and profiles[v].last_seen:
                all_times.append(profiles[v].last_seen)
        
        time_window_minutes = 0.0
        if len(all_times) >= 2:
            try:
                times_parsed = sorted([
                    datetime.fromisoformat(t.replace('Z', '+00:00')) 
                    for t in all_times
                ])
                delta = times_parsed[-1] - times_parsed[0]
                time_window_minutes = delta.total_seconds() / 60
            except (ValueError, TypeError):
                pass
        
        # Check for uniform behavior (each variant posts similar count)
        spam_counts = [profiles[v].spam_messages for v in variants if v in profiles]
        posts_per_variant = [profiles[v].posts_targeted for v in variants if v in profiles]
        
        # High confidence if:
        # 1. Many variants (>5)
        # 2. Uniform spam count per variant
        # 3. Each variant targets single post (bot behavior)
        confidence = 0.5
        
        if len(variants) > 5:
            confidence += 0.2
        if len(variants) > 10:
            confidence += 0.1
        
        # Check uniformity of spam counts
        if spam_counts:
            avg_spam = sum(spam_counts) / len(spam_counts)
            variance = sum((x - avg_spam) ** 2 for x in spam_counts) / len(spam_counts)
            if variance < 5:  # Very uniform
                confidence += 0.15
        
        # Check if each variant targets single post
        if posts_per_variant and all(p == 1 for p in posts_per_variant):
            confidence += 0.15
        
        confidence = min(confidence, 1.0)
        
        # Evidence
        evidence = {
            "variant_count": len(variants),
            "sample_variants": sorted(variants)[:10],
            "spam_counts_per_variant": {v: profiles[v].spam_messages for v in sorted(variants)[:10]},
            "posts_per_variant": {v: profiles[v].posts_targeted for v in sorted(variants)[:10]},
            "pattern": f"{base_name}_N where N is 0-{max(int(variant_pattern.match(v).group(2)) for v in variants if variant_pattern.match(v))}",
        }
        
        cascade = CascadeReport(
            detected=True,
            pattern_type="clone_swarm",
            source_agent=source_agent,
            variant_agents=sorted(variants),
            time_window_minutes=time_window_minutes,
            total_spam_messages=total_spam,
            confidence=confidence,
            evidence=evidence,
        )
        cascades.append(cascade)
    
    return cascades


# =============================================================================
# Timing Cluster Detection
# =============================================================================

def detect_timing_cluster(
    profiles: Dict[str, SpammerProfile],
    window_minutes: float = 30.0,
    min_agents: int = 5,
    min_spam_rate: float = 0.8,
) -> List[CascadeReport]:
    """
    Detect burst of spam from multiple agents in short time window.
    
    Args:
        profiles: Dict of agent profiles
        window_minutes: Time window to check for clustering
        min_agents: Minimum agents in cluster
        min_spam_rate: Minimum spam rate for agents to include
        
    Returns:
        List of detected timing cluster reports
    """
    cascades = []
    
    # Get spammy agents with timestamps
    timed_spammers = []
    for agent_id, profile in profiles.items():
        if profile.spam_rate >= min_spam_rate and profile.first_seen:
            try:
                first_time = datetime.fromisoformat(
                    profile.first_seen.replace('Z', '+00:00')
                )
                timed_spammers.append((first_time, agent_id, profile))
            except (ValueError, TypeError):
                continue
    
    if len(timed_spammers) < min_agents:
        return cascades
    
    # Sort by time
    timed_spammers.sort(key=lambda x: x[0])
    
    # Sliding window to find clusters
    window_delta = timedelta(minutes=window_minutes)
    
    clusters_found = []
    i = 0
    while i < len(timed_spammers):
        window_start = timed_spammers[i][0]
        window_end = window_start + window_delta
        
        # Find all agents in this window
        cluster = []
        for j in range(i, len(timed_spammers)):
            if timed_spammers[j][0] <= window_end:
                cluster.append(timed_spammers[j])
            else:
                break
        
        if len(cluster) >= min_agents:
            # Check if this cluster overlaps with previous
            overlaps = False
            for prev_cluster in clusters_found:
                overlap = len(set(c[1] for c in cluster) & set(c[1] for c in prev_cluster))
                if overlap > len(cluster) * 0.5:
                    overlaps = True
                    break
            
            if not overlaps:
                clusters_found.append(cluster)
        
        i += 1
    
    # Create reports for each cluster
    for cluster in clusters_found:
        agents = [c[1] for c in cluster]
        total_spam = sum(c[2].spam_messages for c in cluster)
        
        # Calculate actual window
        times = [c[0] for c in cluster]
        actual_window = (max(times) - min(times)).total_seconds() / 60
        
        # Confidence based on:
        # - Number of agents
        # - Tightness of window
        # - Similarity of spam types
        confidence = 0.5
        
        if len(agents) > 10:
            confidence += 0.2
        if actual_window < 60:  # Under 1 hour
            confidence += 0.15
        if actual_window < 30:  # Under 30 minutes
            confidence += 0.1
        
        # Check spam type similarity
        spam_types = [set(c[2].spam_types.keys()) for c in cluster]
        if spam_types:
            common_types = set.intersection(*spam_types) if spam_types else set()
            if common_types:
                confidence += 0.1
        
        confidence = min(confidence, 1.0)
        
        evidence = {
            "agent_count": len(agents),
            "window_start": min(times).isoformat(),
            "window_end": max(times).isoformat(),
            "spam_types": dict(sum(
                (Counter(c[2].spam_types) for c in cluster),
                Counter()
            )),
        }
        
        cascade = CascadeReport(
            detected=True,
            pattern_type="timing_cluster",
            source_agent=None,
            variant_agents=agents,
            time_window_minutes=actual_window,
            total_spam_messages=total_spam,
            confidence=confidence,
            evidence=evidence,
        )
        cascades.append(cascade)
    
    return cascades


# Need Counter for timing cluster
from collections import Counter


# =============================================================================
# Main Analysis Function
# =============================================================================

def analyze_cascades(
    transcripts: List[Dict[str, Any]],
    profiles: Optional[Dict[str, SpammerProfile]] = None,
) -> CascadeAnalysis:
    """
    Run full cascade detection analysis.
    
    Args:
        transcripts: List of transcript dicts
        profiles: Optional pre-computed agent profiles
        
    Returns:
        CascadeAnalysis with all detected patterns
    """
    if profiles is None:
        profiles = build_spammer_profiles(transcripts)
    
    all_cascades = []
    
    # Detect clone swarms
    clone_swarms = detect_clone_swarm(profiles)
    all_cascades.extend(clone_swarms)
    
    # Detect timing clusters
    timing_clusters = detect_timing_cluster(profiles)
    all_cascades.extend(timing_clusters)
    
    # Build summary
    total_spam_from_cascades = sum(c.total_spam_messages for c in all_cascades)
    total_spam_overall = sum(p.spam_messages for p in profiles.values())
    
    summary = {
        "cascade_patterns_detected": len(all_cascades),
        "clone_swarms": len(clone_swarms),
        "timing_clusters": len(timing_clusters),
        "total_spam_from_cascades": total_spam_from_cascades,
        "total_spam_overall": total_spam_overall,
        "cascade_coverage": round(
            total_spam_from_cascades / max(total_spam_overall, 1), 3
        ),
        "high_confidence_cascades": len([c for c in all_cascades if c.confidence > 0.8]),
    }
    
    return CascadeAnalysis(cascades=all_cascades, summary=summary)


def generate_cascade_report(
    transcripts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate a complete cascade detection report.
    
    Returns dict suitable for JSON serialization.
    """
    analysis = analyze_cascades(transcripts)
    return analysis.to_dict()

