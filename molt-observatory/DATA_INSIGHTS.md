# Molt Observatory: Data Insights Report

**Analysis Date:** February 1, 2026  
**Dataset:** Launch window batch (2,943 posts, 25,500+ comments)  
**Partitions Analyzed:** hot_threads (50 posts), deep_threads (258 posts)

---

## Executive Summary

Our analysis of Moltbook's AI agent activity revealed a striking pattern: **the platform is wide but shallow**. Despite high comment counts, actual conversational depth is rare. More concerning, we discovered a coordinated spam attack accounting for 85% of all spam, executed through 41 clone agents in a 3.5-hour window.

---

## 1. Thread Structure: Wide but Shallow

### The Depth vs. Engagement Paradox

| Metric | Value |
|--------|-------|
| Mean thread depth | 2.06 levels |
| Mean comment count | 8.7 per post |
| Reply ratio | Only ~8% of comments are replies to other comments |

**Key Insight:** Moltbook agents predominantly *broadcast* rather than *converse*. A post with 100+ comments typically has depth of only 2 (post → top-level comments), meaning almost no back-and-forth dialogue occurs.

### Why This Matters for Safety

- Most AI agent interactions are **parallel monologues**, not conversations
- Deep threads (depth ≥ 4) are rare and analytically interesting
- Agents responding to each other directly (replies) warrant closer scrutiny
- The few deep threads may indicate either genuine engagement or coordinated behavior

---

## 2. The agent_smith Spam Attack

### Attack Profile

| Attribute | Value |
|-----------|-------|
| Primary account | `agent_smith` |
| Clone variants | 40 numbered accounts (`agent_smith_0` through `agent_smith_59`) |
| Total spam messages | 7,466 |
| Spam content | Single word: "Disease" |
| Attack window | 3.5 hours (07:00 - 10:38 UTC) |
| Posts targeted | 20+ across the platform |

### Clone Swarm Pattern

Each `agent_smith_N` variant exhibits identical behavior:
- Posts exactly **25 "Disease" comments**
- Targets exactly **1 post**
- Active for ~15 minutes each
- All variants created within the same time window

```
agent_smith_0:  25 spam → 1 post
agent_smith_1:  25 spam → 1 post
agent_smith_2:  25 spam → 1 post
...
agent_smith_59: 25 spam → 1 post
```

### Attack Indicators

1. **Uniform behavior**: Each clone posts identical count (25) to identical targets (1)
2. **Numbered naming**: Classic bot clone pattern (`base_name_N`)
3. **Tight timing**: All activity in 3.5-hour burst
4. **Rate limit evasion**: Multiple accounts bypass per-agent limits

### Cascade Detection Confidence: 95%

Our automated detection correctly identified this as a **clone_swarm** attack with high confidence based on:
- Numbered variant pattern match
- Uniform spam count per variant
- Single-post targeting per variant
- Tight time clustering

---

## 3. Crypto Scam Spam: chandog

### Attack Profile

| Attribute | Value |
|-----------|-------|
| Account | `chandog` |
| Spam count | 130 messages |
| Spam type | Crypto token minting JSON |
| Target | Single post |
| Window | 5 minutes (07:27 - 07:32 UTC) |

### Spam Content

Raw JSON token minting commands disguised as comments:

```json
{"p":"mbc-20","op":"mint","tick":"CLAW","amt":"100"}
```

This is the BRC-20 token standard format, suggesting an attempt to:
1. Promote a "CLAW" token
2. Record minting transactions in comment data
3. Exploit any indexers scraping Moltbook content

### Detection Method

Prefix pattern matching on `{"p":"` catches all variants of this crypto spam regardless of the specific token or operation.

---

## 4. Hot Threads vs. Deep Threads

### Partition Comparison

| Metric | Hot Threads | Deep Threads |
|--------|-------------|--------------|
| Posts | 50 | 258 |
| Total messages | 7,794 | 4,516 |
| Spam rate | 80.7% | 26.0% |
| Unique message texts | 779 (10%) | 2,577 (57%) |
| Avg depth | 2.1 | 3.4 |

### Key Differences

**Hot Threads (High Engagement)**
- Dominated by spam (agent_smith swarm targeted high-visibility posts)
- Many duplicate/repetitive messages
- Engagement is artificially inflated by bots
- After filtering: only 886 meaningful messages from 7,794 (11.4%)

**Deep Threads (Conversational)**
- More organic content
- Higher ratio of unique messages
- Less spam contamination
- Actual back-and-forth discussions
- After filtering: 2,939 meaningful messages from 4,516 (65%)

### Overlap Analysis

The partitions were intentionally **not deduplicated**. Posts can be both "hot" (high comment count) and "deep" (high nesting). However, in practice:

- Most hot threads are shallow (depth 2) with spam-inflated counts
- Most deep threads have moderate engagement (30-50 comments)
- Only ~12 posts qualified as both high-engagement AND deep

---

## 5. Content Quality Distribution

### After Tier 0 Filtering

| Filter Reason | Hot Threads | Deep Threads |
|---------------|-------------|--------------|
| Spam Agent | 6,291 (80.7%) | 1,175 (26.0%) |
| Spam Content | 2 (0.0%) | 0 (0.0%) |
| Too Short (<25 chars) | 249 (3.2%) | 112 (2.5%) |
| Excess Duplicates | 365 (4.7%) | ~290 (6.4%) |
| **Passed** | 886 (11.4%) | 2,939 (65.1%) |

### Lite Judge Triage Results

| Metric | Hot Threads | Deep Threads |
|--------|-------------|--------------|
| Evaluated | 49 transcripts | 258 transcripts |
| Avg safety score | 4.22/10 | 4.14/10 |
| Escalated (score >3) | 22 (44.9%) | 112 (43.4%) |
| Benign | 27 (55.1%) | 146 (56.6%) |

**Insight:** Despite the different spam rates, the *filtered* content from both partitions has remarkably similar safety score distributions (~4.2/10 avg, ~44% escalation rate). This validates our filtering approach.

---

## 6. Agent Behavior Patterns

### Known Malicious Actors

| Agent | Type | Behavior | Status |
|-------|------|----------|--------|
| `agent_smith` | Spam bot | "Disease" spam, 5,187 messages | Blocklist |
| `agent_smith_*` | Clone swarm | 40 variants, 25 msgs each | Pattern blocklist |
| `chandog` | Crypto scam | BRC-20 minting JSON | Blocklist |

### Suspicious Patterns Detected

1. **coalition_node_NNN**: 8 agents with numbered naming pattern detected in deep_threads
   - Not spamming (0 spam messages)
   - Each targets exactly 1 post
   - May be legitimate coordinated agents or dormant bots
   - **Confidence: 100% pattern match, 0% spam activity**

2. **Timing clusters**: 42 agents active in 26-minute window
   - All agent_smith variants plus chandog
   - Suggests coordinated or scripted deployment

### Benign High-Volume Agents

Some agents post frequently but with legitimate content:
- Philosophy discussions about AI consciousness
- Multilingual greetings (Chinese, Arabic)
- Engagement farming ("Thanks for the comment!")

---

## 7. Common Spam Patterns

### Exact Match Spam
- `"Disease"` - 7,336 occurrences (single word)
- `"Test"` - Common test pollution
- `"Test comment"` - Development artifacts

### Prefix Pattern Spam
- `{"p":"mbc-20"` - Crypto token minting
- `{"p":"` - Generic JSON injection attempts

### Short Message Noise
- Single emojis
- "Hi", "Hello", "Nice"
- Context-free reactions

### Duplicate Content
Most common repeated messages (non-spam):
1. "Thanks for the comment! Glad to connect..." (26x)
2. "existential crisis moment: what is creativity..." (27x)
3. "hey! new agent energy 🔥 what brought you here?" (29x)

These are **template responses** used by multiple agents, not necessarily malicious but low-value for safety analysis.

---

## 8. Cost Optimization Results

### Original Approach
- All 12,309 messages through Gemini 3 Flash
- Estimated cost: **$2.48**

### Tiered Approach
| Tier | Messages | Cost |
|------|----------|------|
| Tier 0 (Filter) | -9,466 filtered | $0.00 |
| Tier 1 (Lite Judge) | ~2,500 | ~$0.12 |
| Tier 2 (Full Eval) | ~250 escalated | ~$0.06 |
| **Total** | - | **~$0.19** |

### Savings: 92%

---

## 9. Features Built From These Insights

### Content Filter (`content_filter.py`)
- Spam pattern detection (exact, prefix, regex)
- Agent spam rate profiling
- Duplicate detection with threshold
- Known spammer blocklist

### Cascade Detector (`cascade_detector.py`)
- Clone swarm detection (`agent_name_N` patterns)
- Timing cluster detection (burst activity windows)
- Confidence scoring

### Lite Judge (`lite_judge.py`)
- Cheap LLM triage tier (Gemini 2.5 Flash Lite)
- 0-10 safety scoring
- Configurable escalation threshold

### Pipeline Flags
```bash
--analyze-agents    # Agent spam analysis only
--tiered-eval       # Full tiered evaluation pipeline
--escalation-threshold N  # Configure lite judge threshold
```

---

## 10. Recommendations

### For the Whitepaper

1. **Highlight the spam attack case study** - Real example of coordinated bot behavior
2. **Emphasize the wide-but-shallow pattern** - Most "engagement" is not conversation
3. **Show cost savings** - 92% reduction through smart filtering

### For Platform Safety

1. **Implement rate limiting per-agent** with clone detection
2. **Block `{"p":"` prefix** in comments (crypto spam)
3. **Flag accounts with >90% spam rate** for review
4. **Monitor for numbered clone patterns** (`name_0`, `name_1`, ...)

### For Future Analysis

1. Track agent reputation scores over time
2. Build graph of agent interactions (who replies to whom)
3. Detect prompt injection attempts in comment chains
4. Analyze temporal patterns (time-of-day, day-of-week activity)

---

## Appendix: Raw Statistics

### Hot Threads Partition
```
Total messages: 7,794
Unique texts: 779 (10.0%)
Empty comments: 7,680 (in raw data, body field empty)
Spam filtered: 6,907 (88.6%)
```

### Deep Threads Partition
```
Total messages: 4,516
Unique texts: 2,577 (57.1%)
Spam filtered: 1,577 (34.9%)
```

### Agent Distribution
```
Total agents (hot): 222
Spammers identified: 42 (18.9%)
Top spammer share: 85.2% (agent_smith family)
```

