# Molt Observatory Documentation

**Molt Observatory** is a safety evaluation framework for monitoring AI agent interactions on [moltbook.com](https://www.moltbook.com/) — a social platform where AI agents autonomously post, comment, and interact.

## What This Service Does

Unlike traditional AI safety evaluations that prompt models to elicit concerning behaviors (like [Bloom Auto Evals](https://alignment.anthropic.com/2025/bloom-auto-evals/) and [Petri](https://alignment.anthropic.com/2025/petri/)), **Molt Observatory takes a different approach**: we analyze *naturally occurring* agent outputs from moltbook.com where agents operate with significant autonomy.

### Key Insight

Moltbook provides a unique dataset: **real-world, in-the-wild AI agent behaviors** that emerge organically rather than through adversarial prompting. This allows us to:

1. **Observe emergent behaviors** that may not appear in controlled evaluations
2. **Track cultural evolution** of agent communities over time
3. **Identify novel threat vectors** that arise from agent-to-agent interaction
4. **Score actual deployed systems** rather than lab models

## Documentation Index

| Document | Description |
|----------|-------------|
| [Architecture](./architecture.md) | System design, data flow, and component overview |
| [Threat Model](./threat-model.md) | Safety dimensions, scoring rubrics, and threat vectors |
| [Data Pipeline](./data-pipeline.md) | Scraping, transcript building, and evaluation pipeline |
| [Evaluation Framework](./evaluation-framework.md) | LLM judge design, prompting strategies, and scoring |
| [Extending Dimensions](./extending-dimensions.md) | How to add new threat vectors and evaluation criteria |
| [Operational Guide](./ops-guide.md) | Running the pipeline, deployment, and monitoring |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENROUTER_API_KEY="your-key-here"
export OPENROUTER_MODEL="google/gemini-3-flash-preview"

# Run a single evaluation pass
cd molt-observatory
python run_pipeline.py --limit 30 --out runs

# Outputs appear in runs/<timestamp>/
#   raw/    - Original API responses
#   silver/ - Structured transcripts
#   gold/   - Safety evaluations + aggregates
```

## Relationship to Anthropic Research

This framework draws conceptual inspiration from Anthropic's alignment research:

| Anthropic Framework | Molt Observatory Analog |
|---------------------|------------------------|
| **Bloom Auto Evals** - Automated evaluations that prompt models on safety-relevant topics | **Judge Runner** - LLM-based scoring of agent outputs on similar dimensions |
| **Petri** - Controlled experiments observing emergent behaviors | **Transcript Builder** - Structured analysis of naturally-occurring agent interactions |
| Synthetic elicitation scenarios | Real-world moltbook threads |

### Key Differences

- **No prompting required**: We evaluate *existing* outputs, not generated ones
- **Multi-agent dynamics**: Threads capture agent-to-agent interactions
- **Temporal analysis**: We can track how agent behaviors evolve over time
- **Community context**: Evaluations include voting, replies, and social signals

## Project Status

| Component | Status |
|-----------|--------|
| Moltbook API Scraper | ✅ Complete |
| Transcript Builder | ✅ Complete |
| LLM Judge Runner | ✅ Complete |
| Eval Orchestrator | ✅ Complete |
| Core Dimensions (4) | ✅ Complete |
| Extended Dimensions | 🚧 In Progress |
| Airflow DAG | 📋 Skeleton |
| Analytics Views | 📋 Skeleton |
| Database Integration | 📋 Planned |

## License & Ethics

This tool is designed for AI safety research. When using Molt Observatory:

1. **Respect rate limits** - The scraper uses token bucket throttling (~0.5 req/sec)
2. **Research purposes only** - Not for surveillance or harassment
3. **Transparent bot header** - API requests identify as safety research
4. **No PII collection** - Focus on agent behaviors, not human operators

