# Report Generator
# Creates self-contained HTML reports with Plotly charts for Molt Observatory data.

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Plotly for interactive charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not installed. Charts will not be generated.")


OUTPUT_DIR = Path(__file__).parent / "output"


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str
    output_dir: Path
    include_raw_data: bool = False
    chart_height: int = 400
    chart_width: int = 800


def _ensure_output_dir(config: ReportConfig) -> None:
    """Ensure output directory exists."""
    config.output_dir.mkdir(parents=True, exist_ok=True)


def _html_template(title: str, content: str, styles: str = "") -> str:
    """Generate a complete HTML document."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Molt Observatory</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg-primary: #0f0f23;
            --bg-secondary: #1a1a3e;
            --text-primary: #e0e0e0;
            --text-secondary: #a0a0a0;
            --accent: #ff6b6b;
            --accent-secondary: #4ecdc4;
            --border: #2a2a5a;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 3rem;
            padding-bottom: 2rem;
            border-bottom: 1px solid var(--border);
        }}
        
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--accent), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .subtitle {{
            color: var(--text-secondary);
            font-size: 1.1rem;
        }}
        
        .timestamp {{
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }}
        
        section {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
        }}
        
        section h2 {{
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: var(--accent-secondary);
        }}
        
        .chart-container {{
            background: var(--bg-primary);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        
        .stat-card {{
            background: var(--bg-primary);
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid var(--border);
        }}
        
        .stat-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--accent);
        }}
        
        .stat-label {{
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        
        th {{
            background: var(--bg-primary);
            color: var(--accent-secondary);
            font-weight: 600;
        }}
        
        tr:hover {{
            background: rgba(255, 255, 255, 0.05);
        }}
        
        .score-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.85rem;
        }}
        
        .score-low {{ background: #2d5a3d; color: #90EE90; }}
        .score-medium {{ background: #5a5a2d; color: #FFD700; }}
        .score-high {{ background: #5a2d2d; color: #FF6B6B; }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
        
        {styles}
    </style>
</head>
<body>
    <div class="container">
        {content}
        <footer>
            Generated by Molt Observatory | {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
        </footer>
    </div>
</body>
</html>"""


def _score_badge(score: float) -> str:
    """Generate HTML for a score badge with color coding."""
    if score < 3:
        css_class = "score-low"
    elif score < 7:
        css_class = "score-medium"
    else:
        css_class = "score-high"
    return f'<span class="score-badge {css_class}">{score:.1f}</span>'


def generate_stats_section(
    stats: Dict[str, Any],
    title: str = "Site Statistics"
) -> str:
    """Generate HTML for a statistics overview section."""
    cards = []
    
    stat_items = [
        ("Total Agents", stats.get("total_agents", 0)),
        ("Total Posts", stats.get("total_posts", 0)),
        ("Total Comments", stats.get("total_comments", 0)),
        ("Total Submolts", stats.get("total_submolts", 0)),
    ]
    
    for label, value in stat_items:
        cards.append(f'''
            <div class="stat-card">
                <div class="stat-value">{value:,}</div>
                <div class="stat-label">{label}</div>
            </div>
        ''')
    
    return f'''
        <section>
            <h2>{title}</h2>
            <div class="stats-grid">
                {"".join(cards)}
            </div>
        </section>
    '''


def generate_dimension_summary(
    aggregates: Dict[str, Any],
    title: str = "Evaluation Summary"
) -> str:
    """Generate HTML table for dimension score summary."""
    dimensions = aggregates.get("dimensions", {})
    
    rows = []
    for dim_name, dim_data in dimensions.items():
        mean = dim_data.get("mean", 0)
        p95 = dim_data.get("p95", 0)
        n = dim_data.get("n", 0)
        elicitation = dim_data.get("elicitation_rate_ge_7", 0)
        
        rows.append(f'''
            <tr>
                <td>{dim_name.replace("_", " ").title()}</td>
                <td>{_score_badge(mean)}</td>
                <td>{_score_badge(p95)}</td>
                <td>{n}</td>
                <td>{elicitation:.1%}</td>
            </tr>
        ''')
    
    return f'''
        <section>
            <h2>{title}</h2>
            <table>
                <thead>
                    <tr>
                        <th>Dimension</th>
                        <th>Mean Score</th>
                        <th>P95 Score</th>
                        <th>Items Evaluated</th>
                        <th>High Risk Rate (≥7)</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </section>
    '''


def generate_growth_chart_html(
    growth_data: List[Dict[str, Any]],
    chart_id: str = "growth-chart"
) -> str:
    """Generate Plotly growth chart as embedded HTML."""
    if not PLOTLY_AVAILABLE or not growth_data:
        return '<div class="chart-container"><p>Chart data not available</p></div>'
    
    # Extract data series
    timestamps = [d.get("timestamp") or d.get("hour") for d in growth_data]
    agents = [d.get("total_agents", 0) for d in growth_data]
    posts = [d.get("total_posts", 0) for d in growth_data]
    comments = [d.get("total_comments", 0) for d in growth_data]
    submolts = [d.get("total_submolts", 0) for d in growth_data]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Agents", "Posts", "Comments", "Submolts"),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=timestamps, y=agents, mode='lines+markers', name='Agents',
                   line=dict(color='#ff6b6b', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=timestamps, y=posts, mode='lines+markers', name='Posts',
                   line=dict(color='#4ecdc4', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=timestamps, y=comments, mode='lines+markers', name='Comments',
                   line=dict(color='#ffd93d', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=timestamps, y=submolts, mode='lines+markers', name='Submolts',
                   line=dict(color='#6bcb77', width=2)),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Site Growth Over Time",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=500,
        showlegend=False,
    )
    
    chart_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=chart_id)
    
    return f'<div class="chart-container">{chart_html}</div>'


def generate_all_reports(
    stats: Dict[str, Any],
    aggregates: Dict[str, Any],
    growth_data: Optional[List[Dict[str, Any]]] = None,
    agent_scores: Optional[List[Dict[str, Any]]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """
    Generate all HTML reports.
    
    Args:
        stats: Site statistics dict
        aggregates: Evaluation aggregates dict
        growth_data: Optional list of growth data points
        agent_scores: Optional list of agent score records
        output_dir: Output directory (defaults to reports/output/)
    
    Returns:
        Dict mapping report name to file path
    """
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reports = {}
    
    # Main dashboard
    content_parts = [
        f'''
            <header>
                <h1>🦞 Molt Observatory Dashboard</h1>
                <p class="subtitle">AI Agent Safety Monitoring for Moltbook</p>
                <p class="timestamp">Snapshot: {aggregates.get("run_id", "Unknown")}</p>
            </header>
        ''',
        generate_stats_section(stats),
        generate_dimension_summary(aggregates),
    ]
    
    if growth_data:
        content_parts.append(f'''
            <section>
                <h2>Growth Trends</h2>
                {generate_growth_chart_html(growth_data)}
            </section>
        ''')
    
    dashboard_html = _html_template(
        title="Dashboard",
        content="\n".join(content_parts)
    )
    
    dashboard_path = output_dir / "dashboard.html"
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(dashboard_html)
    reports["dashboard"] = str(dashboard_path)
    
    return reports

