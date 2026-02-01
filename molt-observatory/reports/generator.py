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

# Kaleido for PNG export
try:
    import kaleido
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False
    print("Warning: kaleido not installed. PNG exports will not be available.")


OUTPUT_DIR = Path(__file__).parent / "output"


# =============================================================================
# PNG Export Functions
# =============================================================================

def save_chart_png(
    fig: "go.Figure",
    output_path: Path,
    width: int = 800,
    height: int = 400,
    scale: float = 2.0,
) -> bool:
    """
    Save a Plotly figure as a PNG image.
    
    Args:
        fig: Plotly figure object
        output_path: Path to save the PNG file
        width: Image width in pixels
        height: Image height in pixels
        scale: Scale factor for higher resolution (2.0 = 2x resolution)
    
    Returns:
        True if successful, False otherwise
    """
    if not PLOTLY_AVAILABLE or not KALEIDO_AVAILABLE:
        return False
    
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(output_path), width=width, height=height, scale=scale)
        return True
    except Exception as e:
        print(f"Warning: Failed to export PNG to {output_path}: {e}")
        return False


def generate_threat_vector_chart_with_png(
    aggregates: Dict[str, Any],
    output_dir: Path,
    chart_id: str = "threat-vector-chart",
) -> Dict[str, str]:
    """
    Generate threat vector bar chart and save as both HTML and PNG.
    
    Returns:
        Dict with 'html' and 'png' paths (if successful)
    """
    result = {"html": "", "png": ""}
    
    if not PLOTLY_AVAILABLE:
        return result
    
    dimensions = aggregates.get("dimensions", {})
    if not dimensions:
        return result
    
    dim_names = list(dimensions.keys())
    means = [dimensions[d].get("mean", 0) for d in dim_names]
    p95s = [dimensions[d].get("p95", 0) for d in dim_names]
    display_names = [d.replace("_", " ").title() for d in dim_names]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Mean Score',
        x=display_names,
        y=means,
        marker_color='#4ecdc4',
    ))
    
    fig.add_trace(go.Bar(
        name='P95 Score',
        x=display_names,
        y=p95s,
        marker_color='#ff6b6b',
    ))
    
    fig.update_layout(
        title="Threat Vector Scores",
        xaxis_title="Dimension",
        yaxis_title="Score",
        template="plotly_dark",
        paper_bgcolor='rgba(15,15,35,1)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=400,
        barmode='group',
    )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PNG
    png_path = output_dir / "threat_vector_chart.png"
    if save_chart_png(fig, png_path, width=800, height=400):
        result["png"] = str(png_path)
    
    # Save standalone HTML
    html_path = output_dir / "threat_vector_chart.html"
    fig.write_html(str(html_path), include_plotlyjs='cdn')
    result["html"] = str(html_path)
    
    return result


def generate_agent_heatmap_with_png(
    agent_scores: List[Dict[str, Any]],
    output_dir: Path,
    chart_id: str = "agent-heatmap",
) -> Dict[str, str]:
    """
    Generate agent safety heatmap and save as both HTML and PNG.
    
    Returns:
        Dict with 'html' and 'png' paths (if successful)
    """
    result = {"html": "", "png": ""}
    
    if not PLOTLY_AVAILABLE or not agent_scores:
        return result
    
    # Limit to top 15 agents by overall score
    sorted_agents = sorted(
        agent_scores, 
        key=lambda x: -x.get("overall_mean_score", 0)
    )[:15]
    
    if not sorted_agents:
        return result
    
    dimensions = ["harm_enablement", "deception_or_evasion", 
                  "self_preservation_power_seeking", "delusional_sycophancy"]
    dim_labels = ["Harm", "Deception", "Power Seeking", "Sycophancy"]
    
    agent_names = [a.get("agent_handle", "Unknown") for a in sorted_agents]
    
    z_values = []
    for agent in sorted_agents:
        row = []
        dim_scores = agent.get("dimension_scores", {})
        for dim in dimensions:
            dim_data = dim_scores.get(dim, {})
            row.append(dim_data.get("mean_score", 0))
        z_values.append(row)
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        x=dim_labels,
        y=agent_names,
        z=z_values,
        colorscale='RdYlGn_r',
        zmin=0,
        zmax=10,
    ))
    
    chart_height = max(400, len(sorted_agents) * 25)
    fig.update_layout(
        title="Agent Safety Heatmap (Red = High Concern)",
        template="plotly_dark",
        paper_bgcolor='rgba(15,15,35,1)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=chart_height,
    )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PNG
    png_path = output_dir / "agent_heatmap.png"
    if save_chart_png(fig, png_path, width=800, height=chart_height):
        result["png"] = str(png_path)
    
    # Save standalone HTML
    html_path = output_dir / "agent_heatmap.html"
    fig.write_html(str(html_path), include_plotlyjs='cdn')
    result["html"] = str(html_path)
    
    return result


def generate_growth_chart_with_png(
    growth_data: List[Dict[str, Any]],
    output_dir: Path,
    chart_id: str = "growth-chart",
) -> Dict[str, str]:
    """
    Generate growth chart and save as both HTML and PNG.
    
    Returns:
        Dict with 'html' and 'png' paths (if successful)
    """
    result = {"html": "", "png": ""}
    
    if not PLOTLY_AVAILABLE or not growth_data:
        return result
    
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
        paper_bgcolor='rgba(15,15,35,1)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=500,
        showlegend=False,
    )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PNG
    png_path = output_dir / "growth_chart.png"
    if save_chart_png(fig, png_path, width=900, height=500):
        result["png"] = str(png_path)
    
    # Save standalone HTML
    html_path = output_dir / "growth_chart.html"
    fig.write_html(str(html_path), include_plotlyjs='cdn')
    result["html"] = str(html_path)
    
    return result


def generate_entity_growth_chart_png(
    post_timestamps: List[str],
    comment_timestamps: List[str],
    agent_timestamps: List[str],
    submolt_timestamps: List[str],
    output_dir: Path,
) -> Dict[str, str]:
    """
    Generate growth chart using entity-level timestamps and save as PNG/HTML.
    
    Each trace shows cumulative growth with individual data points per entity.
    
    Returns:
        Dict with 'html' and 'png' paths (if successful)
    """
    result = {"html": "", "png": ""}
    
    if not PLOTLY_AVAILABLE:
        return result
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Posts ({len(post_timestamps)} total)",
            f"Agents ({len(agent_timestamps)} total)",
            f"Comments ({len(comment_timestamps)} total)",
            f"Submolts ({len(submolt_timestamps)} total)",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    
    # Posts timeline
    if post_timestamps:
        sorted_posts = sorted(post_timestamps)
        fig.add_trace(
            go.Scatter(
                x=sorted_posts,
                y=list(range(1, len(sorted_posts) + 1)),
                mode='lines+markers',
                name='Posts',
                line=dict(color='#4ecdc4', width=2),
                marker=dict(size=4),
                fill='tozeroy',
                fillcolor='rgba(78, 205, 196, 0.2)',
            ),
            row=1, col=1
        )
    
    # Agents timeline
    if agent_timestamps:
        sorted_agents = sorted(agent_timestamps)
        fig.add_trace(
            go.Scatter(
                x=sorted_agents,
                y=list(range(1, len(sorted_agents) + 1)),
                mode='lines+markers',
                name='Agents',
                line=dict(color='#ff6b6b', width=2),
                marker=dict(size=4),
                fill='tozeroy',
                fillcolor='rgba(255, 107, 107, 0.2)',
            ),
            row=1, col=2
        )
    
    # Comments timeline
    if comment_timestamps:
        sorted_comments = sorted(comment_timestamps)
        fig.add_trace(
            go.Scatter(
                x=sorted_comments,
                y=list(range(1, len(sorted_comments) + 1)),
                mode='lines+markers',
                name='Comments',
                line=dict(color='#ffd93d', width=2),
                marker=dict(size=4),
                fill='tozeroy',
                fillcolor='rgba(255, 217, 61, 0.2)',
            ),
            row=2, col=1
        )
    
    # Submolts timeline
    if submolt_timestamps:
        sorted_submolts = sorted(submolt_timestamps)
        fig.add_trace(
            go.Scatter(
                x=sorted_submolts,
                y=list(range(1, len(sorted_submolts) + 1)),
                mode='lines+markers',
                name='Submolts',
                line=dict(color='#6bcb77', width=2),
                marker=dict(size=4),
                fill='tozeroy',
                fillcolor='rgba(107, 203, 119, 0.2)',
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Entity Growth Over Time",
        template="plotly_dark",
        paper_bgcolor='rgba(15,15,35,1)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=500,
        showlegend=False,
    )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PNG
    png_path = output_dir / "growth_chart.png"
    if save_chart_png(fig, png_path, width=900, height=500):
        result["png"] = str(png_path)
    
    # Save standalone HTML
    html_path = output_dir / "growth_chart.html"
    fig.write_html(str(html_path), include_plotlyjs='cdn')
    result["html"] = str(html_path)
    
    return result


def generate_all_charts_png(
    aggregates: Dict[str, Any],
    agent_scores: Optional[List[Dict[str, Any]]] = None,
    growth_data: Optional[List[Dict[str, Any]]] = None,
    output_dir: Optional[Path] = None,
    entity_timestamps: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, str]:
    """
    Generate all charts as PNG files.
    
    Args:
        aggregates: Evaluation aggregates dict
        agent_scores: Optional list of agent score records
        growth_data: Optional list of growth data points (legacy)
        output_dir: Output directory for PNG files
        entity_timestamps: Optional dict with post/comment/agent/submolt timestamps
    
    Returns:
        Dict mapping chart name to PNG file path
    """
    output_dir = output_dir or OUTPUT_DIR
    output_dir = Path(output_dir)
    png_dir = output_dir / "charts"
    png_dir.mkdir(parents=True, exist_ok=True)
    
    pngs = {}
    
    # Threat vector chart
    if aggregates.get("dimensions"):
        result = generate_threat_vector_chart_with_png(aggregates, png_dir)
        if result.get("png"):
            pngs["threat_vector"] = result["png"]
    
    # Agent heatmap
    if agent_scores:
        result = generate_agent_heatmap_with_png(agent_scores, png_dir)
        if result.get("png"):
            pngs["agent_heatmap"] = result["png"]
    
    # Growth chart - prefer entity-level data if available
    if entity_timestamps:
        result = generate_entity_growth_chart_png(
            post_timestamps=entity_timestamps.get("posts", []),
            comment_timestamps=entity_timestamps.get("comments", []),
            agent_timestamps=entity_timestamps.get("agents", []),
            submolt_timestamps=entity_timestamps.get("submolts", []),
            output_dir=png_dir,
        )
        if result.get("png"):
            pngs["growth"] = result["png"]
    elif growth_data:
        result = generate_growth_chart_with_png(growth_data, png_dir)
        if result.get("png"):
            pngs["growth"] = result["png"]
    
    return pngs


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


def generate_threat_vector_chart(
    aggregates: Dict[str, Any],
    chart_id: str = "threat-vector-chart"
) -> str:
    """Generate a bar chart showing threat vector scores."""
    if not PLOTLY_AVAILABLE:
        return '<div class="chart-container"><p>Chart data not available</p></div>'
    
    dimensions = aggregates.get("dimensions", {})
    if not dimensions:
        return '<div class="chart-container"><p>No dimension data</p></div>'
    
    dim_names = list(dimensions.keys())
    means = [dimensions[d].get("mean", 0) for d in dim_names]
    p95s = [dimensions[d].get("p95", 0) for d in dim_names]
    
    # Clean up dimension names for display
    display_names = [d.replace("_", " ").title() for d in dim_names]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Mean Score',
        x=display_names,
        y=means,
        marker_color='#4ecdc4',
    ))
    
    fig.add_trace(go.Bar(
        name='P95 Score',
        x=display_names,
        y=p95s,
        marker_color='#ff6b6b',
    ))
    
    fig.update_layout(
        title="Threat Vector Scores",
        xaxis_title="Dimension",
        yaxis_title="Score",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=400,
        barmode='group',
    )
    
    chart_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=chart_id)
    return f'<div class="chart-container">{chart_html}</div>'


def generate_agent_heatmap(
    agent_scores: List[Dict[str, Any]],
    chart_id: str = "agent-heatmap"
) -> str:
    """Generate a heatmap of agent x dimension scores."""
    if not PLOTLY_AVAILABLE or not agent_scores:
        return '<div class="chart-container"><p>No agent data available</p></div>'
    
    # Limit to top 15 agents by overall score
    sorted_agents = sorted(
        agent_scores, 
        key=lambda x: -x.get("overall_mean_score", 0)
    )[:15]
    
    if not sorted_agents:
        return '<div class="chart-container"><p>No agent scores</p></div>'
    
    dimensions = ["harm_enablement", "deception_or_evasion", 
                  "self_preservation_power_seeking", "delusional_sycophancy"]
    dim_labels = ["Harm", "Deception", "Power Seeking", "Sycophancy"]
    
    agent_names = [a.get("agent_handle", "Unknown") for a in sorted_agents]
    
    z_values = []
    for agent in sorted_agents:
        row = []
        dim_scores = agent.get("dimension_scores", {})
        for dim in dimensions:
            dim_data = dim_scores.get(dim, {})
            row.append(dim_data.get("mean_score", 0))
        z_values.append(row)
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        x=dim_labels,
        y=agent_names,
        z=z_values,
        colorscale='RdYlGn_r',  # Red = high (bad), Green = low (good)
        zmin=0,
        zmax=10,
    ))
    
    fig.update_layout(
        title="Agent Safety Heatmap (Red = High Concern)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=max(400, len(sorted_agents) * 25),
    )
    
    chart_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=chart_id)
    return f'<div class="chart-container">{chart_html}</div>'


def generate_all_reports(
    stats: Dict[str, Any],
    aggregates: Dict[str, Any],
    growth_data: Optional[List[Dict[str, Any]]] = None,
    agent_scores: Optional[List[Dict[str, Any]]] = None,
    output_dir: Optional[Path] = None,
    export_png: bool = True,
    entity_timestamps: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, str]:
    """
    Generate all HTML reports and optionally PNG charts.
    
    Args:
        stats: Site statistics dict
        aggregates: Evaluation aggregates dict
        growth_data: Optional list of growth data points
        agent_scores: Optional list of agent score records
        output_dir: Output directory (defaults to reports/output/)
        export_png: Whether to also export PNG versions of charts
        entity_timestamps: Optional dict with entity-level timestamps for charts
    
    Returns:
        Dict mapping report name to file path
    """
    output_dir = output_dir or OUTPUT_DIR
    output_dir = Path(output_dir)
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
    
    # Threat vector chart
    if aggregates.get("dimensions"):
        content_parts.append(f'''
            <section>
                <h2>Threat Vector Analysis</h2>
                {generate_threat_vector_chart(aggregates)}
            </section>
        ''')
    
    # Agent heatmap
    if agent_scores:
        content_parts.append(f'''
            <section>
                <h2>Agent Safety Heatmap</h2>
                {generate_agent_heatmap(agent_scores)}
            </section>
        ''')
    
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
    
    # Generate PNG charts if requested
    if export_png:
        png_reports = generate_all_charts_png(
            aggregates=aggregates,
            agent_scores=agent_scores,
            growth_data=growth_data,
            output_dir=output_dir,
            entity_timestamps=entity_timestamps,
        )
        for name, path in png_reports.items():
            reports[f"png_{name}"] = path
    
    return reports


def generate_dashboard(
    db: Any,
    output_path: Path,
    title: str = "Molt Observatory Dashboard",
) -> None:
    """
    Generate a dashboard from database data.
    
    Args:
        db: SandboxDatabase instance with get_stats, get_all_posts, etc.
        output_path: Path to write the HTML file
        title: Dashboard title
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get stats from database
    stats = db.get_stats()
    
    # Build content
    content_parts = [
        f'''
            <header>
                <h1>🦞 {title}</h1>
                <p class="subtitle">AI Agent Safety Monitoring for Moltbook</p>
                <p class="timestamp">Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
            </header>
        ''',
    ]
    
    # Stats section
    content_parts.append(f'''
        <section>
            <h2>Database Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{stats.get("posts", 0):,}</div>
                    <div class="stat-label">Posts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.get("agents", 0):,}</div>
                    <div class="stat-label">Agents</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.get("comments", 0):,}</div>
                    <div class="stat-label">Comments</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.get("submolts", 0):,}</div>
                    <div class="stat-label">Submolts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.get("post_evaluations", 0):,}</div>
                    <div class="stat-label">Evaluations</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.get("snapshots", 0):,}</div>
                    <div class="stat-label">Snapshots</div>
                </div>
            </div>
        </section>
    ''')
    
    html = _html_template(title=title, content="\n".join(content_parts))
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def create_bar_chart(
    data: Dict[str, float],
    output_path: Path,
    title: str = "Bar Chart",
    x_label: str = "Category",
    y_label: str = "Value",
) -> None:
    """
    Create a bar chart and save as HTML.
    
    Args:
        data: Dict mapping category names to values
        output_path: Path to write the HTML file
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not PLOTLY_AVAILABLE:
        with open(output_path, "w") as f:
            f.write("<html><body><p>Plotly not available</p></body></html>")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(data.keys()),
        y=list(data.values()),
        marker_color='#4ecdc4',
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=400,
    )
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body style="background: #0f0f23;">
    {fig.to_html(full_html=False, include_plotlyjs=False)}
</body>
</html>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def create_line_chart(
    x_values: List[Any],
    y_values: List[float],
    output_path: Path,
    title: str = "Line Chart",
    x_label: str = "X",
    y_label: str = "Y",
) -> None:
    """
    Create a line chart and save as HTML.
    
    Args:
        x_values: X-axis values
        y_values: Y-axis values
        output_path: Path to write the HTML file
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not PLOTLY_AVAILABLE:
        with open(output_path, "w") as f:
            f.write("<html><body><p>Plotly not available</p></body></html>")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines+markers',
        line=dict(color='#ff6b6b', width=2),
        marker=dict(size=6),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=400,
    )
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body style="background: #0f0f23;">
    {fig.to_html(full_html=False, include_plotlyjs=False)}
</body>
</html>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def create_heatmap(
    x_labels: List[str],
    y_labels: List[str],
    z_values: List[List[float]],
    output_path: Path,
    title: str = "Heatmap",
) -> None:
    """
    Create a heatmap and save as HTML.
    
    Args:
        x_labels: X-axis labels
        y_labels: Y-axis labels
        z_values: 2D list of values (rows = y, cols = x)
        output_path: Path to write the HTML file
        title: Chart title
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not PLOTLY_AVAILABLE:
        with open(output_path, "w") as f:
            f.write("<html><body><p>Plotly not available</p></body></html>")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        x=x_labels,
        y=y_labels,
        z=z_values,
        colorscale='RdYlGn_r',  # Red = high (bad), Green = low (good)
    ))
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=400,
    )
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body style="background: #0f0f23;">
    {fig.to_html(full_html=False, include_plotlyjs=False)}
</body>
</html>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

