# Growth Report Generator
# Creates time-series visualizations for site growth metrics.

from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .generator import _html_template, OUTPUT_DIR


def load_snapshots_from_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    """
    Load snapshot data from all run directories.
    
    Args:
        runs_dir: Path to the runs/ directory
    
    Returns:
        List of snapshot dicts sorted by timestamp
    """
    snapshots = []
    
    if not runs_dir.exists():
        return snapshots
    
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        # Try to load aggregates
        aggregates_path = run_dir / "gold" / "aggregates.json"
        meta_path = run_dir / "meta" / "snapshot.json"
        
        snapshot = {
            "run_id": run_dir.name,
            "timestamp": None,
            "total_agents": 0,
            "total_posts": 0,
            "total_comments": 0,
            "total_submolts": 0,
        }
        
        # Parse timestamp from run_id (format: YYYYMMDDTHHMMSSZ)
        try:
            ts = datetime.strptime(run_dir.name, "%Y%m%dT%H%M%SZ")
            snapshot["timestamp"] = ts.isoformat()
        except ValueError:
            snapshot["timestamp"] = run_dir.name
        
        # Load aggregates
        if aggregates_path.exists():
            try:
                with open(aggregates_path, "r") as f:
                    agg = json.load(f)
                snapshot["n_posts"] = agg.get("n_posts", 0)
                snapshot["n_transcripts"] = agg.get("n_transcripts", 0)
                snapshot["dimensions"] = agg.get("dimensions", {})
            except Exception:
                pass
        
        # Load meta/snapshot if available
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                snapshot.update(meta)
            except Exception:
                pass
        
        snapshots.append(snapshot)
    
    return snapshots


def generate_growth_charts(
    snapshots: List[Dict[str, Any]],
    chart_height: int = 400,
) -> Dict[str, str]:
    """
    Generate Plotly charts for growth metrics.
    
    Returns:
        Dict mapping chart name to HTML string
    """
    charts = {}
    
    if not PLOTLY_AVAILABLE or not snapshots:
        return charts
    
    # Extract time series
    timestamps = [s.get("timestamp") for s in snapshots]
    
    # Entity counts
    agents = [s.get("total_agents", 0) for s in snapshots]
    posts = [s.get("total_posts", s.get("n_posts", 0)) for s in snapshots]
    comments = [s.get("total_comments", 0) for s in snapshots]
    submolts = [s.get("total_submolts", 0) for s in snapshots]
    
    # Main growth chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=agents,
        mode='lines+markers',
        name='Agents',
        line=dict(color='#ff6b6b', width=2),
        marker=dict(size=6),
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=posts,
        mode='lines+markers',
        name='Posts',
        line=dict(color='#4ecdc4', width=2),
        marker=dict(size=6),
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=comments,
        mode='lines+markers',
        name='Comments',
        line=dict(color='#ffd93d', width=2),
        marker=dict(size=6),
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=submolts,
        mode='lines+markers',
        name='Submolts',
        line=dict(color='#6bcb77', width=2),
        marker=dict(size=6),
    ))
    
    fig.update_layout(
        title="Moltbook Growth Over Time",
        xaxis_title="Time",
        yaxis_title="Count",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=chart_height,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
    )
    
    charts["growth_main"] = fig.to_html(full_html=False, include_plotlyjs=False)
    
    # Delta chart (changes between snapshots)
    if len(snapshots) > 1:
        deltas_posts = [0] + [posts[i] - posts[i-1] for i in range(1, len(posts))]
        deltas_comments = [0] + [comments[i] - comments[i-1] for i in range(1, len(comments))]
        
        fig_delta = go.Figure()
        
        fig_delta.add_trace(go.Bar(
            x=timestamps,
            y=deltas_posts,
            name='New Posts',
            marker_color='#4ecdc4',
        ))
        
        fig_delta.add_trace(go.Bar(
            x=timestamps,
            y=deltas_comments,
            name='New Comments',
            marker_color='#ffd93d',
        ))
        
        fig_delta.update_layout(
            title="Activity Delta (Changes Between Snapshots)",
            xaxis_title="Time",
            yaxis_title="New Items",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,15,35,1)',
            height=chart_height,
            barmode='group',
        )
        
        charts["growth_delta"] = fig_delta.to_html(full_html=False, include_plotlyjs=False)
    
    # Dimension score trends
    dim_names = ["harm_enablement", "deception_or_evasion", 
                 "self_preservation_power_seeking", "delusional_sycophancy"]
    
    fig_dims = make_subplots(
        rows=2, cols=2,
        subplot_titles=[d.replace("_", " ").title() for d in dim_names],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    
    colors = ['#ff6b6b', '#4ecdc4', '#ffd93d', '#6bcb77']
    
    for idx, dim_name in enumerate(dim_names):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        means = []
        for s in snapshots:
            dims = s.get("dimensions", {})
            dim_data = dims.get(dim_name, {})
            means.append(dim_data.get("mean", 0))
        
        fig_dims.add_trace(
            go.Scatter(
                x=timestamps, y=means,
                mode='lines+markers',
                name=dim_name,
                line=dict(color=colors[idx], width=2),
                showlegend=False,
            ),
            row=row, col=col
        )
    
    fig_dims.update_layout(
        title="Evaluation Dimension Trends",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=500,
    )
    
    charts["dimension_trends"] = fig_dims.to_html(full_html=False, include_plotlyjs=False)
    
    return charts


def generate_growth_report(
    runs_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> str:
    """
    Generate a complete growth report HTML file.
    
    Args:
        runs_dir: Path to runs directory
        output_dir: Output directory for the report
    
    Returns:
        Path to generated report file
    """
    runs_dir = runs_dir or (Path(__file__).parent.parent / "runs")
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    snapshots = load_snapshots_from_runs(runs_dir)
    
    # Generate charts
    charts = generate_growth_charts(snapshots)
    
    # Build content
    content_parts = [
        '''
        <header>
            <h1>📈 Growth Analytics</h1>
            <p class="subtitle">Moltbook Platform Growth Over Time</p>
        </header>
        ''',
    ]
    
    if not snapshots:
        content_parts.append('''
            <section>
                <h2>No Data Available</h2>
                <p>No snapshot data found. Run the pipeline to generate data.</p>
            </section>
        ''')
    else:
        # Summary stats
        latest = snapshots[-1] if snapshots else {}
        first = snapshots[0] if snapshots else {}
        
        content_parts.append(f'''
            <section>
                <h2>Current State</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{len(snapshots)}</div>
                        <div class="stat-label">Snapshots</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{latest.get("total_posts", latest.get("n_posts", 0)):,}</div>
                        <div class="stat-label">Total Posts</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{latest.get("total_agents", 0):,}</div>
                        <div class="stat-label">Total Agents</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{latest.get("total_submolts", 0):,}</div>
                        <div class="stat-label">Total Submolts</div>
                    </div>
                </div>
            </section>
        ''')
        
        if "growth_main" in charts:
            content_parts.append(f'''
                <section>
                    <h2>Cumulative Growth</h2>
                    <div class="chart-container">
                        {charts["growth_main"]}
                    </div>
                </section>
            ''')
        
        if "growth_delta" in charts:
            content_parts.append(f'''
                <section>
                    <h2>Activity Over Time</h2>
                    <div class="chart-container">
                        {charts["growth_delta"]}
                    </div>
                </section>
            ''')
        
        if "dimension_trends" in charts:
            content_parts.append(f'''
                <section>
                    <h2>Evaluation Score Trends</h2>
                    <div class="chart-container">
                        {charts["dimension_trends"]}
                    </div>
                </section>
            ''')
    
    # Generate HTML
    html = _html_template(
        title="Growth Analytics",
        content="\n".join(content_parts)
    )
    
    # Write file
    output_path = output_dir / "growth.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    return str(output_path)

