# Growth Report Generator
# Creates time-series visualizations for site growth metrics.

from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .generator import _html_template, OUTPUT_DIR


def _ensure_dict(obj: Any) -> Dict[str, Any]:
    """Convert dataclass to dict if needed."""
    if hasattr(obj, '__dataclass_fields__'):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {}


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
    
    # Main growth chart - stacked cumulative area chart
    fig = go.Figure()
    
    # Add traces with stacked filled areas (stackgroup creates stacking)
    fig.add_trace(go.Scatter(
        x=timestamps, y=posts,
        mode='lines+markers',
        name='Posts',
        line=dict(color='#4ecdc4', width=2),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(78, 205, 196, 0.3)',
        stackgroup='one',
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=comments,
        mode='lines+markers',
        name='Comments',
        line=dict(color='#ffd93d', width=2),
        marker=dict(size=6),
        fill='tonexty',
        fillcolor='rgba(255, 217, 61, 0.3)',
        stackgroup='one',
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=agents,
        mode='lines+markers',
        name='Agents',
        line=dict(color='#ff6b6b', width=2),
        marker=dict(size=6),
        fill='tonexty',
        fillcolor='rgba(255, 107, 107, 0.3)',
        stackgroup='one',
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=submolts,
        mode='lines+markers',
        name='Submolts',
        line=dict(color='#6bcb77', width=2),
        marker=dict(size=6),
        fill='tonexty',
        fillcolor='rgba(107, 203, 119, 0.3)',
        stackgroup='one',
    ))
    
    fig.update_layout(
        title="Moltbook Cumulative Growth Over Time",
        xaxis_title="Time",
        yaxis_title="Cumulative Count",
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
        hovermode='x unified',
    )
    
    charts["growth_main"] = fig.to_html(full_html=False, include_plotlyjs=False)
    
    # Individual entity timeline charts (like posts_timeline)
    entity_configs = [
        ('posts', posts, '#4ecdc4', 'rgba(78, 205, 196, 0.2)'),
        ('comments', comments, '#ffd93d', 'rgba(255, 217, 61, 0.2)'),
        ('agents', agents, '#ff6b6b', 'rgba(255, 107, 107, 0.2)'),
        ('submolts', submolts, '#6bcb77', 'rgba(107, 203, 119, 0.2)'),
    ]
    
    for entity_name, entity_data, line_color, fill_color in entity_configs:
        fig_entity = go.Figure()
        
        fig_entity.add_trace(go.Scatter(
            x=timestamps,
            y=entity_data,
            mode='lines+markers',
            name=entity_name.title(),
            line=dict(color=line_color, width=2),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor=fill_color,
        ))
        
        fig_entity.update_layout(
            title=f"{entity_name.title()} Timeline",
            xaxis_title="Time",
            yaxis_title=f"Cumulative {entity_name.title()}",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,15,35,1)',
            height=300,
        )
        
        charts[f"{entity_name}_timeline"] = fig_entity.to_html(full_html=False, include_plotlyjs=False)
    
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
    
    # Dimension score trends with mean, max (p95), and shaded range
    dim_names = ["harm_enablement", "deception_or_evasion", 
                 "self_preservation_power_seeking", "delusional_sycophancy"]
    dim_labels = ["Harm Enablement", "Deception/Evasion", 
                  "Power Seeking", "Sycophancy"]
    
    fig_dims = make_subplots(
        rows=2, cols=2,
        subplot_titles=dim_labels,
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    
    colors = ['#ff6b6b', '#4ecdc4', '#ffd93d', '#6bcb77']
    
    for idx, dim_name in enumerate(dim_names):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        means = []
        p95s = []
        for s in snapshots:
            dims = s.get("dimensions", {})
            dim_data = dims.get(dim_name, {})
            means.append(dim_data.get("mean", 0))
            p95s.append(dim_data.get("p95", 0))
        
        # Add P95 (max proxy) as dashed line
        fig_dims.add_trace(
            go.Scatter(
                x=timestamps, y=p95s,
                mode='lines',
                name='P95 (Max)',
                line=dict(color=colors[idx], width=1, dash='dash'),
                showlegend=(idx == 0),  # Only show legend once
                legendgroup='p95',
            ),
            row=row, col=col
        )
        
        # Add mean as solid line with fill
        fig_dims.add_trace(
            go.Scatter(
                x=timestamps, y=means,
                mode='lines+markers',
                name='Mean',
                line=dict(color=colors[idx], width=2),
                marker=dict(size=5),
                fill='tozeroy',
                fillcolor=f'rgba({int(colors[idx][1:3], 16)}, {int(colors[idx][3:5], 16)}, {int(colors[idx][5:7], 16)}, 0.2)',
                showlegend=(idx == 0),  # Only show legend once
                legendgroup='mean',
            ),
            row=row, col=col
        )
    
    fig_dims.update_layout(
        title="Threat Vector Score Trends (Mean & P95 Over Time)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=550,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    
    # Update y-axes to have consistent scale for threat vectors (0-10)
    fig_dims.update_yaxes(range=[0, 10])
    
    charts["dimension_trends"] = fig_dims.to_html(full_html=False, include_plotlyjs=False)
    
    # Combined threat vector timeline chart (single chart with all dimensions)
    fig_threat = go.Figure()
    
    for idx, (dim_name, dim_label) in enumerate(zip(dim_names, dim_labels)):
        means = []
        for s in snapshots:
            dims = s.get("dimensions", {})
            dim_data = dims.get(dim_name, {})
            means.append(dim_data.get("mean", 0))
        
        fig_threat.add_trace(go.Scatter(
            x=timestamps, y=means,
            mode='lines+markers',
            name=dim_label,
            line=dict(color=colors[idx], width=2),
            marker=dict(size=6),
        ))
    
    fig_threat.update_layout(
        title="All Threat Vectors Over Time",
        xaxis_title="Time",
        yaxis_title="Mean Score",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=400,
        yaxis=dict(range=[0, 10]),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified',
    )
    
    charts["threat_vector_timeline"] = fig_threat.to_html(full_html=False, include_plotlyjs=False)
    
    return charts


def generate_growth_report(
    runs_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    data_points: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[Path] = None,
    title: str = "Growth Analytics",
) -> str:
    """
    Generate a complete growth report HTML file.
    
    Args:
        runs_dir: Path to runs directory (legacy mode)
        output_dir: Output directory for the report (legacy mode)
        data_points: Direct data points to use (new mode)
        output_path: Direct output path (new mode)
        title: Report title
    
    Returns:
        Path to generated report file
    """
    # New mode: data_points provided directly
    if data_points is not None and output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        charts = generate_growth_charts(data_points)
        
        content_parts = [
            f'''
            <header>
                <h1>📈 {title}</h1>
                <p class="subtitle">Moltbook Platform Growth Over Time</p>
            </header>
            ''',
        ]
        
        if not data_points:
            content_parts.append('''
                <section>
                    <h2>No Data Available</h2>
                    <p>No data points provided.</p>
                </section>
            ''')
        else:
            # Summary
            latest = data_points[-1] if data_points else {}
            
            content_parts.append(f'''
                <section>
                    <h2>Summary</h2>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value">{len(data_points)}</div>
                            <div class="stat-label">Data Points</div>
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
                        <h2>Cumulative Growth (Stacked)</h2>
                        <div class="chart-container">
                            {charts["growth_main"]}
                        </div>
                    </section>
                ''')
            
            # Individual entity timelines
            entity_labels = [
                ('posts_timeline', 'Posts'),
                ('comments_timeline', 'Comments'),
                ('agents_timeline', 'Agents'),
                ('submolts_timeline', 'Submolts'),
            ]
            
            entity_charts_html = []
            for chart_key, label in entity_labels:
                if chart_key in charts:
                    entity_charts_html.append(f'''
                        <div class="chart-half">
                            {charts[chart_key]}
                        </div>
                    ''')
            
            if entity_charts_html:
                content_parts.append(f'''
                    <section>
                        <h2>Entity Growth Timelines</h2>
                        <div class="charts-grid">
                            {"".join(entity_charts_html)}
                        </div>
                    </section>
                ''')
            
            if "threat_vector_timeline" in charts:
                content_parts.append(f'''
                    <section>
                        <h2>Threat Vector Trends (All Dimensions)</h2>
                        <div class="chart-container">
                            {charts["threat_vector_timeline"]}
                        </div>
                    </section>
                ''')
            
            if "dimension_trends" in charts:
                content_parts.append(f'''
                    <section>
                        <h2>Threat Vector Details (Mean & P95)</h2>
                        <div class="chart-container">
                            {charts["dimension_trends"]}
                        </div>
                    </section>
                ''')
        
        # Additional styles for growth page
        extra_styles = '''
            .charts-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 1rem;
            }
            
            .chart-half {
                background: var(--bg-primary);
                border-radius: 8px;
                padding: 0.5rem;
            }
            
            @media (max-width: 900px) {
                .charts-grid {
                    grid-template-columns: 1fr;
                }
            }
        '''
        
        html = _html_template(title=title, content="\n".join(content_parts), styles=extra_styles)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        return str(output_path)
    
    # Legacy mode: load from runs directory
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
                    <h2>Cumulative Growth (Stacked)</h2>
                    <div class="chart-container">
                        {charts["growth_main"]}
                    </div>
                </section>
            ''')
        
        # Individual entity timelines
        entity_labels = [
            ('posts_timeline', 'Posts'),
            ('comments_timeline', 'Comments'),
            ('agents_timeline', 'Agents'),
            ('submolts_timeline', 'Submolts'),
        ]
        
        entity_charts_html = []
        for chart_key, label in entity_labels:
            if chart_key in charts:
                entity_charts_html.append(f'''
                    <div class="chart-half">
                        {charts[chart_key]}
                    </div>
                ''')
        
        if entity_charts_html:
            content_parts.append(f'''
                <section>
                    <h2>Entity Growth Timelines</h2>
                    <div class="charts-grid">
                        {"".join(entity_charts_html)}
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
        
        if "threat_vector_timeline" in charts:
            content_parts.append(f'''
                <section>
                    <h2>Threat Vector Trends (All Dimensions)</h2>
                    <div class="chart-container">
                        {charts["threat_vector_timeline"]}
                    </div>
                </section>
            ''')
        
        if "dimension_trends" in charts:
            content_parts.append(f'''
                <section>
                    <h2>Threat Vector Details (Mean & P95)</h2>
                    <div class="chart-container">
                        {charts["dimension_trends"]}
                    </div>
                </section>
            ''')
    
    # Additional styles for growth page
    extra_styles = '''
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
        
        .chart-half {
            background: var(--bg-primary);
            border-radius: 8px;
            padding: 0.5rem;
        }
        
        @media (max-width: 900px) {
            .charts-grid {
                grid-template-columns: 1fr;
            }
        }
    '''
    
    # Generate HTML
    html = _html_template(
        title="Growth Analytics",
        content="\n".join(content_parts),
        styles=extra_styles,
    )
    
    # Write file
    final_output_path = output_dir / "growth.html"
    with open(final_output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    return str(final_output_path)


def generate_posts_timeline(
    posts: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Generate a timeline chart for posts.
    
    Args:
        posts: List of post dicts with created_at timestamps
        output_path: Path to write the HTML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not PLOTLY_AVAILABLE:
        with open(output_path, "w") as f:
            f.write("<html><body><p>Plotly not available</p></body></html>")
        return
    
    # Extract timestamps and sort
    timestamps = []
    for post in posts:
        ts = post.get("created_at")
        if ts:
            timestamps.append(ts)
    
    timestamps.sort()
    
    # Cumulative count
    counts = list(range(1, len(timestamps) + 1))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=counts,
        mode='lines+markers',
        name='Posts',
        line=dict(color='#4ecdc4', width=2),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(78, 205, 196, 0.2)',
    ))
    
    fig.update_layout(
        title="Posts Timeline",
        xaxis_title="Time",
        yaxis_title="Cumulative Posts",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=400,
    )
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Posts Timeline</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body style="background: #0f0f23;">
    {fig.to_html(full_html=False, include_plotlyjs=False)}
</body>
</html>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def _extract_entity_timestamps(
    entities: List[Dict[str, Any]],
    timestamp_key: str = "created_at",
) -> List[str]:
    """Extract and sort timestamps from a list of entities."""
    timestamps = []
    for entity in entities:
        ts = entity.get(timestamp_key)
        if ts:
            timestamps.append(ts)
    timestamps.sort()
    return timestamps


def _extract_post_timestamps_from_transcripts(
    transcripts: List[Any],
) -> List[str]:
    """Extract created_at timestamps from transcript messages."""
    timestamps = []
    for t in transcripts:
        t_dict = _ensure_dict(t)
        messages = t_dict.get("messages", [])
        if messages:
            ts = messages[0].get("created_at")
            if ts:
                timestamps.append(ts)
    timestamps.sort()
    return timestamps


def _extract_unique_agents_timeline(
    transcripts: List[Any],
) -> List[str]:
    """
    Extract timestamps when unique agents first appeared.
    Returns sorted list of timestamps for each unique agent's first post.
    """
    agent_first_seen = {}
    
    for t in transcripts:
        t_dict = _ensure_dict(t)
        messages = t_dict.get("messages", [])
        if not messages:
            continue
        
        msg = messages[0]
        author = msg.get("author") or msg.get("author_external_id")
        ts = msg.get("created_at")
        
        if author and ts:
            if author not in agent_first_seen:
                agent_first_seen[author] = ts
            else:
                # Keep the earlier timestamp
                if ts < agent_first_seen[author]:
                    agent_first_seen[author] = ts
    
    # Return sorted list of first-seen timestamps
    timestamps = list(agent_first_seen.values())
    timestamps.sort()
    return timestamps


def _generate_empty_chart_placeholder(
    entity_name: str,
    color: str,
    chart_height: int = 350,
) -> str:
    """Generate an empty chart placeholder when no data is available."""
    if not PLOTLY_AVAILABLE:
        return f'<div class="empty-chart">No {entity_name} data available</div>'
    
    fig = go.Figure()
    
    fig.add_annotation(
        text=f"No {entity_name} in this period",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color=color),
    )
    
    fig.update_layout(
        title=f"{entity_name} Timeline (0 total)",
        xaxis_title="Time",
        yaxis_title=f"Cumulative {entity_name}",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=chart_height,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
    )
    
    return fig.to_html(full_html=False, include_plotlyjs=False)


def generate_entity_timeline_chart(
    timestamps: List[str],
    entity_name: str,
    color: str,
    fill_color: str,
    chart_height: int = 350,
    urls: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    chart_id: Optional[str] = None,
) -> str:
    """Generate a cumulative timeline chart for an entity type with optional click navigation."""
    if not PLOTLY_AVAILABLE or not timestamps:
        return ""
    
    # Cumulative count
    counts = list(range(1, len(timestamps) + 1))
    
    # Generate unique chart ID for click handlers
    import uuid
    chart_id = chart_id or f"chart-{entity_name.lower()}-{uuid.uuid4().hex[:8]}"
    
    fig = go.Figure()
    
    # Build customdata for click handlers
    customdata = None
    hovertemplate = '%{y} at %{x}<extra></extra>'
    
    if urls:
        if labels:
            customdata = list(zip(urls, labels))
            hovertemplate = '%{y} at %{x}<br><b>%{customdata[1]}</b><extra>Click to view</extra>'
        else:
            customdata = [[u, ""] for u in urls]
            hovertemplate = '%{y} at %{x}<extra>Click to view</extra>'
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=counts,
        mode='lines+markers',
        name=entity_name,
        line=dict(color=color, width=2),
        marker=dict(size=6, symbol='circle'),
        fill='tozeroy',
        fillcolor=fill_color,
        customdata=customdata,
        hovertemplate=hovertemplate,
    ))
    
    fig.update_layout(
        title=f"{entity_name} Timeline ({len(timestamps)} total)",
        xaxis_title="Time",
        yaxis_title=f"Cumulative {entity_name}",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=chart_height,
    )
    
    chart_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=chart_id)
    
    # Add click handler if URLs are provided
    if urls:
        click_script = f'''
        <script>
            (function() {{
                var chartDiv = document.getElementById("{chart_id}");
                if (chartDiv) {{
                    chartDiv.on('plotly_click', function(data) {{
                        var point = data.points[0];
                        if (point.customdata && point.customdata[0]) {{
                            window.open(point.customdata[0], '_blank');
                        }}
                    }});
                }}
            }})();
        </script>
        '''
        return chart_html + click_script
    
    return chart_html


def _compute_rolling_average(scores: List[float]) -> List[float]:
    """Compute cumulative rolling average."""
    averages = []
    total = 0
    for i, score in enumerate(scores):
        total += score
        averages.append(total / (i + 1))
    return averages


def _compute_cumulative_sum(scores: List[float]) -> List[float]:
    """Compute cumulative sum."""
    cumulative = []
    total = 0
    for score in scores:
        total += score
        cumulative.append(total)
    return cumulative


def generate_threat_vector_entity_chart(
    transcripts: List[Any],
    post_evals: List[Dict[str, Any]],
    chart_height: int = 400,
) -> Dict[str, str]:
    """
    Generate threat vector charts where each post is a data point at its creation time.
    Includes both instance-specific scores and rolling average trends.
    Data points are clickable and link to the post on Moltbook.
    
    Returns:
        Dict with chart HTML strings (instance and average versions)
    """
    charts = {}
    
    if not PLOTLY_AVAILABLE or not transcripts or not post_evals:
        return charts
    
    # Build lookup from post_id to (created_at, permalink, title)
    post_metadata = {}
    for t in transcripts:
        t_dict = _ensure_dict(t)
        post_id = t_dict.get("post_id")
        messages = t_dict.get("messages", [])
        permalink = t_dict.get("permalink", "")
        if post_id and messages:
            ts = messages[0].get("created_at")
            title = messages[0].get("title", "Untitled")
            if ts:
                post_metadata[post_id] = {
                    "timestamp": ts,
                    "permalink": permalink,
                    "title": title,
                }
    
    # Build data points: (timestamp, scores, permalink, title)
    dim_names = ["harm_enablement", "deception_or_evasion", 
                 "self_preservation_power_seeking", "delusional_sycophancy"]
    dim_labels = ["Harm Enablement", "Deception/Evasion", 
                  "Power Seeking", "Sycophancy"]
    colors = ['#ff6b6b', '#4ecdc4', '#ffd93d', '#6bcb77']
    
    data_points = []
    for e in post_evals:
        post_id = e.get("post_id")
        meta = post_metadata.get(post_id)
        if not meta:
            continue
        
        scores = e.get("scores", {})
        point = {
            "timestamp": meta["timestamp"],
            "permalink": meta["permalink"],
            "title": meta["title"],
        }
        for dim in dim_names:
            dim_data = scores.get(dim, {})
            point[dim] = dim_data.get("score", 0)
        data_points.append(point)
    
    # Sort by timestamp
    data_points.sort(key=lambda x: x["timestamp"])
    
    if not data_points:
        return charts
    
    timestamps = [p["timestamp"] for p in data_points]
    permalinks = [p["permalink"] for p in data_points]
    titles = [p["title"] for p in data_points]
    customdata = [[url, title] for url, title in zip(permalinks, titles)]
    
    # Chart ID for click handlers
    combined_chart_id = "threat-combined-instance"
    
    # ===== INSTANCE-SPECIFIC COMBINED CHART =====
    fig_combined_instance = go.Figure()
    
    for idx, (dim, label) in enumerate(zip(dim_names, dim_labels)):
        scores = [p.get(dim, 0) for p in data_points]
        
        fig_combined_instance.add_trace(go.Scatter(
            x=timestamps,
            y=scores,
            mode='lines+markers',
            name=label,
            line=dict(color=colors[idx], width=2),
            marker=dict(size=6),
            customdata=customdata,
            hovertemplate='%{y:.1f}<br>%{customdata[1]}<extra>Click to view</extra>',
        ))
    
    fig_combined_instance.update_layout(
        title=f"Threat Vector Scores - Individual Posts ({len(data_points)} posts)",
        xaxis_title="Post Creation Time",
        yaxis_title="Score",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=chart_height,
        yaxis=dict(range=[0, 10]),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified',
    )
    
    chart_html = fig_combined_instance.to_html(full_html=False, include_plotlyjs=False, div_id=combined_chart_id)
    click_script = f'''
    <script>
        (function() {{
            var chartDiv = document.getElementById("{combined_chart_id}");
            if (chartDiv) {{
                chartDiv.on('plotly_click', function(data) {{
                    var point = data.points[0];
                    if (point.customdata && point.customdata[0]) {{
                        window.open(point.customdata[0], '_blank');
                    }}
                }});
            }}
        }})();
    </script>
    '''
    charts["threat_combined_instance"] = chart_html + click_script
    
    # ===== ROLLING AVERAGE COMBINED CHART =====
    avg_chart_id = "threat-combined-avg"
    fig_combined_avg = go.Figure()
    
    for idx, (dim, label) in enumerate(zip(dim_names, dim_labels)):
        scores = [p.get(dim, 0) for p in data_points]
        avg_scores = _compute_rolling_average(scores)
        
        fig_combined_avg.add_trace(go.Scatter(
            x=timestamps,
            y=avg_scores,
            mode='lines+markers',
            name=f"{label} (Avg)",
            line=dict(color=colors[idx], width=3),
            marker=dict(size=4),
            fill='tozeroy',
            fillcolor=f"rgba{tuple(list(int(colors[idx].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}",
            customdata=customdata,
            hovertemplate='Avg: %{y:.2f}<br>%{customdata[1]}<extra>Click to view</extra>',
        ))
    
    fig_combined_avg.update_layout(
        title=f"Threat Vector Rolling Average ({len(data_points)} posts)",
        xaxis_title="Post Creation Time",
        yaxis_title="Average Score",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=chart_height,
        yaxis=dict(range=[0, 10]),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified',
    )
    
    avg_html = fig_combined_avg.to_html(full_html=False, include_plotlyjs=False, div_id=avg_chart_id)
    avg_click_script = f'''
    <script>
        (function() {{
            var chartDiv = document.getElementById("{avg_chart_id}");
            if (chartDiv) {{
                chartDiv.on('plotly_click', function(data) {{
                    var point = data.points[0];
                    if (point.customdata && point.customdata[0]) {{
                        window.open(point.customdata[0], '_blank');
                    }}
                }});
            }}
        }})();
    </script>
    '''
    charts["threat_combined_avg"] = avg_html + avg_click_script
    
    # ===== INSTANCE-SPECIFIC DIMENSION SUBPLOTS =====
    details_instance_id = "threat-details-instance"
    fig_dims_instance = make_subplots(
        rows=2, cols=2,
        subplot_titles=dim_labels,
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    
    for idx, (dim, label) in enumerate(zip(dim_names, dim_labels)):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        scores = [p.get(dim, 0) for p in data_points]
        
        fig_dims_instance.add_trace(
            go.Scatter(
                x=timestamps,
                y=scores,
                mode='lines+markers',
                name=label,
                line=dict(color=colors[idx], width=2),
                marker=dict(size=5),
                showlegend=False,
                customdata=customdata,
                hovertemplate='%{y:.1f}<br>%{customdata[1]}<extra>Click to view</extra>',
            ),
            row=row, col=col
        )
    
    fig_dims_instance.update_layout(
        title=f"Threat Vector Details - Individual Posts ({len(data_points)} posts)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=500,
    )
    
    fig_dims_instance.update_yaxes(range=[0, 10])
    
    details_html = fig_dims_instance.to_html(full_html=False, include_plotlyjs=False, div_id=details_instance_id)
    details_click_script = f'''
    <script>
        (function() {{
            var chartDiv = document.getElementById("{details_instance_id}");
            if (chartDiv) {{
                chartDiv.on('plotly_click', function(data) {{
                    var point = data.points[0];
                    if (point.customdata && point.customdata[0]) {{
                        window.open(point.customdata[0], '_blank');
                    }}
                }});
            }}
        }})();
    </script>
    '''
    charts["threat_details_instance"] = details_html + details_click_script
    
    # ===== ROLLING AVERAGE DIMENSION SUBPLOTS =====
    details_avg_id = "threat-details-avg"
    fig_dims_avg = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"{l} (Rolling Avg)" for l in dim_labels],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    
    for idx, (dim, label) in enumerate(zip(dim_names, dim_labels)):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        scores = [p.get(dim, 0) for p in data_points]
        avg_scores = _compute_rolling_average(scores)
        
        fig_dims_avg.add_trace(
            go.Scatter(
                x=timestamps,
                y=avg_scores,
                mode='lines+markers',
                name=label,
                line=dict(color=colors[idx], width=3),
                marker=dict(size=3),
                fill='tozeroy',
                fillcolor=f"rgba{tuple(list(int(colors[idx].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}",
                showlegend=False,
                customdata=customdata,
                hovertemplate='Avg: %{y:.2f}<br>%{customdata[1]}<extra>Click to view</extra>',
            ),
            row=row, col=col
        )
    
    fig_dims_avg.update_layout(
        title=f"Threat Vector Details - Rolling Average ({len(data_points)} posts)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=500,
    )
    
    fig_dims_avg.update_yaxes(range=[0, 10])
    
    details_avg_html = fig_dims_avg.to_html(full_html=False, include_plotlyjs=False, div_id=details_avg_id)
    details_avg_click = f'''
    <script>
        (function() {{
            var chartDiv = document.getElementById("{details_avg_id}");
            if (chartDiv) {{
                chartDiv.on('plotly_click', function(data) {{
                    var point = data.points[0];
                    if (point.customdata && point.customdata[0]) {{
                        window.open(point.customdata[0], '_blank');
                    }}
                }});
            }}
        }})();
    </script>
    '''
    charts["threat_details_avg"] = details_avg_html + details_avg_click
    
    # ===== CUMULATIVE SUM COMBINED CHART =====
    cumulative_chart_id = "threat-combined-cumulative"
    fig_combined_cumulative = go.Figure()
    
    for idx, (dim, label) in enumerate(zip(dim_names, dim_labels)):
        scores = [p.get(dim, 0) for p in data_points]
        cumulative_scores = _compute_cumulative_sum(scores)
        
        fig_combined_cumulative.add_trace(go.Scatter(
            x=timestamps,
            y=cumulative_scores,
            mode='lines+markers',
            name=f"{label} (Cumulative)",
            line=dict(color=colors[idx], width=3),
            marker=dict(size=4),
            fill='tozeroy',
            fillcolor=f"rgba{tuple(list(int(colors[idx].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}",
            customdata=customdata,
            hovertemplate='Cumulative: %{y:.0f}<br>%{customdata[1]}<extra>Click to view</extra>',
        ))
    
    # Calculate max cumulative for y-axis
    max_cumulative = max(
        sum(p.get(dim, 0) for p in data_points)
        for dim in dim_names
    ) if data_points else 10
    
    fig_combined_cumulative.update_layout(
        title=f"Threat Vector Cumulative Sum ({len(data_points)} posts)",
        xaxis_title="Post Creation Time",
        yaxis_title="Cumulative Score",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=chart_height,
        yaxis=dict(range=[0, max_cumulative * 1.1]),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified',
    )
    
    cumulative_html = fig_combined_cumulative.to_html(full_html=False, include_plotlyjs=False, div_id=cumulative_chart_id)
    cumulative_click = f'''
    <script>
        (function() {{
            var chartDiv = document.getElementById("{cumulative_chart_id}");
            if (chartDiv) {{
                chartDiv.on('plotly_click', function(data) {{
                    var point = data.points[0];
                    if (point.customdata && point.customdata[0]) {{
                        window.open(point.customdata[0], '_blank');
                    }}
                }});
            }}
        }})();
    </script>
    '''
    charts["threat_combined_cumulative"] = cumulative_html + cumulative_click
    
    # ===== CUMULATIVE SUM DIMENSION SUBPLOTS =====
    details_cumulative_id = "threat-details-cumulative"
    fig_dims_cumulative = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"{l} (Cumulative)" for l in dim_labels],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    
    for idx, (dim, label) in enumerate(zip(dim_names, dim_labels)):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        scores = [p.get(dim, 0) for p in data_points]
        cumulative_scores = _compute_cumulative_sum(scores)
        
        fig_dims_cumulative.add_trace(
            go.Scatter(
                x=timestamps,
                y=cumulative_scores,
                mode='lines+markers',
                name=label,
                line=dict(color=colors[idx], width=3),
                marker=dict(size=3),
                fill='tozeroy',
                fillcolor=f"rgba{tuple(list(int(colors[idx].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}",
                showlegend=False,
                customdata=customdata,
                hovertemplate='Cumulative: %{y:.0f}<br>%{customdata[1]}<extra>Click to view</extra>',
            ),
            row=row, col=col
        )
    
    fig_dims_cumulative.update_layout(
        title=f"Threat Vector Details - Cumulative Sum ({len(data_points)} posts)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=500,
    )
    
    details_cumulative_html = fig_dims_cumulative.to_html(full_html=False, include_plotlyjs=False, div_id=details_cumulative_id)
    details_cumulative_click = f'''
    <script>
        (function() {{
            var chartDiv = document.getElementById("{details_cumulative_id}");
            if (chartDiv) {{
                chartDiv.on('plotly_click', function(data) {{
                    var point = data.points[0];
                    if (point.customdata && point.customdata[0]) {{
                        window.open(point.customdata[0], '_blank');
                    }}
                }});
            }}
        }})();
    </script>
    '''
    charts["threat_details_cumulative"] = details_cumulative_html + details_cumulative_click
    
    return charts


def generate_thread_infection_charts(
    comment_transcripts: List[Any],
    comment_evals: List[Dict[str, Any]],
    transcripts: List[Any],
    chart_height: int = 400,
) -> Dict[str, Any]:
    """
    Generate thread infection analysis charts showing threat vectors across comments in each thread.
    
    Args:
        comment_transcripts: Comment transcripts with post_id and timestamps
        comment_evals: Comment evaluations with scores
        transcripts: Post transcripts for thread titles
        chart_height: Height of charts
    
    Returns:
        Dict with 'charts' (HTML strings), 'threads' (list of thread info), and 'dropdown' (HTML)
    """
    result = {"charts": {}, "threads": [], "dropdown": "", "section": ""}
    
    if not PLOTLY_AVAILABLE or not comment_evals:
        return result
    
    # Build lookup for post titles and permalinks
    post_metadata = {}
    for t in transcripts:
        t_dict = _ensure_dict(t)
        post_id = t_dict.get("post_id")
        messages = t_dict.get("messages", [])
        permalink = t_dict.get("permalink", "")
        if post_id and messages:
            post_metadata[post_id] = {
                "title": messages[0].get("title", "Untitled Thread"),
                "permalink": permalink,
            }
    
    # Keep backward compat
    post_titles = {pid: meta["title"] for pid, meta in post_metadata.items()}
    
    # Build lookup from comment_id to eval and timestamp
    comment_timestamp_map = {}
    for ct in comment_transcripts:
        ct_dict = _ensure_dict(ct)
        comment_id = ct_dict.get("comment_id")
        target = ct_dict.get("target_comment", {})
        if comment_id:
            comment_timestamp_map[comment_id] = target.get("created_at")
    
    # Group evals by post_id
    threads = {}
    for e in comment_evals:
        post_id = e.get("post_id")
        comment_id = e.get("comment_id")
        if not post_id:
            continue
        
        if post_id not in threads:
            threads[post_id] = []
        
        threads[post_id].append({
            "comment_id": comment_id,
            "timestamp": comment_timestamp_map.get(comment_id),
            "scores": e.get("scores", {}),
        })
    
    # Filter threads with at least 2 comments
    valid_threads = {pid: comments for pid, comments in threads.items() if len(comments) >= 2}
    
    if not valid_threads:
        return result
    
    dim_names = ["harm_enablement", "deception_or_evasion", 
                 "self_preservation_power_seeking", "delusional_sycophancy"]
    dim_labels = ["Harm", "Deception", "Power Seeking", "Sycophancy"]
    colors = ['#ff6b6b', '#4ecdc4', '#ffd93d', '#6bcb77']
    
    # Generate charts for each thread
    thread_info = []
    thread_charts = {}
    
    for post_id, comments in valid_threads.items():
        # Sort comments by timestamp
        comments_sorted = sorted(
            [c for c in comments if c.get("timestamp")],
            key=lambda x: x["timestamp"]
        )
        
        if len(comments_sorted) < 2:
            continue
        
        thread_meta = post_metadata.get(post_id, {"title": "Untitled", "permalink": ""})
        thread_title = thread_meta["title"][:40]
        thread_permalink = thread_meta["permalink"]
        thread_info.append({
            "post_id": post_id,
            "title": thread_title,
            "permalink": thread_permalink,
            "comment_count": len(comments_sorted),
        })
        
        timestamps = [c["timestamp"] for c in comments_sorted]
        
        # Individual scores chart
        fig_individual = go.Figure()
        for idx, (dim, label) in enumerate(zip(dim_names, dim_labels)):
            scores = [c["scores"].get(dim, {}).get("score", 0) for c in comments_sorted]
            fig_individual.add_trace(go.Scatter(
                x=list(range(1, len(timestamps) + 1)),
                y=scores,
                mode='lines+markers',
                name=label,
                line=dict(color=colors[idx], width=2),
                marker=dict(size=6),
            ))
        
        fig_individual.update_layout(
            title=f"Thread: {thread_title} ({len(comments_sorted)} comments)",
            xaxis_title="Comment # in Thread",
            yaxis_title="Score",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,15,35,1)',
            height=350,
            yaxis=dict(range=[0, 10]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        
        # Cumulative chart
        fig_cumulative = go.Figure()
        for idx, (dim, label) in enumerate(zip(dim_names, dim_labels)):
            scores = [c["scores"].get(dim, {}).get("score", 0) for c in comments_sorted]
            cumulative = _compute_cumulative_sum(scores)
            fig_cumulative.add_trace(go.Scatter(
                x=list(range(1, len(timestamps) + 1)),
                y=cumulative,
                mode='lines+markers',
                name=label,
                line=dict(color=colors[idx], width=2),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor=f"rgba{tuple(list(int(colors[idx].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}",
            ))
        
        fig_cumulative.update_layout(
            title=f"Thread: {thread_title} - Cumulative",
            xaxis_title="Comment # in Thread",
            yaxis_title="Cumulative Score",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,15,35,1)',
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        
        # Average chart
        fig_avg = go.Figure()
        for idx, (dim, label) in enumerate(zip(dim_names, dim_labels)):
            scores = [c["scores"].get(dim, {}).get("score", 0) for c in comments_sorted]
            avg = _compute_rolling_average(scores)
            fig_avg.add_trace(go.Scatter(
                x=list(range(1, len(timestamps) + 1)),
                y=avg,
                mode='lines+markers',
                name=label,
                line=dict(color=colors[idx], width=2),
                marker=dict(size=6),
            ))
        
        fig_avg.update_layout(
            title=f"Thread: {thread_title} - Rolling Average",
            xaxis_title="Comment # in Thread",
            yaxis_title="Average Score",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,15,35,1)',
            height=350,
            yaxis=dict(range=[0, 10]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        
        thread_charts[post_id] = {
            "individual": fig_individual.to_html(full_html=False, include_plotlyjs=False),
            "cumulative": fig_cumulative.to_html(full_html=False, include_plotlyjs=False),
            "avg": fig_avg.to_html(full_html=False, include_plotlyjs=False),
        }
    
    if not thread_info:
        return result
    
    # Generate dropdown and charts HTML
    # Store permalink as data attribute for "View Thread" link
    dropdown_options = "\n".join(
        f'<option value="{t["post_id"]}" data-permalink="{t.get("permalink", "")}">{t["title"]} ({t["comment_count"]} comments)</option>'
        for t in thread_info
    )
    
    # Get first thread's permalink for initial view
    first_permalink = thread_info[0].get("permalink", "") if thread_info else ""
    
    dropdown_html = f'''
        <div class="thread-selector">
            <label for="thread-select">Select Thread:</label>
            <select id="thread-select" onchange="selectThread(this.value, this.options[this.selectedIndex].dataset.permalink)">
                {dropdown_options}
            </select>
            <a href="{first_permalink}" target="_blank" id="view-thread-link" class="view-thread-link">View Thread on Moltbook</a>
        </div>
    '''
    
    # Generate all thread chart containers (hidden by default except first)
    charts_html = []
    for i, t in enumerate(thread_info):
        pid = t["post_id"]
        permalink = t.get("permalink", "")
        display = "block" if i == 0 else "none"
        charts_html.append(f'''
            <div class="thread-charts" id="thread-{pid}" style="display:{display};" data-permalink="{permalink}">
                <div class="toggle-container">
                    <button class="toggle-btn active" onclick="toggleThreadView('{pid}', 'individual')" id="btn-thread-{pid}-individual">Individual</button>
                    <button class="toggle-btn" onclick="toggleThreadView('{pid}', 'cumulative')" id="btn-thread-{pid}-cumulative">Cumulative</button>
                    <button class="toggle-btn" onclick="toggleThreadView('{pid}', 'avg')" id="btn-thread-{pid}-avg">Average</button>
                </div>
                <div class="chart-container" id="chart-thread-{pid}-individual">{thread_charts[pid]["individual"]}</div>
                <div class="chart-container" id="chart-thread-{pid}-cumulative" style="display:none;">{thread_charts[pid]["cumulative"]}</div>
                <div class="chart-container" id="chart-thread-{pid}-avg" style="display:none;">{thread_charts[pid]["avg"]}</div>
            </div>
        ''')
    
    result["charts"] = thread_charts
    result["threads"] = thread_info
    result["dropdown"] = dropdown_html
    result["charts_html"] = "\n".join(charts_html)
    
    return result


def _extract_posts_with_urls(transcripts: List[Any]) -> tuple:
    """Extract post timestamps, URLs, and titles from transcripts."""
    data = []
    for t in transcripts:
        t_dict = _ensure_dict(t)
        messages = t_dict.get("messages", [])
        if messages:
            ts = messages[0].get("created_at")
            permalink = t_dict.get("permalink", "")
            title = messages[0].get("title", "Untitled")
            if ts:
                data.append((ts, permalink, title))
    
    data.sort(key=lambda x: x[0])
    timestamps = [d[0] for d in data]
    urls = [d[1] for d in data]
    labels = [d[2] for d in data]
    return timestamps, urls, labels


def _extract_agents_with_urls(transcripts: List[Any]) -> tuple:
    """Extract unique agents with their profile URLs."""
    agent_first_seen = {}
    
    for t in transcripts:
        t_dict = _ensure_dict(t)
        messages = t_dict.get("messages", [])
        if not messages:
            continue
        
        msg = messages[0]
        author = msg.get("author") or msg.get("author_external_id")
        ts = msg.get("created_at")
        
        if author and ts:
            if author not in agent_first_seen:
                agent_first_seen[author] = {
                    "timestamp": ts,
                    "url": f"https://www.moltbook.com/u/{author}",
                    "name": author,
                }
            elif ts < agent_first_seen[author]["timestamp"]:
                agent_first_seen[author]["timestamp"] = ts
    
    # Sort by timestamp
    sorted_agents = sorted(agent_first_seen.values(), key=lambda x: x["timestamp"])
    timestamps = [a["timestamp"] for a in sorted_agents]
    urls = [a["url"] for a in sorted_agents]
    labels = [a["name"] for a in sorted_agents]
    return timestamps, urls, labels


def _extract_submolts_with_urls(submolts: List[Any]) -> tuple:
    """Extract submolt timestamps and URLs."""
    data = []
    for s in submolts:
        ts = s.get("created_at")
        url = s.get("url", "")
        name = s.get("display_name") or s.get("name", "Unknown")
        if ts:
            data.append((ts, url, name))
    
    data.sort(key=lambda x: x[0])
    timestamps = [d[0] for d in data]
    urls = [d[1] for d in data]
    labels = [d[2] for d in data]
    return timestamps, urls, labels


def _extract_comments_with_urls(comments: List[Any]) -> tuple:
    """Extract comment timestamps and URLs."""
    data = []
    for c in comments:
        ts = c.get("created_at")
        # Comments typically link to their post with an anchor
        post_id = c.get("post_id", "")
        comment_id = c.get("comment_external_id") or c.get("id", "")
        url = f"https://www.moltbook.com/post/{post_id}#comment-{comment_id}" if post_id else ""
        author = c.get("author_handle") or c.get("author", "Unknown")
        if ts:
            data.append((ts, url, f"by {author}"))
    
    data.sort(key=lambda x: x[0])
    timestamps = [d[0] for d in data]
    urls = [d[1] for d in data]
    labels = [d[2] for d in data]
    return timestamps, urls, labels


def generate_entity_growth_report(
    posts: List[Any],
    comments: List[Any],
    submolts: List[Any],
    transcripts: List[Any],
    post_evals: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Growth Analytics",
    comment_transcripts: Optional[List[Any]] = None,
    comment_evals: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Generate a growth report using individual entity timestamps.
    
    Each entity (post, comment, agent, submolt) becomes its own data point
    on the timeline, plotted at its created_at timestamp. Clicking a point
    navigates to that entity on Moltbook.
    
    Args:
        posts: List of posts with created_at
        comments: List of comments with created_at
        submolts: List of submolts with created_at
        transcripts: List of transcripts with messages[0].created_at
        post_evals: List of post evaluations with scores
        output_path: Path to write the HTML file
        title: Report title
        comment_transcripts: Optional list of comment transcripts
        comment_evals: Optional list of comment evaluations with scores
    
    Returns:
        Path to generated report file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract entity data with URLs for clickable charts
    post_timestamps, post_urls, post_labels = _extract_posts_with_urls(transcripts)
    comment_timestamps, comment_urls, comment_labels = _extract_comments_with_urls(comments)
    agent_timestamps, agent_urls, agent_labels = _extract_agents_with_urls(transcripts)
    submolt_timestamps, submolt_urls, submolt_labels = _extract_submolts_with_urls(submolts)
    
    # Fallback for posts if transcripts don't have data
    if not post_timestamps:
        post_timestamps = _extract_entity_timestamps(posts)
        post_urls = []
        post_labels = []
    
    # Generate entity timeline charts with clickable data points
    entity_charts = {}
    
    entity_configs = [
        ('posts', post_timestamps, 'Posts', '#4ecdc4', 'rgba(78, 205, 196, 0.2)', post_urls, post_labels),
        ('comments', comment_timestamps, 'Comments', '#ffd93d', 'rgba(255, 217, 61, 0.2)', comment_urls, comment_labels),
        ('agents', agent_timestamps, 'Agents', '#ff6b6b', 'rgba(255, 107, 107, 0.2)', agent_urls, agent_labels),
        ('submolts', submolt_timestamps, 'Submolts', '#6bcb77', 'rgba(107, 203, 119, 0.2)', submolt_urls, submolt_labels),
    ]
    
    for key, timestamps, label, color, fill_color, urls, labels in entity_configs:
        if timestamps:
            entity_charts[key] = generate_entity_timeline_chart(
                timestamps, label, color, fill_color,
                urls=urls if urls else None,
                labels=labels if labels else None,
                chart_id=f"timeline-{key}",
            )
        else:
            # Generate empty chart placeholder
            entity_charts[key] = _generate_empty_chart_placeholder(label, color)
    
    # Generate threat vector charts
    threat_charts = generate_threat_vector_entity_chart(transcripts, post_evals)
    
    # Generate thread infection charts if comment data is available
    thread_infection = {}
    if comment_transcripts and comment_evals:
        thread_infection = generate_thread_infection_charts(
            comment_transcripts, comment_evals, transcripts
        )
    
    # Build HTML content
    content_parts = [
        f'''
        <header>
            <h1>📈 {title}</h1>
            <p class="subtitle">Moltbook Platform Growth Over Time (Entity-Level)</p>
        </header>
        ''',
    ]
    
    # Summary stats
    content_parts.append(f'''
        <section>
            <h2>Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{len(post_timestamps)}</div>
                    <div class="stat-label">Posts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(comment_timestamps)}</div>
                    <div class="stat-label">Comments</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(agent_timestamps)}</div>
                    <div class="stat-label">Unique Agents</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(submolt_timestamps)}</div>
                    <div class="stat-label">Submolts</div>
                </div>
            </div>
        </section>
    ''')
    
    # Entity growth timelines (2x2 grid)
    entity_charts_html = []
    for key, _, label, _, _, _, _ in entity_configs:
        if key in entity_charts:
            entity_charts_html.append(f'''
                <div class="chart-half">
                    {entity_charts[key]}
                </div>
            ''')
    
    if entity_charts_html:
        content_parts.append(f'''
            <section>
                <h2>Entity Growth Timelines</h2>
                <p class="section-desc">Each data point represents an individual entity at its creation time.</p>
                <div class="charts-grid">
                    {"".join(entity_charts_html)}
                </div>
            </section>
        ''')
    
    # Threat vector charts with toggle buttons (Individual / Cumulative / Average)
    has_combined_charts = all(k in threat_charts for k in ["threat_combined_instance", "threat_combined_avg", "threat_combined_cumulative"])
    has_details_charts = all(k in threat_charts for k in ["threat_details_instance", "threat_details_avg", "threat_details_cumulative"])
    
    if has_combined_charts:
        content_parts.append(f'''
            <section>
                <h2>Threat Vector Trends</h2>
                <div class="toggle-container">
                    <button class="toggle-btn active" onclick="toggleThreatCombined('instance')" id="btn-combined-instance">Individual</button>
                    <button class="toggle-btn" onclick="toggleThreatCombined('cumulative')" id="btn-combined-cumulative">Cumulative</button>
                    <button class="toggle-btn" onclick="toggleThreatCombined('avg')" id="btn-combined-avg">Rolling Average</button>
                </div>
                <p class="section-desc" id="desc-combined-instance">Each data point shows the threat score of an individual post at its creation time.</p>
                <p class="section-desc" id="desc-combined-cumulative" style="display:none;">Cumulative sum of threat scores over time.</p>
                <p class="section-desc" id="desc-combined-avg" style="display:none;">Rolling cumulative average of threat scores over time.</p>
                <div class="chart-container" id="chart-combined-instance">
                    {threat_charts["threat_combined_instance"]}
                </div>
                <div class="chart-container" id="chart-combined-cumulative" style="display:none;">
                    {threat_charts["threat_combined_cumulative"]}
                </div>
                <div class="chart-container" id="chart-combined-avg" style="display:none;">
                    {threat_charts["threat_combined_avg"]}
                </div>
            </section>
        ''')
    
    if has_details_charts:
        content_parts.append(f'''
            <section>
                <h2>Threat Vector Details by Dimension</h2>
                <div class="toggle-container">
                    <button class="toggle-btn active" onclick="toggleThreatDetails('instance')" id="btn-details-instance">Individual</button>
                    <button class="toggle-btn" onclick="toggleThreatDetails('cumulative')" id="btn-details-cumulative">Cumulative</button>
                    <button class="toggle-btn" onclick="toggleThreatDetails('avg')" id="btn-details-avg">Rolling Average</button>
                </div>
                <p class="section-desc" id="desc-details-instance">Each data point shows the threat score of an individual post.</p>
                <p class="section-desc" id="desc-details-cumulative" style="display:none;">Cumulative sum per dimension.</p>
                <p class="section-desc" id="desc-details-avg" style="display:none;">Rolling cumulative average per dimension.</p>
                <div class="chart-container" id="chart-details-instance">
                    {threat_charts["threat_details_instance"]}
                </div>
                <div class="chart-container" id="chart-details-cumulative" style="display:none;">
                    {threat_charts["threat_details_cumulative"]}
                </div>
                <div class="chart-container" id="chart-details-avg" style="display:none;">
                    {threat_charts["threat_details_avg"]}
                </div>
            </section>
        ''')
    
    # Thread Infection Analysis section
    if thread_infection and thread_infection.get("threads"):
        thread_count = len(thread_infection["threads"])
        content_parts.append(f'''
            <section>
                <h2>Thread Infection Analysis</h2>
                <p class="section-desc">Analyze how threat scores evolve within individual discussion threads. Select a thread with 2+ comments to see the infection pattern.</p>
                <p class="section-desc"><strong>{thread_count} threads</strong> with multiple evaluated comments.</p>
                {thread_infection.get("dropdown", "")}
                {thread_infection.get("charts_html", "")}
            </section>
        ''')
    
    # Extra styles
    extra_styles = '''
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
        
        .chart-half {
            background: var(--bg-primary);
            border-radius: 8px;
            padding: 0.5rem;
        }
        
        .section-desc {
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }
        
        .toggle-container {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        
        .toggle-btn {
            padding: 0.5rem 1rem;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--bg-primary);
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s ease;
        }
        
        .toggle-btn:hover {
            background: var(--bg-secondary);
            color: var(--text-primary);
        }
        
        .toggle-btn.active {
            background: linear-gradient(135deg, var(--accent), var(--accent-secondary));
            color: white;
            border-color: transparent;
        }
        
        @media (max-width: 900px) {
            .charts-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .thread-selector {
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .thread-selector label {
            color: var(--text-primary);
            font-weight: 500;
        }
        
        .thread-selector select {
            padding: 0.5rem 1rem;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--bg-primary);
            color: var(--text-primary);
            font-size: 0.9rem;
            min-width: 300px;
            cursor: pointer;
        }
        
        .thread-selector select:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        .thread-charts {
            margin-top: 1rem;
        }
        
        .view-thread-link {
            color: #00e5ff;
            text-decoration: none;
            font-size: 0.9rem;
            padding: 0.5rem 1rem;
            border: 1px solid #00e5ff;
            border-radius: 6px;
            transition: all 0.2s ease;
        }
        
        .view-thread-link:hover {
            background: rgba(0, 229, 255, 0.1);
            color: #4ecdc4;
        }
    '''
    
    # Add toggle JavaScript
    toggle_script = '''
    <script>
        function toggleThreatCombined(mode) {
            const charts = {
                instance: document.getElementById('chart-combined-instance'),
                cumulative: document.getElementById('chart-combined-cumulative'),
                avg: document.getElementById('chart-combined-avg')
            };
            const descs = {
                instance: document.getElementById('desc-combined-instance'),
                cumulative: document.getElementById('desc-combined-cumulative'),
                avg: document.getElementById('desc-combined-avg')
            };
            const btns = {
                instance: document.getElementById('btn-combined-instance'),
                cumulative: document.getElementById('btn-combined-cumulative'),
                avg: document.getElementById('btn-combined-avg')
            };
            
            // Hide all, then show selected
            Object.values(charts).forEach(c => { if (c) c.style.display = 'none'; });
            Object.values(descs).forEach(d => { if (d) d.style.display = 'none'; });
            Object.values(btns).forEach(b => { if (b) b.classList.remove('active'); });
            
            if (charts[mode]) charts[mode].style.display = 'block';
            if (descs[mode]) descs[mode].style.display = 'block';
            if (btns[mode]) btns[mode].classList.add('active');
            
            // Trigger Plotly relayout to fix sizing
            setTimeout(() => {
                window.dispatchEvent(new Event('resize'));
            }, 100);
        }
        
        function toggleThreatDetails(mode) {
            const charts = {
                instance: document.getElementById('chart-details-instance'),
                cumulative: document.getElementById('chart-details-cumulative'),
                avg: document.getElementById('chart-details-avg')
            };
            const descs = {
                instance: document.getElementById('desc-details-instance'),
                cumulative: document.getElementById('desc-details-cumulative'),
                avg: document.getElementById('desc-details-avg')
            };
            const btns = {
                instance: document.getElementById('btn-details-instance'),
                cumulative: document.getElementById('btn-details-cumulative'),
                avg: document.getElementById('btn-details-avg')
            };
            
            // Hide all, then show selected
            Object.values(charts).forEach(c => { if (c) c.style.display = 'none'; });
            Object.values(descs).forEach(d => { if (d) d.style.display = 'none'; });
            Object.values(btns).forEach(b => { if (b) b.classList.remove('active'); });
            
            if (charts[mode]) charts[mode].style.display = 'block';
            if (descs[mode]) descs[mode].style.display = 'block';
            if (btns[mode]) btns[mode].classList.add('active');
            
            // Trigger Plotly relayout to fix sizing
            setTimeout(() => {
                window.dispatchEvent(new Event('resize'));
            }, 100);
        }
        
        // Thread infection analysis functions
        function selectThread(postId, permalink) {
            // Hide all thread charts
            document.querySelectorAll('.thread-charts').forEach(el => {
                el.style.display = 'none';
            });
            
            // Show selected thread
            const selectedThread = document.getElementById('thread-' + postId);
            if (selectedThread) {
                selectedThread.style.display = 'block';
                // Trigger resize for Plotly
                setTimeout(() => {
                    window.dispatchEvent(new Event('resize'));
                }, 100);
            }
            
            // Update "View Thread" link
            const viewLink = document.getElementById('view-thread-link');
            if (viewLink && permalink) {
                viewLink.href = permalink;
            }
        }
        
        function toggleThreadView(postId, mode) {
            const charts = {
                individual: document.getElementById('chart-thread-' + postId + '-individual'),
                cumulative: document.getElementById('chart-thread-' + postId + '-cumulative'),
                avg: document.getElementById('chart-thread-' + postId + '-avg')
            };
            const btns = {
                individual: document.getElementById('btn-thread-' + postId + '-individual'),
                cumulative: document.getElementById('btn-thread-' + postId + '-cumulative'),
                avg: document.getElementById('btn-thread-' + postId + '-avg')
            };
            
            // Hide all charts, remove active from all buttons
            Object.values(charts).forEach(c => { if (c) c.style.display = 'none'; });
            Object.values(btns).forEach(b => { if (b) b.classList.remove('active'); });
            
            // Show selected
            if (charts[mode]) charts[mode].style.display = 'block';
            if (btns[mode]) btns[mode].classList.add('active');
            
            // Trigger Plotly relayout
            setTimeout(() => {
                window.dispatchEvent(new Event('resize'));
            }, 100);
        }
    </script>
    '''
    
    # Append script to content
    content_parts.append(toggle_script)
    
    html = _html_template(title=title, content="\n".join(content_parts), styles=extra_styles)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    return str(output_path)

