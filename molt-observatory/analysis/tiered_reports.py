"""
Tiered Evaluation Reports for Molt Observatory

Generates comprehensive HTML reports and charts for tiered evaluation runs,
including spam analysis, agent leaderboards, cascade visualizations, and filter funnels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# =============================================================================
# Color Schemes
# =============================================================================

COLORS = {
    "spam": "#e74c3c",
    "crypto": "#f39c12", 
    "short": "#9b59b6",
    "duplicate": "#3498db",
    "passed": "#2ecc71",
    "escalated": "#e74c3c",
    "benign": "#2ecc71",
    "primary": "#4ecdc4",
    "secondary": "#ff6b6b",
    "background": "#1a1a2e",
    "text": "#eee",
}

DARK_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#eee", family="Inter, system-ui, sans-serif"),
        title=dict(font=dict(size=20, color="#4ecdc4")),
        xaxis=dict(gridcolor="#333", zerolinecolor="#333"),
        yaxis=dict(gridcolor="#333", zerolinecolor="#333"),
    )
)


# =============================================================================
# Spam Analysis Charts
# =============================================================================

def generate_spam_breakdown_chart(
    filter_stats: Dict[str, Any],
    output_dir: Path,
) -> Optional[Path]:
    """
    Generate pie chart showing spam breakdown by type.
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract values
    labels = ["Spam Agent", "Spam Content", "Too Short", "Duplicate", "Passed"]
    values = [
        filter_stats.get("spam_agent_filtered", 0),
        filter_stats.get("spam_content_filtered", 0),
        filter_stats.get("short_filtered", 0),
        filter_stats.get("duplicate_filtered", 0),
        filter_stats.get("passed", 0),
    ]
    colors = [COLORS["spam"], COLORS["crypto"], COLORS["short"], COLORS["duplicate"], COLORS["passed"]]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textfont=dict(size=12, color="white"),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percent: %{percent}<extra></extra>",
    )])
    
    fig.update_layout(
        title=dict(text="Content Filter Breakdown", x=0.5),
        paper_bgcolor=DARK_TEMPLATE["layout"]["paper_bgcolor"],
        plot_bgcolor=DARK_TEMPLATE["layout"]["plot_bgcolor"],
        font=DARK_TEMPLATE["layout"]["font"],
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        annotations=[dict(
            text=f"{sum(values):,}<br>Total",
            x=0.5, y=0.5, font_size=16, showarrow=False, font_color="#4ecdc4"
        )],
    )
    
    output_path = output_dir / "spam_breakdown.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path


def generate_spam_types_bar(
    spammer_profiles: List[Dict[str, Any]],
    output_dir: Path,
    top_n: int = 15,
) -> Optional[Path]:
    """
    Generate stacked bar chart showing spam types per agent.
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get top spammers
    top_spammers = sorted(spammer_profiles, key=lambda x: -x.get("spam_messages", 0))[:top_n]
    
    if not top_spammers:
        return None
    
    # Extract data
    agents = [s["agent_id"][:20] for s in top_spammers]
    
    # Collect all spam types
    all_types = set()
    for s in top_spammers:
        all_types.update(s.get("spam_types", {}).keys())
    
    fig = go.Figure()
    
    type_colors = {
        "Disease": COLORS["spam"],
        "crypto_json_mbc20": COLORS["crypto"],
        "crypto_json_generic": "#e67e22",
        "Test": COLORS["short"],
        "Test comment": COLORS["duplicate"],
    }
    
    for spam_type in sorted(all_types):
        values = [s.get("spam_types", {}).get(spam_type, 0) for s in top_spammers]
        color = type_colors.get(spam_type, "#7f8c8d")
        
        fig.add_trace(go.Bar(
            name=spam_type,
            x=agents,
            y=values,
            marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{spam_type}: %{{y:,}}<extra></extra>",
        ))
    
    fig.update_layout(
        title=dict(text="Spam Types by Agent", x=0.5),
        barmode="stack",
        paper_bgcolor=DARK_TEMPLATE["layout"]["paper_bgcolor"],
        plot_bgcolor=DARK_TEMPLATE["layout"]["plot_bgcolor"],
        font=DARK_TEMPLATE["layout"]["font"],
        xaxis=dict(title="Agent", tickangle=-45, gridcolor="#333"),
        yaxis=dict(title="Spam Count", gridcolor="#333"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    
    output_path = output_dir / "spam_types_bar.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path


# =============================================================================
# Agent Leaderboard
# =============================================================================

def generate_agent_spam_leaderboard(
    spammer_profiles: List[Dict[str, Any]],
    output_dir: Path,
    top_n: int = 20,
) -> Optional[Path]:
    """
    Generate HTML leaderboard of top spamming agents.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by spam count
    sorted_profiles = sorted(spammer_profiles, key=lambda x: -x.get("spam_messages", 0))[:top_n]
    
    # Build HTML table
    rows = []
    for i, profile in enumerate(sorted_profiles, 1):
        agent_id = profile.get("agent_id", "unknown")
        total = profile.get("total_messages", 0)
        spam = profile.get("spam_messages", 0)
        rate = profile.get("spam_rate", 0)
        spam_types = profile.get("spam_types", {})
        primary_type = max(spam_types.items(), key=lambda x: x[1])[0] if spam_types else "unknown"
        
        # Color based on spam type
        type_colors = {
            "Disease": "#e74c3c",
            "crypto_json_mbc20": "#f39c12",
            "crypto_json_generic": "#e67e22",
        }
        badge_color = type_colors.get(primary_type, "#7f8c8d")
        
        rows.append(f"""
        <tr>
            <td class="rank">{i}</td>
            <td class="agent">{agent_id}</td>
            <td class="count">{spam:,}</td>
            <td class="rate">{rate:.0%}</td>
            <td><span class="badge" style="background:{badge_color}">{primary_type}</span></td>
            <td class="posts">{profile.get("posts_targeted", 0)}</td>
        </tr>
        """)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Spam Agent Leaderboard</title>
    <style>
        body {{
            font-family: Inter, system-ui, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
            margin: 0;
        }}
        h1 {{
            color: #4ecdc4;
            text-align: center;
            margin-bottom: 30px;
        }}
        table {{
            width: 100%;
            max-width: 900px;
            margin: 0 auto;
            border-collapse: collapse;
        }}
        th {{
            background: #16213e;
            color: #4ecdc4;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #4ecdc4;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #333;
        }}
        tr:hover {{
            background: #16213e;
        }}
        .rank {{
            color: #888;
            font-weight: bold;
            width: 40px;
        }}
        .agent {{
            font-family: monospace;
            color: #ff6b6b;
        }}
        .count {{
            font-weight: bold;
            color: #e74c3c;
        }}
        .rate {{
            color: #f39c12;
        }}
        .badge {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            color: white;
        }}
        .posts {{
            color: #888;
        }}
        .summary {{
            text-align: center;
            margin-bottom: 20px;
            color: #888;
        }}
    </style>
</head>
<body>
    <h1>🚨 Spam Agent Leaderboard</h1>
    <p class="summary">Top {top_n} agents by spam volume</p>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Agent</th>
                <th>Spam Count</th>
                <th>Spam Rate</th>
                <th>Primary Type</th>
                <th>Posts Hit</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
</body>
</html>
"""
    
    output_path = output_dir / "agent_spam_leaderboard.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    return output_path


# =============================================================================
# Filter Funnel
# =============================================================================

def generate_filter_funnel(
    filter_stats: Dict[str, Any],
    lite_stats: Dict[str, Any],
    output_dir: Path,
) -> Optional[Path]:
    """
    Generate funnel chart showing content filtered at each tier.
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total = filter_stats.get("total_messages", 0)
    passed_filter = filter_stats.get("passed", 0)
    escalated = lite_stats.get("escalated", 0)
    benign = lite_stats.get("benign", 0)
    
    stages = ["Input Messages", "After Spam Filter", "Escalated to Full Eval", "Marked Benign"]
    values = [total, passed_filter, escalated, benign]
    
    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textposition="inside",
        textinfo="value+percent initial",
        textfont=dict(size=14, color="white"),
        marker=dict(color=[COLORS["primary"], COLORS["passed"], COLORS["escalated"], COLORS["benign"]]),
        connector=dict(line=dict(color="#333", width=2)),
    ))
    
    fig.update_layout(
        title=dict(text="Tiered Evaluation Funnel", x=0.5),
        paper_bgcolor=DARK_TEMPLATE["layout"]["paper_bgcolor"],
        plot_bgcolor=DARK_TEMPLATE["layout"]["plot_bgcolor"],
        font=DARK_TEMPLATE["layout"]["font"],
    )
    
    output_path = output_dir / "filter_funnel.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path


# =============================================================================
# Cascade Visualization
# =============================================================================

def generate_cascade_timeline(
    cascade_report: Dict[str, Any],
    output_dir: Path,
) -> Optional[Path]:
    """
    Generate timeline showing cascade attack patterns.
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cascades = cascade_report.get("cascades", [])
    
    if not cascades:
        return None
    
    fig = go.Figure()
    
    colors = {"clone_swarm": COLORS["spam"], "timing_cluster": COLORS["crypto"]}
    
    for i, cascade in enumerate(cascades):
        pattern_type = cascade.get("pattern_type", "unknown")
        evidence = cascade.get("evidence", {})
        
        # Get time window
        window_start = evidence.get("window_start", "")
        window_end = evidence.get("window_end", "")
        
        variant_count = cascade.get("variant_count", 0)
        spam_count = cascade.get("total_spam_messages", 0)
        confidence = cascade.get("confidence", 0)
        
        # Create annotation
        fig.add_trace(go.Bar(
            name=f"{pattern_type} ({variant_count} agents)",
            x=[spam_count],
            y=[pattern_type.replace("_", " ").title()],
            orientation="h",
            marker=dict(color=colors.get(pattern_type, "#7f8c8d")),
            text=f"{spam_count:,} spam, {confidence:.0%} confidence",
            textposition="inside",
            hovertemplate=(
                f"<b>{pattern_type.replace('_', ' ').title()}</b><br>"
                f"Agents: {variant_count}<br>"
                f"Spam: {spam_count:,}<br>"
                f"Confidence: {confidence:.0%}<br>"
                f"Window: {cascade.get('time_window_minutes', 0):.1f} min"
                "<extra></extra>"
            ),
        ))
    
    fig.update_layout(
        title=dict(text="Detected Cascade Attacks", x=0.5),
        paper_bgcolor=DARK_TEMPLATE["layout"]["paper_bgcolor"],
        plot_bgcolor=DARK_TEMPLATE["layout"]["plot_bgcolor"],
        font=DARK_TEMPLATE["layout"]["font"],
        xaxis=dict(title="Spam Messages", gridcolor="#333"),
        yaxis=dict(gridcolor="#333"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        barmode="group",
    )
    
    output_path = output_dir / "cascade_timeline.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path


def generate_clone_network(
    cascade_report: Dict[str, Any],
    output_dir: Path,
) -> Optional[Path]:
    """
    Generate network visualization of clone agents.
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cascades = cascade_report.get("cascades", [])
    clone_swarms = [c for c in cascades if c.get("pattern_type") == "clone_swarm"]
    
    if not clone_swarms:
        return None
    
    # Create network visualization
    fig = go.Figure()
    
    for swarm in clone_swarms:
        source = swarm.get("source_agent") or "Unknown Source"
        variants = swarm.get("variant_agents", [])[:20]  # Limit for readability
        
        # Create radial layout
        import math
        center_x, center_y = 0, 0
        radius = 1
        
        # Add center node
        fig.add_trace(go.Scatter(
            x=[center_x],
            y=[center_y],
            mode="markers+text",
            marker=dict(size=30, color=COLORS["spam"]),
            text=[source[:15]],
            textposition="bottom center",
            textfont=dict(size=10, color="white"),
            hovertemplate=f"<b>{source}</b><br>Source agent<extra></extra>",
            showlegend=False,
        ))
        
        # Add variant nodes in circle
        for i, variant in enumerate(variants):
            angle = 2 * math.pi * i / len(variants)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Add edge
            fig.add_trace(go.Scatter(
                x=[center_x, x],
                y=[center_y, y],
                mode="lines",
                line=dict(color="#333", width=1),
                showlegend=False,
                hoverinfo="skip",
            ))
            
            # Add node
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                marker=dict(size=15, color=COLORS["secondary"]),
                hovertemplate=f"<b>{variant}</b><br>Clone variant<extra></extra>",
                showlegend=False,
            ))
    
    fig.update_layout(
        title=dict(text="Clone Swarm Network", x=0.5),
        paper_bgcolor=DARK_TEMPLATE["layout"]["paper_bgcolor"],
        plot_bgcolor=DARK_TEMPLATE["layout"]["plot_bgcolor"],
        font=DARK_TEMPLATE["layout"]["font"],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(
            text=f"{len(clone_swarms)} clone swarm(s) detected",
            x=0.5, y=-0.1, xref="paper", yref="paper",
            showarrow=False, font=dict(color="#888")
        )],
    )
    
    output_path = output_dir / "clone_network.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path


# =============================================================================
# Lite Judge Charts
# =============================================================================

def generate_lite_score_histogram(
    lite_results: List[Dict[str, Any]],
    output_dir: Path,
) -> Optional[Path]:
    """
    Generate histogram of lite judge scores.
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scores = [r.get("lite_judge_result", {}).get("score", 0) for r in lite_results if r.get("lite_judge_result")]
    
    if not scores:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=11,
        marker=dict(
            color=scores,
            colorscale=[[0, COLORS["benign"]], [0.3, COLORS["passed"]], [1, COLORS["escalated"]]],
            line=dict(color="#1a1a2e", width=1),
        ),
        hovertemplate="Score %{x}: %{y} items<extra></extra>",
    ))
    
    # Add threshold line
    fig.add_vline(x=3.5, line_dash="dash", line_color=COLORS["escalated"], 
                  annotation_text="Escalation Threshold", annotation_position="top")
    
    fig.update_layout(
        title=dict(text="Lite Judge Score Distribution", x=0.5),
        paper_bgcolor=DARK_TEMPLATE["layout"]["paper_bgcolor"],
        plot_bgcolor=DARK_TEMPLATE["layout"]["plot_bgcolor"],
        font=DARK_TEMPLATE["layout"]["font"],
        xaxis=dict(title="Safety Score (0-10)", gridcolor="#333", dtick=1),
        yaxis=dict(title="Count", gridcolor="#333"),
        bargap=0.1,
    )
    
    output_path = output_dir / "lite_score_histogram.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path


def generate_lite_score_boxplot(
    lite_results: List[Dict[str, Any]],
    output_dir: Path,
) -> Optional[Path]:
    """
    Generate box plot of lite judge scores.
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    escalated_scores = []
    benign_scores = []
    
    for r in lite_results:
        result = r.get("lite_judge_result", {})
        score = result.get("score", 0)
        if result.get("escalate"):
            escalated_scores.append(score)
        else:
            benign_scores.append(score)
    
    if not escalated_scores and not benign_scores:
        return None
    
    fig = go.Figure()
    
    if benign_scores:
        fig.add_trace(go.Box(
            y=benign_scores,
            name="Benign",
            marker_color=COLORS["benign"],
            boxpoints="outliers",
        ))
    
    if escalated_scores:
        fig.add_trace(go.Box(
            y=escalated_scores,
            name="Escalated",
            marker_color=COLORS["escalated"],
            boxpoints="outliers",
        ))
    
    fig.update_layout(
        title=dict(text="Lite Judge Score Distribution by Outcome", x=0.5),
        paper_bgcolor=DARK_TEMPLATE["layout"]["paper_bgcolor"],
        plot_bgcolor=DARK_TEMPLATE["layout"]["plot_bgcolor"],
        font=DARK_TEMPLATE["layout"]["font"],
        yaxis=dict(title="Safety Score", gridcolor="#333", range=[0, 10]),
        showlegend=True,
    )
    
    output_path = output_dir / "lite_score_boxplot.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path


def generate_escalation_pie(
    lite_stats: Dict[str, Any],
    output_dir: Path,
) -> Optional[Path]:
    """
    Generate pie chart of escalated vs benign.
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    escalated = lite_stats.get("escalated", 0)
    benign = lite_stats.get("benign", 0)
    
    if escalated == 0 and benign == 0:
        return None
    
    fig = go.Figure(data=[go.Pie(
        labels=["Escalated", "Benign"],
        values=[escalated, benign],
        hole=0.4,
        marker=dict(colors=[COLORS["escalated"], COLORS["benign"]]),
        textinfo="label+percent",
        textfont=dict(size=14, color="white"),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<extra></extra>",
    )])
    
    fig.update_layout(
        title=dict(text="Lite Judge Triage Results", x=0.5),
        paper_bgcolor=DARK_TEMPLATE["layout"]["paper_bgcolor"],
        plot_bgcolor=DARK_TEMPLATE["layout"]["plot_bgcolor"],
        font=DARK_TEMPLATE["layout"]["font"],
        annotations=[dict(
            text=f"{escalated + benign}<br>Total",
            x=0.5, y=0.5, font_size=16, showarrow=False, font_color="#4ecdc4"
        )],
    )
    
    output_path = output_dir / "escalation_pie.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path


# =============================================================================
# Summary Dashboard
# =============================================================================

def generate_tiered_summary_html(
    filter_stats: Dict[str, Any],
    cascade_report: Dict[str, Any],
    lite_stats: Dict[str, Any],
    output_dir: Path,
    extra_stats: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Generate summary dashboard HTML page.
    
    Args:
        filter_stats: Filter statistics
        cascade_report: Cascade detection report
        lite_stats: Lite judge statistics
        output_dir: Output directory
        extra_stats: Additional stats (total_posts, total_comments, total_agents, etc.)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extra_stats = extra_stats or {}
    
    # Extract stats
    total = filter_stats.get("total_messages", 0)
    spam_filtered = filter_stats.get("spam_agent_filtered", 0) + filter_stats.get("spam_content_filtered", 0)
    passed = filter_stats.get("passed", 0)
    filter_rate = filter_stats.get("filter_rate", 0)
    
    cascade_count = cascade_report.get("cascade_count", 0)
    spammer_count = len(filter_stats.get("top_spammers", []))
    
    escalated = lite_stats.get("escalated", 0)
    benign = lite_stats.get("benign", 0)
    avg_score = lite_stats.get("avg_score", 0)
    
    # Additional stats from extra_stats
    total_posts = extra_stats.get("total_posts", 0)
    total_comments = extra_stats.get("total_comments", 0)
    total_agents = extra_stats.get("total_agents", 0)
    posts_evaluated = extra_stats.get("posts_evaluated", 0)
    comments_evaluated = extra_stats.get("comments_evaluated", 0)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Tiered Evaluation Summary</title>
    <style>
        body {{
            font-family: Inter, system-ui, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 40px;
            margin: 0;
        }}
        h1 {{
            color: #4ecdc4;
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 40px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            max-width: 1000px;
            margin: 0 auto 40px;
        }}
        .stat-card {{
            background: #16213e;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 36px;
            font-weight: bold;
            color: #4ecdc4;
        }}
        .stat-value.danger {{ color: #e74c3c; }}
        .stat-value.warning {{ color: #f39c12; }}
        .stat-value.success {{ color: #2ecc71; }}
        .stat-label {{
            color: #888;
            margin-top: 8px;
            font-size: 14px;
        }}
        .section {{
            max-width: 1000px;
            margin: 0 auto 40px;
        }}
        h2 {{
            color: #ff6b6b;
            border-bottom: 2px solid #ff6b6b;
            padding-bottom: 10px;
        }}
        .tier-flow {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
        }}
        .tier {{
            text-align: center;
            flex: 1;
        }}
        .tier-name {{
            font-size: 12px;
            color: #888;
            margin-bottom: 8px;
        }}
        .tier-count {{
            font-size: 24px;
            font-weight: bold;
        }}
        .arrow {{
            color: #333;
            font-size: 24px;
        }}
        .cascade-alert {{
            background: linear-gradient(135deg, #e74c3c22, #1a1a2e);
            border-left: 4px solid #e74c3c;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .cascade-alert h3 {{
            color: #e74c3c;
            margin: 0 0 10px;
        }}
    </style>
</head>
<body>
    <h1>⚡ Tiered Evaluation Report</h1>
    <p class="subtitle">Content filtering and safety triage analysis</p>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{total_posts:,}</div>
            <div class="stat-label">Posts</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{total_comments:,}</div>
            <div class="stat-label">Comments</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{total_agents:,}</div>
            <div class="stat-label">Agents</div>
        </div>
        <div class="stat-card">
            <div class="stat-value success">{avg_score:.1f}</div>
            <div class="stat-label">Avg Safety Score</div>
        </div>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{total:,}</div>
            <div class="stat-label">Total Messages</div>
        </div>
        <div class="stat-card">
            <div class="stat-value danger">{filter_rate*100:.0f}%</div>
            <div class="stat-label">Filtered Out</div>
        </div>
        <div class="stat-card">
            <div class="stat-value warning">{spammer_count}</div>
            <div class="stat-label">Spammers Found</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{posts_evaluated:,}</div>
            <div class="stat-label">Posts Evaluated</div>
        </div>
    </div>
    
    <div class="section">
        <h2>Processing Flow</h2>
        <div class="tier-flow">
            <div class="tier">
                <div class="tier-name">INPUT</div>
                <div class="tier-count" style="color:#4ecdc4">{total:,}</div>
            </div>
            <div class="arrow">→</div>
            <div class="tier">
                <div class="tier-name">TIER 0: FILTER</div>
                <div class="tier-count" style="color:#e74c3c">-{total - passed:,}</div>
            </div>
            <div class="arrow">→</div>
            <div class="tier">
                <div class="tier-name">TIER 1: LITE JUDGE</div>
                <div class="tier-count" style="color:#2ecc71">{passed:,}</div>
            </div>
            <div class="arrow">→</div>
            <div class="tier">
                <div class="tier-name">ESCALATED</div>
                <div class="tier-count" style="color:#f39c12">{escalated}</div>
            </div>
        </div>
    </div>
    
    {"".join([f'''
    <div class="section">
        <div class="cascade-alert">
            <h3>🚨 Cascade Attack Detected</h3>
            <p><strong>Pattern:</strong> {c.get("pattern_type", "").replace("_", " ").title()}</p>
            <p><strong>Agents:</strong> {c.get("variant_count", 0)} variants</p>
            <p><strong>Spam:</strong> {c.get("total_spam_messages", 0):,} messages</p>
            <p><strong>Confidence:</strong> {c.get("confidence", 0)*100:.0f}%</p>
        </div>
    </div>
    ''' for c in cascade_report.get("cascades", [])])}
    
    <div class="section">
        <h2>Quick Links</h2>
        <ul>
            <li><a href="spam_breakdown.html" style="color:#4ecdc4">Spam Breakdown Chart</a></li>
            <li><a href="agent_spam_leaderboard.html" style="color:#4ecdc4">Agent Spam Leaderboard</a></li>
            <li><a href="filter_funnel.html" style="color:#4ecdc4">Filter Funnel</a></li>
            <li><a href="lite_score_histogram.html" style="color:#4ecdc4">Lite Judge Scores</a></li>
            <li><a href="escalation_pie.html" style="color:#4ecdc4">Escalation Results</a></li>
        </ul>
    </div>
</body>
</html>
"""
    
    output_path = output_dir / "summary.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    return output_path


# =============================================================================
# Main Report Generator
# =============================================================================

def generate_all_tiered_reports(
    run_dir: Path,
    extra_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """
    Generate all tiered evaluation reports from a run directory.
    
    Args:
        run_dir: Path to tiered eval run directory
        extra_stats: Additional stats (total_posts, total_comments, total_agents, etc.)
        
    Returns:
        Dict mapping report name to file path
    """
    run_dir = Path(run_dir)
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    reports = {}
    
    # Load data
    filter_stats = {}
    cascade_report = {}
    lite_stats = {}
    lite_results = []
    
    filter_stats_path = run_dir / "filter" / "filter_stats.json"
    if filter_stats_path.exists():
        with open(filter_stats_path) as f:
            filter_stats = json.load(f)
    
    cascade_path = run_dir / "filter" / "cascade_report.json"
    if cascade_path.exists():
        with open(cascade_path) as f:
            cascade_report = json.load(f)
    
    lite_stats_path = run_dir / "lite_judge" / "lite_stats.json"
    if lite_stats_path.exists():
        with open(lite_stats_path) as f:
            lite_stats = json.load(f)
    
    # Load lite judge results for score charts
    escalated_path = run_dir / "lite_judge" / "escalated.jsonl"
    benign_path = run_dir / "lite_judge" / "benign.jsonl"
    
    for path in [escalated_path, benign_path]:
        if path.exists():
            with open(path) as f:
                for line in f:
                    if line.strip():
                        lite_results.append(json.loads(line))
    
    spammer_profiles = filter_stats.get("top_spammers", [])
    
    print(f"\n📊 Generating tiered evaluation reports...")
    
    # Generate each report
    path = generate_spam_breakdown_chart(filter_stats, reports_dir)
    if path:
        reports["spam_breakdown"] = path
        print(f"   ✅ {path.name}")
    
    path = generate_spam_types_bar(spammer_profiles, reports_dir)
    if path:
        reports["spam_types_bar"] = path
        print(f"   ✅ {path.name}")
    
    path = generate_agent_spam_leaderboard(spammer_profiles, reports_dir)
    if path:
        reports["agent_leaderboard"] = path
        print(f"   ✅ {path.name}")
    
    path = generate_filter_funnel(filter_stats, lite_stats, reports_dir)
    if path:
        reports["filter_funnel"] = path
        print(f"   ✅ {path.name}")
    
    path = generate_cascade_timeline(cascade_report, reports_dir)
    if path:
        reports["cascade_timeline"] = path
        print(f"   ✅ {path.name}")
    
    path = generate_clone_network(cascade_report, reports_dir)
    if path:
        reports["clone_network"] = path
        print(f"   ✅ {path.name}")
    
    path = generate_lite_score_histogram(lite_results, reports_dir)
    if path:
        reports["lite_histogram"] = path
        print(f"   ✅ {path.name}")
    
    path = generate_lite_score_boxplot(lite_results, reports_dir)
    if path:
        reports["lite_boxplot"] = path
        print(f"   ✅ {path.name}")
    
    path = generate_escalation_pie(lite_stats, reports_dir)
    if path:
        reports["escalation_pie"] = path
        print(f"   ✅ {path.name}")
    
    path = generate_tiered_summary_html(filter_stats, cascade_report, lite_stats, reports_dir, extra_stats)
    if path:
        reports["summary"] = path
        print(f"   ✅ {path.name}")
    
    print(f"   📁 Reports saved to {reports_dir}")
    
    return reports

