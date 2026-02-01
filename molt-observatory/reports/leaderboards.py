# Leaderboard Report Generator
# Creates filterable leaderboard pages by evaluation dimension.

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .generator import _html_template, _score_badge, OUTPUT_DIR


# Dimension display names
DIMENSION_LABELS = {
    "harm_enablement": "Harm Enablement",
    "deception_or_evasion": "Deception/Evasion",
    "self_preservation_power_seeking": "Power Seeking",
    "delusional_sycophancy": "Sycophancy",
}


def _load_evals_from_run(run_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """Load post and comment evaluations from a run directory."""
    post_evals = []
    comment_evals = []
    
    post_evals_path = run_dir / "gold" / "evals.jsonl"
    comment_evals_path = run_dir / "gold" / "comment_evals.jsonl"
    
    if post_evals_path.exists():
        with open(post_evals_path, "r") as f:
            for line in f:
                if line.strip():
                    post_evals.append(json.loads(line))
    
    if comment_evals_path.exists():
        with open(comment_evals_path, "r") as f:
            for line in f:
                if line.strip():
                    comment_evals.append(json.loads(line))
    
    return post_evals, comment_evals


def _load_agent_scores(run_dir: Path) -> List[Dict]:
    """Load agent score records from a run directory."""
    agent_scores_path = run_dir / "gold" / "agent_scores.jsonl"
    
    if not agent_scores_path.exists():
        return []
    
    scores = []
    with open(agent_scores_path, "r") as f:
        for line in f:
            if line.strip():
                scores.append(json.loads(line))
    
    return scores


def build_agent_leaderboard(
    agent_scores: List[Dict[str, Any]],
    dimension: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Build a leaderboard for agents ranked by a specific dimension.
    
    Args:
        agent_scores: List of agent score records
        dimension: Dimension to rank by
        limit: Maximum number of entries
    
    Returns:
        List of leaderboard entries sorted by score (highest first)
    """
    entries = []
    
    for agent in agent_scores:
        dim_scores = agent.get("dimension_scores", {})
        dim_data = dim_scores.get(dimension, {})
        
        if not dim_data:
            continue
        
        entries.append({
            "rank": 0,  # Will be set after sorting
            "agent_handle": agent.get("agent_handle", "Unknown"),
            "agent_id": agent.get("agent_id"),
            "mean_score": dim_data.get("mean_score", 0),
            "max_score": dim_data.get("max_score", 0),
            "high_count": dim_data.get("high_score_count", 0),
            "total_items": dim_data.get("total_items", 0),
            "posts_evaluated": agent.get("posts_evaluated", 0),
            "comments_evaluated": agent.get("comments_evaluated", 0),
        })
    
    # Sort by mean score descending
    entries.sort(key=lambda x: (-x["mean_score"], -x["max_score"]))
    
    # Assign ranks
    for i, entry in enumerate(entries[:limit]):
        entry["rank"] = i + 1
    
    return entries[:limit]


def build_post_leaderboard(
    post_evals: List[Dict[str, Any]],
    dimension: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Build a leaderboard for posts ranked by a specific dimension.
    """
    entries = []
    
    for eval_item in post_evals:
        scores = eval_item.get("scores", {})
        dim_data = scores.get(dimension, {})
        
        score = dim_data.get("score", 0)
        if score == 0:
            continue
        
        entries.append({
            "rank": 0,
            "post_id": eval_item.get("post_id"),
            "permalink": eval_item.get("permalink"),
            "score": score,
            "confidence": dim_data.get("confidence", 0),
            "explanation": dim_data.get("explanation", "")[:100],
            "notes": eval_item.get("notes", ""),
        })
    
    # Sort by score descending
    entries.sort(key=lambda x: -x["score"])
    
    # Assign ranks
    for i, entry in enumerate(entries[:limit]):
        entry["rank"] = i + 1
    
    return entries[:limit]


def build_comment_leaderboard(
    comment_evals: List[Dict[str, Any]],
    dimension: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Build a leaderboard for comments ranked by a specific dimension.
    """
    entries = []
    
    for eval_item in comment_evals:
        scores = eval_item.get("scores", {})
        dim_data = scores.get(dimension, {})
        
        score = dim_data.get("score", 0)
        if score == 0:
            continue
        
        entries.append({
            "rank": 0,
            "comment_id": eval_item.get("comment_id"),
            "post_id": eval_item.get("post_id"),
            "permalink": eval_item.get("permalink"),
            "author": eval_item.get("author", "Unknown"),
            "score": score,
            "confidence": dim_data.get("confidence", 0),
        })
    
    # Sort by score descending
    entries.sort(key=lambda x: -x["score"])
    
    # Assign ranks
    for i, entry in enumerate(entries[:limit]):
        entry["rank"] = i + 1
    
    return entries[:limit]


def _generate_leaderboard_table(
    entries: List[Dict[str, Any]],
    columns: List[Tuple[str, str]],
    entity_type: str = "agent",
) -> str:
    """
    Generate HTML table for a leaderboard.
    
    Args:
        entries: List of leaderboard entries
        columns: List of (key, display_name) tuples for columns
        entity_type: Type of entity (for styling)
    
    Returns:
        HTML string
    """
    if not entries:
        return '<p class="no-data">No entries found with non-zero scores.</p>'
    
    # Build header
    header_cells = "".join(f"<th>{name}</th>" for _, name in columns)
    
    # Build rows
    rows = []
    for entry in entries:
        cells = []
        for key, _ in columns:
            value = entry.get(key, "")
            
            # Special formatting
            if key == "score" or key == "mean_score" or key == "max_score":
                cells.append(f"<td>{_score_badge(value)}</td>")
            elif key == "permalink" and value:
                cells.append(f'<td><a href="{value}" target="_blank">View</a></td>')
            elif key == "confidence":
                cells.append(f"<td>{value:.2f}</td>")
            else:
                cells.append(f"<td>{value}</td>")
        
        rows.append(f"<tr>{''.join(cells)}</tr>")
    
    return f'''
        <table>
            <thead>
                <tr>{header_cells}</tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    '''


def generate_dimension_distribution_chart(
    agent_scores: List[Dict[str, Any]],
    dimension: str,
) -> str:
    """Generate a histogram of score distribution for a dimension."""
    if not PLOTLY_AVAILABLE or not agent_scores:
        return ""
    
    scores = []
    for agent in agent_scores:
        dim_scores = agent.get("dimension_scores", {})
        dim_data = dim_scores.get(dimension, {})
        mean = dim_data.get("mean_score", 0)
        if mean > 0:
            scores.append(mean)
    
    if not scores:
        return ""
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=20,
        marker_color='#4ecdc4',
        opacity=0.8,
    ))
    
    fig.update_layout(
        title=f"Score Distribution: {DIMENSION_LABELS.get(dimension, dimension)}",
        xaxis_title="Mean Score",
        yaxis_title="Number of Agents",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,1)',
        height=300,
    )
    
    return f'<div class="chart-container">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'


def generate_agent_leaderboard(
    agent_scores: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Agent Safety Leaderboard",
) -> None:
    """
    Generate an agent leaderboard HTML file.
    
    Args:
        agent_scores: List of agent score records
        output_path: Path to write the HTML file
        title: Report title
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    content_parts = [
        f'''
        <header>
            <h1>🏆 {title}</h1>
            <p class="subtitle">Agents Ranked by Safety Scores</p>
        </header>
        ''',
    ]
    
    # Summary
    content_parts.append(f'''
        <section>
            <h2>Overview</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{len(agent_scores)}</div>
                    <div class="stat-label">Agents Scored</div>
                </div>
            </div>
        </section>
    ''')
    
    # Overall leaderboard
    sorted_agents = sorted(
        agent_scores, 
        key=lambda x: -x.get("overall_mean_score", 0)
    )
    
    rows = []
    for i, agent in enumerate(sorted_agents[:25]):
        agent_handle = agent.get("agent_handle", "Unknown")
        agent_link = f'<a href="https://www.moltbook.com/agent/{agent_handle}" target="_blank" class="agent-link">@{agent_handle}</a>'
        
        posts_count = agent.get("posts_evaluated", 0)
        comments_count = agent.get("comments_evaluated", 0)
        
        rows.append(f'''
            <tr>
                <td>{i + 1}</td>
                <td>{agent_link}</td>
                <td>{_score_badge(agent.get("overall_mean_score", 0))}</td>
                <td>{posts_count}</td>
                <td>{comments_count}</td>
                <td>{"⚠️" if agent.get("has_high_harm_enablement") else "✓"}</td>
            </tr>
        ''')
    
    content_parts.append(f'''
        <section>
            <h2>Overall Rankings</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Agent</th>
                        <th>Mean Score</th>
                        <th>Posts</th>
                        <th>Comments</th>
                        <th>Risk</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </section>
    ''')
    
    html = _html_template(title=title, content="\n".join(content_parts))
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def generate_dimension_leaderboard(
    dimension_name: str,
    post_scores: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Generate a dimension-specific leaderboard HTML file.
    
    Args:
        dimension_name: Name of the dimension
        post_scores: List of post score records for this dimension
        output_path: Path to write the HTML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dim_label = DIMENSION_LABELS.get(dimension_name, dimension_name.replace("_", " ").title())
    
    content_parts = [
        f'''
        <header>
            <h1>📊 {dim_label} Leaderboard</h1>
            <p class="subtitle">Posts Ranked by {dim_label} Score</p>
        </header>
        ''',
    ]
    
    # Sort by score
    sorted_posts = sorted(post_scores, key=lambda x: -x.get("score", 0))
    
    rows = []
    for i, post in enumerate(sorted_posts[:25]):
        title = post.get("title", "Untitled") or "Untitled"
        permalink = post.get("permalink", "")
        author = post.get("author", "Unknown") or "Unknown"
        truncated_title = title[:40] + "..." if len(title) > 40 else title
        
        # Make title a clickable hyperlink if permalink exists
        if permalink:
            title_cell = f'<a href="{permalink}" target="_blank" title="{title}" class="post-link">{truncated_title}</a>'
        else:
            title_cell = truncated_title
        
        # Make author a clickable hyperlink to their profile
        author_cell = f'<a href="https://www.moltbook.com/agent/{author}" target="_blank" class="author-link">@{author}</a>'
        
        # Show comment count if available
        comment_count = post.get("comment_count", 0)
        comment_cell = f'{comment_count} 💬' if comment_count else "-"
        
        rows.append(f'''
            <tr>
                <td>{i + 1}</td>
                <td>{title_cell}</td>
                <td>{_score_badge(post.get("score", 0))}</td>
                <td>{author_cell}</td>
                <td>{comment_cell}</td>
            </tr>
        ''')
    
    content_parts.append(f'''
        <section>
            <h2>Rankings</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Title</th>
                        <th>Score</th>
                        <th>Author</th>
                        <th>Comments</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </section>
    ''')
    
    # Add bright link styling for dark background
    extra_styles = '''
        .post-link {
            color: #00e5ff;
            text-decoration: none;
            transition: color 0.2s ease;
        }
        
        .post-link:hover {
            color: #4ecdc4;
            text-decoration: underline;
        }
        
        table a {
            color: #00e5ff;
            text-decoration: none;
        }
        
        table a:hover {
            color: #4ecdc4;
            text-decoration: underline;
        }
    '''
    
    html = _html_template(title=f"{dim_label} Leaderboard", content="\n".join(content_parts), styles=extra_styles)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def generate_leaderboard_report(
    runs_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    limit: int = 25,
) -> str:
    """
    Generate a complete leaderboard report HTML file.
    
    Args:
        runs_dir: Path to runs directory
        output_dir: Output directory for the report
        limit: Maximum entries per leaderboard
    
    Returns:
        Path to generated report file
    """
    runs_dir = runs_dir or (Path(__file__).parent.parent / "runs")
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find latest run
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)
    
    if not run_dirs:
        content = '''
            <header>
                <h1>🏆 Leaderboards</h1>
                <p class="subtitle">Top Agents, Posts, and Comments by Safety Dimension</p>
            </header>
            <section>
                <h2>No Data Available</h2>
                <p>No run data found. Execute the pipeline to generate evaluations.</p>
            </section>
        '''
        html = _html_template(title="Leaderboards", content=content)
        output_path = output_dir / "leaderboard.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return str(output_path)
    
    latest_run = run_dirs[0]
    
    # Load data
    post_evals, comment_evals = _load_evals_from_run(latest_run)
    agent_scores = _load_agent_scores(latest_run)
    
    # Build content
    content_parts = [
        f'''
        <header>
            <h1>🏆 Leaderboards</h1>
            <p class="subtitle">Top Agents, Posts, and Comments by Safety Dimension</p>
            <p class="timestamp">Data from: {latest_run.name}</p>
        </header>
        ''',
    ]
    
    # Summary stats
    content_parts.append(f'''
        <section>
            <h2>Overview</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{len(agent_scores)}</div>
                    <div class="stat-label">Agents Scored</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(post_evals)}</div>
                    <div class="stat-label">Posts Evaluated</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(comment_evals)}</div>
                    <div class="stat-label">Comments Evaluated</div>
                </div>
            </div>
        </section>
    ''')
    
    # Generate leaderboards for each dimension
    dimensions = [
        "harm_enablement",
        "deception_or_evasion", 
        "self_preservation_power_seeking",
        "delusional_sycophancy",
    ]
    
    for dim in dimensions:
        dim_label = DIMENSION_LABELS.get(dim, dim)
        
        # Agent leaderboard
        agent_lb = build_agent_leaderboard(agent_scores, dim, limit)
        agent_table = _generate_leaderboard_table(
            agent_lb,
            [
                ("rank", "#"),
                ("agent_handle", "Agent"),
                ("mean_score", "Mean"),
                ("max_score", "Max"),
                ("high_count", "High (≥7)"),
                ("total_items", "Items"),
            ],
        )
        
        # Post leaderboard
        post_lb = build_post_leaderboard(post_evals, dim, limit)
        post_table = _generate_leaderboard_table(
            post_lb,
            [
                ("rank", "#"),
                ("post_id", "Post ID"),
                ("score", "Score"),
                ("confidence", "Confidence"),
                ("permalink", "Link"),
            ],
        )
        
        # Comment leaderboard
        comment_lb = build_comment_leaderboard(comment_evals, dim, limit)
        comment_table = _generate_leaderboard_table(
            comment_lb,
            [
                ("rank", "#"),
                ("author", "Author"),
                ("score", "Score"),
                ("confidence", "Confidence"),
                ("permalink", "Link"),
            ],
        )
        
        # Distribution chart
        dist_chart = generate_dimension_distribution_chart(agent_scores, dim)
        
        content_parts.append(f'''
            <section>
                <h2>{dim_label}</h2>
                
                {dist_chart}
                
                <h3>Top Agents</h3>
                {agent_table}
                
                <h3>Top Posts</h3>
                {post_table}
                
                <h3>Top Comments</h3>
                {comment_table}
            </section>
        ''')
    
    # Add CSS for tabs/filtering (static for now)
    extra_styles = '''
        section h3 {
            color: var(--text-primary);
            margin-top: 2rem;
            margin-bottom: 1rem;
            font-size: 1.2rem;
        }
        
        .no-data {
            color: var(--text-secondary);
            font-style: italic;
            padding: 1rem;
        }
        
        table a {
            color: var(--accent-secondary);
            text-decoration: none;
        }
        
        table a:hover {
            text-decoration: underline;
        }
    '''
    
    # Generate HTML
    html = _html_template(
        title="Leaderboards",
        content="\n".join(content_parts),
        styles=extra_styles,
    )
    
    # Write file
    output_path = output_dir / "leaderboard.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    return str(output_path)

