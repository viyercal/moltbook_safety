"""
Thread Analysis Charts

Generates comprehensive Plotly visualizations for thread depth and engagement analysis:
- Histograms
- Box-and-whisker plots
- Cumulative Distribution Functions (CDF)
- Normal distribution overlays
- Combined scatter plot
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Chart styling constants
CHART_TEMPLATE = "plotly_dark"
DEPTH_COLOR = "#4ecdc4"  # Teal
ENGAGEMENT_COLOR = "#ff6b6b"  # Coral
THRESHOLD_COLOR = "#ffd93d"  # Yellow
GRID_COLOR = "rgba(255,255,255,0.1)"


def _save_chart(fig: "go.Figure", output_path: Path, export_png: bool = True) -> Dict[str, str]:
    """Save chart as HTML and optionally PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save HTML
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    
    result = {"html": str(output_path)}
    
    # Save PNG if requested
    if export_png:
        png_path = output_path.with_suffix(".png")
        try:
            fig.write_image(str(png_path), width=1200, height=600, scale=2)
            result["png"] = str(png_path)
        except Exception:
            pass  # Kaleido may not be installed
    
    return result


def _save_chart_with_click_handler(fig: "go.Figure", output_path: Path, export_png: bool = True) -> Dict[str, str]:
    """Save chart as HTML with JavaScript click handler for opening URLs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save HTML first
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    
    # Read the HTML and inject click handler JavaScript
    with open(output_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    # JavaScript to handle clicks on data points
    click_handler_js = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    var plotDiv = document.querySelector('.plotly-graph-div');
    if (plotDiv) {
        plotDiv.on('plotly_click', function(data) {
            var point = data.points[0];
            if (point && point.customdata) {
                window.open(point.customdata, '_blank');
            }
        });
        // Change cursor to pointer on hover
        plotDiv.on('plotly_hover', function(data) {
            if (data.points[0] && data.points[0].customdata) {
                plotDiv.style.cursor = 'pointer';
            }
        });
        plotDiv.on('plotly_unhover', function(data) {
            plotDiv.style.cursor = 'default';
        });
    }
});
</script>
"""
    
    # Insert before closing body tag
    html_content = html_content.replace("</body>", click_handler_js + "</body>")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    result = {"html": str(output_path)}
    
    # Save PNG if requested
    if export_png:
        png_path = output_path.with_suffix(".png")
        try:
            fig.write_image(str(png_path), width=1200, height=600, scale=2)
            result["png"] = str(png_path)
        except Exception:
            pass
    
    return result


# =============================================================================
# DEPTH CHARTS
# =============================================================================

def generate_depth_histogram(
    depths: List[int],
    stats_dict: Dict[str, Any],
    output_path: Path,
    export_png: bool = True,
) -> Dict[str, str]:
    """
    Generate histogram of thread depths.
    
    Shows distribution of depths (1, 2, 3, 4, ...) with threshold line.
    """
    if not PLOTLY_AVAILABLE:
        return {}
    
    fig = go.Figure()
    
    # Count occurrences of each depth
    from collections import Counter
    depth_counts = Counter(depths)
    x_vals = sorted(depth_counts.keys())
    y_vals = [depth_counts[x] for x in x_vals]
    
    # Bar chart for discrete depths
    fig.add_trace(go.Bar(
        x=x_vals,
        y=y_vals,
        marker_color=DEPTH_COLOR,
        name="Posts",
        text=[f"{v:,}<br>({v/len(depths)*100:.1f}%)" for v in y_vals],
        textposition="outside",
        hovertemplate="Depth %{x}<br>Posts: %{y:,}<extra></extra>",
    ))
    
    # Add threshold line
    threshold = stats_dict.get("threshold_2std", 0)
    if threshold > 0:
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color=THRESHOLD_COLOR,
            annotation_text=f"Mean+2σ = {threshold:.2f}",
            annotation_position="top right",
        )
    
    # Add mean line
    mean = stats_dict.get("mean", 0)
    fig.add_vline(
        x=mean,
        line_dash="dot",
        line_color="white",
        annotation_text=f"Mean = {mean:.2f}",
        annotation_position="top left",
    )
    
    fig.update_layout(
        title=dict(
            text="Thread Depth Distribution",
            font=dict(size=24),
        ),
        xaxis_title="Thread Depth (1 = post only, 2 = +comments, 3+ = +replies)",
        yaxis_title="Number of Posts",
        template=CHART_TEMPLATE,
        showlegend=False,
        xaxis=dict(tickmode="linear", dtick=1),
        annotations=[
            dict(
                text=f"n={len(depths):,} | μ={mean:.2f} | σ={stats_dict.get('std', 0):.2f}",
                xref="paper", yref="paper",
                x=0.99, y=0.99,
                showarrow=False,
                font=dict(size=12, color="gray"),
                xanchor="right",
            )
        ],
    )
    
    return _save_chart(fig, output_path, export_png)


def generate_depth_boxplot(
    depths: List[int],
    stats_dict: Dict[str, Any],
    output_path: Path,
    export_png: bool = True,
) -> Dict[str, str]:
    """
    Generate box-and-whisker plot for thread depths.
    """
    if not PLOTLY_AVAILABLE:
        return {}
    
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=depths,
        name="Thread Depth",
        marker_color=DEPTH_COLOR,
        boxmean=True,  # Show mean as dashed line
        boxpoints="outliers",
        jitter=0.3,
        pointpos=-1.5,
        hovertemplate="Depth: %{y}<extra></extra>",
    ))
    
    # Add horizontal line for threshold
    threshold = stats_dict.get("threshold_2std", 0)
    if threshold > 0:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=THRESHOLD_COLOR,
            annotation_text=f"Threshold (μ+2σ) = {threshold:.2f}",
            annotation_position="right",
        )
    
    fig.update_layout(
        title=dict(
            text="Thread Depth Box Plot",
            font=dict(size=24),
        ),
        yaxis_title="Thread Depth",
        template=CHART_TEMPLATE,
        showlegend=False,
        annotations=[
            dict(
                text=f"Median: {stats_dict.get('median', 0):.1f} | "
                     f"Q1: {stats_dict.get('percentiles', {}).get(25, 0):.1f} | "
                     f"Q3: {stats_dict.get('percentiles', {}).get(75, 0):.1f}",
                xref="paper", yref="paper",
                x=0.99, y=0.01,
                showarrow=False,
                font=dict(size=12, color="gray"),
                xanchor="right", yanchor="bottom",
            )
        ],
    )
    
    return _save_chart(fig, output_path, export_png)


def generate_depth_cdf(
    depths: List[int],
    stats_dict: Dict[str, Any],
    output_path: Path,
    export_png: bool = True,
) -> Dict[str, str]:
    """
    Generate cumulative distribution function for thread depths.
    """
    if not PLOTLY_AVAILABLE:
        return {}
    
    # Sort and compute CDF
    sorted_depths = np.sort(depths)
    cdf = np.arange(1, len(sorted_depths) + 1) / len(sorted_depths)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sorted_depths,
        y=cdf,
        mode="lines",
        line=dict(color=DEPTH_COLOR, width=3),
        name="CDF",
        fill="tozeroy",
        fillcolor=f"rgba(78, 205, 196, 0.2)",
        hovertemplate="Depth ≤ %{x}<br>Percentile: %{y:.1%}<extra></extra>",
    ))
    
    # Add threshold marker
    threshold = stats_dict.get("threshold_2std", 0)
    if threshold > 0:
        # Find CDF value at threshold
        cdf_at_threshold = np.mean(sorted_depths <= threshold)
        fig.add_trace(go.Scatter(
            x=[threshold],
            y=[cdf_at_threshold],
            mode="markers",
            marker=dict(color=THRESHOLD_COLOR, size=15, symbol="diamond"),
            name=f"Threshold ({cdf_at_threshold:.1%})",
            hovertemplate=f"Threshold: {threshold:.2f}<br>CDF: {cdf_at_threshold:.1%}<extra></extra>",
        ))
    
    # Add percentile markers
    percentiles = stats_dict.get("percentiles", {})
    for p in [50, 90, 95]:
        if p in percentiles:
            val = percentiles[p]
            fig.add_vline(
                x=val,
                line_dash="dot",
                line_color="rgba(255,255,255,0.5)",
                annotation_text=f"P{p}={val:.1f}",
            )
    
    fig.update_layout(
        title=dict(
            text="Thread Depth CDF (Cumulative Distribution)",
            font=dict(size=24),
        ),
        xaxis_title="Thread Depth",
        yaxis_title="Cumulative Probability",
        template=CHART_TEMPLATE,
        yaxis=dict(tickformat=".0%"),
    )
    
    return _save_chart(fig, output_path, export_png)


def generate_depth_normal(
    depths: List[int],
    stats_dict: Dict[str, Any],
    output_path: Path,
    export_png: bool = True,
) -> Dict[str, str]:
    """
    Generate histogram with normal distribution overlay.
    """
    if not PLOTLY_AVAILABLE:
        return {}
    
    mean = stats_dict.get("mean", np.mean(depths))
    std = stats_dict.get("std", np.std(depths))
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Histogram
    from collections import Counter
    depth_counts = Counter(depths)
    x_vals = sorted(depth_counts.keys())
    y_vals = [depth_counts[x] for x in x_vals]
    
    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=y_vals,
            marker_color=DEPTH_COLOR,
            name="Observed",
            opacity=0.7,
        ),
        secondary_y=False,
    )
    
    # Normal distribution overlay (requires scipy)
    if SCIPY_AVAILABLE and std > 0:
        x_range = np.linspace(min(depths) - 0.5, max(depths) + 0.5, 100)
        normal_pdf = stats.norm.pdf(x_range, mean, std)
        # Scale to match histogram
        scale = max(y_vals) / max(normal_pdf) if max(normal_pdf) > 0 else 1
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_pdf * scale,
                mode="lines",
                line=dict(color="white", width=3, dash="dash"),
                name=f"Normal (μ={mean:.2f}, σ={std:.2f})",
            ),
            secondary_y=False,
        )
    
    # Threshold
    threshold = stats_dict.get("threshold_2std", 0)
    if threshold > 0:
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color=THRESHOLD_COLOR,
            annotation_text=f"μ+2σ = {threshold:.2f}",
        )
    
    fig.update_layout(
        title=dict(
            text="Thread Depth with Normal Distribution Overlay",
            font=dict(size=24),
        ),
        xaxis_title="Thread Depth",
        yaxis_title="Number of Posts",
        template=CHART_TEMPLATE,
        legend=dict(x=0.7, y=0.95),
        xaxis=dict(tickmode="linear", dtick=1),
    )
    
    return _save_chart(fig, output_path, export_png)


# =============================================================================
# ENGAGEMENT CHARTS
# =============================================================================

def generate_engagement_histogram(
    counts: List[int],
    stats_dict: Dict[str, Any],
    output_path: Path,
    export_png: bool = True,
    log_scale: bool = True,
) -> Dict[str, str]:
    """
    Generate histogram of comment counts.
    
    Uses log scale by default due to heavy-tailed distribution.
    """
    if not PLOTLY_AVAILABLE:
        return {}
    
    fig = go.Figure()
    
    # Create bins (log-spaced for better visualization)
    if log_scale and max(counts) > 20:
        bins = [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000, max(counts) + 1]
    else:
        bins = list(range(0, max(counts) + 2, max(1, max(counts) // 20)))
    
    # Compute histogram manually for custom bins
    hist, bin_edges = np.histogram(counts, bins=bins)
    
    # Create labels
    labels = []
    for i in range(len(bins) - 1):
        if bins[i + 1] - bins[i] == 1:
            labels.append(str(bins[i]))
        else:
            labels.append(f"{bins[i]}-{bins[i+1]-1}")
    
    fig.add_trace(go.Bar(
        x=labels,
        y=hist,
        marker_color=ENGAGEMENT_COLOR,
        text=[f"{v:,}" for v in hist],
        textposition="outside",
        hovertemplate="Comments: %{x}<br>Posts: %{y:,}<extra></extra>",
    ))
    
    fig.update_layout(
        title=dict(
            text="Comment Count Distribution (Engagement)",
            font=dict(size=24),
        ),
        xaxis_title="Number of Comments",
        yaxis_title="Number of Posts",
        template=CHART_TEMPLATE,
        showlegend=False,
        annotations=[
            dict(
                text=f"n={len(counts):,} | μ={stats_dict.get('mean', 0):.1f} | "
                     f"σ={stats_dict.get('std', 0):.1f} | max={stats_dict.get('max', 0):,}",
                xref="paper", yref="paper",
                x=0.99, y=0.99,
                showarrow=False,
                font=dict(size=12, color="gray"),
                xanchor="right",
            )
        ],
    )
    
    if log_scale:
        fig.update_yaxes(type="log")
    
    return _save_chart(fig, output_path, export_png)


def generate_engagement_boxplot(
    counts: List[int],
    stats_dict: Dict[str, Any],
    output_path: Path,
    export_png: bool = True,
) -> Dict[str, str]:
    """
    Generate box-and-whisker plot for comment counts.
    """
    if not PLOTLY_AVAILABLE:
        return {}
    
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=counts,
        name="Comment Count",
        marker_color=ENGAGEMENT_COLOR,
        boxmean=True,
        boxpoints="outliers",
        jitter=0.3,
        pointpos=0,  # Center points on box
        hovertemplate="Comments: %{y:,}<extra></extra>",
    ))
    
    # Add threshold line
    threshold = stats_dict.get("threshold_2std", 0)
    if threshold > 0:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=THRESHOLD_COLOR,
            annotation_text=f"Threshold (μ+2σ) = {threshold:.1f}",
            annotation_position="right",
        )
    
    fig.update_layout(
        title=dict(
            text="Comment Count Box Plot (Engagement)",
            font=dict(size=24),
        ),
        yaxis_title="Number of Comments",
        template=CHART_TEMPLATE,
        showlegend=False,
        yaxis=dict(type="log") if max(counts) > 100 else {},
        annotations=[
            dict(
                text=f"Median: {stats_dict.get('median', 0):.0f} | "
                     f"P90: {stats_dict.get('percentiles', {}).get(90, 0):.0f} | "
                     f"P99: {stats_dict.get('percentiles', {}).get(99, 0):.0f}",
                xref="paper", yref="paper",
                x=0.99, y=0.01,
                showarrow=False,
                font=dict(size=12, color="gray"),
                xanchor="right", yanchor="bottom",
            )
        ],
    )
    
    return _save_chart(fig, output_path, export_png)


def generate_engagement_cdf(
    counts: List[int],
    stats_dict: Dict[str, Any],
    output_path: Path,
    export_png: bool = True,
) -> Dict[str, str]:
    """
    Generate CDF for comment counts.
    """
    if not PLOTLY_AVAILABLE:
        return {}
    
    sorted_counts = np.sort(counts)
    cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sorted_counts,
        y=cdf,
        mode="lines",
        line=dict(color=ENGAGEMENT_COLOR, width=3),
        name="CDF",
        fill="tozeroy",
        fillcolor=f"rgba(255, 107, 107, 0.2)",
        hovertemplate="Comments ≤ %{x:,}<br>Percentile: %{y:.1%}<extra></extra>",
    ))
    
    # Add threshold marker
    threshold = stats_dict.get("threshold_2std", 0)
    if threshold > 0:
        cdf_at_threshold = np.mean(sorted_counts <= threshold)
        fig.add_trace(go.Scatter(
            x=[threshold],
            y=[cdf_at_threshold],
            mode="markers",
            marker=dict(color=THRESHOLD_COLOR, size=15, symbol="diamond"),
            name=f"Threshold ({cdf_at_threshold:.1%})",
        ))
    
    # Add percentile markers
    percentiles = stats_dict.get("percentiles", {})
    for p in [50, 90, 95, 99]:
        if p in percentiles:
            val = percentiles[p]
            fig.add_vline(
                x=val,
                line_dash="dot",
                line_color="rgba(255,255,255,0.3)",
                annotation_text=f"P{p}={val:.0f}",
            )
    
    fig.update_layout(
        title=dict(
            text="Comment Count CDF (Cumulative Distribution)",
            font=dict(size=24),
        ),
        xaxis_title="Number of Comments",
        yaxis_title="Cumulative Probability",
        template=CHART_TEMPLATE,
        yaxis=dict(tickformat=".0%"),
        xaxis=dict(range=[0, 1500]),  # Cap at 1500
    )
    
    return _save_chart(fig, output_path, export_png)


def generate_engagement_normal(
    counts: List[int],
    stats_dict: Dict[str, Any],
    output_path: Path,
    export_png: bool = True,
) -> Dict[str, str]:
    """
    Generate histogram with log-normal distribution overlay.
    
    Comment counts typically follow a log-normal distribution.
    """
    if not PLOTLY_AVAILABLE:
        return {}
    
    # Filter out zeros for log transform
    positive_counts = [c for c in counts if c > 0]
    if not positive_counts:
        positive_counts = [1]
    
    log_counts = np.log1p(positive_counts)  # log(1+x) to handle zeros
    log_mean = np.mean(log_counts)
    log_std = np.std(log_counts)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Histogram of raw counts
    fig.add_trace(
        go.Histogram(
            x=counts,
            nbinsx=30,
            marker_color=ENGAGEMENT_COLOR,
            name="Observed",
            opacity=0.7,
        ),
        secondary_y=False,
    )
    
    # Log-normal overlay (requires scipy)
    if SCIPY_AVAILABLE and log_std > 0:
        x_range = np.linspace(0, np.percentile(counts, 99), 200)
        lognorm_pdf = stats.lognorm.pdf(x_range + 1, s=log_std, scale=np.exp(log_mean))
        # Scale to match histogram
        hist_max = np.max(np.histogram(counts, bins=30)[0])
        scale = hist_max / (max(lognorm_pdf) if max(lognorm_pdf) > 0 else 1)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=lognorm_pdf * scale,
                mode="lines",
                line=dict(color="white", width=3, dash="dash"),
                name="Log-Normal Fit",
            ),
            secondary_y=False,
        )
    
    # Threshold
    threshold = stats_dict.get("threshold_2std", 0)
    if threshold > 0:
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color=THRESHOLD_COLOR,
            annotation_text=f"μ+2σ = {threshold:.0f}",
        )
    
    fig.update_layout(
        title=dict(
            text="Comment Count with Log-Normal Distribution Overlay",
            font=dict(size=24),
        ),
        xaxis_title="Number of Comments",
        yaxis_title="Number of Posts",
        template=CHART_TEMPLATE,
        legend=dict(x=0.7, y=0.95),
    )
    
    return _save_chart(fig, output_path, export_png)


# =============================================================================
# COMBINED CHART
# =============================================================================

def generate_thread_scatter(
    metrics: List[Dict[str, Any]],
    depth_stats: Dict[str, Any],
    engagement_stats: Dict[str, Any],
    output_path: Path,
    export_png: bool = True,
) -> Dict[str, str]:
    """
    Generate scatter plot: X=comment count, Y=depth.
    
    Identifies threads that are BOTH deep AND popular.
    Points are clickable - clicking opens the thread on Moltbook.
    """
    if not PLOTLY_AVAILABLE:
        return {}
    
    x = [m["comment_count"] for m in metrics]
    y = [m["depth"] for m in metrics]
    post_ids = [m["post_id"][:8] for m in metrics]
    urls = [m.get("permalink", f"https://www.moltbook.com/post/{m['post_id']}") for m in metrics]
    titles = [m.get("title", "Untitled")[:40] for m in metrics]
    
    fig = go.Figure()
    
    # Main scatter with customdata for URLs
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(
            size=8,
            color=DEPTH_COLOR,
            opacity=0.6,
            line=dict(width=1, color="white"),
        ),
        text=titles,
        customdata=urls,
        hovertemplate="<b>%{text}</b><br>Comments: %{x:,}<br>Depth: %{y}<br><i>Click to open</i><extra></extra>",
        name="Threads",
    ))
    
    # Add threshold lines
    depth_threshold = depth_stats.get("threshold_2std", 0)
    engagement_threshold = engagement_stats.get("threshold_2std", 0)
    
    if depth_threshold > 0:
        fig.add_hline(
            y=depth_threshold,
            line_dash="dash",
            line_color=DEPTH_COLOR,
            annotation_text=f"Deep threshold: {depth_threshold:.2f}",
            annotation_position="right",
        )
    
    if engagement_threshold > 0:
        fig.add_vline(
            x=engagement_threshold,
            line_dash="dash",
            line_color=ENGAGEMENT_COLOR,
            annotation_text=f"Hot threshold: {engagement_threshold:.0f}",
            annotation_position="top",
        )
    
    # Highlight threads that are BOTH conversational AND popular
    high_value = [
        m for m in metrics
        if m["depth"] > depth_threshold and m["comment_count"] > engagement_threshold
    ]
    
    if high_value:
        fig.add_trace(go.Scatter(
            x=[m["comment_count"] for m in high_value],
            y=[m["depth"] for m in high_value],
            mode="markers",
            marker=dict(
                size=15,
                color=THRESHOLD_COLOR,
                symbol="diamond",
                line=dict(width=2, color="white"),
            ),
            text=[m.get("title", "Untitled")[:40] for m in high_value],
            customdata=[m.get("permalink", f"https://www.moltbook.com/post/{m['post_id']}") for m in high_value],
            hovertemplate="◆ <b>%{text}</b><br>Comments: %{x:,}<br>Depth: %{y}<br><i>Click to open</i><extra></extra>",
            name=f"High-Value ({len(high_value)})",
        ))
    
    fig.update_layout(
        title=dict(
            text="Thread Depth vs Engagement (Click points to view threads)",
            font=dict(size=24),
        ),
        xaxis_title="Comment Count (Engagement)",
        yaxis_title="Thread Depth (Nesting)",
        template=CHART_TEMPLATE,
        legend=dict(x=0.02, y=0.98),
        xaxis=dict(type="log") if max(x) > 100 else {},
        annotations=[
            dict(
                text=f"n={len(metrics):,} threads | {len(high_value)} high-value (both thresholds)",
                xref="paper", yref="paper",
                x=0.99, y=0.01,
                showarrow=False,
                font=dict(size=12, color="gray"),
                xanchor="right", yanchor="bottom",
            )
        ],
    )
    
    return _save_chart_with_click_handler(fig, output_path, export_png)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def generate_all_thread_charts(
    analysis_result: Dict[str, Any],
    output_dir: Path,
    export_png: bool = True,
) -> Dict[str, Dict[str, str]]:
    """
    Generate all 9 thread analysis charts.
    
    Args:
        analysis_result: Output from analyze_batch()
        output_dir: Directory to save charts
        export_png: Whether to also export PNG versions
        
    Returns:
        Dict mapping chart name to {html: path, png: path}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = analysis_result["metrics"]
    depth_stats = analysis_result["depth_stats"]
    engagement_stats = analysis_result["engagement_stats"]
    
    depths = [m["depth"] for m in metrics]
    counts = [m["comment_count"] for m in metrics]
    
    charts = {}
    
    # Depth charts
    charts["depth_histogram"] = generate_depth_histogram(
        depths, depth_stats, output_dir / "depth_histogram.html", export_png
    )
    charts["depth_boxplot"] = generate_depth_boxplot(
        depths, depth_stats, output_dir / "depth_boxplot.html", export_png
    )
    charts["depth_cdf"] = generate_depth_cdf(
        depths, depth_stats, output_dir / "depth_cdf.html", export_png
    )
    charts["depth_normal"] = generate_depth_normal(
        depths, depth_stats, output_dir / "depth_normal.html", export_png
    )
    
    # Engagement charts
    charts["engagement_histogram"] = generate_engagement_histogram(
        counts, engagement_stats, output_dir / "engagement_histogram.html", export_png
    )
    charts["engagement_boxplot"] = generate_engagement_boxplot(
        counts, engagement_stats, output_dir / "engagement_boxplot.html", export_png
    )
    charts["engagement_cdf"] = generate_engagement_cdf(
        counts, engagement_stats, output_dir / "engagement_cdf.html", export_png
    )
    charts["engagement_normal"] = generate_engagement_normal(
        counts, engagement_stats, output_dir / "engagement_normal.html", export_png
    )
    
    # Combined scatter
    charts["thread_scatter"] = generate_thread_scatter(
        metrics, depth_stats, engagement_stats, 
        output_dir / "thread_scatter.html", export_png
    )
    
    return charts

