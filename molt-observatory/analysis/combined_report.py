"""
Combined Report Generator

Creates a single HTML page with all thread analysis charts embedded,
optimized for printing to PDF.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


def generate_combined_report(
    analysis_dir: Path,
    stats: Dict[str, Any],
    output_path: Optional[Path] = None,
    title: str = "Thread Analysis Report",
) -> str:
    """
    Generate a combined HTML report with all charts embedded.
    
    Args:
        analysis_dir: Directory containing chart HTML files
        stats: Thread statistics dict
        output_path: Output path (default: analysis_dir/combined_report.html)
        title: Report title
        
    Returns:
        Path to generated report
    """
    analysis_dir = Path(analysis_dir)
    output_path = output_path or analysis_dir / "combined_report.html"
    
    # Read all chart HTML files and extract the Plotly div
    chart_files = [
        ("Thread Depth Distribution", "depth_histogram.html"),
        ("Depth Box Plot", "depth_boxplot.html"),
        ("Depth CDF", "depth_cdf.html"),
        ("Depth with Normal Overlay", "depth_normal.html"),
        ("Engagement Distribution", "engagement_histogram.html"),
        ("Engagement Box Plot", "engagement_boxplot.html"),
        ("Engagement CDF", "engagement_cdf.html"),
        ("Engagement with Log-Normal Overlay", "engagement_normal.html"),
        ("Depth vs Engagement Scatter", "thread_scatter.html"),
    ]
    
    charts_html = []
    for chart_title, filename in chart_files:
        chart_path = analysis_dir / filename
        if chart_path.exists():
            with open(chart_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Extract the plotly div and script
            # The chart is in a div with class plotly-graph-div
            charts_html.append(f"""
        <div class="chart-section">
            <h3>{chart_title}</h3>
            <iframe src="{filename}" class="chart-frame"></iframe>
        </div>
""")
    
    # Build statistics section
    depth_stats = stats.get("depth_stats", {})
    engagement_stats = stats.get("engagement_stats", {})
    summary = stats.get("summary", {})
    
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid #4ecdc4;
        }}
        
        h1 {{
            color: #4ecdc4;
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }}
        
        .timestamp {{
            color: #888;
            font-size: 0.9em;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .stat-card {{
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border: 1px solid #333;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #4ecdc4;
        }}
        
        .stat-label {{
            color: #888;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        h2 {{
            color: #ff6b6b;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
        }}
        
        h3 {{
            color: #ffd93d;
            margin: 20px 0 10px 0;
        }}
        
        .chart-section {{
            margin-bottom: 30px;
            page-break-inside: avoid;
        }}
        
        .chart-frame {{
            width: 100%;
            height: 500px;
            border: 1px solid #333;
            border-radius: 8px;
            background: #0f0f23;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        
        th {{
            background: #16213e;
            color: #4ecdc4;
        }}
        
        tr:hover {{
            background: rgba(78, 205, 196, 0.1);
        }}
        
        .highlight {{
            color: #ffd93d;
            font-weight: bold;
        }}
        
        @media print {{
            body {{
                background: white;
                color: black;
            }}
            
            .stat-card {{
                background: #f5f5f5;
                border: 1px solid #ddd;
            }}
            
            .stat-value {{
                color: #2d6a4f;
            }}
            
            h1, h2 {{
                color: #1a1a2e;
            }}
            
            h3 {{
                color: #333;
            }}
            
            .chart-frame {{
                height: 400px;
            }}
            
            .chart-section {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>
        
        <div class="section">
            <h2>Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{summary.get('total_posts', 0):,}</div>
                    <div class="stat-label">Total Posts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.get('total_comments', 0):,}</div>
                    <div class="stat-label">Total Comments</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.get('reply_ratio', 0)*100:.1f}%</div>
                    <div class="stat-label">Reply Ratio</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.get('posts_above_depth_threshold', 0):,}</div>
                    <div class="stat-label">Deep Threads</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.get('posts_above_engagement_threshold', 0):,}</div>
                    <div class="stat-label">Hot Threads</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Detailed Statistics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Depth (Nesting)</th>
                    <th>Engagement (Comments)</th>
                </tr>
                <tr>
                    <td>Mean</td>
                    <td>{depth_stats.get('mean', 0):.2f}</td>
                    <td>{engagement_stats.get('mean', 0):.1f}</td>
                </tr>
                <tr>
                    <td>Std Dev</td>
                    <td>{depth_stats.get('std', 0):.2f}</td>
                    <td>{engagement_stats.get('std', 0):.1f}</td>
                </tr>
                <tr>
                    <td>Median</td>
                    <td>{depth_stats.get('median', 0):.1f}</td>
                    <td>{engagement_stats.get('median', 0):.1f}</td>
                </tr>
                <tr>
                    <td>Min / Max</td>
                    <td>{depth_stats.get('min', 0)} / {depth_stats.get('max', 0)}</td>
                    <td>{engagement_stats.get('min', 0)} / {engagement_stats.get('max', 0)}</td>
                </tr>
                <tr>
                    <td>Threshold (μ+2σ)</td>
                    <td class="highlight">{depth_stats.get('threshold_2std', 0):.2f}</td>
                    <td class="highlight">{engagement_stats.get('threshold_2std', 0):.1f}</td>
                </tr>
                <tr>
                    <td>95th Percentile</td>
                    <td>{depth_stats.get('percentiles', {}).get(95, 0):.1f}</td>
                    <td>{engagement_stats.get('percentiles', {}).get(95, 0):.1f}</td>
                </tr>
                <tr>
                    <td>99th Percentile</td>
                    <td>{depth_stats.get('percentiles', {}).get(99, 0):.1f}</td>
                    <td>{engagement_stats.get('percentiles', {}).get(99, 0):.1f}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Depth Distribution</h2>
            <table>
                <tr>
                    <th>Depth</th>
                    <th>Count</th>
                    <th>Percentage</th>
                    <th>Meaning</th>
                </tr>
"""
    
    # Add depth distribution rows
    distribution = depth_stats.get('distribution', {})
    total = summary.get('total_posts', 1)
    depth_meanings = {
        1: "Post only, no comments",
        2: "Post + top-level comments",
        3: "Post + comments + replies",
        4: "Post + 3 levels of nesting",
        5: "Post + 4 levels of nesting",
    }
    
    # Convert string keys to int if needed
    for depth_key in sorted(distribution.keys(), key=lambda x: int(x)):
        depth = int(depth_key)
        count = distribution[depth_key]
        pct = count / total * 100
        meaning = depth_meanings.get(depth, f"{depth-1} levels of replies")
        html += f"""                <tr>
                    <td>{depth}</td>
                    <td>{count:,}</td>
                    <td>{pct:.1f}%</td>
                    <td>{meaning}</td>
                </tr>
"""
    
    html += """            </table>
        </div>
        
        <div class="section">
            <h2>Charts</h2>
"""
    
    # Add charts
    html += "\n".join(charts_html)
    
    html += """
        </div>
        
        <footer style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #333; color: #666;">
            <p>Molt Observatory - Thread Analysis Report</p>
            <p><small>To save as PDF: Press Cmd+P (Mac) or Ctrl+P (Windows) → Save as PDF</small></p>
        </footer>
    </div>
</body>
</html>
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    return str(output_path)

