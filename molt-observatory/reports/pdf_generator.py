"""
PDF Generator for Molt Observatory Reports

Uses Playwright to render HTML pages with JavaScript (for Plotly charts)
and exports them as PDF pages, then combines into a single document.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    from PyPDF2 import PdfMerger
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False


def html_to_pdf(html_path: Path, output_path: Path, wait_for_js: int = 2000) -> bool:
    """
    Convert a single HTML file to PDF using Playwright.
    
    Args:
        html_path: Path to HTML file
        output_path: Path for output PDF
        wait_for_js: Milliseconds to wait for JavaScript to render
        
    Returns:
        True if successful, False otherwise
    """
    if not PLAYWRIGHT_AVAILABLE:
        print("Warning: Playwright not available. Cannot generate PDF.")
        return False
    
    html_path = Path(html_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            
            # Load HTML file
            page.goto(f"file://{html_path.absolute()}")
            
            # Wait for JavaScript to render (Plotly charts)
            page.wait_for_timeout(wait_for_js)
            
            # Export to PDF
            page.pdf(
                path=str(output_path),
                format="A4",
                landscape=True,
                print_background=True,
                margin={"top": "0.5in", "bottom": "0.5in", "left": "0.5in", "right": "0.5in"},
            )
            
            browser.close()
            
        return True
    except Exception as e:
        print(f"Error converting {html_path} to PDF: {e}")
        return False


def generate_stats_html(stats: Dict[str, Any], output_path: Path) -> Path:
    """Generate a simple HTML page with statistics summary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    depth_stats = stats.get("depth_stats", {})
    engagement_stats = stats.get("engagement_stats", {})
    summary = stats.get("summary", {})
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Thread Analysis Summary</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 40px;
            max-width: 1000px;
            margin: 0 auto;
        }}
        h1 {{
            color: #4ecdc4;
            border-bottom: 2px solid #4ecdc4;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #ff6b6b;
            margin-top: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 36px;
            font-weight: bold;
            color: #4ecdc4;
        }}
        .stat-label {{
            color: #888;
            margin-top: 5px;
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
        .highlight {{
            color: #ffd93d;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>Thread Analysis Report</h1>
    
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
    </div>
    
    <h2>Thread Depth Statistics</h2>
    <p>Measures reply nesting: Depth 1 = post only, Depth 2 = post + comments, Depth 3+ = nested replies</p>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Mean</td><td>{depth_stats.get('mean', 0):.2f}</td></tr>
        <tr><td>Std Dev</td><td>{depth_stats.get('std', 0):.2f}</td></tr>
        <tr><td>Median</td><td>{depth_stats.get('median', 0):.1f}</td></tr>
        <tr><td>Min / Max</td><td>{depth_stats.get('min', 0)} / {depth_stats.get('max', 0)}</td></tr>
        <tr><td>Threshold (μ+2σ)</td><td class="highlight">{depth_stats.get('threshold_2std', 0):.2f}</td></tr>
        <tr><td>Posts Above Threshold</td><td class="highlight">{summary.get('posts_above_depth_threshold', 0):,}</td></tr>
    </table>
    
    <h2>Engagement Statistics (Comment Count)</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Mean</td><td>{engagement_stats.get('mean', 0):.1f}</td></tr>
        <tr><td>Std Dev</td><td>{engagement_stats.get('std', 0):.1f}</td></tr>
        <tr><td>Median</td><td>{engagement_stats.get('median', 0):.0f}</td></tr>
        <tr><td>Min / Max</td><td>{engagement_stats.get('min', 0)} / {engagement_stats.get('max', 0):,}</td></tr>
        <tr><td>Threshold (μ+2σ)</td><td class="highlight">{engagement_stats.get('threshold_2std', 0):.1f}</td></tr>
        <tr><td>Posts Above Threshold</td><td class="highlight">{summary.get('posts_above_engagement_threshold', 0):,}</td></tr>
    </table>
    
    <h2>Key Insight</h2>
    <p>Only <strong>{summary.get('reply_ratio', 0)*100:.1f}%</strong> of comments are replies to other comments. 
    Agents mostly <em>broadcast</em> rather than <em>converse</em>, making deep threads rare and analytically interesting.</p>
</body>
</html>
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    return output_path


def combine_pdfs(pdf_paths: List[Path], output_path: Path) -> bool:
    """
    Combine multiple PDFs into a single document.
    
    Args:
        pdf_paths: List of PDF file paths
        output_path: Output path for combined PDF
        
    Returns:
        True if successful
    """
    if not PYPDF2_AVAILABLE:
        # Fallback: just copy the first PDF
        print("Warning: PyPDF2 not available. Using first PDF only.")
        if pdf_paths:
            import shutil
            shutil.copy(pdf_paths[0], output_path)
            return True
        return False
    
    try:
        merger = PdfMerger()
        for pdf_path in pdf_paths:
            if Path(pdf_path).exists():
                merger.append(str(pdf_path))
        
        merger.write(str(output_path))
        merger.close()
        return True
    except Exception as e:
        print(f"Error combining PDFs: {e}")
        return False


def generate_thread_analysis_pdf(
    analysis_dir: Path,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Generate a combined PDF report for thread analysis.
    
    Args:
        analysis_dir: Directory containing HTML charts and thread_stats.json
        output_path: Output PDF path (default: analysis_dir/thread_analysis_report.pdf)
        
    Returns:
        Path to generated PDF, or None if failed
    """
    analysis_dir = Path(analysis_dir)
    
    if output_path is None:
        output_path = analysis_dir / "thread_analysis_report.pdf"
    
    print(f"\n📄 Generating PDF report...")
    print(f"   Source: {analysis_dir}")
    print(f"   Output: {output_path}")
    
    # Load stats
    stats_path = analysis_dir / "thread_stats.json"
    stats = {}
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
    
    # Define pages in order
    pages = [
        ("summary", None),  # Will generate HTML
        ("depth_histogram", analysis_dir / "depth_histogram.html"),
        ("depth_boxplot", analysis_dir / "depth_boxplot.html"),
        ("depth_cdf", analysis_dir / "depth_cdf.html"),
        ("depth_normal", analysis_dir / "depth_normal.html"),
        ("engagement_histogram", analysis_dir / "engagement_histogram.html"),
        ("engagement_boxplot", analysis_dir / "engagement_boxplot.html"),
        ("engagement_cdf", analysis_dir / "engagement_cdf.html"),
        ("engagement_normal", analysis_dir / "engagement_normal.html"),
        ("thread_scatter", analysis_dir / "thread_scatter.html"),
    ]
    
    # Create temp directory for individual PDFs
    temp_dir = analysis_dir / ".pdf_temp"
    temp_dir.mkdir(exist_ok=True)
    
    pdf_paths = []
    
    for name, html_path in pages:
        print(f"   Converting {name}...", end=" ")
        
        if name == "summary":
            # Generate summary HTML
            html_path = temp_dir / "summary.html"
            generate_stats_html(stats, html_path)
        
        if html_path and html_path.exists():
            pdf_path = temp_dir / f"{name}.pdf"
            if html_to_pdf(html_path, pdf_path):
                pdf_paths.append(pdf_path)
                print("✅")
            else:
                print("❌")
        else:
            print(f"⚠️ (file not found)")
    
    # Combine PDFs
    if pdf_paths:
        print(f"   Combining {len(pdf_paths)} pages...")
        if combine_pdfs(pdf_paths, output_path):
            print(f"   ✅ PDF saved to {output_path}")
            
            # Cleanup temp files
            for pdf in pdf_paths:
                try:
                    pdf.unlink()
                except:
                    pass
            try:
                (temp_dir / "summary.html").unlink()
            except:
                pass
            try:
                temp_dir.rmdir()
            except:
                pass
            
            return output_path
    
    print("   ❌ Failed to generate PDF")
    return None


def generate_pipeline_report_pdf(
    reports_dir: Path,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Generate a combined PDF report for standard pipeline reports.
    
    Args:
        reports_dir: Directory containing HTML reports
        output_path: Output PDF path
        
    Returns:
        Path to generated PDF, or None if failed
    """
    reports_dir = Path(reports_dir)
    
    if output_path is None:
        output_path = reports_dir / "pipeline_report.pdf"
    
    # Find all HTML files
    html_files = sorted(reports_dir.glob("*.html"))
    
    if not html_files:
        print(f"No HTML files found in {reports_dir}")
        return None
    
    print(f"\n📄 Generating PDF report from {len(html_files)} HTML files...")
    
    temp_dir = reports_dir / ".pdf_temp"
    temp_dir.mkdir(exist_ok=True)
    
    pdf_paths = []
    
    for html_path in html_files:
        print(f"   Converting {html_path.name}...", end=" ")
        pdf_path = temp_dir / f"{html_path.stem}.pdf"
        if html_to_pdf(html_path, pdf_path):
            pdf_paths.append(pdf_path)
            print("✅")
        else:
            print("❌")
    
    if pdf_paths:
        print(f"   Combining {len(pdf_paths)} pages...")
        if combine_pdfs(pdf_paths, output_path):
            print(f"   ✅ PDF saved to {output_path}")
            
            # Cleanup
            for pdf in pdf_paths:
                try:
                    pdf.unlink()
                except:
                    pass
            try:
                temp_dir.rmdir()
            except:
                pass
            
            return output_path
    
    return None


def generate_tiered_eval_pdf(
    run_dir: Path,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Generate a combined PDF report for tiered evaluation runs.
    
    Args:
        run_dir: Path to tiered eval run directory
        output_path: Output PDF path
        
    Returns:
        Path to generated PDF, or None if failed
    """
    run_dir = Path(run_dir)
    reports_dir = run_dir / "reports"
    
    if output_path is None:
        output_path = run_dir / "tiered_eval_report.pdf"
    
    # Find all HTML files in reports dir
    html_files = sorted(reports_dir.glob("*.html"))
    
    if not html_files:
        print(f"No HTML files found in {reports_dir}")
        return None
    
    # Order pages for better reading flow
    page_order = [
        # Summary and dashboard
        "summary.html",           # Tiered eval overview
        "dashboard.html",         # Standard dashboard
        # Spam and filtering
        "spam_breakdown.html",    # Filter results
        "spam_types_bar.html",    # Spam details
        "agent_spam_leaderboard.html",  # Spammer ranking
        "filter_funnel.html",     # Filter flow
        "cascade_timeline.html",  # Attack patterns
        "clone_network.html",     # Clone visualization
        # Lite judge
        "lite_score_histogram.html",  # Lite judge scores
        "lite_score_boxplot.html",    # Score distribution
        "escalation_pie.html",    # Triage results
        # Standard reports
        "growth.html",            # Growth and timeline
        "agent_leaderboard.html", # Agent safety rankings
        "leaderboard_harm_enablement.html",
        "leaderboard_deception_or_evasion.html",
        "leaderboard_self_preservation_power_seeking.html",
        "leaderboard_delusional_sycophancy.html",
        "posts_timeline.html",    # Posts over time
    ]
    
    ordered_files = []
    for name in page_order:
        for f in html_files:
            if f.name == name:
                ordered_files.append(f)
                break
    
    # Add any remaining files not in our order list
    for f in html_files:
        if f not in ordered_files:
            ordered_files.append(f)
    
    print(f"\n📄 Generating tiered evaluation PDF report...")
    print(f"   Source: {reports_dir}")
    print(f"   Output: {output_path}")
    print(f"   Pages: {len(ordered_files)}")
    
    temp_dir = reports_dir / ".pdf_temp"
    temp_dir.mkdir(exist_ok=True)
    
    pdf_paths = []
    
    for html_path in ordered_files:
        print(f"   Converting {html_path.name}...", end=" ")
        pdf_path = temp_dir / f"{html_path.stem}.pdf"
        if html_to_pdf(html_path, pdf_path):
            pdf_paths.append(pdf_path)
            print("✅")
        else:
            print("❌")
    
    if pdf_paths:
        print(f"   Combining {len(pdf_paths)} pages...")
        if combine_pdfs(pdf_paths, output_path):
            print(f"   ✅ PDF saved to {output_path}")
            print(f"   📁 File size: {output_path.stat().st_size / 1024:.1f} KB")
            
            # Cleanup
            for pdf in pdf_paths:
                try:
                    pdf.unlink()
                except:
                    pass
            try:
                temp_dir.rmdir()
            except:
                pass
            
            return output_path
    
    print("   ❌ Failed to generate PDF")
    return None

