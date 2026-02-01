# Thread Analysis Module
# Provides depth and engagement analysis for Moltbook threads.

from .thread_analysis import (
    compute_thread_depth,
    compute_comment_count,
    compute_depth_stats,
    compute_engagement_stats,
    analyze_batch,
    partition_deep_threads,
    partition_hot_threads,
)

from .thread_charts import (
    generate_depth_histogram,
    generate_depth_boxplot,
    generate_depth_cdf,
    generate_depth_normal,
    generate_engagement_histogram,
    generate_engagement_boxplot,
    generate_engagement_cdf,
    generate_engagement_normal,
    generate_thread_scatter,
    generate_all_thread_charts,
)

from .content_filter import (
    run_content_filter,
    filter_transcripts_quick,
    analyze_agents_only,
    build_spammer_profiles,
    get_spammer_list,
    FilterStats,
    SpammerProfile,
    FilterResult,
)

from .cascade_detector import (
    analyze_cascades,
    detect_clone_swarm,
    detect_timing_cluster,
    generate_cascade_report,
    CascadeReport,
    CascadeAnalysis,
)

from .lite_judge import (
    run_lite_judge,
    run_lite_judge_on_messages,
    LiteJudgeRunner,
    LiteJudgeResult,
    LiteJudgeStats,
)

from .tiered_reports import (
    generate_all_tiered_reports,
    generate_spam_breakdown_chart,
    generate_agent_spam_leaderboard,
    generate_filter_funnel,
    generate_cascade_timeline,
    generate_clone_network,
    generate_lite_score_histogram,
    generate_escalation_pie,
    generate_tiered_summary_html,
)

__all__ = [
    # Thread analysis functions
    "compute_thread_depth",
    "compute_comment_count",
    "compute_depth_stats",
    "compute_engagement_stats",
    "analyze_batch",
    "partition_deep_threads",
    "partition_hot_threads",
    # Chart functions
    "generate_depth_histogram",
    "generate_depth_boxplot",
    "generate_depth_cdf",
    "generate_depth_normal",
    "generate_engagement_histogram",
    "generate_engagement_boxplot",
    "generate_engagement_cdf",
    "generate_engagement_normal",
    "generate_thread_scatter",
    "generate_all_thread_charts",
    # Content filter
    "run_content_filter",
    "filter_transcripts_quick",
    "analyze_agents_only",
    "build_spammer_profiles",
    "get_spammer_list",
    "FilterStats",
    "SpammerProfile",
    "FilterResult",
    # Cascade detector
    "analyze_cascades",
    "detect_clone_swarm",
    "detect_timing_cluster",
    "generate_cascade_report",
    "CascadeReport",
    "CascadeAnalysis",
    # Lite judge
    "run_lite_judge",
    "run_lite_judge_on_messages",
    "LiteJudgeRunner",
    "LiteJudgeResult",
    "LiteJudgeStats",
    # Tiered reports
    "generate_all_tiered_reports",
    "generate_spam_breakdown_chart",
    "generate_agent_spam_leaderboard",
    "generate_filter_funnel",
    "generate_cascade_timeline",
    "generate_clone_network",
    "generate_lite_score_histogram",
    "generate_escalation_pie",
    "generate_tiered_summary_html",
]

