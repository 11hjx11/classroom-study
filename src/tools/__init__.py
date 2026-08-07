"""
Agent Tools 包
自动注册所有工具到 ToolRegistry
"""

from .base import BaseTool, ToolParameter, ToolRegistry
from .video_tools import (
    GetVideoInfoTool,
    AnalyzeVideoTool,
    ListVideosTool,
    ListCSVsTool,
)
from .analysis_tools import (
    ComputeMetricsTool,
    AnalyzeTrendTool,
    QueryCSVDataTool,
    CompareMetricsTool,
)
from .report_tools import GenerateReportTool, QuickSummaryTool


def create_default_registry() -> ToolRegistry:
    """创建包含所有默认工具的注册表"""
    registry = ToolRegistry()

    video_tools = [
        GetVideoInfoTool(),
        AnalyzeVideoTool(),
        ListVideosTool(),
        ListCSVsTool(),
    ]
    analysis_tools = [
        ComputeMetricsTool(),
        AnalyzeTrendTool(),
        QueryCSVDataTool(),
        CompareMetricsTool(),
    ]
    report_tools = [
        GenerateReportTool(),
        QuickSummaryTool(),
    ]

    for t in video_tools + analysis_tools + report_tools:
        registry.register(t)

    return registry


__all__ = [
    "BaseTool",
    "ToolParameter",
    "ToolRegistry",
    "GetVideoInfoTool",
    "AnalyzeVideoTool",
    "ListVideosTool",
    "ListCSVsTool",
    "ComputeMetricsTool",
    "AnalyzeTrendTool",
    "QueryCSVDataTool",
    "CompareMetricsTool",
    "GenerateReportTool",
    "QuickSummaryTool",
    "create_default_registry",
]
