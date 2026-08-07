"""
Agent 工具集 - 报告生成相关工具
封装 ReportGenerator 为 Agent 可调用的 Tool
"""

import os
import sys
from typing import Dict, Any

from .base import BaseTool, ToolParameter


class GenerateReportTool(BaseTool):
    """基于分析结果生成专业的学情分析报告"""

    name = "generate_report"
    description = (
        "基于课堂学情分析结果，调用通义千问大模型生成专业化的课堂学情分析报告。"
        "报告包含：基础数据概况、整体学情评价、分时段深度分析、行为结构分析、"
        "优势与问题、优化建议、综合评分与等级评定。"
    )
    parameters = [
        ToolParameter("csv_path", "string", "CSV 数据文件路径，将自动进行完整分析", required=True),
        ToolParameter("output_dir", "string", "报告输出目录，默认 outputs/reports/", required=False, default="outputs/reports/"),
    ]

    def execute(self, csv_path: str, output_dir: str = "outputs/reports/") -> Dict[str, Any]:
        if not os.path.exists(csv_path):
            return {"error": f"CSV 文件不存在: {csv_path}"}

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.analysis_module.analyzer import ClassroomAnalyzer

        api_key = os.environ.get("QWEN_API_KEY")
        analyzer = ClassroomAnalyzer(api_key=api_key)

        result = analyzer.analyze(csv_path)

        report = result.get("report", {})
        report_content = report.get("report_content", "报告生成失败")

        os.makedirs(output_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(csv_path))[0].replace("_raw", "")
        report_path = os.path.join(output_dir, f"analysis_report_{video_name}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        return {
            "csv_path": csv_path,
            "report_path": report_path,
            "report_content": report_content,
            "overall_metrics": result.get("analysis", {}).get("overall", {}),
            "segments": result.get("analysis", {}).get("segments", {}),
            "trends": result.get("analysis", {}).get("trends", {}),
        }


class QuickSummaryTool(BaseTool):
    """快速生成 CSV 数据的摘要信息（不调用 LLM）"""

    name = "quick_summary"
    description = (
        "快速获取 CSV 数据的统计摘要，不调用大语言模型，返回核心指标的简要描述。"
        "适用于需要快速了解数据概览的场景。"
    )
    parameters = [
        ToolParameter("csv_path", "string", "CSV 数据文件路径", required=True),
    ]

    def execute(self, csv_path: str) -> Dict[str, Any]:
        if not os.path.exists(csv_path):
            return {"error": f"CSV 文件不存在: {csv_path}"}

        import pandas as pd
        from src.analysis_module.metrics_calculator import MetricsCalculator

        df = pd.read_csv(csv_path)
        mc = MetricsCalculator()
        df_m = mc.calculate_frame_metrics(df)
        overall = mc.calculate_overall_metrics(df_m)

        total_s = sum(df_m.get("total_stu", df_m.get("total_students", pd.Series([0]))))
        behavior_cols = [
            "focus_listen", "study_bow", "empty_mind", "sleep_stu",
            "look_side", "talk_discuss", "talk_private",
            "stand_up", "loose_stu", "phone_game",
        ]
        behavior_dist = {}
        for col in behavior_cols:
            if col in df_m.columns:
                behavior_dist[col] = int(df_m[col].sum())

        summary = f"共{len(df_m)}帧，平均{overall.get('avg_total_students', 0)}人/帧。"
        summary += f"有效学习率{overall.get('avg_effective_learning_rate', 0)}%，"
        summary += f"走神率{overall.get('avg_distraction_rate', 0)}%，"
        summary += f"困倦率{overall.get('avg_drowsiness_rate', 0)}%。"

        return {
            "csv_path": csv_path,
            "summary_text": summary,
            "overall": overall,
            "behavior_distribution": behavior_dist,
        }
