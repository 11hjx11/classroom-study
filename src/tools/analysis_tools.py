"""
Agent 工具集 - 数据分析相关工具
封装 MetricsCalculator、TrendAnalyzer、ClassroomAnalyzer 为 Agent 可调用的 Tool
"""

import os
import sys
import pandas as pd
from typing import Dict, Any

from .base import BaseTool, ToolParameter


class ComputeMetricsTool(BaseTool):
    """对已生成的 CSV 数据计算多维度学情指标"""

    name = "compute_metrics"
    description = (
        "对课堂视频分析生成的 CSV 数据计算多维度学情量化指标，"
        "包括有效学习率、走神率、困倦率、互动率、违纪率、注意力衰减等。"
        "返回整体指标和分时段指标。"
    )
    parameters = [
        ToolParameter("csv_path", "string", "CSV 数据文件的绝对路径", required=True),
    ]

    def execute(self, csv_path: str) -> Dict[str, Any]:
        if not os.path.exists(csv_path):
            return {"error": f"CSV 文件不存在: {csv_path}"}

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.analysis_module.analyzer import ClassroomAnalyzer

        analyzer = ClassroomAnalyzer(api_key=self._get_api_key())

        df = pd.read_csv(csv_path)
        df_with_metrics = analyzer.metrics_calculator.calculate_frame_metrics(df)
        overall = analyzer.metrics_calculator.calculate_overall_metrics(df_with_metrics)
        trends = analyzer.metrics_calculator.calculate_temporal_trends(df_with_metrics)

        _, segment_stats = analyzer.temporal_analyzer.analyze(df_with_metrics)

        return {
            "csv_path": csv_path,
            "overall_metrics": overall,
            "temporal_trends": trends,
            "segments": segment_stats,
            "total_frames": len(df),
        }

    def _get_api_key(self) -> str:
        return os.environ.get("QWEN_API_KEY")


class AnalyzeTrendTool(BaseTool):
    """对 CSV 数据进行深度趋势分析"""

    name = "analyze_trend"
    description = (
        "对课堂 CSV 数据进行深度趋势分析，包括：注意力峰值/低谷检测、"
        "行为状态转换分析、学生留存率变化、课堂节奏模式（FFT周期性检测）等。"
    )
    parameters = [
        ToolParameter("csv_path", "string", "CSV 数据文件的绝对路径", required=True),
    ]

    def execute(self, csv_path: str) -> Dict[str, Any]:
        if not os.path.exists(csv_path):
            return {"error": f"CSV 文件不存在: {csv_path}"}

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.analysis_module.trend_analyzer import TrendAnalyzer
        from src.analysis_module.metrics_calculator import MetricsCalculator

        df = pd.read_csv(csv_path)

        mc = MetricsCalculator()
        df_with_metrics = mc.calculate_frame_metrics(df)

        ta = TrendAnalyzer()
        result = ta.analyze(df_with_metrics)

        return {
            "csv_path": csv_path,
            "attention_peaks": result.get("attention_peaks", []),
            "attention_dips": result.get("attention_dips", []),
            "behavior_transitions": result.get("behavior_transitions", {}),
            "student_retention": result.get("student_retention", {}),
            "rhythm_patterns": result.get("rhythm_patterns", {}),
        }


class QueryCSVDataTool(BaseTool):
    """查询 CSV 数据的原始内容，支持时间范围和字段筛选"""

    name = "query_csv_data"
    description = (
        "查询 CSV 数据文件的原始帧数据，可按时间范围筛选，"
        "用于 Agent 获取具体帧的详细数据进行深入分析。"
    )
    parameters = [
        ToolParameter("csv_path", "string", "CSV 数据文件的绝对路径", required=True),
        ToolParameter("time_range", "string", "时间范围，格式 '开始秒-结束秒'，如 '120-300'", required=False, default=None),
        ToolParameter("fields", "string", "要返回的字段，逗号分隔。默认返回全部", required=False, default=None),
        ToolParameter("limit", "integer", "最多返回的帧数，默认50", required=False, default=50),
    ]

    def execute(self, csv_path: str, time_range: str = None,
                fields: str = None, limit: int = 50) -> Dict[str, Any]:
        if not os.path.exists(csv_path):
            return {"error": f"CSV 文件不存在: {csv_path}"}

        df = pd.read_csv(csv_path)
        total_rows = len(df)

        if time_range:
            try:
                start_sec, end_sec = map(float, time_range.split("-"))
                df = df[(df["timestamp"] >= start_sec) & (df["timestamp"] <= end_sec)]
            except ValueError:
                return {"error": f"time_range 格式错误: {time_range}"}

        if fields:
            field_list = [f.strip() for f in fields.split(",")]
            available = [f for f in field_list if f in df.columns]
            if available:
                df = df[available]

        df = df.head(limit)

        def convert_val(val):
            if pd.isna(val):
                return None
            if hasattr(val, "item"):
                return val.item()
            if hasattr(val, "tolist"):
                return val.tolist()
            return val

        records = []
        for _, row in df.iterrows():
            records.append({k: convert_val(v) for k, v in row.items()})

        return {
            "csv_path": csv_path,
            "total_rows": total_rows,
            "returned_rows": len(records),
            "columns": list(df.columns),
            "data": records,
        }


class CompareMetricsTool(BaseTool):
    """对比两个 CSV 数据文件的指标差异"""

    name = "compare_metrics"
    description = (
        "对比两个课堂 CSV 数据文件的学情指标差异，"
        "返回各项指标的差值和百分比变化，用于对比不同课程或不同时间段的课堂表现。"
    )
    parameters = [
        ToolParameter("csv_path_a", "string", "第一个 CSV 文件路径（基准）", required=True),
        ToolParameter("csv_path_b", "string", "第二个 CSV 文件路径（对比）", required=True),
    ]

    def execute(self, csv_path_a: str, csv_path_b: str) -> Dict[str, Any]:
        for path in [csv_path_a, csv_path_b]:
            if not os.path.exists(path):
                return {"error": f"文件不存在: {path}"}

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.analysis_module.metrics_calculator import MetricsCalculator

        mc = MetricsCalculator()

        def load_and_compute(path):
            df = pd.read_csv(path)
            df_m = mc.calculate_frame_metrics(df)
            overall = mc.calculate_overall_metrics(df_m)
            return overall

        metrics_a = load_and_compute(csv_path_a)
        metrics_b = load_and_compute(csv_path_b)

        comparison = {}
        for key in metrics_a:
            if isinstance(metrics_a[key], (int, float)):
                val_a = metrics_a[key]
                val_b = metrics_b.get(key, 0)
                diff = round(val_b - val_a, 1)
                pct = round(diff / max(val_a, 0.001) * 100, 1)
                comparison[key] = {
                    "baseline": val_a,
                    "comparison": val_b,
                    "diff": diff,
                    "diff_percent": pct,
                }

        return {
            "baseline_file": os.path.basename(csv_path_a),
            "comparison_file": os.path.basename(csv_path_b),
            "comparison": comparison,
        }
