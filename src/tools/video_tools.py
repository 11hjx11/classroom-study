"""
Agent 工具集 - 视频处理相关工具
封装 VideoSampler、StudentDetector、CSVSaver 为 Agent 可调用的 Tool
"""

import os
import sys
import yaml
import pandas as pd
from typing import Dict, Any

from .base import BaseTool, ToolParameter


class GetVideoInfoTool(BaseTool):
    """获取视频基本信息（时长、帧率、分辨率等）"""

    name = "get_video_info"
    description = "获取视频的基本信息，包括时长、帧率、总帧数、分辨率等"
    parameters = [
        ToolParameter("video_path", "string", "视频文件的绝对路径", required=True),
    ]

    def execute(self, video_path: str) -> Dict[str, Any]:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.cv_module.video_sampler import VideoSampler

        sampler = VideoSampler(config_path=self._find_config())
        info = sampler.get_video_info(video_path)
        if info is None:
            return {"error": f"无法打开视频: {video_path}"}

        return {
            "video_path": video_path,
            "duration_seconds": round(info["duration"], 1),
            "duration_minutes": round(info["duration"] / 60, 1),
            "fps": round(info["fps"], 1),
            "total_frames": info["total_frames"],
            "width": info["width"],
            "height": info["height"],
        }

    def _find_config(self) -> str:
        candidates = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "config.yaml"),
            os.path.join(os.getcwd(), "config.yaml"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return "config.yaml"


class AnalyzeVideoTool(BaseTool):
    """完整视频分析工具：采样 → 检测 → 分类 → 保存 CSV"""

    name = "analyze_video"
    description = (
        "对课堂视频进行完整分析：抽帧采样、学生检测、头部姿态估计、"
        "行为分类，最终输出 CSV 数据文件。支持指定时间段进行分析。"
    )
    parameters = [
        ToolParameter("video_path", "string", "视频文件的绝对路径", required=True),
        ToolParameter("time_range", "string", "分析时间段，格式 '开始秒-结束秒'，如 '0-900' 表示前15分钟。不指定则分析全程", required=False, default=None),
        ToolParameter("frame_interval", "integer", "抽帧间隔（秒），默认3秒", required=False, default=3),
    ]

    def execute(self, video_path: str, time_range: str = None,
                frame_interval: int = 3) -> Dict[str, Any]:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.cv_module.video_sampler import VideoSampler
        from src.cv_module.student_detector import StudentDetector
        from src.cv_module.csv_saver import CSVSaver

        config = self._load_config()

        sampler = VideoSampler(config_path=self._find_config())
        detector = StudentDetector(config)

        output_dir = config.get("output", {}).get("cache_csv", "cache_csv/")
        saver = CSVSaver(output_dir=output_dir)

        video_info = sampler.get_video_info(video_path)
        if video_info is None:
            return {"error": f"无法打开视频: {video_path}"}

        frames = sampler.get_sample_frames(video_path)

        if time_range:
            try:
                start_sec, end_sec = map(float, time_range.split("-"))
                frames = [f for f in frames if start_sec <= f["timestamp"] <= end_sec]
            except ValueError:
                return {"error": f"time_range 格式错误: {time_range}，应为 '开始秒-结束秒'"}

        results = []
        for frame_info in frames:
            result = detector.process_frame(
                frame_info["frame"],
                frame_info["frame_num"],
                frame_info["timestamp"],
            )
            result["is_valid"] = True
            result["invalid_reason"] = "valid"
            if "total_students" in result and "total_stu" not in result:
                result["total_stu"] = result["total_students"]
            results.append(result)

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = saver.save_frame_data(results, video_name)

        total_s = sum(r.get("total_stu", 0) for r in results)
        behavior_cols = StudentDetector.BEHAVIOR_COLS
        behavior_totals = {}
        for col in behavior_cols:
            behavior_totals[col] = sum(
                r.get("behavior_counts", {}).get(col, 0) for r in results
            )

        return {
            "video_path": video_path,
            "video_info": video_info,
            "total_frames_processed": len(results),
            "total_student_detections": total_s,
            "avg_students_per_frame": round(total_s / max(len(results), 1), 1),
            "behavior_distribution": behavior_totals,
            "csv_path": output_path,
        }

    def _find_config(self) -> str:
        candidates = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "config.yaml"),
            os.path.join(os.getcwd(), "config.yaml"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return "config.yaml"

    def _load_config(self) -> Dict:
        config_path = self._find_config()
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}


class ListVideosTool(BaseTool):
    """列出指定目录下的视频文件"""

    name = "list_videos"
    description = "列出指定目录下所有支持的视频文件（mp4、avi、mov）"
    parameters = [
        ToolParameter("directory", "string", "要扫描的目录路径，默认使用 uploads/ 目录", required=False, default="uploads/"),
    ]

    def execute(self, directory: str = "uploads/") -> Dict[str, Any]:
        if not os.path.exists(directory):
            return {"videos": [], "message": f"目录不存在: {directory}"}

        videos = []
        for f in sorted(os.listdir(directory)):
            if f.lower().endswith((".mp4", ".avi", ".mov")):
                full_path = os.path.join(directory, f)
                size_mb = round(os.path.getsize(full_path) / (1024 * 1024), 1)
                videos.append({
                    "filename": f,
                    "path": full_path,
                    "size_mb": size_mb,
                })

        return {
            "directory": os.path.abspath(directory),
            "video_count": len(videos),
            "videos": videos,
        }


class ListCSVsTool(BaseTool):
    """列出 cache_csv 目录下已有的 CSV 数据文件"""

    name = "list_csvs"
    description = "列出 cache_csv 目录下已生成的 CSV 数据文件，可用于查询历史分析结果"
    parameters = [
        ToolParameter("directory", "string", "CSV 目录路径，默认 cache_csv/", required=False, default="cache_csv/"),
    ]

    def execute(self, directory: str = "cache_csv/") -> Dict[str, Any]:
        if not os.path.exists(directory):
            return {"csvs": [], "message": f"目录不存在: {directory}"}

        csvs = []
        for f in sorted(os.listdir(directory)):
            if f.endswith("_raw.csv"):
                full_path = os.path.join(directory, f)
                csvs.append({
                    "filename": f,
                    "path": full_path,
                    "size_bytes": os.path.getsize(full_path),
                })

        return {
            "directory": os.path.abspath(directory),
            "csv_count": len(csvs),
            "csvs": csvs,
        }
