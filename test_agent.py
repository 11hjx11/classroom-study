"""
Agent 全流程测试脚本
验证 Tools 注册、Agent 初始化、工具调用链路
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_tools_import():
    """测试工具模块导入"""
    print("=" * 60)
    print("测试 1: 工具模块导入")
    from src.tools import (
        BaseTool, ToolParameter, ToolRegistry,
        GetVideoInfoTool, AnalyzeVideoTool, ListVideosTool, ListCSVsTool,
        ComputeMetricsTool, AnalyzeTrendTool, QueryCSVDataTool, CompareMetricsTool,
        GenerateReportTool, QuickSummaryTool,
        create_default_registry,
    )
    print("  ✓ 所有工具类导入成功")


def test_registry():
    """测试工具注册表"""
    print("\n" + "=" * 60)
    print("测试 2: 工具注册表")
    from src.tools import create_default_registry

    registry = create_default_registry()
    tools = registry.get_all_tools()
    print(f"  ✓ 注册工具数量: {len(tools)}")
    for t in tools:
        print(f"    - {t.name}: {t.description[:50]}...")

    schemas = registry.get_function_schemas()
    print(f"  ✓ Function Schema 数量: {len(schemas)}")
    for s in schemas:
        print(f"    - {s['function']['name']}")


def test_list_videos():
    """测试列表工具"""
    print("\n" + "=" * 60)
    print("测试 3: 列表工具")
    from src.tools import ListVideosTool, ListCSVsTool

    list_videos = ListVideosTool()
    result = list_videos.run(directory="uploads/")
    print(f"  ✓ list_videos: success={result['success']}")
    if result['success']:
        data = result['data']
        print(f"    目录: {data['directory']}")
        print(f"    视频数: {data['video_count']}")
        for v in data.get('videos', []):
            print(f"      - {v['filename']} ({v['size_mb']}MB)")

    list_csvs = ListCSVsTool()
    result2 = list_csvs.run(directory="cache_csv/")
    print(f"  ✓ list_csvs: success={result2['success']}")
    if result2['success']:
        data = result2['data']
        print(f"    CSV数: {data['csv_count']}")
        for c in data.get('csvs', []):
            print(f"      - {c['filename']}")


def test_agent_init():
    """测试 Agent 初始化"""
    print("\n" + "=" * 60)
    print("测试 4: Agent 初始化")
    from src.agents.orchestrator import ClassAgent

    agent = ClassAgent()
    print(f"  ✓ Agent 初始化成功")
    print(f"  ✓ 可用工具数: {len(agent.get_available_tools())}")
    print(f"  ✓ 欢迎语: {agent.get_greeting()[:80]}...")


def test_get_video_info():
    """测试视频信息获取"""
    print("\n" + "=" * 60)
    print("测试 5: 视频信息获取")
    from src.tools import GetVideoInfoTool

    get_info = GetVideoInfoTool()
    video_path = os.path.join("uploads", "6ba744434fdd2990ae69b8a01b481547.mp4")
    if os.path.exists(video_path):
        result = get_info.run(video_path=video_path)
        print(f"  ✓ get_video_info: success={result['success']}")
        if result['success']:
            data = result['data']
            print(f"    时长: {data['duration_minutes']} 分钟")
            print(f"    FPS: {data['fps']}")
            print(f"    分辨率: {data['width']}x{data['height']}")
    else:
        print(f"  ⚠ 视频文件不存在: {video_path}")
        print("    跳过此测试")


def test_quick_summary():
    """测试快速摘要"""
    print("\n" + "=" * 60)
    print("测试 6: 快速摘要")
    from src.tools import QuickSummaryTool

    csv_path = os.path.join("cache_csv", "6ba744434fdd2990ae69b8a01b481547_raw.csv")
    if os.path.exists(csv_path):
        summary = QuickSummaryTool()
        result = summary.run(csv_path=csv_path)
        print(f"  ✓ quick_summary: success={result['success']}")
        if result['success']:
            data = result['data']
            print(f"    摘要: {data['summary_text']}")
    else:
        print(f"  ⚠ CSV 文件不存在: {csv_path}")
        print("    跳过此测试")


def test_compute_metrics():
    """测试指标计算"""
    print("\n" + "=" * 60)
    print("测试 7: 指标计算")
    from src.tools import ComputeMetricsTool

    csv_path = os.path.join("cache_csv", "6ba744434fdd2990ae69b8a01b481547_raw.csv")
    if os.path.exists(csv_path):
        metrics = ComputeMetricsTool()
        result = metrics.run(csv_path=csv_path)
        print(f"  ✓ compute_metrics: success={result['success']}")
        if result['success']:
            data = result['data']
            overall = data.get('overall_metrics', {})
            print(f"    有效学习率: {overall.get('avg_effective_learning_rate', 'N/A')}%")
            print(f"    走神率: {overall.get('avg_distraction_rate', 'N/A')}%")
            print(f"    总帧数: {data.get('total_frames', 'N/A')}")
    else:
        print(f"  ⚠ CSV 文件不存在: {csv_path}")
        print("    跳过此测试")


def main():
    print("\n" + "#" * 60)
    print("#  ClassVision Agent 全流程测试")
    print("#" * 60 + "\n")

    test_tools_import()
    test_registry()
    test_list_videos()
    test_agent_init()
    test_get_video_info()
    test_quick_summary()
    test_compute_metrics()

    print("\n" + "#" * 60)
    print("#  所有测试完成!")
    print("#" * 60)


if __name__ == '__main__':
    main()
