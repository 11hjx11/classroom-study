"""
Agent 性能基准测试
对比单 Agent vs 多 Agent 模式的响应时间、Token 消耗、工具调用次数
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 确保有 API Key
if not os.environ.get("QWEN_API_KEY"):
    print("请先设置环境变量 QWEN_API_KEY")
    sys.exit(1)

from src.agents.orchestrator import ClassAgent
from src.agents.multi_agent import MultiAgentOrchestrator

# 测试用例
TEST_CASES = [
    {"name": "简单查询", "message": "列出 uploads 目录下的视频"},
    {"name": "单工具调用", "message": "对第一个 CSV 文件做快速摘要"},
    {"name": "多工具调用", "message": "列出所有视频并告诉我第一个视频的信息"},
    {"name": "分析任务", "message": "计算第一个 CSV 的学情指标"},
]


def benchmark_single_agent():
    """单 Agent 基准测试"""
    print("\n" + "=" * 60)
    print("单 Agent 模式 (LangGraph ReAct)")
    print("=" * 60)

    agent = ClassAgent()
    results = []

    for case in TEST_CASES:
        print(f"\n测试: {case['name']}")
        print(f"  输入: {case['message']}")

        start = time.time()
        result = agent.chat(case["message"])
        elapsed = time.time() - start

        usage = agent.get_last_usage()
        success = result.get("success", False)
        msg_len = len(result.get("message", ""))

        print(f"  耗时: {elapsed:.2f}s")
        print(f"  成功: {success}")
        print(f"  Token: input={usage['input_tokens']}, output={usage['output_tokens']}, total={usage['total_tokens']}")
        print(f"  工具调用: {usage['tool_calls']} 次")
        print(f"  回复长度: {msg_len} 字符")

        results.append({
            "name": case["name"],
            "elapsed": round(elapsed, 2),
            "success": success,
            "usage": usage,
            "response_length": msg_len,
        })

    return results


def benchmark_multi_agent():
    """多 Agent 基准测试"""
    print("\n" + "=" * 60)
    print("多 Agent 模式 (Supervisor)")
    print("=" * 60)

    try:
        agent = MultiAgentOrchestrator()
    except Exception as e:
        print(f"多 Agent 初始化失败: {e}")
        return []

    results = []

    for case in TEST_CASES:
        print(f"\n测试: {case['name']}")
        print(f"  输入: {case['message']}")

        start = time.time()
        result = agent.chat(case["message"])
        elapsed = time.time() - start

        usage = agent.get_last_usage()
        success = result.get("success", False)
        msg_len = len(result.get("message", ""))

        print(f"  耗时: {elapsed:.2f}s")
        print(f"  成功: {success}")
        print(f"  Token: input={usage['input_tokens']}, output={usage['output_tokens']}, total={usage['total_tokens']}")
        print(f"  工具调用: {usage['tool_calls']} 次")
        print(f"  回复长度: {msg_len} 字符")

        results.append({
            "name": case["name"],
            "elapsed": round(elapsed, 2),
            "success": success,
            "usage": usage,
            "response_length": msg_len,
        })

    return results


def print_comparison(single_results, multi_results):
    """打印对比报告"""
    print("\n" + "#" * 60)
    print("#  性能对比报告")
    print("#" * 60)

    print(f"\n{'测试用例':<15} {'单Agent耗时':>12} {'多Agent耗时':>12} {'单Agent Token':>15} {'多Agent Token':>15}")
    print("-" * 75)

    for i, case in enumerate(TEST_CASES):
        s = single_results[i] if i < len(single_results) else {}
        m = multi_results[i] if i < len(multi_results) else {}

        s_time = f"{s.get('elapsed', 0):.2f}s" if s else "N/A"
        m_time = f"{m.get('elapsed', 0):.2f}s" if m else "N/A"
        s_token = str(s.get('usage', {}).get('total_tokens', 0)) if s else "N/A"
        m_token = str(m.get('usage', {}).get('total_tokens', 0)) if m else "N/A"

        print(f"{case['name']:<15} {s_time:>12} {m_time:>12} {s_token:>15} {m_token:>15}")

    # 汇总
    if single_results:
        s_avg_time = sum(r["elapsed"] for r in single_results) / len(single_results)
        s_avg_token = sum(r["usage"]["total_tokens"] for r in single_results) / len(single_results)
        print(f"\n单 Agent 平均: {s_avg_time:.2f}s, {s_avg_token:.0f} tokens")

    if multi_results:
        m_avg_time = sum(r["elapsed"] for r in multi_results) / len(multi_results)
        m_avg_token = sum(r["usage"]["total_tokens"] for r in multi_results) / len(multi_results)
        print(f"多 Agent 平均: {m_avg_time:.2f}s, {m_avg_token:.0f} tokens")


def main():
    print("Agent 性能基准测试")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型: qwen3-max")

    single_results = benchmark_single_agent()
    multi_results = benchmark_multi_agent()
    print_comparison(single_results, multi_results)

    # 保存结果
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "qwen3-max",
        "single_agent": single_results,
        "multi_agent": multi_results,
    }

    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
