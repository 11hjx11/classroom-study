"""
Agent 评估体系
对课堂学情分析 Agent 进行端到端能力评估

评估指标：
  1. 任务完成率 (Task Completion Rate)  - 是否成功返回有效回答
  2. 工具调用准确率 (Tool Accuracy)     - 期望工具是否被正确触发
  3. 关键词命中率 (Keyword Accuracy)    - 期望关键词是否出现在回答中
  4. 综合准确率 (Overall Accuracy)      - 上述指标的加权综合

覆盖 11 个工具 + 5 大类意图（列表查询 / 数据分析 / 报告生成 / 历史检索 / 通用问答）
共 20 条测试 query

用法：
  $env:QWEN_API_KEY = "..."
  python evaluate.py                # 跑全部 20 条
  python evaluate.py --limit 5      # 只跑前 5 条（快速验证）
  python evaluate.py --case 3       # 只跑第 3 条
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if not os.environ.get("QWEN_API_KEY"):
    print("[ERROR] 未设置 QWEN_API_KEY 环境变量")
    sys.exit(1)

from src.agents.orchestrator import ClassAgent


# ============================================================
# 测试集：20 条 query，覆盖全部 11 个工具
# ============================================================

TEST_CASES: List[Dict] = [
    # --- 列表查询类 (5 条) ---
    {
        "id": 1,
        "category": "list_query",
        "query": "列出 uploads 目录下的视频",
        "expected_tools": ["list_videos"],
        "expected_keywords": ["视频"],
    },
    {
        "id": 2,
        "category": "list_query",
        "query": "列出所有可用的 CSV 数据文件",
        "expected_tools": ["list_csvs"],
        "expected_keywords": ["csv"],
    },
    {
        "id": 3,
        "category": "list_query",
        "query": "查看 cache_csv 目录里有什么文件",
        "expected_tools": ["list_csvs"],
        "expected_keywords": ["csv", "文件"],
    },
    {
        "id": 4,
        "category": "list_query",
        "query": "有什么视频文件可以分析",
        "expected_tools": ["list_videos"],
        "expected_keywords": ["视频"],
    },
    {
        "id": 5,
        "category": "list_query",
        "query": "获取 uploads 目录下第一个视频的信息",
        "expected_tools": ["list_videos", "get_video_info"],
        "expected_keywords": ["视频"],
    },

    # --- 数据分析类 (6 条) ---
    {
        "id": 6,
        "category": "analysis",
        "query": "快速摘要一下第一个 CSV 文件",
        "expected_tools": ["list_csvs", "quick_summary"],
        "expected_keywords": ["摘要"],
    },
    {
        "id": 7,
        "category": "analysis",
        "query": "计算第一个 CSV 的学情指标",
        "expected_tools": ["list_csvs", "compute_metrics"],
        "expected_keywords": ["指标"],
    },
    {
        "id": 8,
        "category": "analysis",
        "query": "分析第一个 CSV 的趋势变化",
        "expected_tools": ["list_csvs", "analyze_trend"],
        "expected_keywords": ["趋势"],
    },
    {
        "id": 9,
        "category": "analysis",
        "query": "查询第一个 CSV 文件的前 100 行原始数据",
        "expected_tools": ["list_csvs", "query_csv_data"],
        "expected_keywords": ["数据"],
    },
    {
        "id": 10,
        "category": "analysis",
        "query": "对比前两个 CSV 文件的学情表现",
        "expected_tools": ["list_csvs", "compare_metrics"],
        "expected_keywords": ["对比"],
    },
    {
        "id": 11,
        "category": "analysis",
        "query": "课堂学生的整体学情情况如何",
        "expected_tools": ["list_csvs", "quick_summary"],
        "expected_keywords": ["学情"],
    },

    # --- 报告生成类 (3 条) ---
    {
        "id": 12,
        "category": "report",
        "query": "生成第一个 CSV 的分析报告",
        "expected_tools": ["list_csvs", "generate_report"],
        "expected_keywords": ["报告"],
    },
    {
        "id": 13,
        "category": "report",
        "query": "把第一个 CSV 的分析结果整理成报告",
        "expected_tools": ["list_csvs", "generate_report"],
        "expected_keywords": ["报告"],
    },
    {
        "id": 14,
        "category": "report",
        "query": "学生专注度的时间段分布是怎样的",
        "expected_tools": ["list_csvs", "analyze_trend"],
        "expected_keywords": ["专注"],
    },

    # --- 历史检索类 (3 条) ---
    {
        "id": 15,
        "category": "retrieval",
        "query": "搜索历史报告中关于专注度的内容",
        "expected_tools": ["search_history"],
        "expected_keywords": ["历史"],
    },
    {
        "id": 16,
        "category": "retrieval",
        "query": "查找之前关于走神率的分析报告",
        "expected_tools": ["search_history"],
        "expected_keywords": ["走神"],
    },
    {
        "id": 17,
        "category": "retrieval",
        "query": "在历史报告里搜索学情趋势相关内容",
        "expected_tools": ["search_history"],
        "expected_keywords": ["历史"],
    },

    # --- 视频处理类 (2 条) ---
    {
        "id": 18,
        "category": "video",
        "query": "分析 uploads 目录下的第一个视频",
        "expected_tools": ["list_videos", "analyze_video"],
        "expected_keywords": ["视频"],
    },
    {
        "id": 19,
        "category": "video",
        "query": "帮我处理 uploads 里的课堂视频",
        "expected_tools": ["list_videos", "analyze_video"],
        "expected_keywords": ["视频"],
    },

    # --- 通用问答类 (1 条) ---
    {
        "id": 20,
        "category": "general",
        "query": "告诉我当前系统支持哪些操作",
        "expected_tools": [],
        "expected_keywords": ["视频", "分析"],
    },
]


# ============================================================
# 评估核心
# ============================================================

class AgentEvaluator:
    """Agent 评估器"""

    MIN_RESPONSE_LENGTH = 30  # 任务完成的最低响应长度

    def __init__(self, agent: ClassAgent):
        self.agent = agent

    def _extract_called_tools(self, conversation: List[Dict]) -> Set[str]:
        """从对话历史中提取实际调用的工具集合"""
        called: Set[str] = set()
        for msg in conversation or []:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    name = tc.get("name") if isinstance(tc, dict) else None
                    if name:
                        called.add(name)
        return called

    def evaluate_one(self, case: Dict) -> Dict:
        """评估单条测试用例"""
        case_id = case["id"]
        query = case["query"]
        expected_tools = set(case.get("expected_tools", []))
        expected_keywords = case.get("expected_keywords", [])

        print(f"\n[Case {case_id:>2}/{len(TEST_CASES)}] ({case['category']}) {query}")
        print("  running...", end=" ", flush=True)

        # 用独立 thread_id，避免污染
        thread_id = f"eval_{case_id}"
        start = time.time()
        try:
            result = self.agent.chat(query, thread_id=thread_id)
        except Exception as e:
            result = {"success": False, "message": f"[eval exception] {e}", "conversation": []}
        elapsed = round(time.time() - start, 2)

        print(f"done ({elapsed}s)")

        success = bool(result.get("success", False))
        message = result.get("message", "") or ""
        conversation = result.get("conversation", []) or []
        usage = self.agent.get_last_usage()
        called_tools = self._extract_called_tools(conversation)

        # --- 任务完成率 ---
        task_completed = success and len(message) >= self.MIN_RESPONSE_LENGTH

        # --- 工具调用准确率 ---
        if expected_tools:
            matched = expected_tools & called_tools
            tool_recall = len(matched) / len(expected_tools)
            # precision: 多调用的工具是否过载
            extra = called_tools - expected_tools
            tool_precision = (
                len(matched) / len(called_tools) if called_tools else 0.0
            )
        else:
            # 没有期望工具的 case：只要不调用任何工具也算准确
            tool_recall = 1.0 if not called_tools else 0.5
            tool_precision = 1.0 if not called_tools else 0.5

        tool_f1 = (
            2 * tool_precision * tool_recall / (tool_precision + tool_recall)
            if (tool_precision + tool_recall) > 0
            else 0.0
        )

        # --- 关键词命中率 ---
        msg_lower = message.lower()
        matched_kw = [kw for kw in expected_keywords if kw.lower() in msg_lower]
        keyword_accuracy = (
            len(matched_kw) / len(expected_keywords) if expected_keywords else 1.0
        )

        # --- 综合判定 ---
        # 通过条件：任务完成 + 工具召回 >= 0.5 + 关键词命中 >= 0.5
        passed = (
            task_completed
            and tool_recall >= 0.5
            and keyword_accuracy >= 0.5
        )

        detail = {
            "id": case_id,
            "category": case["category"],
            "query": query,
            "expected_tools": sorted(expected_tools),
            "called_tools": sorted(called_tools),
            "tool_recall": round(tool_recall, 3),
            "tool_precision": round(tool_precision, 3),
            "tool_f1": round(tool_f1, 3),
            "expected_keywords": expected_keywords,
            "matched_keywords": matched_kw,
            "keyword_accuracy": round(keyword_accuracy, 3),
            "task_completed": task_completed,
            "passed": passed,
            "elapsed_sec": elapsed,
            "usage": usage,
            "response_preview": message[:300] + ("..." if len(message) > 300 else ""),
        }

        status = "PASS" if passed else "FAIL"
        print(
            f"  -> {status} | tools: {sorted(called_tools)} "
            f"(recall={tool_recall:.2f}, prec={tool_precision:.2f}) | "
            f"kw: {matched_kw}/{expected_keywords} ({keyword_accuracy:.2f})"
        )
        return detail


def run_evaluation(limit: int = None, case_id: int = None) -> Dict:
    """执行评估并返回完整报告"""
    print("=" * 70)
    print("Agent 评估体系启动")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型: qwen3-max")
    print(f"测试集大小: {len(TEST_CASES)} 条")
    print("=" * 70)

    agent = ClassAgent()
    evaluator = AgentEvaluator(agent)

    # 选择要跑的用例
    cases = TEST_CASES
    if case_id is not None:
        cases = [c for c in TEST_CASES if c["id"] == case_id]
        if not cases:
            print(f"[ERROR] 未找到 id={case_id} 的用例")
            sys.exit(1)
    elif limit:
        cases = TEST_CASES[:limit]

    results: List[Dict] = []
    for case in cases:
        results.append(evaluator.evaluate_one(case))

    # ---------- 汇总统计 ----------
    total = len(results)
    completed = sum(1 for r in results if r["task_completed"])
    passed = sum(1 for r in results if r["passed"])

    avg_tool_recall = sum(r["tool_recall"] for r in results) / total
    avg_tool_precision = sum(r["tool_precision"] for r in results) / total
    avg_tool_f1 = sum(r["tool_f1"] for r in results) / total
    avg_keyword_acc = sum(r["keyword_accuracy"] for r in results) / total
    avg_elapsed = sum(r["elapsed_sec"] for r in results) / total
    total_tokens = sum(r["usage"].get("total_tokens", 0) for r in results)
    total_tool_calls = sum(r["usage"].get("tool_calls", 0) for r in results)

    # 分类别统计
    categories: Dict[str, Dict] = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "completed": 0}
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1
        if r["task_completed"]:
            categories[cat]["completed"] += 1

    for cat, stats in categories.items():
        stats["pass_rate"] = round(stats["passed"] / stats["total"], 3) if stats["total"] else 0
        stats["completion_rate"] = (
            round(stats["completed"] / stats["total"], 3) if stats["total"] else 0
        )

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "qwen3-max",
        "total_cases": total,
        "summary": {
            "task_completion_rate": round(completed / total, 3),
            "overall_pass_rate": round(passed / total, 3),
            "avg_tool_recall": round(avg_tool_recall, 3),
            "avg_tool_precision": round(avg_tool_precision, 3),
            "avg_tool_f1": round(avg_tool_f1, 3),
            "avg_keyword_accuracy": round(avg_keyword_acc, 3),
            "avg_elapsed_sec": round(avg_elapsed, 2),
            "total_tokens": total_tokens,
            "total_tool_calls": total_tool_calls,
        },
        "by_category": categories,
        "details": results,
    }

    # ---------- 控制台打印 ----------
    print("\n" + "=" * 70)
    print("评估汇总")
    print("=" * 70)
    print(f"任务完成率       : {completed}/{total} = {completed/total:.1%}")
    print(f"综合通过率       : {passed}/{total} = {passed/total:.1%}")
    print(f"工具召回率 (avg) : {avg_tool_recall:.1%}")
    print(f"工具精确率 (avg) : {avg_tool_precision:.1%}")
    print(f"工具 F1    (avg) : {avg_tool_f1:.1%}")
    print(f"关键词命中率(avg): {avg_keyword_acc:.1%}")
    print(f"平均耗时         : {avg_elapsed:.2f}s")
    print(f"总 Token 消耗    : {total_tokens}")
    print(f"总工具调用次数   : {total_tool_calls}")

    print("\n--- 分类别通过率 ---")
    for cat in sorted(categories.keys()):
        s = categories[cat]
        print(
            f"  {cat:<12} : {s['passed']}/{s['total']} "
            f"(pass={s['pass_rate']:.1%}, complete={s['completion_rate']:.1%})"
        )

    # 失败用例详情
    failed = [r for r in results if not r["passed"]]
    if failed:
        print(f"\n--- 失败用例 ({len(failed)} 条) ---")
        for r in failed:
            print(
                f"  #{r['id']} [{r['category']}] {r['query']}\n"
                f"     expected: {r['expected_tools']}, called: {r['called_tools']}, "
                f"tool_recall={r['tool_recall']}, kw_acc={r['keyword_accuracy']}"
            )

    return report


def main():
    parser = argparse.ArgumentParser(description="Agent 评估体系")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="只跑前 N 条用例（快速验证）"
    )
    parser.add_argument(
        "--case", type=int, default=None,
        help="只跑指定 id 的用例"
    )
    parser.add_argument(
        "--output", type=str, default="reports/evaluation_report.json",
        help="报告输出路径"
    )
    args = parser.parse_args()

    report = run_evaluation(limit=args.limit, case_id=args.case)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n报告已保存: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
