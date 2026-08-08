"""
多 Agent 协作架构（Supervisor 模式）
将课堂学情分析任务拆分给 3 个专职子 Agent，由 Supervisor 统一调度

架构：
  Supervisor（路由器）
  ├── video_agent     → 视频管理（列出/信息/分析/CSV列表）
  ├── analysis_agent  → 数据分析（指标/趋势/查询/对比/摘要）
  └── report_agent    → 报告生成（报告/历史检索）

Supervisor 通过 LLM 分类用户意图，路由到对应子 Agent，
子 Agent 独立执行 ReAct 循环，完成后返回 Supervisor 汇总
"""

import json
import os
import traceback
from typing import Any, Dict, Generator, List, Optional

from typing_extensions import Annotated, TypedDict

from src.tools import create_default_registry, ToolRegistry
from src.agents.prompts import SYSTEM_PROMPT, FIRST_GREETING
from src.agents.orchestrator import (
    AgentState,
    _wrap_as_langchain_tool,
    TokenUsage,
    _setup_langsmith,
)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# ============================================================
# 子 Agent 工具分组
# ============================================================

VIDEO_TOOLS = ["list_videos", "get_video_info", "analyze_video", "list_csvs"]
ANALYSIS_TOOLS = ["compute_metrics", "analyze_trend", "query_csv_data", "compare_metrics", "quick_summary"]
REPORT_TOOLS = ["generate_report", "search_history"]


# ============================================================
# Supervisor 状态
# ============================================================

class MultiAgentState(TypedDict):
    """多 Agent 状态"""
    messages: Annotated[list, add_messages]
    next_agent: str  # supervisor 决定的下一个子 agent


# ============================================================
# Supervisor Prompt
# ============================================================

SUPERVISOR_PROMPT = """你是课堂学情分析系统的任务路由器（Supervisor）。
你的职责是根据用户问题，决定由哪个子 Agent 来处理。

可选的子 Agent：
1. **video_agent** - 负责视频相关操作：列出视频、获取视频信息、分析视频、列出CSV数据文件
2. **analysis_agent** - 负责数据分析：计算学情指标、趋势分析、数据查询、对比分析、快速摘要
3. **report_agent** - 负责报告生成：生成学情分析报告、检索历史报告
4. **FINISH** - 任务已完成，不需要调用子 Agent

请只返回一个 JSON 对象，格式如下：
{"next": "video_agent" | "analysis_agent" | "report_agent" | "FINISH"}

注意：只返回 JSON，不要有其他内容。"""

SUB_AGENT_PROMPTS = {
    "video_agent": """你是课堂视频管理专家。你可以列出视频、获取视频详情、分析视频、列出CSV数据文件。
请使用你的工具完成用户的请求，然后给出简洁清晰的回答。""",

    "analysis_agent": """你是课堂数据分析专家。你可以计算学情指标、分析趋势、查询原始数据、对比分析、快速摘要。
请使用你的工具完成用户的请求，然后给出包含数据支撑的专业分析。""",

    "report_agent": """你是课堂学情报告专家。你可以生成专业的学情分析报告，也可以检索历史报告。
请使用你的工具完成用户的请求，然后给出格式化的报告内容。""",
}


# ============================================================
# 多 Agent 编排器
# ============================================================

class MultiAgentOrchestrator:
    """多 Agent 协作编排器（Supervisor 模式）"""

    def __init__(self, api_key: str = None, model: str = "qwen3-max"):
        self.api_key = api_key or os.environ.get("QWEN_API_KEY")
        if not self.api_key:
            raise RuntimeError("未配置通义千问 API Key，请设置环境变量 QWEN_API_KEY")

        self.model = model
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.registry = create_default_registry()
        self.conversation_history: List[Dict] = []
        self._thread_id = "default"
        self._last_usage = TokenUsage()

        # 构建 LLM
        self._llm = ChatOpenAI(
            model=model, api_key=self.api_key, base_url=self.base_url,
            temperature=0.3, max_tokens=4000, timeout=120,
        )

        # 为每个子 Agent 构建专属工具集
        self._sub_agent_tools = self._build_sub_agent_tools()
        self._sub_agent_llms = {}
        for agent_name, tools in self._sub_agent_tools.items():
            if tools:
                self._sub_agent_llms[agent_name] = self._llm.bind_tools(tools)
            else:
                self._sub_agent_llms[agent_name] = self._llm

        self._memory = MemorySaver()
        self._graph = self._build_graph()

    def _build_sub_agent_tools(self) -> Dict[str, List]:
        """为每个子 Agent 构建专属工具集"""
        all_tools = {t.name: _wrap_as_langchain_tool(t) for t in self.registry.get_all_tools()}

        result = {}
        for agent_name, tool_names in [
            ("video_agent", VIDEO_TOOLS),
            ("analysis_agent", ANALYSIS_TOOLS),
            ("report_agent", REPORT_TOOLS),
        ]:
            result[agent_name] = [all_tools[n] for n in tool_names if n in all_tools]

        return result

    def _build_graph(self):
        """构建 Supervisor 多 Agent 图"""
        graph = StateGraph(MultiAgentState)

        # 添加节点
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("video_agent", self._sub_agent_node("video_agent"))
        graph.add_node("analysis_agent", self._sub_agent_node("analysis_agent"))
        graph.add_node("report_agent", self._sub_agent_node("report_agent"))

        # 入口
        graph.set_entry_point("supervisor")

        # Supervisor 条件边：路由到子 Agent 或结束
        graph.add_conditional_edges(
            "supervisor",
            self._route,
            {
                "video_agent": "video_agent",
                "analysis_agent": "analysis_agent",
                "report_agent": "report_agent",
                "FINISH": END,
            },
        )

        # 子 Agent 执行完后回到 Supervisor
        graph.add_edge("video_agent", "supervisor")
        graph.add_edge("analysis_agent", "supervisor")
        graph.add_edge("report_agent", "supervisor")

        return graph.compile(checkpointer=self._memory)

    # ---------------- 节点实现 ----------------

    def _supervisor_node(self, state: MultiAgentState) -> Dict:
        """Supervisor 节点：用 LLM 分类意图，决定路由（不添加消息到状态）"""
        # 只取最后几条消息给 Supervisor 做分类，避免上下文过长
        recent_messages = state["messages"][-6:]
        messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + recent_messages

        try:
            response = self._llm.invoke(messages)
            content = response.content.strip()
            # 尝试提取 JSON
            import re
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                decision = json.loads(json_match.group())
                next_agent = decision.get("next", "FINISH")
            else:
                # 如果没有 JSON，尝试从文本中匹配
                if "video" in content.lower():
                    next_agent = "video_agent"
                elif "analysis" in content.lower():
                    next_agent = "analysis_agent"
                elif "report" in content.lower():
                    next_agent = "report_agent"
                else:
                    next_agent = "FINISH"
        except Exception:
            next_agent = "FINISH"

        # Supervisor 不添加消息到状态，只做路由决策
        return {"next_agent": next_agent}

    def _sub_agent_node(self, agent_name: str):
        """生成子 Agent 节点函数"""
        def node_fn(state: MultiAgentState) -> Dict:
            llm = self._sub_agent_llms[agent_name]
            prompt = SUB_AGENT_PROMPTS[agent_name]

            messages = [SystemMessage(content=prompt)] + state["messages"]

            try:
                response = llm.invoke(messages)

                # 如果有 tool_calls，执行工具
                if hasattr(response, "tool_calls") and response.tool_calls:
                    tool_results = []
                    for tc in response.tool_calls:
                        name = tc.get("name", "")
                        args = tc.get("args") or {}
                        self._last_usage.tool_calls += 1
                        result = self.registry.execute_tool(name, **args)
                        tool_results.append(
                            ToolMessage(
                                content=json.dumps(result, ensure_ascii=False, default=str),
                                name=name,
                                tool_call_id=tc.get("id", f"call_{name}"),
                            )
                        )

                    # 再次调用 LLM 汇总工具结果
                    messages = messages + [response] + tool_results
                    final_response = llm.invoke(messages)
                    return {"messages": [response] + tool_results + [final_response]}

                return {"messages": [response]}
            except Exception as e:
                error_msg = AIMessage(content=f"子 Agent {agent_name} 执行出错: {e}")
                return {"messages": [error_msg]}

        return node_fn

    def _route(self, state: MultiAgentState) -> str:
        """路由函数"""
        return state.get("next_agent", "FINISH")

    def _get_config(self) -> Dict:
        return {
            "configurable": {"thread_id": self._thread_id},
            "recursion_limit": 20,
        }

    # ---------------- 公共接口 ----------------

    def set_thread_id(self, thread_id: str):
        self._thread_id = thread_id

    def reset(self):
        self.conversation_history = []
        self._thread_id = "default"

    def get_available_tools(self) -> List[Dict]:
        return self.registry.get_function_schemas()

    def get_greeting(self) -> str:
        return FIRST_GREETING

    def get_last_usage(self) -> Dict:
        return self._last_usage.to_dict()

    def chat(self, user_message: str, thread_id: str = None) -> Dict:
        """多 Agent 对话（非流式）"""
        if thread_id:
            self.set_thread_id(thread_id)

        self._last_usage = TokenUsage()
        messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in self.conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=user_message))

        try:
            final_state = self._graph.invoke(
                {"messages": messages, "next_agent": ""},
                config=self._get_config(),
            )
            final_messages = final_state["messages"]

            self.conversation_history = self._messages_to_dicts(final_messages[1:])

            last_ai_content = ""
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    last_ai_content = msg.content
                    break

            return {
                "success": True,
                "message": last_ai_content,
                "conversation": self.conversation_history,
                "usage": self._last_usage.to_dict(),
                "mode": "multi_agent",
            }
        except Exception as e:
            traceback.print_exc()
            return {
                "success": False,
                "message": f"多 Agent 模式出错: {e}",
                "conversation": self.conversation_history,
            }

    def chat_stream(self, user_message: str, thread_id: str = None) -> Generator:
        """多 Agent 对话（流式）"""
        if thread_id:
            self.set_thread_id(thread_id)

        self._last_usage = TokenUsage()
        messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in self.conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=user_message))

        yield {"type": "status", "data": "thinking", "message": "Supervisor 正在分析任务..."}

        try:
            for event in self._graph.stream(
                {"messages": messages, "next_agent": ""},
                config=self._get_config(),
                stream_mode="updates",
            ):
                for node_name, node_output in event.items():
                    if node_name == "supervisor":
                        next_agent = node_output.get("next_agent", "")
                        if next_agent and next_agent != "FINISH":
                            yield {
                                "type": "status",
                                "data": "routing",
                                "message": f"路由到子 Agent: {next_agent}",
                            }

                    new_msgs = node_output.get("messages", []) or []
                    for msg in new_msgs:
                        if isinstance(msg, AIMessage):
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                func_names = [tc.get("name", "") for tc in msg.tool_calls]
                                yield {
                                    "type": "tool_call",
                                    "data": {"tools": func_names, "agent": node_name},
                                    "message": f"[{node_name}] 调用工具: {', '.join(func_names)}",
                                }
                            if msg.content:
                                yield {"type": "content", "data": msg.content}

                        elif isinstance(msg, ToolMessage):
                            try:
                                parsed = json.loads(msg.content)
                                yield {
                                    "type": "tool_result",
                                    "data": {
                                        "tool": msg.name,
                                        "success": parsed.get("success", True),
                                        "agent": node_name,
                                    },
                                    "message": f"[{node_name}] {msg.name} 执行{'成功' if parsed.get('success') else '失败'}",
                                }
                            except json.JSONDecodeError:
                                yield {
                                    "type": "tool_result",
                                    "data": {"tool": msg.name, "agent": node_name},
                                    "message": f"[{node_name}] {msg.name} 执行完成",
                                }

            yield {"type": "done", "data": None, "message": "多 Agent 协作完成"}

        except Exception as e:
            traceback.print_exc()
            yield {"type": "error", "data": None, "message": str(e)}
            yield {"type": "done", "data": None, "message": "执行出错"}

    @staticmethod
    def _messages_to_dicts(messages) -> List[Dict]:
        """将消息列表转为字典"""
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                continue
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                result.append({"role": "tool", "content": msg.content, "name": msg.name})
        return result
