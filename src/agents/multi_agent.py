"""
多 Agent 协作架构（Supervisor 模式 - 深化版）
将课堂学情分析任务拆分给 3 个专职子 Agent，由 Supervisor 统一调度

架构：
  Supervisor（路由器，可连续路由多个子 Agent 形成协作链路）
  ├── video_agent     → 视频管理（列出/信息/分析/CSV列表）
  ├── analysis_agent  → 数据分析（指标/趋势/查询/对比/摘要）
  └── report_agent    → 报告生成（报告/历史检索）
  └── synthesis       → 汇总各子 Agent 结果，产出最终回答

协作链路示例：
  用户："分析视频并生成学情报告"
  → Supervisor 路由 video_agent（分析视频生成 CSV）
  → Supervisor 看到 video_agent 已完成，路由 analysis_agent（基于 CSV 计算指标）
  → Supervisor 看到 analysis_agent 已完成，路由 report_agent（基于指标生成报告）
  → Supervisor 路由 FINISH → synthesis 节点汇总输出

每个子 Agent 都能感知前序 Agent 的输出，形成协作链路而非孤立执行
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
    """多 Agent 状态

    - messages: 累积的消息流（含 user/assistant/tool），用 add_messages reducer
    - next_agent: Supervisor 决定的下一个子 Agent
    - agent_history: 已调用的子 Agent 列表（按顺序）
    - agent_outputs: 每个子 Agent 的最终输出（agent_name -> summary）
    - turn_count: Supervisor 已路由的次数（防止无限循环）
    """
    messages: Annotated[list, add_messages]
    next_agent: str
    agent_history: List[str]
    agent_outputs: Dict[str, str]
    turn_count: int


# 单次会话内 Supervisor 最多路由次数（避免无限循环）
MAX_SUPERVISOR_TURNS = 5


# ============================================================
# Supervisor Prompt
# ============================================================

SUPERVISOR_PROMPT = """你是课堂学情分析系统的任务路由器（Supervisor）。
你的职责是根据【用户原始问题】、【已调用的子 Agent 列表】和【各子 Agent 的输出】，决定下一步由哪个子 Agent 处理，直到任务真正完成。

可选的子 Agent：
1. **video_agent** - 视频管理：列出视频、获取视频信息、分析视频生成 CSV、列出已有 CSV 文件
2. **analysis_agent** - 数据分析：基于 CSV 计算学情指标、趋势分析、数据查询、对比分析、快速摘要
3. **report_agent** - 报告生成：基于分析结果生成学情报告、检索历史报告
4. **FINISH** - 所有必要工作都已完成，进入最终汇总

协作链路设计原则：
- 复杂任务应拆分给多个子 Agent 顺序执行，形成协作链路
- 典型链路：video_agent（生成 CSV）→ analysis_agent（基于 CSV 计算指标）→ report_agent（基于指标生成报告）→ FINISH
- 每次只路由 1 个子 Agent，子 Agent 完成后由你重新决策下一步
- 已调用过的子 Agent 可以再次调用（如需补充数据）
- 若用户的请求已被现有子 Agent 的输出完全覆盖，路由 FINISH
- 简单任务（如只需一步）路由 1 次子 Agent 后即可 FINISH

请只返回一个 JSON 对象，格式如下：
{"next": "video_agent" | "analysis_agent" | "report_agent" | "FINISH", "reason": "简要说明为什么这样决策"}

注意：只返回 JSON，不要有其他内容。"""

SUB_AGENT_PROMPTS = {
    "video_agent": """你是课堂视频管理专家。你可以列出视频、获取视频详情、分析视频生成 CSV、列出已有 CSV 文件。

工作要求：
1. 使用你的工具完成用户的请求
2. 若收到前序 Agent 的输出作为上下文，请基于其结果工作（例如 analysis_agent 已计算指标，你可以基于其结论生成报告）
3. 完成后给出简洁清晰的回答，包含关键数据/路径""",

    "analysis_agent": """你是课堂数据分析专家。你可以计算学情指标、分析趋势、查询原始数据、对比分析、快速摘要。

工作要求：
1. 使用你的工具完成用户的请求
2. 若前序 Agent（如 video_agent）已生成 CSV 数据，请优先使用该数据进行分析
3. 给出包含数据支撑的专业分析，包含具体数值""",

    "report_agent": """你是课堂学情报告专家。你可以生成专业的学情分析报告，也可以检索历史报告。

工作要求：
1. 使用你的工具完成用户的请求
2. 若前序 Agent（如 analysis_agent）已完成数据分析，请基于其分析结论生成报告，不要重复计算
3. 给出格式化的报告内容，包含关键结论和建议""",
}

# 最终汇总 Prompt：在所有子 Agent 完成后，由 Supervisor 合成最终回答
SYNTHESIS_PROMPT = """你是课堂学情分析系统的最终汇总器。
请基于用户原始问题与各子 Agent 的输出，合成一份结构清晰、面向用户的最终回答。

要求：
1. 直接回应用户问题，不要赘述协作过程
2. 整合各子 Agent 的关键数据、结论、报告路径
3. 使用 Markdown 格式，包含标题、要点列表
4. 若某个子 Agent 执行失败，简要说明影响
5. 篇幅适中（300-800 字），避免冗长"""


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
        """构建 Supervisor 多 Agent 图（含连续路由 + 最终汇总）"""
        graph = StateGraph(MultiAgentState)

        # 添加节点
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("video_agent", self._sub_agent_node("video_agent"))
        graph.add_node("analysis_agent", self._sub_agent_node("analysis_agent"))
        graph.add_node("report_agent", self._sub_agent_node("report_agent"))
        graph.add_node("synthesis", self._synthesis_node)

        # 入口
        graph.set_entry_point("supervisor")

        # Supervisor 条件边：路由到子 Agent / 进入 synthesis / 强制结束
        graph.add_conditional_edges(
            "supervisor",
            self._route,
            {
                "video_agent": "video_agent",
                "analysis_agent": "analysis_agent",
                "report_agent": "report_agent",
                "FINISH": "synthesis",
            },
        )

        # 子 Agent 执行完后回到 Supervisor（形成连续路由循环）
        graph.add_edge("video_agent", "supervisor")
        graph.add_edge("analysis_agent", "supervisor")
        graph.add_edge("report_agent", "supervisor")

        # synthesis 节点为终点
        graph.add_edge("synthesis", END)

        return graph.compile(checkpointer=self._memory)

    # ---------------- 节点实现 ----------------

    def _supervisor_node(self, state: MultiAgentState) -> Dict:
        """Supervisor 节点：基于用户问题 + 已调用 Agent 历史，决定下一步路由"""
        turn_count = state.get("turn_count", 0) or 0
        agent_history: List[str] = list(state.get("agent_history", []) or [])
        agent_outputs: Dict[str, str] = dict(state.get("agent_outputs", {}) or {})

        # 强制结束条件：超过最大路由次数
        if turn_count >= MAX_SUPERVISOR_TURNS:
            return {"next_agent": "FINISH"}

        # 提取当前用户问题（取最后一条 HumanMessage，避免多轮对话时取到历史 query）
        original_query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                original_query = msg.content
                break

        # 构造 Supervisor 上下文：用户问题 + 各 Agent 已完成的工作
        context_parts = [f"【用户原始问题】\n{original_query}"]
        if agent_history:
            context_parts.append("\n【已调用的子 Agent 及其输出】")
            for i, name in enumerate(agent_history, 1):
                output = agent_outputs.get(name, "(无输出)")
                # 截断过长输出，避免 token 浪费
                if len(output) > 800:
                    output = output[:800] + "..."
                context_parts.append(f"{i}. {name}:\n{output}")
        else:
            context_parts.append("\n【已调用的子 Agent 及其输出】\n（尚无子 Agent 被调用）")

        # 取最近几条消息作为对话流上下文（限制长度）
        recent_messages = state["messages"][-4:]

        messages = (
            [SystemMessage(content=SUPERVISOR_PROMPT)]
            + [HumanMessage(content="\n\n".join(context_parts))]
            + recent_messages
        )

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

            # 防御：若同一个 agent 已经连续调用 2 次，强制 FINISH（避免死循环）
            if (
                next_agent != "FINISH"
                and len(agent_history) >= 2
                and agent_history[-1] == next_agent
                and agent_history[-2] == next_agent
            ):
                next_agent = "FINISH"
        except Exception:
            next_agent = "FINISH"

        return {
            "next_agent": next_agent,
            "turn_count": turn_count + 1,
        }

    def _sub_agent_node(self, agent_name: str):
        """生成子 Agent 节点函数

        每个子 Agent 执行后：
        1. 将其最终输出记录到 agent_outputs（供后续 Agent 引用）
        2. 将其加入 agent_history（供 Supervisor 决策）
        """
        def node_fn(state: MultiAgentState) -> Dict:
            llm = self._sub_agent_llms[agent_name]
            prompt = SUB_AGENT_PROMPTS[agent_name]

            # 构造子 Agent 的消息：system prompt + 前序 Agent 的输出 + 当前消息流
            agent_outputs: Dict[str, str] = dict(state.get("agent_outputs", {}) or {})
            prior_context_parts: List[str] = []
            for prior_name, prior_output in agent_outputs.items():
                if prior_name == agent_name:
                    continue
                if len(prior_output) > 500:
                    prior_output = prior_output[:500] + "..."
                prior_context_parts.append(f"[前序 Agent: {prior_name} 的输出]\n{prior_output}")

            messages: List[BaseMessage] = [SystemMessage(content=prompt)]
            if prior_context_parts:
                messages.append(
                    HumanMessage(
                        content="以下是前序子 Agent 的执行输出，请基于它们工作（若相关）：\n\n"
                        + "\n\n".join(prior_context_parts)
                    )
                )
            # 当前消息流（含 user 问题 + 之前的 assistant/tool 消息）
            messages.extend(state["messages"])

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
                    followup_messages = messages + [response] + tool_results
                    final_response = llm.invoke(followup_messages)
                    new_messages = [response] + tool_results + [final_response]
                    output_text = final_response.content or ""
                else:
                    new_messages = [response]
                    output_text = response.content or ""

                # 更新 agent_outputs / agent_history
                # 用 agent_name 作为 key（多次调用会被覆盖为最新输出）
                new_outputs = dict(agent_outputs)
                new_outputs[agent_name] = output_text[:2000]  # 限制存储长度

                new_history = list(state.get("agent_history", []) or [])
                new_history.append(agent_name)

                return {
                    "messages": new_messages,
                    "agent_outputs": new_outputs,
                    "agent_history": new_history,
                }
            except Exception as e:
                error_msg = AIMessage(content=f"子 Agent {agent_name} 执行出错: {e}")
                new_outputs = dict(agent_outputs)
                new_outputs[agent_name] = f"[执行出错] {e}"
                new_history = list(state.get("agent_history", []) or [])
                new_history.append(agent_name)
                return {
                    "messages": [error_msg],
                    "agent_outputs": new_outputs,
                    "agent_history": new_history,
                }

        return node_fn

    def _synthesis_node(self, state: MultiAgentState) -> Dict:
        """最终汇总节点：基于用户问题 + 各子 Agent 输出，合成最终回答"""
        agent_outputs: Dict[str, str] = dict(state.get("agent_outputs", {}) or {})
        agent_history: List[str] = list(state.get("agent_history", []) or [])

        # 提取当前用户问题（取最后一条 HumanMessage，避免多轮对话时取到历史 query）
        original_query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                original_query = msg.content
                break

        # 若没有任何子 Agent 被调用（用户问题太简单或 Supervisor 直接 FINISH）
        if not agent_history:
            # 直接用 LLM 回答用户问题
            try:
                response = self._llm.invoke(
                    [SystemMessage(content=SYSTEM_PROMPT)]
                    + [m for m in state["messages"] if not isinstance(m, SystemMessage)]
                )
                return {"messages": [response]}
            except Exception as e:
                return {"messages": [AIMessage(content=f"汇总时出错: {e}")]}

        # 构造汇总输入
        context_parts = [f"【用户原始问题】\n{original_query}", "\n【各子 Agent 的输出】"]
        for i, name in enumerate(agent_history, 1):
            output = agent_outputs.get(name, "(无输出)")
            context_parts.append(f"{i}. {name}:\n{output}")

        messages = [
            SystemMessage(content=SYNTHESIS_PROMPT),
            HumanMessage(content="\n\n".join(context_parts)),
        ]

        try:
            response = self._llm.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            # 降级：取最后一个子 Agent 的输出作为最终回答
            last_output = ""
            if agent_history:
                last_output = agent_outputs.get(agent_history[-1], "")
            return {
                "messages": [
                    AIMessage(
                        content=f"{last_output}\n\n[注：最终汇总失败 - {e}]"
                    )
                ]
            }

    def _route(self, state: MultiAgentState) -> str:
        """路由函数：基于 next_agent 字段决定下一步"""
        return state.get("next_agent", "FINISH")

    def _get_config(self) -> Dict:
        # recursion_limit 需 ≥ MAX_SUPERVISOR_TURNS*2+4（每轮 supervisor+sub_agent）
        return {
            "configurable": {"thread_id": self._thread_id},
            "recursion_limit": MAX_SUPERVISOR_TURNS * 3 + 4,
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
        """多 Agent 对话（非流式）

        Supervisor 可连续路由多个子 Agent 形成协作链路，最后由 synthesis 节点汇总
        """
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
                {
                    "messages": messages,
                    "next_agent": "",
                    "agent_history": [],
                    "agent_outputs": {},
                    "turn_count": 0,
                },
                config=self._get_config(),
            )
            final_messages = final_state["messages"]
            agent_history = final_state.get("agent_history", []) or []

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
                "collaboration_chain": agent_history,  # 协作链路（已调用的子 Agent 顺序）
            }
        except Exception as e:
            traceback.print_exc()
            return {
                "success": False,
                "message": f"多 Agent 模式出错: {e}",
                "conversation": self.conversation_history,
            }

    def chat_stream(self, user_message: str, thread_id: str = None) -> Generator:
        """多 Agent 对话（流式）

        Supervisor 连续路由多个子 Agent，最后由 synthesis 节点汇总
        流式事件中包含 collaboration_chain，方便前端展示协作过程
        """
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

        collaboration_chain: List[str] = []

        try:
            for event in self._graph.stream(
                {
                    "messages": messages,
                    "next_agent": "",
                    "agent_history": [],
                    "agent_outputs": {},
                    "turn_count": 0,
                },
                config=self._get_config(),
                stream_mode="updates",
            ):
                for node_name, node_output in event.items():
                    if node_name == "supervisor":
                        next_agent = node_output.get("next_agent", "")
                        turn = node_output.get("turn_count", 0)
                        if next_agent and next_agent != "FINISH":
                            yield {
                                "type": "status",
                                "data": "routing",
                                "message": f"[Turn {turn}] Supervisor 路由到子 Agent: {next_agent}",
                                "collaboration_chain": list(collaboration_chain),
                            }
                        elif next_agent == "FINISH":
                            yield {
                                "type": "status",
                                "data": "synthesizing",
                                "message": f"协作链路完成 ({len(collaboration_chain)} 个子 Agent)，开始汇总...",
                                "collaboration_chain": list(collaboration_chain),
                            }

                    # 子 Agent 执行后，更新协作链
                    if node_name in ("video_agent", "analysis_agent", "report_agent"):
                        new_history = node_output.get("agent_history", []) or []
                        if new_history and new_history[-1] == node_name and node_name not in collaboration_chain:
                            collaboration_chain.append(node_name)
                        elif new_history and new_history[-1] == node_name:
                            # 同一 Agent 再次调用，标记为多次协作
                            collaboration_chain.append(f"{node_name}#2")

                    new_msgs = node_output.get("messages", []) or []
                    for msg in new_msgs:
                        if isinstance(msg, AIMessage):
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                func_names = [tc.get("name", "") for tc in msg.tool_calls]
                                yield {
                                    "type": "tool_call",
                                    "data": {
                                        "tools": func_names,
                                        "agent": node_name,
                                        "collaboration_chain": list(collaboration_chain),
                                    },
                                    "message": f"[{node_name}] 调用工具: {', '.join(func_names)}",
                                }
                            if msg.content and node_name == "synthesis":
                                # synthesis 节点的输出是最终回答
                                yield {"type": "content", "data": msg.content, "final": True}
                            elif msg.content:
                                yield {"type": "content", "data": msg.content, "agent": node_name}

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

            yield {
                "type": "done",
                "data": {"collaboration_chain": list(collaboration_chain)},
                "message": f"多 Agent 协作完成（链路: {' → '.join(collaboration_chain) if collaboration_chain else '直接回答'}）",
            }

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
