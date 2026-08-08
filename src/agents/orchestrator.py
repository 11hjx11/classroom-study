"""
课堂学情分析 Agent 编排器 (LangGraph 版本)
基于 LangGraph StateGraph 实现的 ReAct 模式智能体
负责：理解用户意图 → 规划工具调用 → 执行工具 → 汇总结果 → 生成回答

核心特性：
  1. LangGraph StateGraph ReAct 循环（agent ⇄ tools）
  2. MemorySaver 检查点：跨会话状态持久化，支持 thread_id 多会话隔离
  3. LangSmith 可观测性：环境变量自动启用 trace 可视化
  4. Token 用量追踪：每次对话记录 input/output/total tokens
  5. 工具执行重试：tenacity 指数退避，提升容错能力
  6. 异步支持：achat / achat_stream 异步接口
  7. 降级容错：LLM 不可用时自动切换关键词匹配模式
"""

import json
import os
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Sequence

from typing_extensions import Annotated, TypedDict

from src.tools import create_default_registry, ToolRegistry
from src.agents.prompts import SYSTEM_PROMPT, FIRST_GREETING

# LangGraph / LangChain 依赖
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool as LCBaseTool, StructuredTool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# ============================================================
# LangSmith 可观测性（环境变量自动启用）
# ============================================================

def _setup_langsmith():
    """若设置了 LANGCHAIN_API_KEY，自动启用 LangSmith trace"""
    if os.environ.get("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "classroom-study")


_setup_langsmith()


# ============================================================
# Token 用量追踪
# ============================================================

@dataclass
class TokenUsage:
    """单次对话的 Token 用量"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0

    def add(self, input_tokens: int = 0, output_tokens: int = 0):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.llm_calls += 1

    def to_dict(self) -> Dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
        }


class TokenTrackingCallback(BaseCallbackHandler):
    """LangChain 回调：追踪 LLM token 用量"""

    def __init__(self, usage: TokenUsage):
        self.usage = usage

    def on_llm_end(self, response, **kwargs):
        try:
            if response.llm_output and "token_usage" in response.llm_output:
                tu = response.llm_output["token_usage"]
                self.usage.add(
                    input_tokens=tu.get("prompt_tokens", 0),
                    output_tokens=tu.get("completion_tokens", 0),
                )
        except Exception:
            pass


# ============================================================
# LangGraph 状态定义
# ============================================================

class AgentState(TypedDict):
    """LangGraph 状态：消息列表使用 add_messages reducer 自动累加"""
    messages: Annotated[list, add_messages]


# ============================================================
# 工具适配：将项目自有 BaseTool 转换为 LangChain BaseTool
# ============================================================

def _map_json_type(json_type: str) -> type:
    """将 JSON Schema 类型映射为 Python 类型"""
    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }.get(json_type, str)


def _build_args_model(tool) -> type:
    """根据自有工具的 ToolParameter 列表动态构建 Pydantic 模型"""
    from pydantic import Field, create_model

    fields: Dict[str, Any] = {}
    for param in tool.parameters:
        py_type = _map_json_type(param.type)
        description = param.description
        if param.required:
            fields[param.name] = (py_type, Field(..., description=description))
        else:
            default = param.default if param.default is not None else None
            fields[param.name] = (Optional[py_type], Field(default=default, description=description))

    return create_model(f"{tool.name}_ArgsModel", **fields)


def _wrap_as_langchain_tool(original_tool) -> LCBaseTool:
    """将项目自有 BaseTool 包装为 LangChain StructuredTool"""

    args_model = _build_args_model(original_tool)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    def _run(**kwargs):
        return original_tool.run(**kwargs)

    return StructuredTool.from_function(
        name=original_tool.name,
        description=original_tool.description,
        args_schema=args_model,
        func=_run,
    )


# ============================================================
# Agent 主类
# ============================================================

class ClassAgent:
    """课堂学情分析智能体（LangGraph 实现）"""

    def __init__(self, api_key: str = None, model: str = "qwen3-max"):
        self.api_key = api_key or os.environ.get("QWEN_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "未配置通义千问 API Key，请设置环境变量 QWEN_API_KEY 或在初始化时传入 api_key 参数"
            )
        self.model = model
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.registry: ToolRegistry = create_default_registry()
        self.conversation_history: List[Dict] = []
        self._system_prompt = SYSTEM_PROMPT
        self._thread_id: str = "default"
        self._last_usage: TokenUsage = TokenUsage()

        # 初始化 LLM、工具、编译后的图
        self._llm = self._build_llm()
        self._langchain_tools = self._build_langchain_tools()
        self._llm_with_tools = (
            self._llm.bind_tools(self._langchain_tools)
            if self._langchain_tools
            else self._llm
        )
        self._memory = MemorySaver()
        self._graph = self._build_graph()

    # ---------------- 初始化相关 ----------------

    def _build_llm(self) -> ChatOpenAI:
        """构建通义千问 LLM（通过 OpenAI 兼容接口）"""
        return ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.7,
            max_tokens=4000,
            top_p=0.9,
            timeout=120,
        )

    def _build_langchain_tools(self) -> List[LCBaseTool]:
        """将项目自有工具转换为 LangChain BaseTool 列表"""
        return [_wrap_as_langchain_tool(t) for t in self.registry.get_all_tools()]

    def _build_graph(self):
        """构建并编译 LangGraph 状态图（ReAct 模式 + MemorySaver 检查点）"""
        graph = StateGraph(AgentState)

        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)

        graph.set_entry_point("agent")

        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "end": END,
            },
        )

        graph.add_edge("tools", "agent")

        # 编译时传入 MemorySaver，实现状态持久化
        return graph.compile(checkpointer=self._memory)

    # ---------------- LangGraph 节点 ----------------

    def _agent_node(self, state: AgentState) -> Dict:
        """Agent 节点：调用 LLM 进行推理"""
        response: BaseMessage = self._llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    async def _agent_node_async(self, state: AgentState) -> Dict:
        """Agent 节点（异步）：调用 LLM 进行推理"""
        response: BaseMessage = await self._llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    def _tools_node(self, state: AgentState) -> Dict:
        """工具节点：执行 LLM 的工具调用"""
        last_message: AIMessage = state["messages"][-1]
        tool_results: List[ToolMessage] = []
        for tc in last_message.tool_calls:
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
        return {"messages": tool_results}

    async def _tools_node_async(self, state: AgentState) -> Dict:
        """工具节点（异步）：执行 LLM 的工具调用"""
        return self._tools_node(state)

    def _should_continue(self, state: AgentState) -> str:
        """条件边：判断是否继续调用工具"""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "end"

    def _get_config(self) -> Dict:
        """获取 LangGraph 调用配置（含 thread_id 用于会话隔离）"""
        return {
            "configurable": {"thread_id": self._thread_id},
            "recursion_limit": 25,
        }

    # ---------------- 公共接口 ----------------

    def set_thread_id(self, thread_id: str):
        """设置会话线程 ID（用于多会话隔离）"""
        self._thread_id = thread_id

    def reset(self):
        """重置对话历史"""
        self.conversation_history = []
        self._thread_id = "default"

    def get_available_tools(self) -> List[Dict]:
        """获取所有可用工具的 function calling schema"""
        return self.registry.get_function_schemas()

    def get_greeting(self) -> str:
        """获取欢迎语"""
        return FIRST_GREETING

    def get_last_usage(self) -> Dict:
        """获取上次对话的 Token 用量"""
        return self._last_usage.to_dict()

    def chat(self, user_message: str, thread_id: str = None) -> Dict:
        """
        与 Agent 对话（非流式）
        完整流程：用户消息 → LLM → 工具调用 → LLM → ... → 最终回答
        若 LLM 不可用，自动降级为关键词匹配模式
        """
        if thread_id:
            self.set_thread_id(thread_id)

        self._last_usage = TokenUsage()
        messages = self._build_messages(user_message)
        config = self._get_config()

        try:
            final_state = self._graph.invoke(
                {"messages": messages},
                config=config,
            )
            final_messages: List[BaseMessage] = final_state["messages"]

            # 保存对话历史（去掉 system 消息）
            self.conversation_history = self._messages_to_dicts(
                final_messages[1:]
            )

            # 取最后一条 AI 消息作为最终回答
            last_ai_content = ""
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage):
                    last_ai_content = msg.content or ""
                    break

            return {
                "success": True,
                "message": last_ai_content,
                "conversation": self.conversation_history,
                "usage": self._last_usage.to_dict(),
            }
        except Exception as e:
            traceback.print_exc()
            return self._fallback_handle(user_message)

    async def achat(self, user_message: str, thread_id: str = None) -> Dict:
        """与 Agent 对话（异步非流式）"""
        if thread_id:
            self.set_thread_id(thread_id)

        self._last_usage = TokenUsage()
        messages = self._build_messages(user_message)
        config = self._get_config()

        try:
            final_state = await self._graph.ainvoke(
                {"messages": messages},
                config=config,
            )
            final_messages: List[BaseMessage] = final_state["messages"]

            self.conversation_history = self._messages_to_dicts(
                final_messages[1:]
            )

            last_ai_content = ""
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage):
                    last_ai_content = msg.content or ""
                    break

            return {
                "success": True,
                "message": last_ai_content,
                "conversation": self.conversation_history,
                "usage": self._last_usage.to_dict(),
            }
        except Exception as e:
            traceback.print_exc()
            return self._fallback_handle(user_message)

    def chat_stream(self, user_message: str, thread_id: str = None) -> Generator:
        """
        与 Agent 对话（流式）
        逐步 yield 事件，前端通过 SSE 接收
        若 LLM 不可用，自动降级为关键词匹配模式
        """
        if thread_id:
            self.set_thread_id(thread_id)

        self._last_usage = TokenUsage()
        messages = self._build_messages(user_message)
        config = self._get_config()

        yield {"type": "status", "data": "thinking", "message": "正在分析您的问题..."}

        new_messages: List[BaseMessage] = []

        try:
            for event in self._graph.stream(
                {"messages": messages},
                config=config,
                stream_mode="updates",
            ):
                for node_name, node_output in event.items():
                    new_msgs: List[BaseMessage] = node_output.get("messages", []) or []
                    new_messages.extend(new_msgs)

                    if node_name == "agent":
                        for msg in new_msgs:
                            if not isinstance(msg, AIMessage):
                                continue
                            if msg.tool_calls:
                                func_names = [
                                    tc.get("name", "") for tc in msg.tool_calls
                                ]
                                yield {
                                    "type": "tool_call",
                                    "data": {"tools": func_names},
                                    "message": f"调用工具: {', '.join(func_names)}",
                                }
                            if msg.content:
                                yield {"type": "content", "data": msg.content}

                    elif node_name == "tools":
                        yield {
                            "type": "status",
                            "data": "executing_tools",
                            "message": "执行工具调用...",
                        }
                        for msg in new_msgs:
                            if not isinstance(msg, ToolMessage):
                                continue
                            try:
                                parsed = json.loads(msg.content)
                                if parsed.get("success"):
                                    yield {
                                        "type": "tool_result",
                                        "data": {
                                            "tool": msg.name,
                                            "summary": self._summarize_result(
                                                parsed.get("data", {})
                                            ),
                                        },
                                        "message": f"{msg.name} 执行成功",
                                    }
                                else:
                                    yield {
                                        "type": "tool_error",
                                        "data": {
                                            "tool": msg.name,
                                            "error": parsed.get("error", "未知错误"),
                                        },
                                        "message": f"{msg.name} 执行失败",
                                    }
                            except json.JSONDecodeError:
                                yield {
                                    "type": "tool_result",
                                    "data": {"tool": msg.name},
                                    "message": f"{msg.name} 执行完成",
                                }

            # 保存对话历史
            self.conversation_history = self._messages_to_dicts(new_messages)

            yield {"type": "done", "data": None, "message": "分析完成"}

        except Exception as e:
            traceback.print_exc()
            yield {
                "type": "status",
                "data": "fallback",
                "message": "LLM 不可用，使用降级模式...",
            }
            fallback_result = self._fallback_handle(user_message)
            yield {"type": "content", "data": fallback_result.get("message", "")}
            yield {"type": "done", "data": None, "message": "分析完成"}

    async def achat_stream(self, user_message: str, thread_id: str = None) -> AsyncGenerator:
        """与 Agent 对话（异步流式）"""
        if thread_id:
            self.set_thread_id(thread_id)

        self._last_usage = TokenUsage()
        messages = self._build_messages(user_message)
        config = self._get_config()

        yield {"type": "status", "data": "thinking", "message": "正在分析您的问题..."}

        new_messages: List[BaseMessage] = []

        try:
            async for event in self._graph.astream(
                {"messages": messages},
                config=config,
                stream_mode="updates",
            ):
                for node_name, node_output in event.items():
                    new_msgs: List[BaseMessage] = node_output.get("messages", []) or []
                    new_messages.extend(new_msgs)

                    if node_name == "agent":
                        for msg in new_msgs:
                            if not isinstance(msg, AIMessage):
                                continue
                            if msg.tool_calls:
                                func_names = [
                                    tc.get("name", "") for tc in msg.tool_calls
                                ]
                                yield {
                                    "type": "tool_call",
                                    "data": {"tools": func_names},
                                    "message": f"调用工具: {', '.join(func_names)}",
                                }
                            if msg.content:
                                yield {"type": "content", "data": msg.content}

                    elif node_name == "tools":
                        yield {
                            "type": "status",
                            "data": "executing_tools",
                            "message": "执行工具调用...",
                        }
                        for msg in new_msgs:
                            if not isinstance(msg, ToolMessage):
                                continue
                            try:
                                parsed = json.loads(msg.content)
                                if parsed.get("success"):
                                    yield {
                                        "type": "tool_result",
                                        "data": {
                                            "tool": msg.name,
                                            "summary": self._summarize_result(
                                                parsed.get("data", {})
                                            ),
                                        },
                                        "message": f"{msg.name} 执行成功",
                                    }
                                else:
                                    yield {
                                        "type": "tool_error",
                                        "data": {
                                            "tool": msg.name,
                                            "error": parsed.get("error", "未知错误"),
                                        },
                                        "message": f"{msg.name} 执行失败",
                                    }
                            except json.JSONDecodeError:
                                yield {
                                    "type": "tool_result",
                                    "data": {"tool": msg.name},
                                    "message": f"{msg.name} 执行完成",
                                }

            self.conversation_history = self._messages_to_dicts(new_messages)
            yield {"type": "done", "data": None, "message": "分析完成"}

        except Exception as e:
            traceback.print_exc()
            yield {
                "type": "status",
                "data": "fallback",
                "message": "LLM 不可用，使用降级模式...",
            }
            fallback_result = self._fallback_handle(user_message)
            yield {"type": "content", "data": fallback_result.get("message", "")}
            yield {"type": "done", "data": None, "message": "分析完成"}

    # ---------------- 消息构建与转换 ----------------

    def _build_messages(self, user_message: str) -> List[BaseMessage]:
        """构建发送给 LLM 的消息列表（LangChain BaseMessage 格式）"""
        messages: List[BaseMessage] = [SystemMessage(content=self._system_prompt)]
        for msg in self.conversation_history:
            messages.append(self._dict_to_message(msg))
        messages.append(HumanMessage(content=user_message))
        return messages

    @staticmethod
    def _dict_to_message(d: Dict) -> BaseMessage:
        """将消息字典转换为 LangChain BaseMessage"""
        role = d.get("role", "user")
        content = d.get("content", "")

        if role == "system":
            return SystemMessage(content=content)
        if role == "user":
            return HumanMessage(content=content)
        if role == "assistant":
            msg = AIMessage(content=content)
            if d.get("tool_calls"):
                msg.tool_calls = d["tool_calls"]
            return msg
        if role == "tool":
            return ToolMessage(
                content=content,
                name=d.get("name", ""),
                tool_call_id=d.get("tool_call_id", ""),
            )
        return HumanMessage(content=content)

    @staticmethod
    def _messages_to_dicts(messages: Sequence[BaseMessage]) -> List[Dict]:
        """将 LangChain BaseMessage 列表转换为消息字典列表（不含 system 消息）"""
        result: List[Dict] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                continue
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                d: Dict[str, Any] = {"role": "assistant", "content": msg.content}
                if getattr(msg, "tool_calls", None):
                    d["tool_calls"] = msg.tool_calls
                result.append(d)
            elif isinstance(msg, ToolMessage):
                result.append(
                    {
                        "role": "tool",
                        "content": msg.content,
                        "name": msg.name,
                        "tool_call_id": msg.tool_call_id,
                    }
                )
        return result

    # ---------------- 降级模式（LLM 不可用时使用） ----------------

    def _fallback_handle(self, user_message: str) -> Dict:
        """
        降级模式：当 LLM 不可用时，基于关键词匹配直接调用工具
        确保核心功能在无 LLM 时仍可用
        """
        msg = user_message.lower()

        # 关键词匹配规则
        rules = [
            # 视频信息
            (["视频信息", "视频详情", "video info"], "get_video_info", self._guess_video_path(user_message)),
            # 列出视频
            (["列出视频", "视频列表", "list video", "有什么视频"], "list_videos", {"directory": "uploads/"}),
            # 列出CSV
            (["列出csv", "csv列表", "数据列表", "历史数据"], "list_csvs", {"directory": "cache_csv/"}),
            # 快速摘要
            (["摘要", "summary", "概览", "快速看"], "quick_summary", self._guess_csv_path(user_message)),
            # 计算指标
            (["指标", "metrics", "学情", "有效学习率", "走神率"], "compute_metrics", self._guess_csv_path(user_message)),
            # 趋势分析
            (["趋势", "trend", "峰值", "低谷"], "analyze_trend", self._guess_csv_path(user_message)),
            # 生成报告
            (["报告", "report", "生成报告", "分析报告"], "generate_report", self._guess_csv_path(user_message)),
            # 对比分析
            (["对比", "比较", "compare"], "compare_metrics", self._guess_compare_paths(user_message)),
            # 查询数据
            (["查询", "query", "原始数据", "帧数据"], "query_csv_data", self._guess_csv_path(user_message)),
            # 分析视频
            (["分析视频", "分析课堂", "analyze", "处理视频"], "analyze_video", self._guess_video_path(user_message)),
            # RAG 检索
            (["历史", "历史报告", "之前", "搜索报告"], "search_history", {"query": user_message}),
        ]

        for keywords, tool_name, args in rules:
            if any(kw in msg for kw in keywords):
                if isinstance(args, dict) and "csv_path" in args and args["csv_path"] is None:
                    # 没有指定 CSV 路径，先列出可用的
                    csv_result = self.registry.execute_tool("list_csvs", directory="cache_csv/")
                    if csv_result["success"] and csv_result["data"]["csv_count"] > 0:
                        first_csv = csv_result["data"]["csvs"][0]["path"]
                        args["csv_path"] = first_csv
                    else:
                        return {
                            "success": False,
                            "message": "降级模式：未找到可用的 CSV 数据文件。请先分析视频生成数据。",
                            "conversation": self.conversation_history,
                        }
                if isinstance(args, dict) and "video_path" in args and args["video_path"] is None:
                    video_result = self.registry.execute_tool("list_videos", directory="uploads/")
                    if video_result["success"] and video_result["data"]["video_count"] > 0:
                        first_video = video_result["data"]["videos"][0]["path"]
                        args["video_path"] = first_video
                    else:
                        return {
                            "success": False,
                            "message": "降级模式：未找到可用的视频文件。请先上传视频。",
                            "conversation": self.conversation_history,
                        }

                result = self.registry.execute_tool(tool_name, **args)
                return self._format_fallback_result(tool_name, args, result, user_message)

        # 默认：尝试快速摘要第一个CSV
        csv_result = self.registry.execute_tool("list_csvs", directory="cache_csv/")
        if csv_result["success"] and csv_result["data"]["csv_count"] > 0:
            first_csv = csv_result["data"]["csvs"][0]["path"]
            result = self.registry.execute_tool("quick_summary", csv_path=first_csv)
            return self._format_fallback_result("quick_summary", {"csv_path": first_csv}, result, user_message)

        # 最后返回帮助信息
        return {
            "success": True,
            "message": (
                "当前处于降级模式（LLM 不可用）。\n"
                "你可以尝试以下操作：\n"
                "1. 输入「列出视频」查看可用视频\n"
                "2. 输入「分析视频」分析最新视频\n"
                "3. 输入「生成报告」生成学情报告\n"
                "4. 输入「对比」对比两份数据\n"
                "5. 输入「历史报告」检索历史分析\n\n"
                "请更换 API Key 后重启以获得完整 Agent 体验。"
            ),
            "conversation": self.conversation_history,
        }

    def _guess_video_path(self, user_message: str) -> Dict:
        """从用户消息中猜测视频路径"""
        match = re.search(r'[\w]+\.(mp4|avi|mov)', user_message, re.IGNORECASE)
        if match:
            filename = match.group(0)
            path = os.path.join("uploads", filename)
            if os.path.exists(path):
                return {"video_path": path}
        return {"video_path": None}

    def _guess_csv_path(self, user_message: str) -> Dict:
        """从用户消息中猜测 CSV 路径"""
        match = re.search(r'[\w]+_raw\.csv', user_message, re.IGNORECASE)
        if match:
            filename = match.group(0)
            path = os.path.join("cache_csv", filename)
            if os.path.exists(path):
                return {"csv_path": path}
        # 尝试匹配视频名
        match = re.search(r'[\w]+\.(mp4|avi|mov)', user_message, re.IGNORECASE)
        if match:
            video_name = os.path.splitext(match.group(0))[0]
            csv_path = os.path.join("cache_csv", f"{video_name}_raw.csv")
            if os.path.exists(csv_path):
                return {"csv_path": csv_path}
        return {"csv_path": None}

    def _guess_compare_paths(self, user_message: str) -> Dict:
        """从用户消息中猜测两个 CSV 路径"""
        # 查找所有CSV文件引用
        matches = re.findall(r'[\w]+_raw\.csv', user_message, re.IGNORECASE)
        if len(matches) >= 2:
            path_a = os.path.join("cache_csv", matches[0])
            path_b = os.path.join("cache_csv", matches[1])
            if os.path.exists(path_a) and os.path.exists(path_b):
                return {"csv_path_a": path_a, "csv_path_b": path_b}
        # 查找视频文件名
        video_matches = re.findall(r'[\w]+\.(mp4|avi|mov)', user_message, re.IGNORECASE)
        if len(video_matches) >= 2:
            name_a = os.path.splitext(video_matches[0])[0]
            name_b = os.path.splitext(video_matches[1])[0]
            path_a = os.path.join("cache_csv", f"{name_a}_raw.csv")
            path_b = os.path.join("cache_csv", f"{name_b}_raw.csv")
            if os.path.exists(path_a) and os.path.exists(path_b):
                return {"csv_path_a": path_a, "csv_path_b": path_b}
        return {"csv_path_a": None, "csv_path_b": None}

    def _format_fallback_result(self, tool_name: str, args: Dict, result: Dict, user_message: str) -> Dict:
        """格式化降级模式的结果"""
        if not result["success"]:
            return {
                "success": False,
                "message": f"降级模式：工具 {tool_name} 执行失败 - {result.get('error', '未知错误')}",
                "conversation": self.conversation_history,
            }

        data = result["data"]
        tool_names = {
            "list_videos": "📹 视频列表",
            "list_csvs": "📂 数据文件",
            "get_video_info": "🎬 视频信息",
            "quick_summary": "📋 数据摘要",
            "compute_metrics": "📊 学情指标",
            "analyze_trend": "📈 趋势分析",
            "generate_report": "📝 分析报告",
            "compare_metrics": "⚖️ 对比结果",
            "query_csv_data": "🔍 数据查询",
            "analyze_video": "🎥 视频分析",
            "search_history": "📚 历史检索",
        }

        title = tool_names.get(tool_name, tool_name)
        response_parts = [f"**[降级模式] {title}**\n"]

        if tool_name == "list_videos":
            response_parts.append(f"共 {data.get('video_count', 0)} 个视频：\n")
            for v in data.get("videos", []):
                response_parts.append(f"  - {v['filename']} ({v['size_mb']}MB)")
        elif tool_name == "list_csvs":
            response_parts.append(f"共 {data.get('csv_count', 0)} 个数据文件：\n")
            for c in data.get("csvs", []):
                response_parts.append(f"  - {c['filename']}")
        elif tool_name == "quick_summary":
            response_parts.append(f"{data.get('summary_text', '')}")
            overall = data.get("overall", {})
            if overall:
                response_parts.append(f"\n详细指标：")
                for k, v in overall.items():
                    response_parts.append(f"  {k}: {v}")
        elif tool_name == "compute_metrics":
            response_parts.append("**整体指标：**\n")
            for k, v in data.get("overall_metrics", {}).items():
                response_parts.append(f"  {k}: {v}")
            response_parts.append("\n**时段指标：**\n")
            for seg, metrics in data.get("segments", {}).items():
                response_parts.append(f"  {seg}:")
                for k, v in metrics.items():
                    response_parts.append(f"    {k}: {v}")
        elif tool_name == "generate_report":
            response_parts.append(f"报告已生成：{data.get('report_path', '')}\n")
            response_parts.append(f"\n**报告内容预览：**\n")
            content = data.get("report_content", "")
            response_parts.append(content[:2000] + "..." if len(content) > 2000 else content)
        elif tool_name == "analyze_video":
            response_parts.append(f"视频分析完成！\n")
            response_parts.append(f"处理帧数：{data.get('total_frames_processed', 0)}")
            response_parts.append(f"学生检测总数：{data.get('total_student_detections', 0)}")
            response_parts.append(f"CSV 输出：{data.get('csv_path', '')}")
        elif tool_name == "search_history":
            response_parts.append(f"检索到 {data.get('total_results', 0)} 条历史报告：\n")
            for r in data.get("results", []):
                response_parts.append(f"  - **{r['filename']}** (相似度: {r['score']:.2f})")
                response_parts.append(f"    {r['snippet'][:150]}...")
        else:
            response_parts.append(f"执行成功，结果：\n```json\n{json.dumps(data, ensure_ascii=False, indent=2, default=str)[:2000]}\n```")

        response_parts.append("\n---\n💡 提示：当前为降级模式，建议更换 API Key 后使用完整 Agent 对话功能。")

        return {
            "success": True,
            "message": "\n".join(response_parts),
            "conversation": self.conversation_history,
        }

    def _summarize_result(self, data: Any) -> str:
        """将工具结果数据精简为可读摘要"""
        if isinstance(data, dict):
            keys = list(data.keys())[:5]
            parts = []
            for k in keys:
                v = data[k]
                if isinstance(v, (dict, list)):
                    parts.append(f"{k}: {type(v).__name__}")
                elif isinstance(v, float):
                    parts.append(f"{k}: {v:.1f}")
                else:
                    parts.append(f"{k}: {v}")
            return ", ".join(parts)
        if isinstance(data, list):
            return f"{len(data)} 条结果"
        return str(data)
