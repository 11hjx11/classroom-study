"""
Agent Tools 基础模块
定义工具基类、参数 Schema、工具注册机制
所有 Agent 可调用的工具都继承自 BaseTool
"""

import json
import traceback
from typing import Dict, Any, Optional, Callable, List


class ToolParameter:
    """工具参数定义"""

    def __init__(self, name: str, type: str, description: str,
                 required: bool = True, default: Any = None,
                 enum: List[str] = None):
        self.name = name
        self.type = type
        self.description = description
        self.required = required
        self.default = default
        self.enum = enum

    def to_dict(self) -> Dict:
        d = {
            "name": self.name,
            "type": self.type,
            "description": self.description,
        }
        if self.default is not None:
            d["default"] = self.default
        if not self.required:
            d["required"] = False
        if self.enum:
            d["enum"] = self.enum
        return d


class BaseTool:
    """工具基类，所有 Agent 工具继承此类"""

    name: str = "base_tool"
    description: str = "基础工具"
    parameters: List[ToolParameter] = []

    def to_function_schema(self) -> Dict:
        """转换为 LLM function calling schema"""
        properties = {}
        required = []
        for p in self.parameters:
            properties[p.name] = {
                "type": p.type,
                "description": p.description,
            }
            if p.default is not None:
                properties[p.name]["default"] = p.default
            if p.enum:
                properties[p.name]["enum"] = p.enum
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }

    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行工具，子类必须实现"""
        raise NotImplementedError("子类必须实现 execute 方法")

    def run(self, **kwargs) -> Dict[str, Any]:
        """安全执行，捕获异常"""
        try:
            result = self.execute(**kwargs)
            return {
                "success": True,
                "data": result,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }


class ToolRegistry:
    """工具注册表，管理所有可用工具"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def get_all_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    def get_function_schemas(self) -> List[Dict]:
        """获取所有工具的 function calling schema"""
        return [t.to_function_schema() for t in self._tools.values()]

    def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """执行指定工具"""
        tool = self._tools.get(name)
        if tool is None:
            return {
                "success": False,
                "data": None,
                "error": f"工具 '{name}' 不存在"
            }
        return tool.run(**kwargs)
