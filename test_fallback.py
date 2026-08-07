import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.orchestrator import ClassAgent

agent = ClassAgent()

print("测试降级模式 - 列出视频...")
result = agent._fallback_handle("列出 uploads 目录下的视频")
print(f"成功: {result['success']}")
print(f"消息: {result['message'][:300]}")
print()

print("测试降级模式 - 快速摘要...")
result2 = agent._fallback_handle("快速看一下数据摘要")
print(f"成功: {result2['success']}")
print(f"消息: {result2['message'][:300]}")
print()

print("测试降级模式 - 生成报告...")
result3 = agent._fallback_handle("生成报告")
print(f"成功: {result3['success']}")
print(f"消息: {result3['message'][:300]}")
