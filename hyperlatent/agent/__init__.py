"""Agent subsystems: speculative rollouts, MCP tools, ReAct loop, critic."""

from hyperlatent.agent.speculative import SpeculativeRolloutEngine
from hyperlatent.agent.mcp_tools import Tool, ToolRegistry
from hyperlatent.agent.react_loop import ReActAgent
from hyperlatent.agent.critic import SelfCorrectionCritic

__all__ = [
    "SpeculativeRolloutEngine",
    "Tool",
    "ToolRegistry",
    "ReActAgent",
    "SelfCorrectionCritic",
]