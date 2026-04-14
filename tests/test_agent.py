"""Tests for MCP tools and speculative rollout."""
from __future__ import annotations
import json
import torch
from hyperlatent.agent.mcp_tools import ToolRegistry, build_default_registry
from hyperlatent.agent.speculative import SpeculativeRolloutEngine
from hyperlatent.memory import SemanticWorldModel

def test_mcp_registry_schema() -> None:
    reg: ToolRegistry = build_default_registry()
    schema = reg.to_mcp_schema()
    assert isinstance(schema, list) and len(schema) >= 1
    for tool in schema:
        assert {"name", "description", "inputSchema"} <= set(tool.keys())
    # Round-trip JSON serialisable.
    json.dumps(schema)

def test_speculative_engine_returns_action() -> None:
    wm = SemanticWorldModel(latent_dim=16, action_dim=4)
    eng = SpeculativeRolloutEngine(world_model=wm, horizon=2, num_samples=4)
    z = torch.randn(16)
    actions = torch.randn(3, 4)  # 3 candidates
    result = eng.plan(z, actions)
    assert 0 <= int(result.best_action_index) < 3
    assert result.values.shape[0] == 3
