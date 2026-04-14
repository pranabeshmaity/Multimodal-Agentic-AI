"""ReAct-style agent loop with speculative planning and self-correction.

The `ReActAgent` interleaves Thought -> Action -> Observation steps. Before
each action it queries the `SpeculativeRolloutEngine` to pick the best
candidate via latent-space simulation. After every action the
`SelfCorrectionCritic` scores the observation; low scores trigger a re-plan
with the critique appended to the running context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from hyperlatent.agent.critic import Critique, SelfCorrectionCritic
from hyperlatent.agent.mcp_tools import Tool, ToolRegistry
from hyperlatent.agent.speculative import SpeculativeRolloutEngine


@dataclass
class ReActStep:
    """One (thought, action, observation) record from the trajectory.

    Attributes:
        thought: The rationale string emitted before the action.
        action_name: Name of the tool invoked.
        action_input: Payload passed to the tool.
        observation: Raw observation returned by the tool.
        critique: Self-correction critique of this step.
        replanned: Whether this step was produced after a re-plan.
        predicted_value: Value estimate from the speculative rollout.
    """

    thought: str
    action_name: str
    action_input: Dict[str, Any]
    observation: Any
    critique: Optional[Critique] = None
    replanned: bool = False
    predicted_value: float = 0.0


@dataclass
class ReActCandidate:
    """A candidate action considered during planning.

    Attributes:
        name: Tool name.
        action_input: Payload to pass to the tool.
        action_embedding: Learned embedding of this candidate.
        thought: Rationale associated with the candidate.
    """

    name: str
    action_input: Dict[str, Any]
    action_embedding: torch.Tensor
    thought: str = ""


ObservationEncoder = Callable[[Any], torch.Tensor]
CandidateProposer = Callable[[torch.Tensor, List[str]], List[ReActCandidate]]


class ReActAgent:
    """Thought -> Action -> Observation loop with self-correction.

    The agent delegates candidate generation to a user-provided
    `candidate_proposer` so that symbolic planners or an LLM can be plugged
    in. The speculative engine selects among candidates by simulating `K`
    latent futures, and the critic re-plans when confidence is too low.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        speculative_engine: SpeculativeRolloutEngine,
        critic: SelfCorrectionCritic,
        observation_encoder: ObservationEncoder,
        candidate_proposer: CandidateProposer,
        max_steps: int = 8,
        max_replans: int = 2,
    ) -> None:
        """Initialize the agent.

        Args:
            tool_registry: Registry containing all callable tools.
            speculative_engine: Planner used to pick the best candidate.
            critic: Self-correction critic used after each action.
            observation_encoder: Function mapping raw tool observations to
                fixed-dim latent tensors.
            candidate_proposer: Callable returning candidate actions for the
                current latent and tool-name list.
            max_steps: Hard cap on loop iterations.
            max_replans: Max re-plans per step before giving up.
        """
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if max_replans < 0:
            raise ValueError("max_replans must be non-negative")

        self.tools = tool_registry
        self.engine = speculative_engine
        self.critic = critic
        self.encode_obs = observation_encoder
        self.propose = candidate_proposer
        self.max_steps = max_steps
        self.max_replans = max_replans

        self.trajectory: List[ReActStep] = []
        self.context: List[str] = []

    # --------------------------------------------------------------- helpers
    def _pick_tool(self, name: str) -> Tool:
        return self.tools.get(name)

    def _plan(
        self, z: torch.Tensor, candidates: List[ReActCandidate]
    ) -> Tuple[int, float]:
        """Select the best candidate via speculative rollouts."""
        stack = torch.stack([c.action_embedding for c in candidates], dim=0)
        result = self.engine.plan(z, stack)
        return result.best_action_index, result.best_value

    # ----------------------------------------------------------------- loop
    def step(self, z: torch.Tensor, goal: str) -> ReActStep:
        """Run a single Thought->Action->Observation cycle with self-correction.

        Args:
            z: Current latent observation `(latent_dim,)`.
            goal: High-level goal string exposed to the candidate proposer
                and prepended to the context.

        Returns:
            The completed `ReActStep`.
        """
        if not self.context or self.context[0] != f"GOAL: {goal}":
            self.context.insert(0, f"GOAL: {goal}")

        tool_names = [t.name for t in self.tools.list()]
        candidates = self.propose(z, tool_names)
        if not candidates:
            raise RuntimeError("candidate_proposer returned no candidates")

        idx, value = self._plan(z, candidates)
        chosen = candidates[idx]
        tool = self._pick_tool(chosen.name)
        observation = tool.invoke(chosen.action_input)

        obs_vec = self.encode_obs(observation)
        critique = self.critic.critique(
            z, chosen.action_embedding, obs_vec, action_name=chosen.name
        )

        replanned = False
        attempts = 0
        while (
            not self.critic.is_acceptable(critique) and attempts < self.max_replans
        ):
            attempts += 1
            replanned = True
            self.context.append(f"CRITIQUE[{attempts}]: {critique.text}")
            candidates = self.propose(z, tool_names)
            # Remove the failed candidate to encourage diversity.
            candidates = [c for c in candidates if c.name != chosen.name] or candidates
            idx, value = self._plan(z, candidates)
            chosen = candidates[idx]
            tool = self._pick_tool(chosen.name)
            observation = tool.invoke(chosen.action_input)
            obs_vec = self.encode_obs(observation)
            critique = self.critic.critique(
                z, chosen.action_embedding, obs_vec, action_name=chosen.name
            )

        step = ReActStep(
            thought=chosen.thought,
            action_name=chosen.name,
            action_input=chosen.action_input,
            observation=observation,
            critique=critique,
            replanned=replanned,
            predicted_value=value,
        )
        self.trajectory.append(step)
        self.context.append(
            f"THOUGHT: {step.thought}\nACTION: {step.action_name}\n"
            f"OBSERVATION: {str(step.observation)[:512]}"
        )
        return step

    def run(
        self,
        z0: torch.Tensor,
        goal: str,
        update_latent: Callable[[torch.Tensor, ReActStep], torch.Tensor],
        is_done: Optional[Callable[[ReActStep], bool]] = None,
    ) -> List[ReActStep]:
        """Drive the agent to completion.

        Args:
            z0: Initial latent state.
            goal: High-level task description.
            update_latent: Callable `(prev_z, step) -> new_z` invoked after
                each step so the agent can incorporate the observation.
            is_done: Optional termination predicate; if omitted the loop
                runs until `max_steps` or until the critic accepts a step
                whose observation contains the literal key ``"done": True``.

        Returns:
            The full trajectory.
        """
        self.trajectory = []
        self.context = []
        z = z0
        for _ in range(self.max_steps):
            step = self.step(z, goal)
            z = update_latent(z, step)
            if is_done is not None:
                if is_done(step):
                    break
            else:
                obs = step.observation
                if isinstance(obs, dict) and obs.get("done") is True:
                    break
        return self.trajectory

    def reset(self) -> None:
        """Clear trajectory and context for a new episode."""
        self.trajectory = []
        self.context = []
