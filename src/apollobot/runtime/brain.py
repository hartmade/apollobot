"""
ResearchBrain — the autonomous reasoning engine.

Each tick, assembles full runtime context (state, history, memory,
constraints) into a layered prompt, asks the LLM what to do next,
and parses the structured response into actions.

Directly analogous to OpenCat's AgentBrain.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from apollobot.agents import LLMProvider
from apollobot.runtime.config import RuntimeConfig
from apollobot.runtime.storage import RunnerStorage
from apollobot.runtime.types import (
    ActionRecord,
    ActionType,
    BrainAction,
    BrainDecision,
    DecisionRecord,
    RunnerState,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
LLM_CALL_MAX_ATTEMPTS = 3
LLM_CALL_BASE_DELAY = 1  # seconds; doubles each retry

# Exception types that should NOT be retried (budget/auth problems).
_NON_RETRYABLE_SUBSTRINGS = ("authentication", "auth", "api key", "budget", "quota", "billing")


class ResearchBrain:
    """
    Autonomous reasoning engine for continuous research.

    Assembles layered context and asks the LLM to decide:
    - What to research next (or whether to wait)
    - How long to wait before next check-in
    - What to remember for future ticks
    """

    def __init__(
        self,
        llm: LLMProvider,
        storage: RunnerStorage,
        config: RuntimeConfig,
    ) -> None:
        self.llm = llm
        self.storage = storage
        self.config = config
        self.memory: dict[str, str] = {}

    async def load_memory(self) -> None:
        """Load persistent memory from storage."""
        self.memory = self.storage.load_memory()
        logger.info("Brain loaded %d memory entries", len(self.memory))

    @staticmethod
    def _is_non_retryable(exc: Exception) -> bool:
        """Return True for auth/budget errors that should not be retried."""
        msg = str(exc).lower()
        return any(s in msg for s in _NON_RETRYABLE_SUBSTRINGS)

    async def reason(self, state: RunnerState) -> BrainDecision:
        """
        Core reasoning loop — called once per tick.

        Returns actions + next_check_in + memory updates.
        Retries transient LLM failures with exponential backoff.
        """
        system_prompt = self._build_system_prompt(state)
        user_message = self._build_user_message(state)

        last_exc: Exception | None = None
        for attempt in range(1, LLM_CALL_MAX_ATTEMPTS + 1):
            try:
                raw = await self.llm.complete_json(
                    messages=[{"role": "user", "content": user_message}],
                    system=system_prompt,
                    retries=MAX_RETRIES,
                )
                decision = self._parse_decision(raw)
                break  # success
            except Exception as e:
                last_exc = e
                if self._is_non_retryable(e):
                    logger.error("Brain reasoning failed (non-retryable): %s", e)
                    return BrainDecision(
                        actions=[],
                        reasoning=f"Reasoning failed: {e}",
                        next_check_in=self.config.error_interval,
                    )
                if attempt < LLM_CALL_MAX_ATTEMPTS:
                    delay = LLM_CALL_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Brain LLM call failed (attempt %d/%d), retrying in %ds: %s",
                        attempt,
                        LLM_CALL_MAX_ATTEMPTS,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
        else:
            # All attempts exhausted — return safe fallback
            logger.error(
                "Brain reasoning failed after %d attempts: %s",
                LLM_CALL_MAX_ATTEMPTS,
                last_exc,
            )
            return BrainDecision(
                actions=[],
                reasoning="LLM unavailable \u2014 backing off",
                next_check_in=600,
            )

        # Persist memory updates
        if decision.memory:
            self.memory.update(decision.memory)
            self.storage.save_memory(self.memory)

        # Record decision in history
        self.storage.record_decision(
            DecisionRecord(
                tick=state.tick_number,
                reasoning=decision.reasoning,
                actions=[a.type.value for a in decision.actions],
                next_check_in=decision.next_check_in,
            )
        )

        return decision

    def record_action_result(
        self, tick: int, action: BrainAction, result: str, details: str = ""
    ) -> None:
        """Record the result of an executed action."""
        self.storage.record_action(
            ActionRecord(
                tick=tick,
                action_type=action.type.value,
                objective=action.objective,
                result=result,
                details=details,
            )
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self, state: RunnerState) -> str:
        parts = [
            "You are ApolloBot, an autonomous scientific research agent running continuously.",
            "Your job is to decide what research to pursue, when to act, and when to wait.",
            "",
            "You operate under strict safety guardrails. You MUST respect them.",
            "You have persistent memory across ticks — use it to track leads, dead ends,",
            "and cross-session insights.",
            "",
            "Be strategic: not every tick needs an action. Good research requires patience.",
            "Consider what you've already investigated, what worked, what didn't,",
            "and what new questions emerged from prior sessions.",
        ]
        return "\n".join(parts)

    def _build_user_message(self, state: RunnerState) -> str:
        # Action history
        recent_actions = self.storage.recent_actions(self.config.memory_window)
        if recent_actions:
            action_lines = []
            for a in recent_actions:
                line = f"  tick {a.tick}: {a.action_type}"
                if a.objective:
                    line += f" — {a.objective[:80]}"
                line += f" [{a.result}]"
                action_lines.append(line)
            action_text = "\n".join(action_lines)
        else:
            action_text = "  No actions taken yet."

        # Decision history
        recent_decisions = self.storage.recent_decisions(self.config.reasoning_window)
        if recent_decisions:
            decision_lines = []
            for d in recent_decisions:
                decision_lines.append(f"  tick {d.tick}: {d.reasoning[:120]}")
            decision_text = "\n".join(decision_lines)
        else:
            decision_text = "  No previous reasoning."

        # Memory
        if state.memory:
            memory_lines = [f"  {k}: {v}" for k, v in state.memory.items()]
            memory_text = "\n".join(memory_lines)
        else:
            memory_text = "  Empty — this is a fresh start."

        # Completed sessions
        if state.completed_sessions:
            session_lines = []
            for s in state.completed_sessions[:10]:
                line = f"  - [{s.domain}] {s.objective[:60]}"
                if s.translation_score > 0:
                    line += f" (translation: {s.translation_score:.1f}/10)"
                if s.key_findings:
                    line += f" findings: {', '.join(s.key_findings[:2])}"
                session_lines.append(line)
            completed_text = "\n".join(session_lines)
        else:
            completed_text = "  None yet."

        # Active sessions
        if state.active_sessions:
            active_lines = [
                f"  - {s.session_id}: {s.objective[:50]} (phase: {s.phase})"
                for s in state.active_sessions
            ]
            active_text = "\n".join(active_lines)
        else:
            active_text = "  None running."

        # User instructions
        instructions = ""
        if self.config.user_instructions:
            instructions = f"\n## Operator Instructions\n{self.config.user_instructions}\n"

        prompt = f"""## Identity
- Agent: ApolloBot continuous researcher
- Domain focus: {state.domain}
- Tick: {state.tick_number}
- Uptime: {state.uptime_seconds / 3600:.1f} hours

## Research State
- Active sessions: {len(state.active_sessions)}
- Completed papers: {state.total_papers}
- Total cost: ${state.total_cost_usd:.2f}

## Active Sessions
{active_text}

## Completed Sessions (most recent)
{completed_text}

## Guardrails (MUST NOT be violated)
- Daily compute budget remaining: ${state.guardrails_remaining_budget:.2f}
- Max concurrent sessions: {state.guardrails_max_concurrent}
- Daily sessions started today: {state.daily_sessions_started}
- Watchdog state: {state.watchdog_state}

## Recent Actions (last {self.config.memory_window})
{action_text}

## Previous Reasoning (last {self.config.reasoning_window})
{decision_text}

## Persistent Memory
{memory_text}
{instructions}
Evaluate the current state and decide what to do next.

Respond with JSON matching this schema:
{{
  "actions": [
    {{
      "type": "start_research | scan_literature | review_session | idle",
      "objective": "Research question (required for start_research)",
      "mode": "hypothesis | exploratory | meta-analysis | replication | simulation",
      "domain": "{state.domain}",
      "reasoning": "Why this specific action now"
    }}
  ],
  "reasoning": "Overall decision explanation",
  "nextCheckIn": 300,
  "memory": {{
    "key": "value to persist across ticks"
  }}
}}

Rules:
- "actions" can be empty if no action is needed (prefer this over idle action)
- "reasoning" is required — explain your logic
- "nextCheckIn" in seconds — how long to wait before next evaluation
- "memory" — key-value pairs to remember; omit to keep existing memory unchanged
- Be strategic: not every tick needs action. Quality over quantity.
- Build on prior sessions — pursue promising leads, avoid dead ends"""

        return prompt

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_decision(self, raw: dict) -> BrainDecision:
        """Parse the LLM JSON response into a BrainDecision."""
        actions = []
        for a in raw.get("actions", []):
            action_type_str = a.get("type", "idle")
            try:
                action_type = ActionType(action_type_str)
            except ValueError:
                logger.warning("Unknown action type: %s, treating as idle", action_type_str)
                action_type = ActionType.IDLE

            actions.append(
                BrainAction(
                    type=action_type,
                    objective=a.get("objective", ""),
                    mode=a.get("mode", "hypothesis"),
                    domain=a.get("domain", self.config.domain),
                    session_id=a.get("session_id", ""),
                    reasoning=a.get("reasoning", ""),
                )
            )

        return BrainDecision(
            actions=actions,
            reasoning=raw.get("reasoning", ""),
            next_check_in=int(raw.get("nextCheckIn", self.config.default_interval)),
            memory=raw.get("memory", {}),
        )
