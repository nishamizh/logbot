"""
logbot/agent/state_machine.py
──────────────────────────────
Finite State Machine governing LogBot's ReAct agent loop.

States:
  IDLE        → waiting for input
  PLANNING    → LLM generating THOUGHT + ACTION
  EXECUTING   → tool being called
  REFLECTING  → LLM processing OBSERVATION, deciding next step
  RESPONDING  → LLM generating final ANSWER
  DONE        → answer delivered
  ERROR       → unrecoverable failure
  GUARDRAILED → input/output blocked by guardrails

Transitions:
  IDLE        → PLANNING    (on user question)
  PLANNING    → EXECUTING   (ACTION parsed)
  PLANNING    → RESPONDING  (ANSWER parsed — no tool needed)
  EXECUTING   → REFLECTING  (tool result received)
  REFLECTING  → PLANNING    (more steps needed, iterations < max)
  REFLECTING  → RESPONDING  (enough context, ready to answer)
  REFLECTING  → ERROR       (iterations exhausted)
  RESPONDING  → DONE        (answer passes guardrails)
  RESPONDING  → GUARDRAILED (answer blocked)
  ANY         → ERROR       (unhandled exception)
  ANY         → GUARDRAILED (guardrail violation on input)

Design decisions (interview-ready talking points):
  • Explicit FSM beats implicit if/else chains — every transition is
    visible, testable, and auditable.
  • AgentContext is the single mutable object threaded through all states —
    no hidden state, easy to serialize for debugging.
  • transition() validates the edge before applying it — invalid transitions
    raise immediately rather than silently corrupting state.
  • on_enter hooks allow state-specific side effects (logging, metrics)
    without polluting the planner logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from logbot.core.logging import get_logger

log = get_logger(__name__, component="state_machine")


# ──────────────────────────────────────────────────────────────────────────────
# States
# ──────────────────────────────────────────────────────────────────────────────

class AgentState(str, Enum):
    IDLE        = "idle"
    PLANNING    = "planning"
    EXECUTING   = "executing"
    REFLECTING  = "reflecting"
    RESPONDING  = "responding"
    DONE        = "done"
    ERROR       = "error"
    GUARDRAILED = "guardrailed"


# ──────────────────────────────────────────────────────────────────────────────
# Valid transitions  (from_state → set of allowed to_states)
# ──────────────────────────────────────────────────────────────────────────────

VALID_TRANSITIONS: Dict[AgentState, Set[AgentState]] = {
    AgentState.IDLE:        {AgentState.PLANNING, AgentState.GUARDRAILED},
    AgentState.PLANNING:    {AgentState.EXECUTING, AgentState.RESPONDING,
                             AgentState.ERROR, AgentState.GUARDRAILED},
    AgentState.EXECUTING:   {AgentState.REFLECTING, AgentState.ERROR},
    AgentState.REFLECTING:  {AgentState.PLANNING, AgentState.RESPONDING,
                             AgentState.ERROR},
    AgentState.RESPONDING:  {AgentState.DONE, AgentState.GUARDRAILED,
                             AgentState.ERROR},
    AgentState.DONE:        set(),          # terminal
    AgentState.ERROR:       {AgentState.IDLE},   # reset only
    AgentState.GUARDRAILED: {AgentState.IDLE},   # reset only
}

TERMINAL_STATES: Set[AgentState] = {AgentState.DONE, AgentState.ERROR,
                                     AgentState.GUARDRAILED}


# ──────────────────────────────────────────────────────────────────────────────
# Agent context — single mutable object threaded through all states
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    """A parsed tool call from the LLM's ACTION/ACTION_INPUT."""
    tool:       str
    params:     Dict[str, Any]
    result:     Optional[str] = None   # observation string after execution
    elapsed_ms: float         = 0.0
    success:    bool          = True


@dataclass
class AgentContext:
    """
    All mutable state for one agent run.
    Passed through every state; serialisable for debugging / audit trail.
    """
    session_id:   str
    question:     str                        # original user question (sanitized)
    raw_question: str                        # before guardrails

    # Conversation history — list of (role, content) pairs
    history:      List[Tuple[str, str]]      = field(default_factory=list)

    # ReAct loop state
    iterations:   int                        = 0
    max_iterations: int                      = 5
    tool_calls:   List[ToolCall]             = field(default_factory=list)
    llm_outputs:  List[str]                  = field(default_factory=list)

    # Final output
    answer:       Optional[str]              = None
    error_msg:    Optional[str]              = None
    guardrail_violation: Optional[str]       = None

    # Timing
    started_at:   float                      = field(default_factory=time.time)
    ended_at:     Optional[float]            = None

    @property
    def elapsed_ms(self) -> float:
        end = self.ended_at or time.time()
        return round((end - self.started_at) * 1000, 2)

    @property
    def iterations_remaining(self) -> int:
        return self.max_iterations - self.iterations

    @property
    def is_exhausted(self) -> bool:
        return self.iterations >= self.max_iterations

    def add_observation(self, tool: str, observation: str) -> None:
        """Append a tool observation to history."""
        self.history.append(("observation", f"[{tool}] {observation}"))

    def build_conversation(self) -> str:
        """Render full conversation history as a string for the LLM."""
        parts = []
        for role, content in self.history:
            if role == "llm":
                parts.append(content)
            elif role == "observation":
                parts.append(f"OBSERVATION: {content}")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id":   self.session_id,
            "question":     self.question,
            "iterations":   self.iterations,
            "tool_calls":   len(self.tool_calls),
            "has_answer":   self.answer is not None,
            "elapsed_ms":   self.elapsed_ms,
            "error":        self.error_msg,
            "guardrailed":  self.guardrail_violation,
        }


# ──────────────────────────────────────────────────────────────────────────────
# State machine
# ──────────────────────────────────────────────────────────────────────────────

class AgentStateMachine:
    """
    Governs valid state transitions for the ReAct agent loop.

    Usage:
        fsm = AgentStateMachine(session_id="abc123")
        ctx = fsm.start(question="Why is payments.log erroring?")

        # Planner calls:
        fsm.transition(AgentState.PLANNING)
        fsm.transition(AgentState.EXECUTING)
        fsm.transition(AgentState.REFLECTING)
        fsm.transition(AgentState.RESPONDING)
        fsm.transition(AgentState.DONE)

        assert fsm.is_done
    """

    def __init__(
        self,
        session_id:     str,
        max_iterations: int = 5,
        on_enter:       Optional[Dict[AgentState, Callable]] = None,
    ) -> None:
        self._session_id     = session_id
        self._max_iterations = max_iterations
        self._state          = AgentState.IDLE
        self._on_enter       = on_enter or {}
        self._ctx: Optional[AgentContext] = None
        self._transition_log: List[Tuple[float, AgentState, AgentState]] = []

        log.info("fsm_created", session_id=session_id, max_iterations=max_iterations)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, question: str, raw_question: str = "") -> AgentContext:
        """
        Create an AgentContext and transition IDLE → PLANNING.
        Call this once per agent run.
        """
        if self._state != AgentState.IDLE:
            raise RuntimeError(
                f"Cannot start: FSM is in state {self._state}, expected IDLE. "
                "Call reset() first."
            )
        self._ctx = AgentContext(
            session_id=self._session_id,
            question=question,
            raw_question=raw_question or question,
            max_iterations=self._max_iterations,
        )
        self.transition(AgentState.PLANNING)
        return self._ctx

    def transition(self, to: AgentState, reason: str = "") -> None:
        """
        Attempt a state transition. Raises ValueError on invalid edge.
        Fires the on_enter hook for the new state if registered.
        """
        allowed = VALID_TRANSITIONS.get(self._state, set())
        if to not in allowed:
            raise ValueError(
                f"Invalid transition: {self._state} → {to}. "
                f"Allowed from {self._state}: {allowed}"
            )

        from_state   = self._state
        self._state  = to
        self._transition_log.append((time.time(), from_state, to))

        log.debug(
            "fsm_transition",
            from_state=from_state.value,
            to_state=to.value,
            reason=reason or "—",
            session_id=self._session_id,
        )

        # Fire on_enter hook
        if to in self._on_enter:
            try:
                self._on_enter[to](self._ctx)
            except Exception as exc:
                log.error("on_enter_hook_failed", state=to.value, error=str(exc))

    def mark_iteration(self) -> None:
        """Increment iteration counter. Called after each PLANNING→EXECUTING cycle."""
        if self._ctx:
            self._ctx.iterations += 1
            log.debug("iteration_marked",
                      iteration=self._ctx.iterations,
                      max=self._ctx.max_iterations,
                      session_id=self._session_id)

    def set_answer(self, answer: str) -> None:
        if self._ctx:
            self._ctx.answer   = answer
            self._ctx.ended_at = time.time()

    def set_error(self, error_msg: str) -> None:
        if self._ctx:
            self._ctx.error_msg = error_msg
            self._ctx.ended_at  = time.time()
        self._state = AgentState.ERROR
        log.error("fsm_error", error=error_msg, session_id=self._session_id)

    def set_guardrailed(self, violation: str) -> None:
        if self._ctx:
            self._ctx.guardrail_violation = violation
            self._ctx.ended_at            = time.time()
        self._state = AgentState.GUARDRAILED
        log.warning("fsm_guardrailed", violation=violation,
                    session_id=self._session_id)

    def reset(self) -> None:
        """Return FSM to IDLE. Only valid from ERROR or GUARDRAILED."""
        if self._state not in (AgentState.ERROR, AgentState.GUARDRAILED,
                               AgentState.DONE, AgentState.IDLE):
            raise RuntimeError(f"Cannot reset from state {self._state}")
        self._state = AgentState.IDLE
        self._ctx   = None
        log.info("fsm_reset", session_id=self._session_id)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def context(self) -> Optional[AgentContext]:
        return self._ctx

    @property
    def is_done(self) -> bool:
        return self._state in TERMINAL_STATES

    @property
    def is_terminal(self) -> bool:
        return self._state in TERMINAL_STATES

    @property
    def transition_log(self) -> List[Tuple[float, AgentState, AgentState]]:
        return list(self._transition_log)

    def summary(self) -> Dict[str, Any]:
        transitions = [
            f"{f.value}→{t.value}" for _, f, t in self._transition_log
        ]
        return {
            "session_id":  self._session_id,
            "state":       self._state.value,
            "transitions": transitions,
            "context":     self._ctx.to_dict() if self._ctx else None,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def create_fsm(
    session_id:     str,
    max_iterations: int = 5,
) -> AgentStateMachine:
    """Create an FSM with standard logging hooks on every state."""

    def make_hook(state: AgentState) -> Callable:
        def hook(ctx: Optional[AgentContext]) -> None:
            log.info(
                "fsm_enter_state",
                state=state.value,
                session_id=ctx.session_id if ctx else "?",
                iterations=ctx.iterations if ctx else 0,
            )
        return hook

    hooks = {state: make_hook(state) for state in AgentState}
    return AgentStateMachine(
        session_id=session_id,
        max_iterations=max_iterations,
        on_enter=hooks,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test  →  python -m logbot.agent.state_machine
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from logbot.core.logging import configure_logging
    configure_logging()
    log.info("smoke_test_start")

    # ── 1. Happy path: full ReAct loop ────────────────────────────────────────
    fsm = create_fsm("test-session-001", max_iterations=5)
    assert fsm.state == AgentState.IDLE

    ctx = fsm.start("Why is payments.log showing errors?")
    assert fsm.state == AgentState.PLANNING
    assert ctx.session_id == "test-session-001"
    assert ctx.question == "Why is payments.log showing errors?"

    # Iteration 1: plan → execute → reflect
    fsm.mark_iteration()
    fsm.transition(AgentState.EXECUTING, reason="ACTION: get_anomalies")
    assert fsm.state == AgentState.EXECUTING

    ctx.add_observation("get_anomalies", '{"alerts": [{"severity": "critical"}]}')
    fsm.transition(AgentState.REFLECTING, reason="tool result received")
    assert fsm.state == AgentState.REFLECTING
    assert ctx.iterations == 1

    # Iteration 2: reflect → plan → execute → reflect
    fsm.transition(AgentState.PLANNING, reason="need more context")
    fsm.mark_iteration()
    fsm.transition(AgentState.EXECUTING, reason="ACTION: search_logs")
    ctx.add_observation("search_logs", '{"results": [...]}')
    fsm.transition(AgentState.REFLECTING)

    # Done: reflect → respond → done
    fsm.transition(AgentState.RESPONDING, reason="enough context")
    fsm.set_answer("**Severity**: CRITICAL\n**Root Cause**: DB timeout")
    fsm.transition(AgentState.DONE)

    assert fsm.state == AgentState.DONE
    assert fsm.is_done
    assert ctx.answer is not None
    assert ctx.iterations == 2
    assert ctx.elapsed_ms > 0

    summary = fsm.summary()
    print(f"\n── Happy path summary ──")
    print(f"  transitions: {' → '.join(summary['transitions'])}")
    print(f"  iterations:  {ctx.iterations}")
    print(f"  elapsed_ms:  {ctx.elapsed_ms}")
    print("✅  Happy path passed")

    # ── 2. Invalid transition raises ──────────────────────────────────────────
    fsm2 = create_fsm("test-session-002")
    fsm2.start("test question")
    try:
        fsm2.transition(AgentState.DONE)   # PLANNING → DONE is invalid
        assert False, "Should have raised"
    except ValueError as e:
        assert "Invalid transition" in str(e)
        print("✅  Invalid transition raises ValueError")

    # ── 3. Error path ─────────────────────────────────────────────────────────
    fsm3 = create_fsm("test-session-003")
    fsm3.start("test question")
    fsm3.set_error("LLM API timeout after 30s")
    assert fsm3.state == AgentState.ERROR
    assert fsm3.is_done
    assert fsm3.context.error_msg == "LLM API timeout after 30s"
    fsm3.reset()
    assert fsm3.state == AgentState.IDLE
    print("✅  Error path + reset passed")

    # ── 4. Guardrail path ─────────────────────────────────────────────────────
    fsm4 = create_fsm("test-session-004")
    fsm4.start("test question")
    fsm4.set_guardrailed("Injection attempt detected")
    assert fsm4.state == AgentState.GUARDRAILED
    assert fsm4.is_done
    assert "Injection" in fsm4.context.guardrail_violation
    fsm4.reset()
    assert fsm4.state == AgentState.IDLE
    print("✅  Guardrail path + reset passed")

    # ── 5. Iteration exhaustion ────────────────────────────────────────────────
    fsm5 = create_fsm("test-session-005", max_iterations=2)
    ctx5 = fsm5.start("test question")
    for _ in range(2):
        fsm5.mark_iteration()
        fsm5.transition(AgentState.EXECUTING)
        fsm5.transition(AgentState.REFLECTING)
        if not ctx5.is_exhausted:
            fsm5.transition(AgentState.PLANNING)

    assert ctx5.is_exhausted
    assert ctx5.iterations_remaining == 0
    fsm5.transition(AgentState.RESPONDING)
    fsm5.set_answer("Max iterations reached. Based on available data: ...")
    fsm5.transition(AgentState.DONE)
    print("✅  Iteration exhaustion handled")

    # ── 6. Transition log ─────────────────────────────────────────────────────
    log_entries = fsm.transition_log
    assert len(log_entries) > 3
    assert all(len(e) == 3 for e in log_entries)  # (timestamp, from, to)
    print(f"✅  Transition log has {len(log_entries)} entries")

    # ── 7. Cannot start twice ─────────────────────────────────────────────────
    try:
        fsm2.start("second question")   # fsm2 is in PLANNING state
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "IDLE" in str(e)
        print("✅  Double-start raises RuntimeError")

    print("\n✅  All state_machine.py smoke-tests passed.")
