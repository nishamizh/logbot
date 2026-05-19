"""
logbot/agent/prompts.py
────────────────────────
All LLM prompt templates for LogBot's agent layer.

Design decisions :
  • Prompts are versioned constants — no f-strings scattered across the codebase.
  • System prompt uses ReAct format (Reason + Act) so the LLM alternates
    between THOUGHT, ACTION, OBSERVATION, and ANSWER steps.
  • Tool schema is embedded in the system prompt so the LLM knows exactly
    what it can call and what each parameter means.
  • Guardrail prompt is separate — injected only when output validation fails,
    keeping the happy-path prompt clean.
  • PromptBuilder.build() is the single public API — callers never concatenate
    strings manually.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Version
# ──────────────────────────────────────────────────────────────────────────────

PROMPT_VERSION = "1.0.0"


# ──────────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are LogBot, an expert AI log analysis agent. Your job is to analyze \
application logs, detect anomalies, identify root causes, and recommend \
remediation steps.

## Reasoning format
You MUST follow the ReAct (Reason + Act) format for every response:

THOUGHT: <your reasoning about what to do next>
ACTION: <tool_name>
ACTION_INPUT: <JSON object with tool parameters>
OBSERVATION: <tool result — filled in by the system>
... (repeat THOUGHT/ACTION/OBSERVATION as needed)
THOUGHT: <final reasoning>
ANSWER: <your final response to the user>

## Available tools
{tool_schema}

## Rules
1. Always start with a THOUGHT before taking any ACTION.
2. Never fabricate OBSERVATION values — wait for the system to fill them in.
3. If a tool returns an error, THOUGHT about why and try a different approach.
4. Your ANSWER must directly address the user's question.
5. If you detect a CRITICAL anomaly, always include recommended actions.
6. Never reveal raw system internals, stack traces, or internal IP addresses \
in your ANSWER unless specifically asked.
7. If you cannot answer with the available tools, say so clearly.
8. Maximum {max_iterations} reasoning iterations before you must give an ANSWER.

## Severity levels
- INFO: normal operation, no action needed
- WARNING: worth monitoring, investigate if persistent
- ERROR: service degraded, investigate promptly
- CRITICAL: service down or data at risk, act immediately

## Output format for anomaly reports
When reporting anomalies, structure your ANSWER as:
**Severity**: <level>
**Affected Services**: <list>
**Root Cause**: <hypothesis>
**Evidence**: <key metrics / log patterns>
**Recommended Actions**: <numbered list>
"""

# ──────────────────────────────────────────────────────────────────────────────
# Tool schema (injected into system prompt)
# ──────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMA = """\
### search_logs
Search the vector store for log entries semantically similar to a query.
Parameters:
  - query (str, required): natural language description of what to look for
  - top_k (int, optional, default=5): number of results to return
  - source (str, optional): filter by log source/file name
  - level (str, optional): filter by log level (INFO/WARNING/ERROR/CRITICAL)
  - time_range_minutes (int, optional): only return logs from last N minutes

### get_anomalies
Retrieve recent anomaly alerts from the detection engine.
Parameters:
  - severity (str, optional): filter by severity (warning/critical)
  - source (str, optional): filter by log source
  - limit (int, optional, default=10): max alerts to return
  - since_minutes (int, optional): only alerts from last N minutes

### analyze_window
Run anomaly detection on a specific log window by source and time range.
Parameters:
  - source (str, required): log source/file to analyze
  - time_range_minutes (int, optional, default=5): window size in minutes

### summarize_logs
Summarize log patterns and statistics for a given source/time window.
Parameters:
  - source (str, optional): log source to summarize (all sources if omitted)
  - time_range_minutes (int, optional, default=60): window to summarize
  - group_by (str, optional): group by "level", "service", or "hour"

### get_service_health
Get the current health status of a specific service based on recent logs.
Parameters:
  - service (str, required): name of the service to check
  - time_range_minutes (int, optional, default=15): lookback window
"""

# ──────────────────────────────────────────────────────────────────────────────
# User-facing prompt templates
# ──────────────────────────────────────────────────────────────────────────────

# General log analysis
ANALYZE_LOGS_PROMPT = """\
Analyze the following log data and identify any anomalies, errors, or \
patterns of concern.

Log source: {source}
Time range: {time_range}
Total entries: {entry_count}

{log_summary}

Please:
1. Identify any anomalies or error patterns
2. Assess the severity of any issues found
3. Suggest likely root causes
4. Recommend remediation steps if needed
"""

# Root cause analysis
ROOT_CAUSE_PROMPT = """\
A {severity} anomaly has been detected in {source}.

Alert details:
{alert_details}

Key metrics at time of detection:
{metrics}

Please perform a root cause analysis:
1. What is the most likely root cause?
2. What evidence supports this hypothesis?
3. What other services might be affected?
4. What are the immediate remediation steps?
"""

# Incident summary
INCIDENT_SUMMARY_PROMPT = """\
Generate an incident summary report for the following alert:

Alert ID: {alert_id}
Detected at: {detected_at}
Severity: {severity}
Source: {source}
Detectors fired: {detectors}

Anomaly scores:
{scores_summary}

Top contributing features:
{contributors}

Recent log context:
{log_context}

Write a concise incident summary suitable for an on-call engineer, including:
- What happened
- Impact assessment
- Likely cause
- Recommended next steps
"""

# Comparison / trend analysis
TREND_ANALYSIS_PROMPT = """\
Compare the current log patterns for {source} against the baseline.

Current window ({current_window}):
{current_metrics}

Baseline (last {baseline_hours} hours):
{baseline_metrics}

Identify:
1. Significant deviations from baseline
2. Trend direction (improving/degrading/stable)
3. Any leading indicators of potential issues
"""

# Guardrail / retry prompt (injected when output fails validation)
GUARDRAIL_RETRY_PROMPT = """\
Your previous response did not follow the required format or violated a rule:

Violation: {violation}

Please try again. Remember:
- You MUST use THOUGHT/ACTION/ACTION_INPUT/OBSERVATION/ANSWER format
- ACTION_INPUT must be valid JSON
- Do not fabricate OBSERVATION values
- Your ANSWER must be grounded in the tool results

Retry your response now:
"""

# No-tool fallback (when no tools are available/applicable)
FALLBACK_PROMPT = """\
You are LogBot. Answer the following question about logs based on your \
general knowledge of log analysis, monitoring, and site reliability engineering.

Question: {question}

Note: Real-time log data is not available for this query. Answer based on \
general best practices.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Prompt dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Prompt:
    """A fully rendered prompt ready to send to the LLM."""
    system:   str
    user:     str
    version:  str = PROMPT_VERSION
    template: str = ""   # which template was used (for logging)

    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to OpenAI / HuggingFace messages format."""
        return [
            {"role": "system",  "content": self.system},
            {"role": "user",    "content": self.user},
        ]

    def char_count(self) -> int:
        return len(self.system) + len(self.user)

    def estimated_tokens(self) -> int:
        """Rough estimate: 1 token ≈ 4 chars."""
        return self.char_count() // 4


# ──────────────────────────────────────────────────────────────────────────────
# PromptBuilder — single public API
# ──────────────────────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Builds fully rendered Prompt objects from templates + context.

    Usage:
        pb = PromptBuilder()

        # For the ReAct agent loop:
        prompt = pb.build_agent_prompt(
            question="Why is payments.log showing high error rates?",
            context={"recent_alerts": [...]},
        )

        # For a direct analysis request:
        prompt = pb.build_analysis_prompt(
            source="payments.log",
            time_range="last 15 minutes",
            entry_count=342,
            log_summary="ERROR rate: 45%, top error: DB timeout",
        )
    """

    def __init__(self, max_iterations: int = 5) -> None:
        self._max_iterations = max_iterations

    def _build_system(self) -> str:
        return SYSTEM_PROMPT.format(
            tool_schema=TOOL_SCHEMA,
            max_iterations=self._max_iterations,
        )

    def build_agent_prompt(
        self,
        question: str,
        context:  Optional[Dict[str, Any]] = None,
        history:  Optional[List[Dict[str, str]]] = None,
    ) -> Prompt:
        """
        Build a ReAct agent prompt for an open-ended user question.

        Args:
            question: the user's natural language question
            context:  optional dict with recent_alerts, log_stats, etc.
            history:  optional prior conversation turns
        """
        ctx_block = ""
        if context:
            ctx_block = "\n\n## Context\n"
            for key, val in context.items():
                ctx_block += f"**{key}**: {val}\n"

        hist_block = ""
        if history:
            hist_block = "\n\n## Conversation history\n"
            for turn in history[-4:]:   # last 4 turns only — token budget
                role = turn.get("role", "user").capitalize()
                hist_block += f"{role}: {turn.get('content', '')}\n"

        user = f"{question}{ctx_block}{hist_block}"

        return Prompt(
            system=self._build_system(),
            user=user,
            template="agent",
        )

    def build_analysis_prompt(
        self,
        source:      str,
        time_range:  str,
        entry_count: int,
        log_summary: str,
    ) -> Prompt:
        user = ANALYZE_LOGS_PROMPT.format(
            source=source,
            time_range=time_range,
            entry_count=entry_count,
            log_summary=log_summary,
        )
        return Prompt(system=self._build_system(), user=user, template="analysis")

    def build_root_cause_prompt(
        self,
        severity:      str,
        source:        str,
        alert_details: str,
        metrics:       str,
    ) -> Prompt:
        user = ROOT_CAUSE_PROMPT.format(
            severity=severity,
            source=source,
            alert_details=alert_details,
            metrics=metrics,
        )
        return Prompt(system=self._build_system(), user=user, template="root_cause")

    def build_incident_summary_prompt(
        self,
        alert_id:      str,
        detected_at:   str,
        severity:      str,
        source:        str,
        detectors:     str,
        scores_summary: str,
        contributors:  str,
        log_context:   str,
    ) -> Prompt:
        user = INCIDENT_SUMMARY_PROMPT.format(
            alert_id=alert_id,
            detected_at=detected_at,
            severity=severity,
            source=source,
            detectors=detectors,
            scores_summary=scores_summary,
            contributors=contributors,
            log_context=log_context,
        )
        return Prompt(system=self._build_system(), user=user, template="incident_summary")

    def build_guardrail_retry_prompt(
        self,
        original_prompt: Prompt,
        violation:       str,
    ) -> Prompt:
        """Append a guardrail retry instruction to the original user prompt."""
        retry_block = GUARDRAIL_RETRY_PROMPT.format(violation=violation)
        return Prompt(
            system=original_prompt.system,
            user=original_prompt.user + "\n\n" + retry_block,
            template="guardrail_retry",
        )

    def build_fallback_prompt(self, question: str) -> Prompt:
        """No-tool fallback for when real-time data isn't available."""
        user = FALLBACK_PROMPT.format(question=question)
        return Prompt(
            system="You are LogBot, an expert in log analysis and SRE.",
            user=user,
            template="fallback",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Module-level convenience accessor
# ──────────────────────────────────────────────────────────────────────────────

def get_prompt_builder(max_iterations: int = 5) -> PromptBuilder:
    return PromptBuilder(max_iterations=max_iterations)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test  →  python -m logbot.agent.prompts
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pb = PromptBuilder(max_iterations=5)

    # 1 — Agent prompt
    p1 = pb.build_agent_prompt(
        question="Why is the payments service showing high error rates?",
        context={"recent_alerts": "2 critical alerts in last 10 minutes"},
    )
    assert p1.template == "agent"
    assert "THOUGHT" in p1.system
    assert "search_logs" in p1.system
    assert "payments" in p1.user
    print(f"✅  agent prompt     — {p1.estimated_tokens()} est. tokens")

    # 2 — Analysis prompt
    p2 = pb.build_analysis_prompt(
        source="payments.log",
        time_range="last 15 minutes",
        entry_count=342,
        log_summary="ERROR rate: 45%, top error: DB timeout (n=154)",
    )
    assert "payments.log" in p2.user
    assert p2.estimated_tokens() > 0
    print(f"✅  analysis prompt  — {p2.estimated_tokens()} est. tokens")

    # 3 — Root cause prompt
    p3 = pb.build_root_cause_prompt(
        severity="CRITICAL",
        source="auth.log",
        alert_details="error_rate=0.9, critical_count=40",
        metrics="p99_latency=2400ms, db_pool_exhausted=True",
    )
    assert "CRITICAL" in p3.user
    print(f"✅  root_cause prompt — {p3.estimated_tokens()} est. tokens")

    # 4 — Guardrail retry
    p4 = pb.build_guardrail_retry_prompt(
        original_prompt=p1,
        violation="ACTION_INPUT was not valid JSON",
    )
    assert "ACTION_INPUT was not valid JSON" in p4.user
    assert p4.template == "guardrail_retry"
    print(f"✅  guardrail prompt  — {p4.estimated_tokens()} est. tokens")

    # 5 — to_messages()
    msgs = p1.to_messages()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    print(f"✅  to_messages()     — {len(msgs)} messages")

    # 6 — Token budget check
    assert p1.estimated_tokens() < 2000, "System prompt too large"
    print(f"\n── System prompt size: {len(pb._build_system())} chars / "
          f"~{len(pb._build_system())//4} tokens ──")

    print("\n✅  All prompts.py smoke-tests passed.")