"""mcp-judge -- a calibrated, multi-persona LLM-as-judge with an MCP server and an eval harness.

Public API::

    from mcp_judge import score, Rubric, Persona, Dimension
    from mcp_judge.personas import get_rubric, list_rubrics

    result = score(document, get_rubric("document_quality"), caller=my_llm_fn)
    print(result.panel_avg, result.passed)
"""
from .judge import (
    ADVANCE_FLOOR,
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    Dimension,
    JudgeError,
    JudgeResult,
    Persona,
    PersonaResult,
    Rubric,
    aggregate,
    anthropic_caller,
    band_for,
    build_prompt,
    extract_json,
    score,
    self_consistent,
    tiered_pass,
    vote_coupled,
    weighted_composite,
)

__all__ = [
    "ADVANCE_FLOOR",
    "DEFAULT_MODEL",
    "DEFAULT_THRESHOLD",
    "Dimension",
    "JudgeError",
    "JudgeResult",
    "Persona",
    "PersonaResult",
    "Rubric",
    "aggregate",
    "anthropic_caller",
    "band_for",
    "build_prompt",
    "extract_json",
    "score",
    "self_consistent",
    "tiered_pass",
    "vote_coupled",
    "weighted_composite",
]

__version__ = "0.1.0"
