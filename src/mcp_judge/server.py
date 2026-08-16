"""MCP server exposing the calibrated judge as callable tools.

Run it (stdio transport, the default an MCP client speaks)::

    python -m mcp_judge.server
    # or, once installed:  mcp-judge-server

Tools exposed:

  * ``score(document, rubric_name, context=None, samples=3)`` -- score a document
    against a named rubric and return the full calibrated result.
  * ``list_rubrics()`` -- enumerate the available rubrics and their dimensions.

Scoring makes live Claude API calls, so the server needs ``ANTHROPIC_API_KEY`` in
the environment. A missing key (or any API/parse failure) is returned as a
structured ``{"error": ...}`` payload rather than crashing the tool call.
"""
from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from .judge import DEFAULT_MODEL, JudgeError
from .judge import score as _judge_score
from .personas import get_rubric, list_rubrics as _list_rubrics

mcp = FastMCP("mcp-judge")


@mcp.tool()
def score(document: str, rubric_name: str, context: Optional[str] = None,
          samples: int = 3) -> dict:
    """Score a document against a named rubric with a calibrated multi-persona panel.

    Args:
        document: The text to evaluate.
        rubric_name: One of the registered rubrics (see ``list_rubrics``), e.g.
            ``"document_quality"`` or ``"resume_vs_jd"``.
        context: Optional reference material to judge the document against (for
            ``resume_vs_jd`` this is the job description). Ignored by rubrics that
            do not expect it.
        samples: Self-consistency samples per persona (>= 1). Default 3.

    Returns:
        The full ``JudgeResult`` as a dict -- panel average, per-persona scores,
        votes, pass/fail, self-consistency SD, and dimension disagreements -- or
        ``{"error": "..."}`` on failure.
    """
    try:
        rubric = get_rubric(rubric_name)
    except KeyError as exc:
        return {"error": str(exc)}
    try:
        result = _judge_score(
            document, rubric, context=context, samples=samples, model=DEFAULT_MODEL
        )
    except (JudgeError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return result.to_dict()


@mcp.tool()
def list_rubrics() -> list[dict]:
    """List the available rubrics, with their dimensions, weights, pass policy, and
    whether each expects reference ``context``."""
    return _list_rubrics()


def main() -> None:
    """Console-script entry point: run the server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
