# mcp-judge

**A calibrated, multi-persona LLM-as-judge — exposed as an MCP server, shipped with an eval harness that proves the calibration.**

Most "LLM grades your X" tools are uncalibrated: their scores cluster in a narrow mid-band (55–72), a skeptical persona quietly drags the mean down, and the numeric score drifts out of sync with the actual verdict ("yes, advance this" while scoring it a 70). `mcp-judge` fixes those failure modes on purpose, and the `eval/` harness measures that it did.

It scores **any document against any rubric** — PR reviews, design docs, support-ticket triage, RFP responses, resume-vs-job-description — with a configurable panel of reviewer personas. Any MCP client (Claude Desktop, an agent) can call it as a tool.

## Why it exists

This is a small, self-contained demonstration of two things that are hard to fake: **building reliable LLM systems** and **designing evals that hold them honest**. The interesting part isn't "call an LLM and ask for a score" — it's the calibration, and the harness that proves it.

## The four LLM-judge failure modes it fixes

| Failure mode | What goes wrong | Fix in `mcp-judge` |
|---|---|---|
| **Mid-band compression** | LLM judges cluster scores in the 55–72 range and rarely use the extremes, so they can't discriminate. | **Outcome-anchored scale** (a score band *is* a decision) + **full-range anchor exemplars** (~42/60/78/85/92/98) so "90" has a concrete referent. |
| **Severity bias** | "Be strict / reject most" without anchors shifts the whole distribution down instead of spreading it. | Keep the skeptical stance but **bind each persona to the anchored scale**, so harshness *spreads* scores rather than dragging the mean. |
| **Score↔decision decoupling** | A persona votes "yes, advance" while its dimension scores sit in the 70s. | **Vote-coupling**: a persona's "advance" vote floors its contribution at 85 (`ADVANCE_FLOOR`), so a genuine yes can't be sunk by residual compression. |
| **Self-inconsistency** | Single-pass judges self-agree only ~61% run-to-run. | **N-sample self-consistency**: sample each persona 3× (default), average the dimensions (fractional averages break integer-clustering), majority-vote the decision. |

## How it works

```
document + rubric + persona panel
        │
        ▼
 per persona ×N samples ──► anchored-scale prompt ──► {dimensions, decision}
        │
        ▼
 self-consistency (avg dims, majority-vote)  ──►  PersonaResult
        │
        ▼
 vote-coupled aggregation (yes ⇒ floored at 85)  ──►  calibrated panel_avg + raw_panel_avg
        │
        ▼
 tiered pass (avg ≥ threshold [+ majority for high-stakes])  ──►  JudgeResult
```

Rubrics and personas are data, not hardcoded (`src/mcp_judge/rubrics.py`). Ships with two: `document-quality` (a generic writing/eng-doc panel) and `resume-fit` (resume vs. job description).

## Quickstart

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...

# score a document from Python
python -c "
from mcp_judge import score
from mcp_judge.rubrics import get_rubric
rubric, panel = get_rubric('document-quality')
r = score(open('README.md').read(), rubric, panel, samples=3)
print(r.panel_avg, r.band, r.votes, '/', len(r.personas))
"

# run the eval harness (needs a key)
python eval/run_eval.py

# run as an MCP server (stdio)
python -m mcp_judge.server
```

Offline unit tests (LLM mocked, no key needed):

```bash
python -m pytest -q
```

## Eval

The credibility of a judge is its eval, not its prose. `eval/` contains a small **synthetic, labeled** dataset (`dataset.jsonl` — invented PR reviews, design notes, postmortems, RFP responses tagged `strong` / `borderline` / `mismatch`) and a harness that reports **separation**, **calibration** (does the score band match the label), and **self-consistency** (score SD across samples). See **[eval/RESULTS.md](eval/RESULTS.md)** for the metrics and a failure-mode analysis.

## License

MIT. Author: JD Davenport.
