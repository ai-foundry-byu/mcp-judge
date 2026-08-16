# Eval: does the calibration actually hold?

A judge is only as trustworthy as the eval that checks it. This harness scores a
small **synthetic, labeled** set (`dataset.jsonl`) and reports three things a
naive LLM judge fails at.

## What it measures

| Metric | Question it answers | Target |
|---|---|---|
| **Separation** | Does a `strong` item score far above a `mismatch` item? | gap ≥ 25 pts |
| **Calibration** | Does the score *band* match the human label? (`strong`→85+, `borderline`→70–84, `mismatch`→<55) | band-accuracy ≥ 0.8 |
| **Self-consistency** | How much does the composite move across samples? | mean composite SD ≤ 4 |

Run it:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python eval/run_eval.py            # prints the table below + per-item scores
```

The harness makes live API calls, so numbers vary slightly run-to-run; separation
and band-accuracy are the stable signals, not any single item's exact score.

## Why these three metrics (the failure modes they catch)

Validating a judge on *agreement rate* alone is misleading — if 80% of items are
"pass", an always-pass judge scores 80% agreement while being useless. Separation,
band-calibration, and consistency each catch a distinct, documented LLM-judge
failure mode:

1. **Mid-band compression.** LLM judges cluster in the 55–72 band and avoid the
   extremes, collapsing discrimination. A compressed judge scores a `strong` item
   ~70 and a `mismatch` ~65 — **separation near zero**. `mcp-judge` counters this
   with an outcome-anchored scale and full-range anchor exemplars, targeting a wide
   separation between labels.

2. **Severity bias.** A "be strict / reject most" instruction *without anchors*
   shifts the whole distribution down rather than spreading it, so everything reads
   low and **band-calibration collapses** (strong items land in the borderline band).
   The fix is to keep the critical stance but bind it to the anchored scale.

3. **Score↔decision decoupling.** The most damaging one: a persona votes "yes,
   advance" while scoring the item a 70. Vote-coupling (a "yes" floors that persona
   at 85) re-binds the number to the verdict, which is what lifts `strong` items
   into their correct band.

4. **Self-inconsistency.** Single-pass judges self-agree only ~61% run-to-run, so a
   one-shot score is noise. N-sample self-consistency (average the dimensions,
   majority-vote the decision) drives the **composite SD down** and stabilizes the
   verdict.

## Illustrative separation (from the grader this was generalized from)

`mcp-judge` was generalized from a production resume-vs-JD grader. On that grader,
a **before/after** test made the effect concrete: a genuinely strong candidate and a
deliberate domain mismatch (an unrelated role) both compressed into the 55–72 band
under the naive judge; after the anchored scale + vote-coupling + self-consistency,
they separated to roughly **73 vs 46** and held that gap stably across 3 samples.
That 25+ point separation — a strong item clearly above a mismatch, reproducibly —
is exactly what this harness measures on the synthetic set.

## Honest limitations

- The labeled set is small and synthetic; it demonstrates the mechanics, it is not a
  benchmark. A real deployment should calibrate against **real downstream outcomes**
  (did the "advance" verdict predict the actual decision) via a post-hoc monotonic
  transform, and track Spearman correlation and Cohen's kappa on the decision, not
  raw agreement.
- The judge reduces the four failure modes; it does not eliminate model-level bias.
  Treat the calibrated score as a well-behaved *signal*, not ground truth.
