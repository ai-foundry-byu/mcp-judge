#!/usr/bin/env python3
"""Eval harness for the calibrated judge.

Scores a small labeled set (``dataset.jsonl``) and reports the three metrics that
tell you whether the judge is calibrated rather than just plausible:

  * separation  -- the gap between the mean score of STRONG items and MISMATCH items.
                   A calibrated judge opens a wide, stable gap; a naive one compresses
                   both into the mid-band.
  * calibration -- does each item's score-band match its expert label
                   (strong / borderline / mismatch), and do the label means order
                   monotonically?
  * self-consistency -- the mean standard deviation of an item's score across the N
                   self-consistency samples. Lower is steadier.

Usage::

    export ANTHROPIC_API_KEY=sk-ant-...
    python eval/run_eval.py                 # score the whole set (live API)
    python eval/run_eval.py --samples 3     # 3 self-consistency samples per persona
    python eval/run_eval.py --dry-run       # validate wiring only, no API calls
    python eval/run_eval.py --json out.json # also write raw results as JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean, pstdev

# Run from a clone without installing: make ``src/`` importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mcp_judge.judge import JudgeError, anthropic_caller, score  # noqa: E402
from mcp_judge.personas import get_rubric  # noqa: E402

DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.jsonl")

# Nominal score bands each expert label should land in (illustrative, for a synthetic set).
NOMINAL_BANDS = {
    "strong": (80, 100),
    "borderline": (60, 84),
    "mismatch": (0, 59),
}


def load_dataset(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{i}: invalid JSON: {exc}")
            for key in ("id", "rubric", "label", "document"):
                if key not in rec:
                    raise SystemExit(f"{path}:{i}: missing required field {key!r}")
            if rec["label"] not in NOMINAL_BANDS:
                raise SystemExit(f"{path}:{i}: unknown label {rec['label']!r}")
            # Validate the rubric name resolves and any expected context is present.
            rubric = get_rubric(rec["rubric"])
            if rubric.context_hint and not rec.get("context"):
                raise SystemExit(
                    f"{path}:{i}: rubric {rubric.name!r} expects context ({rubric.context_hint}) "
                    f"but the entry has none"
                )
            rows.append(rec)
    return rows


# --------------------------------------------------------------------------- #
# Metrics (pure functions over scored rows: [{"label":..., "panel_avg":..., "sd":...}])
# --------------------------------------------------------------------------- #
def separation_gap(rows: list[dict]) -> float:
    strong = [r["panel_avg"] for r in rows if r["label"] == "strong"]
    mismatch = [r["panel_avg"] for r in rows if r["label"] == "mismatch"]
    if not strong or not mismatch:
        return float("nan")
    return round(mean(strong) - mean(mismatch), 1)


def label_means(rows: list[dict]) -> dict:
    out = {}
    for label in NOMINAL_BANDS:
        vals = [r["panel_avg"] for r in rows if r["label"] == label]
        out[label] = round(mean(vals), 1) if vals else float("nan")
    return out


def band_hit_rate(rows: list[dict]) -> float:
    hits = sum(1 for r in rows if NOMINAL_BANDS[r["label"]][0] <= r["panel_avg"] <= NOMINAL_BANDS[r["label"]][1])
    return round(100 * hits / len(rows), 1) if rows else float("nan")


def is_monotonic(means: dict) -> bool:
    return means["strong"] > means["borderline"] > means["mismatch"]


def mean_self_consistency_sd(rows: list[dict]) -> float:
    sds = [r["sd"] for r in rows if r.get("sd") is not None]
    return round(mean(sds), 2) if sds else float("nan")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run(rows: list[dict], samples: int, model: str) -> list[dict]:
    caller = anthropic_caller(model)  # raises a clear JudgeError if the key is missing
    scored = []
    for rec in rows:
        rubric = get_rubric(rec["rubric"])
        result = score(
            rec["document"], rubric, samples=samples,
            context=rec.get("context"), model=model, caller=caller,
        )
        scored.append({
            "id": rec["id"],
            "rubric": rec["rubric"],
            "label": rec["label"],
            "panel_avg": result.panel_avg,
            "raw_panel_avg": result.raw_panel_avg,
            "band": result.band,
            "votes": result.votes,
            "passed": result.passed,
            "sd": result.mean_composite_sd,
        })
        print(f"  scored {rec['id']:<16} label={rec['label']:<10} "
              f"panel_avg={result.panel_avg:<5} band={result.band}")
    return scored


def report(scored: list[dict], samples: int) -> None:
    gap = separation_gap(scored)
    means = label_means(scored)
    hit = band_hit_rate(scored)
    mono = is_monotonic(means)
    sd = mean_self_consistency_sd(scored)

    print("\n" + "=" * 64)
    print(f"EVAL RESULTS  ({len(scored)} items, {samples} self-consistency sample(s)/persona)")
    print("=" * 64)
    print("\nPer-item scores:")
    print(f"  {'id':<16}{'label':<12}{'panel_avg':>10}{'band':>14}{'SD':>7}")
    for r in sorted(scored, key=lambda x: (x["label"], x["id"])):
        print(f"  {r['id']:<16}{r['label']:<12}{r['panel_avg']:>10}{r['band']:>14}{r['sd']:>7}")

    print("\nMetrics:")
    print(f"  separation gap (strong - mismatch) : {gap} points")
    print(f"  label means  strong / borderline / mismatch : "
          f"{means['strong']} / {means['borderline']} / {means['mismatch']}")
    print(f"  label ordering monotonic           : {mono}")
    print(f"  band hit-rate (score-band == label): {hit}%")
    print(f"  mean self-consistency SD           : {sd} points")
    print("=" * 64)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the calibrated-judge eval.")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--samples", type=int, default=3, help="self-consistency samples per persona")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N entries")
    ap.add_argument("--dry-run", action="store_true", help="validate wiring only; no API calls")
    ap.add_argument("--json", metavar="PATH", default=None, help="write raw scored results as JSON")
    args = ap.parse_args()

    rows = load_dataset(args.dataset)
    if args.limit:
        rows = rows[: args.limit]

    if args.dry_run:
        print(f"OK: loaded {len(rows)} entries; every rubric resolves and required context is present.")
        by_label = {lbl: sum(1 for r in rows if r["label"] == lbl) for lbl in NOMINAL_BANDS}
        print(f"     label counts: {by_label}")
        print("     (dry run -- no API calls made; drop --dry-run to score against the live model.)")
        return 0

    try:
        scored = run(rows, args.samples, args.model)
    except JudgeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("Set ANTHROPIC_API_KEY to score against the live model, or use --dry-run to "
              "validate the harness offline.", file=sys.stderr)
        return 1

    report(scored, args.samples)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(scored, fh, indent=2)
        print(f"\nWrote raw results to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
