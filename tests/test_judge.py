"""Offline unit tests — the LLM is mocked via an injected caller, so these run
without an API key. They pin the calibration mechanics: brace-balanced JSON
extraction, vote-coupling, self-consistency, and the end-to-end pass logic."""
import json

from mcp_judge.judge import (
    Dimension, Persona, PersonaResult, Rubric, aggregate, extract_json, score,
    _score_persona,
)

RUBRIC = Rubric(name="t", decision_question="advance?",
                dimensions=[Dimension("a", 50, "d"), Dimension("b", 50, "d")])
PANEL = [Persona("p1", "f"), Persona("p2", "f"), Persona("p3", "f")]


def _payload(a, b, advance):
    return json.dumps({
        "dimensions": {"a": {"score": a, "rationale": "x"}, "b": {"score": b, "rationale": "y"}},
        "decision": {"would_advance": advance, "confidence": 90, "reason": "r"},
    })


def _caller(a, b, advance):
    return lambda prompt: _payload(a, b, advance)


def test_extract_json_handles_braces_in_strings():
    got = extract_json('noise {"decision": {"reason": "a } b"}} tail')
    assert got["decision"]["reason"] == "a } b"


def test_vote_coupling_floors_a_yes_vote():
    prs = [
        PersonaResult("p", {}, 72.0, True, "r"),
        PersonaResult("q", {}, 72.0, True, "r"),
        PersonaResult("s", {}, 40.0, False, "r"),
    ]
    res = aggregate(prs, RUBRIC, threshold=90)
    assert res.raw_panel_avg == round((72 + 72 + 40) / 3, 1)   # untouched arithmetic mean
    assert res.panel_avg == round((85 + 85 + 40) / 3, 1)        # each yes floored to 85
    assert res.votes == 2
    assert res.passed is False                                  # 70.0 calibrated < 90 threshold


def test_no_vote_means_no_floor():
    res = aggregate([PersonaResult("p", {}, 72.0, False, "r")], RUBRIC, threshold=90)
    assert res.panel_avg == res.raw_panel_avg == 72.0
    assert res.votes == 0 and res.passed is False


def test_score_strong_passes():
    res = score("doc", RUBRIC, PANEL, samples=1, threshold=85, caller=_caller(90, 88, True))
    assert res.votes == 3
    assert res.panel_avg >= 85
    assert res.passed is True


def test_score_mismatch_is_low():
    res = score("doc", RUBRIC, PANEL, samples=1, threshold=85, caller=_caller(40, 44, False))
    assert res.votes == 0
    assert res.panel_avg < 55
    assert res.passed is False


def test_self_consistency_averages_and_majority_votes():
    rounds = [(90, 90, True), (80, 80, True), (70, 70, False)]
    seq = iter(rounds * 5)

    def caller(prompt):
        a, b, adv = next(seq)
        return _payload(a, b, adv)

    pr = _score_persona(caller, PANEL[0], RUBRIC, "doc", None, 3)
    assert pr.dimensions["a"]["score"] == 80.0          # mean(90, 80, 70)
    assert pr.composite == 80.0
    assert pr.would_advance is True                     # 2 of 3 said yes
    assert pr.confidence == 67
