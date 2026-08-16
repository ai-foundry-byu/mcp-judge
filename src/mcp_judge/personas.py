"""Example personas and rubrics for the calibrated judge.

Two ready-to-use configurations, plus a small registry the MCP server and the eval
harness look rubrics up in:

  * ``document_quality`` -- a generic three-seat panel for scoring any written
    artifact (a PR description, a design doc, an RFP answer, a support reply) against
    a general quality bar. Normal-stakes pass policy.

  * ``resume_vs_jd`` -- the flagship worked example: score a resume against a job
    description. High-stakes pass policy (majority vote + a 90 bar). All example data
    is synthetic; nothing here is anyone's real resume.

Everything is data. Personas, dimensions, weights, thresholds, and the pass policy
are all just configuration you can copy and edit -- the judge machinery in
``mcp_judge.judge`` is entirely agnostic to what is being scored.
"""
from __future__ import annotations

from .judge import Dimension, Persona, Rubric

# --------------------------------------------------------------------------- #
# 1) Generic document-quality panel
# --------------------------------------------------------------------------- #
DOCUMENT_QUALITY_PERSONAS = [
    Persona(
        key="editor",
        framing=(
            "You are a senior editor who has shipped thousands of documents. Your lens: is this "
            "CLEAR, well-structured, and free of filler and hedging? You reward a strong lead, a "
            "clean hierarchy, and prose that respects the reader's time; you are not swayed by length "
            "or jargon. Calibrate to the anchored scale below: a document you would publish as-is is "
            "an 85 or higher, and you use the full range to place a strong one precisely."
        ),
    ),
    Persona(
        key="domain_expert",
        framing=(
            "You are a domain expert who knows this subject deeply. Your lens: are the claims "
            "ACCURATE and genuinely SUPPORTED -- specific evidence, correct reasoning, honest scope -- "
            "rather than confident-sounding but thin? Unsupported assertions and hand-waving score low; "
            "concrete, verifiable substance scores high. If you would stake your name on this being "
            "correct, that is an 85 or higher on the anchored scale below."
        ),
    ),
    Persona(
        key="skeptical_reader",
        framing=(
            "You are the skeptical first reader this document must convince. Your lens: does it actually "
            "ANSWER the question it set out to, and would a busy, doubtful reader come away persuaded and "
            "informed? You gate on completeness, relevance, and whether real objections are addressed -- "
            "not on polish alone. Be discriminating, not dismissive: a document that clearly does the job "
            "and one that clearly does not must land far apart on the anchored scale below."
        ),
    ),
]

DOCUMENT_QUALITY = Rubric(
    name="document_quality",
    description="Score any written artifact against a general quality bar.",
    decision_question="Would you advance this document as-is -- publish it, or send it onward without a rewrite?",
    dimensions=[
        Dimension("clarity_structure", 25,
                  "Clear lead, clean hierarchy, easy to scan; no filler, hedging, or padding."),
        Dimension("accuracy_support", 30,
                  "Claims are correct and backed by specific, verifiable evidence -- not thin assertions."),
        Dimension("completeness", 25,
                  "Answers the question it set out to; addresses the obvious objections and gaps."),
        Dimension("relevance_focus", 20,
                  "Stays on the point for its intended reader; no off-topic or generic boilerplate."),
    ],
    personas=DOCUMENT_QUALITY_PERSONAS,
    pass_threshold=70,
    require_majority=False,
)


# --------------------------------------------------------------------------- #
# 2) Flagship example: resume vs. job description
# --------------------------------------------------------------------------- #
RESUME_VS_JD_PERSONAS = [
    Persona(
        key="hiring_manager",
        framing=(
            "You are the hiring manager who owns this role and will manage whoever is hired. Your lens: "
            "WOULD I MOVE THIS FORWARD and want this person on my team? You reward genuine role-fit and "
            "believable, correctly-scoped impact; you are not swayed by polish alone, and you note inflated "
            "or vaguely-attributed claims. Calibrate to the anchored scale below: if you would interview "
            "this candidate, that is an 85 or higher, and a genuine strong-yes is a 90+, not a 72."
        ),
    ),
    Persona(
        key="recruiter",
        framing=(
            "You are the recruiter who screens this role first. Your lens: DO I PASS THIS TO THE HIRING "
            "MANAGER? You look for the target level, the job's must-have skills present and easy to find, a "
            "believable progression, and a skimmable layout. You gate on match, clarity, and real red flags, "
            "not deep technical merit. Be discriminating, not reflexively dismissive: a resume that clearly "
            "meets the bar and one that clearly misses it must land far apart on the anchored scale below."
        ),
    ),
    Persona(
        key="technical_peer",
        framing=(
            "You are a senior technical peer who would vet this candidate's real substance on a screen. You "
            "build the kind of systems this role owns. Your lens: DOES THIS PERSON ACTUALLY BUILD? You reward "
            "concrete systems, real architecture decisions, honest scope, and specific tools/metrics only a "
            "builder would know; vague buzzword filler scores low, hard-to-fabricate work scores high. If you "
            "would want this person on a technical screen, that is an 85 or higher on the anchored scale below."
        ),
    ),
]

RESUME_VS_JD = Rubric(
    name="resume_vs_jd",
    description="Score a resume against a specific job description (the flagship worked example).",
    decision_question=(
        "Given only this resume for this exact posting, would you advance this candidate to an interview?"
    ),
    dimensions=[
        Dimension("quantified_impact", 25,
                  "Bullets carry specific, internally-consistent, plausibly-scoped metrics -- not duty statements."),
        Dimension("requirement_coverage", 25,
                  "The resume naturally uses the JD's own must-have terms and required experience, load-bearing in real bullets."),
        Dimension("domain_relevance", 25,
                  "Career trajectory, seniority, and any named prior title/function the JD requires are a direct or clearly-adjacent match."),
        Dimension("structure_clarity", 25,
                  "Skimmable, consistent, appropriate length; no filler phrases or templated/AI-tell padding."),
    ],
    personas=RESUME_VS_JD_PERSONAS,
    pass_threshold=90,          # high-stakes bar
    require_majority=True,      # a majority of the panel must also vote "advance"
    context_hint="the job description to score the resume against",
)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
RUBRICS = {
    DOCUMENT_QUALITY.name: DOCUMENT_QUALITY,
    RESUME_VS_JD.name: RESUME_VS_JD,
}


def get_rubric(name: str) -> Rubric:
    """Look up a rubric by name. Raises KeyError with the valid names on a miss."""
    try:
        return RUBRICS[name]
    except KeyError:
        raise KeyError(f"unknown rubric {name!r}; available: {sorted(RUBRICS)}") from None


def list_rubrics() -> list[dict]:
    """Machine-readable summary of every registered rubric (used by the MCP server)."""
    out = []
    for name, r in RUBRICS.items():
        out.append({
            "name": name,
            "description": r.description,
            "decision_question": r.decision_question,
            "dimensions": [{"key": d.key, "weight": d.weight, "description": d.description}
                           for d in r.dimensions],
            "personas": [p.key for p in r.personas],
            "pass_threshold": r.pass_threshold,
            "require_majority": r.require_majority,
            "expects_context": bool(r.context_hint),
            "context_hint": r.context_hint,
        })
    return out
