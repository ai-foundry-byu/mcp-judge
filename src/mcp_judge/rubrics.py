"""Example rubrics + persona panels.

These are configuration, not the engine — swap dimensions and personas for your
own task. Two panels ship as examples:

  - "document-quality": a general editorial panel (editor / domain expert / skeptic)
  - "resume-fit":       score a candidate document against a role's requirements
                        (recruiter / hiring manager / technical peer)

Nothing here is specific to any real person or company.
"""
from __future__ import annotations

from .judge import Dimension, Persona, Rubric

# --------------------------------------------------------------------------- #
# Panel 1: general document quality
# --------------------------------------------------------------------------- #
DOC_QUALITY = Rubric(
    name="document-quality",
    decision_question="Would you publish this document as-is?",
    dimensions=[
        Dimension("correctness", 30, "Are the claims accurate, internally consistent, and well supported?"),
        Dimension("clarity", 25, "Is it easy to follow, well structured, and free of filler?"),
        Dimension("completeness", 25, "Does it fully address its stated purpose with no important gaps?"),
        Dimension("specificity", 20, "Concrete detail and evidence rather than vague generalities?"),
    ],
)

DOC_QUALITY_PANEL = [
    Persona("editor",
            "You are a senior editor. Your lens: is this clear, correct, and ready for a reader? "
            "Be discriminating, not dismissive, and calibrate to the anchored scale: if you would "
            "publish it, that is an 85 or higher."),
    Persona("domain_expert",
            "You are a subject-matter expert in this document's topic. Your lens: is it substantively "
            "right and non-superficial? Reward specific, correct detail; a genuinely wrong or shallow "
            "claim scores low, but do not penalize a correct, well-argued piece for being concise. "
            "Calibrate to the anchored scale."),
    Persona("skeptic",
            "You are a careful skeptic. Your lens: what is unsupported, overstated, or hand-waved? "
            "You reward claims backed by evidence and flag those that are not, but you place a genuinely "
            "solid document high on the anchored scale rather than defaulting to the middle."),
]

# --------------------------------------------------------------------------- #
# Panel 2: resume-vs-role fit (generic; the DOCUMENT is the resume text with the
# role's requirements appended as context — no real resume is bundled here).
# --------------------------------------------------------------------------- #
RESUME_FIT = Rubric(
    name="resume-fit",
    decision_question="Would you advance this candidate to an interview for the role described?",
    dimensions=[
        Dimension("impact_credibility", 25, "Specific, scoped, believable outcomes (with denominators), not duty statements."),
        Dimension("requirement_coverage", 20, "Are the role's must-have skills present and load-bearing in real evidence?"),
        Dimension("relevance", 20, "Does the trajectory and functional experience genuinely fit the role?"),
        Dimension("convention_fit", 20, "Does it follow the norms this kind of employer expects?"),
        Dimension("clarity", 15, "Clean, scannable, varied structure, no filler or generated-text tells."),
    ],
)

RESUME_FIT_PANEL = [
    Persona("recruiter",
            "You are the recruiter screening first. Your lens: do I pass this to the hiring manager? "
            "Look for the level, the must-have skills present and easy to find, a believable progression, "
            "and a clean layout. Be discriminating, not reflexively dismissive: a clear fit and a clear "
            "miss must land far apart. If you would advance it, that is an 85 or higher."),
    Persona("hiring_manager",
            "You are the hiring manager who owns the role. Your lens: would I move this forward and want "
            "them on my team? Reward genuine role-fit and believable, correctly-scoped impact; note "
            "inflated or vaguely-attributed claims. Calibrate to the anchored scale — a genuine strong-yes "
            "is a 90+, not a 72."),
    Persona("technical_peer",
            "You are a senior practitioner who would work with this person. Your lens: do they actually do "
            "the work? Reward concrete systems, real decisions, honest scope, and specifics only a "
            "practitioner would know; vague buzzword filler scores low, but real, specific, hard-to-fabricate "
            "work scores high. Do not penalize a clean, specific document for being polished."),
]

RUBRICS: dict[str, tuple[Rubric, list[Persona]]] = {
    DOC_QUALITY.name: (DOC_QUALITY, DOC_QUALITY_PANEL),
    RESUME_FIT.name: (RESUME_FIT, RESUME_FIT_PANEL),
}


def list_rubrics() -> list[str]:
    return sorted(RUBRICS)


def get_rubric(name: str) -> tuple[Rubric, list[Persona]]:
    if name not in RUBRICS:
        raise KeyError(f"unknown rubric '{name}'. Available: {', '.join(list_rubrics())}")
    return RUBRICS[name]
