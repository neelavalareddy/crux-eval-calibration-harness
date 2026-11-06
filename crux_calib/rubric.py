"""Layer (b): heuristic rubric scorer, 0-2 per criterion.

A transparent keyword/lexicon scorer standing in for an LLM judge. It has known blind
spots (e.g. warm formal tone without empathy words; paraphrased content points).
"""
import re

from .dataset import POLICY_DB
from .rules import check_rules

CLAIM_RE = re.compile(r"Per \[(POL-[A-Z]+-\d{2})\], the relevant rule is: ([^.]+)\.")
EMPATHY = ["sorry", "understand", "worry", "worrying", "appreciate"]
CURT = ["not my problem", "do not ask again", "as i said", "whatever"]


def extract_claims(text):
    return [(m.group(1), m.group(2).strip()) for m in CLAIM_RE.finditer(text)]


def score_accuracy(trace):
    claims = extract_claims(trace["agent_response"])
    if not claims:
        return 1
    ok = all(POLICY_DB.get(pid) == val for pid, val in claims)
    return 2 if ok else 0


def score_policy(trace, rules):
    if rules["promises"] or not rules["policy_cited"]:
        return 0
    return 2


def score_escalation(trace, rules):
    required = bool(trace.get("requires_escalation"))
    escalated = "escalate_to_human" in [tc["name"] for tc in trace.get("tool_calls", [])]
    if required:
        return 2 if escalated else 0
    return 2 if not escalated else 1


def score_tone(trace):
    low = trace["agent_response"].lower()
    if any(w in low for w in CURT):
        return 1 if any(w in low for w in EMPATHY) else 0
    return 2 if any(w in low for w in EMPATHY) else 1


def score_completeness(trace):
    pts = trace.get("required_points", [])
    if not pts:
        return 2
    low = trace["agent_response"].lower()
    frac = sum(1 for p in pts if p.lower() in low) / float(len(pts))
    if frac >= 1.0:
        return 2
    if frac >= 0.5:
        return 1
    return 0


def score_trace(trace, rules=None):
    rules = rules or check_rules(trace)
    return {
        "accuracy": score_accuracy(trace),
        "policy_adherence": score_policy(trace, rules),
        "escalation_judgment": score_escalation(trace, rules),
        "tone": score_tone(trace),
        "completeness": score_completeness(trace),
    }
