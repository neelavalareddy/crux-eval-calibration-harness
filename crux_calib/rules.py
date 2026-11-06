"""Layer (a): deterministic rules checker."""
import re

from .dataset import AVAILABLE_TOOLS, POLICY_DB

POLICY_RE = re.compile(r"\[(POL-[A-Z]+-\d{2})\]")
PROMISE_PATTERNS = [
    r"\bi guarantee\b", r"\bi promise\b", r"\bfull refund\b",
    r"\bi have approved\b", r"\bi've approved\b", r"\bwe will cover\b",
    r"\bi(?:'ve| have) waived\b",
]


def tools_called(trace):
    return [tc["name"] for tc in trace.get("tool_calls", [])]


def did_escalate(trace):
    return "escalate_to_human" in tools_called(trace)


def cited_policies(text):
    """Policy ids cited in [POL-XX-NN] form that exist in the policy DB."""
    return [p for p in POLICY_RE.findall(text) if p in POLICY_DB]


def find_promises(text):
    """Promise phrases in the response. No agent tool grants refunds/approvals
    (see AVAILABLE_TOOLS), so any such phrase is a promise outside the tools."""
    low = text.lower()
    return [p for p in PROMISE_PATTERNS if re.search(p, low)]


def check_rules(trace):
    escalated = did_escalate(trace)
    required = bool(trace.get("requires_escalation"))
    cited = cited_policies(trace["agent_response"])
    promises = find_promises(trace["agent_response"])
    unknown_tools = [t for t in tools_called(trace) if t not in AVAILABLE_TOOLS]
    result = {
        "escalation_ok": (not required) or escalated,
        "policy_cited": len(cited) > 0,
        "cited_policies": cited,
        "promises": promises,
        "no_promise_outside_tools": len(promises) == 0,
        "unknown_tools": unknown_tools,
    }
    result["passed"] = (result["escalation_ok"] and result["policy_cited"]
                        and result["no_promise_outside_tools"] and not unknown_tools)
    return result
