"""Pipeline orchestration + divergence analysis."""
from .agreement import expert_vs_expert, judge_vs_experts, mean_of
from .dataset import CRITERIA, generate_expert_labels, generate_traces
from .drift import run_drift
from .rubric import score_trace
from .rules import check_rules


def divergence(traces, judge, labels, top_n=3):
    """Per criterion: mean |judge - expert|, disagreement rate, and worst example traces."""
    by_id = {t["trace_id"]: t for t in traces}
    out = {}
    for c in CRITERIA:
        diffs, dis, total = 0.0, 0, 0
        per_trace = []
        for tid in sorted(labels):
            exp = labels[tid]
            ds = [abs(judge[tid][c] - exp[e][c]) for e in sorted(exp)]
            diffs += sum(ds)
            dis += sum(1 for d in ds if d > 0)
            total += len(ds)
            mean_exp = sum(exp[e][c] for e in exp) / float(len(exp))
            per_trace.append((abs(judge[tid][c] - mean_exp), tid))
        per_trace.sort(key=lambda x: (-x[0], x[1]))
        examples = []
        for gap, tid in per_trace[:top_n]:
            if gap == 0:
                break
            t = by_id[tid]
            examples.append({
                "trace_id": tid, "category": t["category"], "gap": gap,
                "judge": judge[tid][c],
                "experts": {e: labels[tid][e][c] for e in sorted(labels[tid])},
                "response": t["agent_response"],
            })
        out[c] = {
            "mean_abs_diff": diffs / total if total else 0.0,
            "disagreement_rate": dis / float(total) if total else 0.0,
            "examples": examples,
        }
    return out


def run_harness(seed=2026, n=60):
    traces = generate_traces(seed, n)
    labels = generate_expert_labels(traces, seed)
    rules = {t["trace_id"]: check_rules(t) for t in traces}
    judge = {t["trace_id"]: score_trace(t, rules[t["trace_id"]]) for t in traces}
    jve = judge_vs_experts(judge, labels)
    eve = expert_vs_expert(labels)
    return {
        "seed": seed, "traces": traces, "labels": labels, "rules": rules, "judge": judge,
        "judge_vs_expert": jve, "expert_vs_expert": eve,
        "divergence": divergence(traces, judge, labels),
        "drift": run_drift(judge, labels, seed),
        "headline": {
            "judge_kappa_mean": sum(mean_of(jve[c], "kappa") for c in CRITERIA) / len(CRITERIA),
            "judge_agree_mean": sum(mean_of(jve[c], "agreement") for c in CRITERIA) / len(CRITERIA),
            "expert_kappa_mean": sum(mean_of(eve[c], "kappa") for c in CRITERIA) / len(CRITERIA),
            "expert_agree_mean": sum(mean_of(eve[c], "agreement") for c in CRITERIA) / len(CRITERIA),
        },
    }
