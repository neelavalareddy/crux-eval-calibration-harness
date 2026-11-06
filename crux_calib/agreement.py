"""Agreement math: % agreement, Cohen's kappa, judge-vs-expert, expert-vs-expert."""
from itertools import combinations

from .dataset import CRITERIA


def percent_agreement(a, b):
    if len(a) != len(b):
        raise ValueError("length mismatch")
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / float(len(a))


def cohens_kappa(a, b, categories=(0, 1, 2)):
    """Unweighted Cohen's kappa. If chance agreement is 1 (both raters constant and
    equal), returns 1.0 when observed agreement is 1, else 0.0."""
    if len(a) != len(b):
        raise ValueError("length mismatch")
    n = len(a)
    if n == 0:
        return 0.0
    po = percent_agreement(a, b)
    pe = 0.0
    for c in categories:
        pe += (sum(1 for x in a if x == c) / float(n)) * (sum(1 for y in b if y == c) / float(n))
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def _stats(a, b):
    return {"n": len(a), "agreement": percent_agreement(a, b), "kappa": cohens_kappa(a, b)}


def judge_vs_experts(judge, labels, criteria=CRITERIA):
    """-> {criterion: {expert: {n, agreement, kappa}}}"""
    out = {}
    experts = sorted({e for tl in labels.values() for e in tl})
    for c in criteria:
        out[c] = {}
        for e in experts:
            a, b = [], []
            for tid, tl in labels.items():
                if e in tl and tid in judge:
                    a.append(judge[tid][c])
                    b.append(tl[e][c])
            if a:
                out[c][e] = _stats(a, b)
    return out


def expert_vs_expert(labels, criteria=CRITERIA):
    """-> {criterion: {"E1-E2": {n, agreement, kappa}, ...}} over co-labelled traces."""
    out = {}
    experts = sorted({e for tl in labels.values() for e in tl})
    for c in criteria:
        out[c] = {}
        for e1, e2 in combinations(experts, 2):
            a, b = [], []
            for tl in labels.values():
                if e1 in tl and e2 in tl:
                    a.append(tl[e1][c])
                    b.append(tl[e2][c])
            if a:
                out[c]["%s-%s" % (e1, e2)] = _stats(a, b)
    return out


def mean_of(stats_by_name, key):
    vals = [s[key] for s in stats_by_name.values()]
    return sum(vals) / len(vals) if vals else 0.0
