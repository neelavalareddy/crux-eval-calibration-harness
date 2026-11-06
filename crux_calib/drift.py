"""Drift simulation: degrade ONE criterion at a time and watch judge-vs-expert kappa."""
import random

from .agreement import cohens_kappa, judge_vs_experts, mean_of
from .dataset import CRITERIA

LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6]


def _replace(v, rng):
    return rng.choice([x for x in (0, 1, 2) if x != v])


def corrupt_labels(labels, criterion, p, rng):
    """Return a copy of labels where `criterion` is replaced by a different value w.p. p."""
    out = {}
    for tid in sorted(labels):
        out[tid] = {}
        for e in sorted(labels[tid]):
            row = dict(labels[tid][e])
            if rng.random() < p:
                row[criterion] = _replace(row[criterion], rng)
            out[tid][e] = row
    return out


def degrade_judge(judge, criterion, p, rng):
    """Return a copy of judge scores where `criterion` is randomised w.p. p."""
    out = {}
    for tid in sorted(judge):
        row = dict(judge[tid])
        if rng.random() < p:
            row[criterion] = _replace(row[criterion], rng)
        out[tid] = row
    return out


def _mean_kappa(judge, labels, criterion):
    res = judge_vs_experts(judge, labels, [criterion])
    return mean_of(res[criterion], "kappa")


def run_drift(judge, labels, seed=2026, levels=LEVELS):
    """-> {"label_noise": {crit: [kappa per level]}, "judge_degradation": {...}, "levels": levels}"""
    result = {"levels": list(levels), "label_noise": {}, "judge_degradation": {}}
    for ci, c in enumerate(CRITERIA):
        ln, jd = [], []
        for li, p in enumerate(levels):
            rng = random.Random(seed * 1000 + ci * 50 + li)
            ln.append(_mean_kappa(judge, corrupt_labels(labels, c, p, rng), c))
            rng = random.Random(seed * 1000 + 500 + ci * 50 + li)
            jd.append(_mean_kappa(degrade_judge(judge, c, p, rng), labels, c))
        result["label_noise"][c] = ln
        result["judge_degradation"][c] = jd
    return result


def rank_by_drop(series_by_criterion, level_index=-1):
    """Rank criteria by kappa drop (level 0 -> level_index), largest first."""
    rows = [(c, s[0] - s[level_index]) for c, s in series_by_criterion.items()]
    return sorted(rows, key=lambda r: (-r[1], r[0]))


def first_level_below(series, levels, threshold_ratio=0.8):
    """First noise level where kappa falls below threshold_ratio * baseline (None if never)."""
    base = series[0]
    for k, p in zip(series, levels):
        if k < threshold_ratio * base:
            return p
    return None
