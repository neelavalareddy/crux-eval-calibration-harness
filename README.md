# crux-eval-calibration-harness

An offline harness that measures how well an automated eval judge agrees with human expert judgment, built around a fully **synthetic** healthcare support-agent scenario. It reports inter-rater agreement, per-criterion divergence, and drift, and writes a Markdown calibration report.

Motivation: CRUX Labs describes evals "calibrated to your organization", and this harness shows what publishable calibration evidence could look like (agreement metrics, divergence report, drift report). It is an independent sketch, not affiliated with CRUX Labs.

## Synthetic-data disclosure

**All data is synthetic.** The 60 traces are templated by a seeded script (the fictional "Meridian Health (SYNTHETIC)" support agent). The 2-3 "expert" labels per trace are seeded random perturbations of a latent truth, with disagreement deliberately planted on subjective criteria (tone, completeness) and near-consensus on objective ones (accuracy, policy adherence, escalation judgment). No real patients, experts, or company data.

## Run

Python 3 standard library only. No network, no API keys.

```
python3 -m unittest discover -s tests -v    # unit tests
python3 demo.py [seed]                      # generate data, run harness, write sample_report.md (default seed 2026)
```

`demo.py` also writes `data/synthetic_traces.json` and `data/synthetic_expert_labels.json`.

## Methodology

- **Layer (a), rules checker** (`crux_calib/rules.py`): escalated when required; cited a known `[POL-XX-NN]` policy; no promise phrase (guarantee, full refund, approved claim...) since no available tool grants such things; no calls to tools outside the allowed set.
- **Layer (b), rubric scorer** (`rubric.py`): 0-2 per criterion (accuracy, policy adherence, escalation judgment, tone, completeness) using claim parsing against a policy table, rule outputs, an empathy/curt lexicon, and required-point keyword coverage. It is a heuristic stand-in for an LLM judge.
- **Agreement** (`agreement.py`): % agreement and unweighted Cohen's kappa, judge vs each expert, and expert vs expert pairs (co-labelled traces only).
- **Divergence** (`harness.py`): per criterion, disagreement rate, mean absolute difference, and the traces with the largest judge-vs-expert-mean gap.
- **Drift** (`drift.py`): corrupt one criterion at a time, either expert labels (label noise) or judge scores (judge degradation), at rates 0 to 0.6, and track that criterion's mean kappa.

## Honest limitations

- Synthetic labels are not real expert judgment; kappa here demonstrates the pipeline, not a production claim.
- The judge and generator share conventions (e.g. the claim sentence format), so objective-criterion agreement is optimistic. Planted blind spots (warm tone without empathy words, paraphrased content) are the main judge errors.
- ~60 traces and 2-3 raters give noisy kappa; kappa is unweighted although the scale is ordinal, and is sensitive to prevalence.
- Drift on a single criterion necessarily only affects that criterion; the ranking shows sensitivity of the metric, not real-world failure modes.
- `sample_report.md` is produced by `demo.py`; see the note at the bottom of this README on its status.

## Status note

The author's environment at build time had no code execution. Tests and demo had not been run when this package was written; `sample_report.md` is only present if someone has run `demo.py`.
