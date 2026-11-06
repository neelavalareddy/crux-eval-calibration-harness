"""Demo: generate synthetic dataset, run harness, write sample_report.md.

Usage: python3 demo.py [seed]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from crux_calib.harness import run_harness  # noqa: E402
from crux_calib.report import render_markdown  # noqa: E402


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    r = run_harness(seed=seed, n=60)
    data_dir = os.path.join(HERE, "data")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "synthetic_traces.json"), "w") as f:
        json.dump(r["traces"], f, indent=2, sort_keys=True)
    with open(os.path.join(data_dir, "synthetic_expert_labels.json"), "w") as f:
        json.dump(r["labels"], f, indent=2, sort_keys=True)
    md = render_markdown(r)
    with open(os.path.join(HERE, "sample_report.md"), "w") as f:
        f.write(md)
    h = r["headline"]
    print("SYNTHETIC DATA. traces=%d labelled=%d" % (len(r["traces"]), len(r["labels"])))
    print("judge-vs-expert mean kappa=%.3f agreement=%.3f" % (h["judge_kappa_mean"], h["judge_agree_mean"]))
    print("expert-vs-expert mean kappa=%.3f agreement=%.3f" % (h["expert_kappa_mean"], h["expert_agree_mean"]))
    print("wrote sample_report.md, data/synthetic_traces.json, data/synthetic_expert_labels.json")


if __name__ == "__main__":
    main()
