# Verification evidence

Verified an unchanged copy of task 630b0d37-9916-4472-a849-adfd800565be/crux-eval-calibration-harness in this task workspace.

Commands run from crux-eval-calibration-harness/:
- python3 -m unittest discover -s tests -v: 21 run, 21 passed, 0 failed, 0 errors, exit 0. See test-output.txt.
- python3 demo.py: exit 0. See demo-output.txt and generated sample_report.md.

No code fixes, threshold changes, label changes, or seed changes. No repository created or GitHub publication.

An initial test command was accidentally launched from the task root, before entering the package directory. Discovery failed with TypeError: expected str, bytes or os.PathLike object, not NoneType. This was an invocation-directory error, not a harness test failure. The requested command then passed from the package directory, and a confirmation run also passed.

The report uses synthetic traces and synthetic expert labels, not real human judgment. Agreement tables average pair-level metrics; divergence rates pool label comparisons, so they need not equal one minus the displayed mean agreement. Full per-expert, divergence examples, and drift tables are in sample_report.md.
