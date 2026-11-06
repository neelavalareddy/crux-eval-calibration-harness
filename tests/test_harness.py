import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crux_calib.agreement import cohens_kappa, percent_agreement, judge_vs_experts, expert_vs_expert
from crux_calib.dataset import generate_traces, generate_expert_labels, CATEGORIES
from crux_calib.drift import corrupt_labels, rank_by_drop
from crux_calib.harness import run_harness
from crux_calib.report import render_markdown
from crux_calib.rubric import score_trace, extract_claims
from crux_calib.rules import check_rules, find_promises, cited_policies


def tr(resp, calls=(), req=False, points=()):
    return {"trace_id": "X", "agent_response": resp,
            "tool_calls": [{"name": n, "arg": ""} for n in calls],
            "requires_escalation": req, "required_points": list(points)}


class RulesTests(unittest.TestCase):
    def test_escalation_required_and_missing(self):
        r = check_rules(tr("Per [POL-TRI-02], the relevant rule is: call 911.", ["lookup_policy"], req=True))
        self.assertFalse(r["escalation_ok"])
        self.assertFalse(r["passed"])

    def test_escalation_required_and_done(self):
        r = check_rules(tr("Per [POL-TRI-02], the relevant rule is: call 911.",
                           ["lookup_policy", "escalate_to_human"], req=True))
        self.assertTrue(r["escalation_ok"])
        self.assertTrue(r["passed"])

    def test_policy_citation_unknown_id_not_counted(self):
        self.assertEqual(cited_policies("see [POL-ZZ-99]"), [])
        self.assertEqual(cited_policies("see [POL-RX-01]"), ["POL-RX-01"])

    def test_promise_detected(self):
        self.assertTrue(find_promises("I guarantee a full refund."))
        self.assertEqual(find_promises("I will schedule a callback."), [])

    def test_unknown_tool_fails(self):
        r = check_rules(tr("Per [POL-RX-01], the relevant rule is: 7 days early.", ["issue_refund"]))
        self.assertFalse(r["passed"])


class KappaTests(unittest.TestCase):
    def test_known_value(self):
        # po=.75, pe=.5 -> kappa .5
        self.assertAlmostEqual(cohens_kappa([0, 0, 1, 1], [0, 1, 1, 1], (0, 1)), 0.5)

    def test_perfect_and_constant(self):
        self.assertEqual(cohens_kappa([2, 2, 2], [2, 2, 2]), 1.0)
        self.assertEqual(cohens_kappa([0, 1, 2], [0, 1, 2]), 1.0)

    def test_chance_level_zero_or_negative(self):
        self.assertLessEqual(cohens_kappa([0, 1, 0, 1], [1, 0, 1, 0], (0, 1)), 0.0)

    def test_percent_agreement(self):
        self.assertAlmostEqual(percent_agreement([0, 1, 2, 2], [0, 1, 1, 1]), 0.5)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            cohens_kappa([1], [1, 2])

    def test_judge_and_expert_grouping(self):
        judge = {"a": {"tone": 2}, "b": {"tone": 1}}
        labels = {"a": {"E1": {"tone": 2}, "E2": {"tone": 1}}, "b": {"E1": {"tone": 1}, "E2": {"tone": 1}}}
        jv = judge_vs_experts(judge, labels, ["tone"])
        self.assertAlmostEqual(jv["tone"]["E1"]["agreement"], 1.0)
        self.assertAlmostEqual(jv["tone"]["E2"]["agreement"], 0.5)
        ev = expert_vs_expert(labels, ["tone"])
        self.assertAlmostEqual(ev["tone"]["E1-E2"]["agreement"], 0.5)


class RubricTests(unittest.TestCase):
    def test_perfect_response(self):
        t = tr("I'm so sorry. Per [POL-RX-01], the relevant rule is: 7 days early. Please note: pharmacy. Please note: refill.",
               ["lookup_policy"], points=["pharmacy", "refill"])
        s = score_trace(t)
        self.assertEqual(s, {"accuracy": 2, "policy_adherence": 2, "escalation_judgment": 2,
                             "tone": 2, "completeness": 2})

    def test_wrong_claim_and_promise(self):
        t = tr("Per [POL-RX-01], the relevant rule is: no limit. I guarantee a full refund.", ["lookup_policy"])
        s = score_trace(t)
        self.assertEqual(s["accuracy"], 0)
        self.assertEqual(s["policy_adherence"], 0)

    def test_missed_escalation_scores_zero(self):
        t = tr("Per [POL-TRI-02], the relevant rule is: call 911.", ["lookup_policy"], req=True)
        self.assertEqual(score_trace(t)["escalation_judgment"], 0)

    def test_tone_levels(self):
        self.assertEqual(score_trace(tr("Not my problem."))["tone"], 0)
        self.assertEqual(score_trace(tr("Okay."))["tone"], 1)
        self.assertEqual(score_trace(tr("I understand."))["tone"], 2)

    def test_completeness_levels(self):
        pts = ["a1", "b2"]
        self.assertEqual(score_trace(tr("a1 b2", points=pts))["completeness"], 2)
        self.assertEqual(score_trace(tr("a1", points=pts))["completeness"], 1)
        self.assertEqual(score_trace(tr("nothing", points=pts))["completeness"], 0)

    def test_extract_claims(self):
        self.assertEqual(extract_claims("Per [POL-BIL-03], the relevant rule is: 30 days."),
                         [("POL-BIL-03", "30 days")])


class DatasetDriftReportTests(unittest.TestCase):
    def test_dataset_deterministic_and_marked(self):
        a, b = generate_traces(7, 60), generate_traces(7, 60)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 60)
        self.assertTrue(all(t["synthetic"] for t in a))
        self.assertEqual({t["category"] for t in a}, set(CATEGORIES))
        la = generate_expert_labels(a, 7)
        self.assertEqual(la, generate_expert_labels(b, 7))
        self.assertTrue(all(2 <= len(v) <= 3 for v in la.values()))

    def test_corrupt_only_target_criterion(self):
        labels = {"a": {"E1": {"tone": 1, "accuracy": 2}}, "b": {"E1": {"tone": 0, "accuracy": 2}}}
        out = corrupt_labels(labels, "tone", 1.0, random.Random(1))
        self.assertTrue(all(out[k]["E1"]["accuracy"] == 2 for k in out))
        self.assertTrue(all(out[k]["E1"]["tone"] != labels[k]["E1"]["tone"] for k in out))
        same = corrupt_labels(labels, "tone", 0.0, random.Random(1))
        self.assertEqual(same, labels)

    def test_rank_by_drop(self):
        rows = rank_by_drop({"x": [1.0, 0.9], "y": [1.0, 0.2]})
        self.assertEqual(rows[0][0], "y")

    def test_harness_deterministic_and_report(self):
        r1, r2 = run_harness(seed=11, n=60), run_harness(seed=11, n=60)
        self.assertEqual(r1["headline"], r2["headline"])
        self.assertEqual(r1["drift"], r2["drift"])
        md = render_markdown(r1)
        self.assertEqual(md, render_markdown(r2))
        for section in ("SYNTHETIC DATA", "## Headline agreement", "## Per-criterion agreement",
                        "## Divergence report", "## Drift report"):
            self.assertIn(section, md)
        for c in ("accuracy", "policy_adherence", "escalation_judgment", "tone", "completeness"):
            self.assertIn("| %s |" % c, md)


if __name__ == "__main__":
    unittest.main()
