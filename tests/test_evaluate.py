"""Pure-logic tests for eval/evaluate.py.

House style, matching tests/test_review_priority.py and test_compare_runs.py:
the scoring functions are tested directly and `main()` is left to manual
verification against real stored data. No store fixture, no network, no spend.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval"))

import evaluate  # noqa: E402


# ---------------------------------------------------------------------------
# wilson
# ---------------------------------------------------------------------------


def test_wilson_brackets_the_point_estimate():
    lo, hi = evaluate.wilson(18, 23)
    assert lo < 18 / 23 < hi


def test_wilson_reproduces_the_design_review_interval():
    """The number the 2026-08-26 review published for Tier B, to 1 decimal.

    Locked in literally rather than recomputed in the test, so a change to the
    interval maths shows up as a failure here instead of quietly moving every
    number in the docs.
    """
    lo, hi = evaluate.wilson(18, 23)
    assert round(lo * 100, 1) == 58.1
    assert round(hi * 100, 1) == 90.3


def test_wilson_on_a_perfect_score_does_not_reach_certainty():
    lo, hi = evaluate.wilson(4, 4)
    assert lo < 1.0
    assert hi == pytest.approx(1.0, abs=1e-9)


def test_wilson_of_an_empty_denominator_is_the_whole_interval():
    assert evaluate.wilson(0, 0) == (0.0, 1.0)


def test_wilson_width_shrinks_as_n_grows():
    narrow = evaluate.wilson(180, 230)
    wide = evaluate.wilson(18, 23)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


# ---------------------------------------------------------------------------
# the two lead binaries
# ---------------------------------------------------------------------------


def test_strict_lead_is_in_house_only():
    """The Phase 7b gate's semantics, unchanged. Do not widen this one."""
    assert evaluate.LEAD == "in_house"
    assert evaluate.is_strict_lead("in_house")
    assert not evaluate.is_strict_lead("likely_in_house")


def test_qualified_lead_includes_likely_in_house():
    """The commercial question: a likely_in_house that survives review IS a lead.

    Measured 2026-08-26: 6 of the 9 correct lead calls on Tier B are
    likely_in_house, so a binary that excludes it under-reports recall badly.
    """
    assert evaluate.is_qualified_lead("in_house")
    assert evaluate.is_qualified_lead("likely_in_house")
    for label in ("outsourced", "no_cleanroom_needed", "unknown"):
        assert not evaluate.is_qualified_lead(label)


def test_lead_counts_separates_the_two_binaries():
    pairs = [
        ("a.com", "in_house", "in_house"),               # both agree, lead
        ("b.com", "likely_in_house", "likely_in_house"),  # qualified only
        ("c.com", "no_cleanroom_needed", "likely_in_house"),  # qualified FP
        ("d.com", "in_house", "no_cleanroom_needed"),    # missed under both
    ]
    strict = evaluate.lead_counts(pairs, evaluate.is_strict_lead)
    assert (strict["tp"], strict["fp"], strict["fn"], strict["tn"]) == (1, 0, 1, 2)

    qualified = evaluate.lead_counts(pairs, evaluate.is_qualified_lead)
    assert (qualified["tp"], qualified["fp"], qualified["fn"], qualified["tn"]) == (2, 1, 1, 0)


def test_qualified_lead_recall_on_the_real_tier_b_shape():
    """Tier B, 2026-08-26: TP 9, FP 1, FN 3, TN 10 at the qualified level."""
    pairs = (
        [(f"tp{i}.com", "likely_in_house", "likely_in_house") for i in range(9)]
        + [("fp.com", "no_cleanroom_needed", "in_house")]
        + [(f"fn{i}.com", "in_house", "unknown") for i in range(3)]
        + [(f"tn{i}.com", "no_cleanroom_needed", "no_cleanroom_needed") for i in range(10)]
    )
    counts = evaluate.lead_counts(pairs, evaluate.is_qualified_lead)
    assert (counts["tp"], counts["fp"], counts["fn"], counts["tn"]) == (9, 1, 3, 10)
    assert counts["precision"] == pytest.approx(0.9)
    assert counts["recall"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# the circularity guard
# ---------------------------------------------------------------------------


def test_circular_rows_counts_labels_reconciled_to_a_pipeline_run():
    labels = {
        "a.com": {"domain": "a.com", "label_provenance": "reconciled_to_pipeline"},
        "b.com": {"domain": "b.com", "label_provenance": "human_reviewed"},
        "c.com": {"domain": "c.com"},  # older file, no provenance recorded
    }
    assert evaluate.circular_rows(labels) == ["a.com"]


def test_circular_rows_is_empty_on_a_label_file_with_no_provenance_field():
    """eval/goldset.jsonl (Tier A) predates the field and must not warn."""
    labels = {"a.com": {"domain": "a.com", "expected_label": "in_house"}}
    assert evaluate.circular_rows(labels) == []


def test_tier_c_is_marked_circular_on_disk():
    """Tier C's 13 reconciled rows were relabelled to match
    20260825-batch1-rescore, so scoring against it cannot measure accuracy.

    This guards the file itself, not the code: if someone strips the provenance
    field, the 96.8% starts looking like a real number again.

    The count is 12, not the original 13. `scanlonintl.com` was upgraded
    to `human_reviewed` on 2026-08-26 when the user re-checked the live site and
    confirmed `no_cleanroom_needed`, resolving a two-reviewer disagreement that
    had been filed as settled. The label did not move; its provenance did, which
    is the only way a row leaves this set legitimately. Expect this number to
    keep falling as rows get independently confirmed — that is the file becoming
    a real test. It must never rise.
    """
    path = REPO_ROOT / "eval" / "goldset_tier_c_batch1.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reconciled = [r for r in rows if r.get("status") == "reconciled"]
    assert len(reconciled) == 12
    assert all(r.get("label_provenance") == "reconciled_to_pipeline" for r in reconciled)

    scorable = [r for r in rows if r.get("expected_label")]
    labels = {r["domain"]: r for r in scorable}
    # All 12 carry a label. ventexlife.com is among them and its fix postdates
    # the run, which is why Tier C scores 30/31 rather than 31/31.
    assert len(evaluate.circular_rows(labels)) == 12


def test_tier_b_is_not_marked_circular():
    """Tier B is dual-reader adjudicated against live sites, independent of any
    run. It is the only honest accuracy instrument and must stay unmarked."""
    path = REPO_ROOT / "eval" / "goldset_tier_b_holdout25.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = {r["domain"]: r for r in rows if r.get("expected_label")}
    assert len(labels) == 23
    assert evaluate.circular_rows(labels) == []
