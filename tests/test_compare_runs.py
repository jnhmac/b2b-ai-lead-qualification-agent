"""Phase 7b: the run diff.

`compare_runs.py` is read-only over the store, so everything here is built from
records in memory. What it has to get right is not arithmetic but bookkeeping:
a diff that mislabels which way a domain moved sends the next prompt edit in
the wrong direction, and nothing downstream would catch it.

The two that matter most are `input_changed` and `direction`. The first is what
stops a re-crawl being read as a prompt improvement; the second is what the
iteration loop actually steers on.
"""

from __future__ import annotations

import csv
import io
from datetime import date

import pytest

from compare_runs import build_csv, compare, direction, transitions
from cleanroom.models import ClassificationRecord, Evidence


def record(
    domain: str,
    status: str,
    *,
    confidence: float = 0.9,
    evidence: tuple[str, ...] = (),
    pages: int = 10,
    crawl_date: str = "2026-08-07",
    iso_class: str | None = None,
    review: bool = False,
    run_id: str = "run",
) -> ClassificationRecord:
    return ClassificationRecord(
        domain=domain,
        cleanroom_status=status,
        iso_class=iso_class,
        confidence=confidence,
        reasoning=f"because {status}",
        evidence=[Evidence(quote=q, source_url=f"https://{domain}/p") for q in evidence],
        pages_analyzed=pages,
        page_types_found=["home"],
        needs_manual_review=review,
        run_id=run_id,
        pack_version="medical_v1",
        source_crawl_date=date.fromisoformat(crawl_date),
    )


def rows_by_domain(a, b):
    return {row["domain"]: row for row in compare(a, b)}


def test_only_shared_domains_are_diffed():
    a = {"x.com": record("x.com", "none"), "gone.com": record("gone.com", "none")}
    b = {"x.com": record("x.com", "none"), "new.com": record("new.com", "none")}
    assert [row["domain"] for row in compare(a, b)] == ["x.com"]


def test_a_verdict_change_is_flagged_and_a_lead_change_is_flagged_separately():
    # none -> outsourced changed the label and sold nothing differently.
    a = {"x.com": record("x.com", "none"), "y.com": record("y.com", "unknown")}
    b = {"x.com": record("x.com", "outsourced"), "y.com": record("y.com", "in_house")}
    rows = rows_by_domain(a, b)
    assert rows["x.com"]["verdict_changed"] and not rows["x.com"]["lead_changed"]
    assert rows["y.com"]["verdict_changed"] and rows["y.com"]["lead_changed"]


def test_a_re_crawl_is_marked_so_it_is_not_read_as_a_prompt_win():
    """The distinction the whole file exists for.

    Same verdict move, two causes. Only the first says anything about a prompt.
    """
    a = {
        "prompt.com": record("prompt.com", "unknown", pages=10),
        "recrawl.com": record("recrawl.com", "unknown", pages=3),
        "redate.com": record("redate.com", "unknown", crawl_date="2026-08-07"),
    }
    b = {
        "prompt.com": record("prompt.com", "in_house", pages=10),
        "recrawl.com": record("recrawl.com", "in_house", pages=10),
        "redate.com": record("redate.com", "in_house", crawl_date="2026-08-11"),
    }
    rows = rows_by_domain(a, b)
    assert not rows["prompt.com"]["input_changed"]
    assert rows["recrawl.com"]["input_changed"]
    assert rows["redate.com"]["input_changed"]


def test_evidence_is_compared_by_text_not_by_count():
    a = {"x.com": record("x.com", "in_house", evidence=("kept quote", "dropped quote"))}
    b = {"x.com": record("x.com", "in_house", evidence=("kept   quote\n", "added quote"))}
    row = rows_by_domain(a, b)["x.com"]
    # The kept quote was re-wrapped, not changed. Whitespace is not a diff.
    assert row["quotes_added"] == ["added quote"]
    assert row["quotes_removed"] == ["dropped quote"]
    assert not row["verdict_changed"]


def test_iso_class_and_review_flag_changes_are_visible_under_an_unchanged_verdict():
    a = {"x.com": record("x.com", "in_house", iso_class=None, review=True)}
    b = {"x.com": record("x.com", "in_house", iso_class="ISO 8", review=False)}
    row = rows_by_domain(a, b)["x.com"]
    assert row["iso_changed"] and row["review_changed"]
    assert not row["verdict_changed"]


def test_direction_names_what_a_change_did_to_the_label():
    fixed = rows_by_domain(
        {"x.com": record("x.com", "unknown")}, {"x.com": record("x.com", "none")}
    )["x.com"]
    broke = rows_by_domain(
        {"x.com": record("x.com", "none")}, {"x.com": record("x.com", "unknown")}
    )["x.com"]
    sideways = rows_by_domain(
        {"x.com": record("x.com", "unknown")}, {"x.com": record("x.com", "outsourced")}
    )["x.com"]
    held = rows_by_domain(
        {"x.com": record("x.com", "none")}, {"x.com": record("x.com", "none")}
    )["x.com"]

    assert direction(fixed, "none") == "FIXED"
    assert direction(broke, "none") == "BROKE"
    assert direction(sideways, "none") == "still wrong"
    assert direction(held, "none") == "unchanged"
    # An unlabelled domain still diffs; it just cannot be judged.
    assert direction(fixed, None) == ""


def test_transitions_counts_pairs():
    a = {d: record(d, "unknown") for d in ("a.com", "b.com", "c.com")}
    b = {
        "a.com": record("a.com", "none"),
        "b.com": record("b.com", "none"),
        "c.com": record("c.com", "in_house"),
    }
    counts = transitions(compare(a, b))
    assert counts[("unknown", "none")] == 2
    assert counts[("unknown", "in_house")] == 1


def test_csv_carries_the_label_join_and_survives_a_multi_sentence_reasoning():
    a = {"x.com": record("x.com", "unknown", evidence=("old",))}
    b = {"x.com": record("x.com", "none", evidence=("new",))}
    text = build_csv(compare(a, b), {"x.com": {"expected_label": "none"}})
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == 1
    row = rows[0]
    assert row["verdict_a"] == "unknown" and row["verdict_b"] == "none"
    assert row["expected_label"] == "none" and row["direction"] == "FIXED"
    assert row["quotes_added"] == "new" and row["quotes_removed"] == "old"


def test_csv_leaves_the_label_columns_empty_when_a_domain_is_unlabelled():
    a = {"x.com": record("x.com", "unknown")}
    b = {"x.com": record("x.com", "none")}
    row = next(iter(csv.DictReader(io.StringIO(build_csv(compare(a, b), {})))))
    assert row["expected_label"] == "" and row["direction"] == ""


class TestMcNemar:
    """The paired test that stops a noise move being reported as an improvement.

    The motivating case: the 2026-08-25 regression run was reported as
    "17/23 -> 18/23 (73.9% -> 78.3%)", read as a real if modest gain, and
    carried into three documents. It is 2 domains fixed against 1 regressed.
    Exact McNemar gives p = 1.00 -- the run measured nothing about accuracy.
    See outputs/2026-08-26_design-review.md F10.
    """

    def test_the_real_2026_08_25_move_is_not_significant(self):
        from compare_runs import mcnemar

        result = mcnemar(fixed=2, broke=1)
        assert result["n_discordant"] == 3
        assert result["p_value"] == pytest.approx(1.0)
        assert result["significant"] is False

    def test_a_one_sided_sweep_of_six_is_the_smallest_significant_result(self):
        """Where the boundary actually sits at these sample sizes, so nobody has
        to guess whether their next run could possibly clear it."""
        from compare_runs import mcnemar

        assert mcnemar(fixed=5, broke=0)["significant"] is False
        assert mcnemar(fixed=6, broke=0)["significant"] is True
        assert mcnemar(fixed=6, broke=0)["p_value"] == pytest.approx(0.03125)

    def test_a_regression_is_flagged_as_readily_as_a_gain(self):
        from compare_runs import mcnemar

        gain = mcnemar(fixed=8, broke=1)
        loss = mcnemar(fixed=1, broke=8)
        assert gain["p_value"] == pytest.approx(loss["p_value"])
        assert gain["significant"] and loss["significant"]

    def test_no_discordant_pairs_is_not_evidence_of_equivalence(self):
        from compare_runs import mcnemar

        result = mcnemar(fixed=0, broke=0)
        assert result["n_discordant"] == 0
        assert result["p_value"] == pytest.approx(1.0)
        assert result["significant"] is False

    def test_p_value_is_two_sided(self):
        from compare_runs import mcnemar

        # 3 of 3 one way: 2 * (1/8) = 0.25, not the one-sided 0.125.
        assert mcnemar(fixed=3, broke=0)["p_value"] == pytest.approx(0.25)

    def test_p_value_never_exceeds_one(self):
        from compare_runs import mcnemar

        for fixed, broke in ((1, 1), (2, 2), (5, 5), (10, 9)):
            assert 0.0 <= mcnemar(fixed=fixed, broke=broke)["p_value"] <= 1.0

    def test_describe_names_the_smallest_move_that_would_have_counted(self):
        from compare_runs import mcnemar

        text = mcnemar(fixed=2, broke=1)["describe"]
        assert "p = 1.00" in text
        assert "6" in text  # the b+c threshold a reader needs to know
