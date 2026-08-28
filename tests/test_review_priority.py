"""review_priority(): a pure ranking function over already-classified,
already-flagged rows. No LLM call, no store, no network — see
eval/review_priority.py's own docstring for why that matters here.
"""

from __future__ import annotations

from datetime import date

from review_priority import review_priority

from cleanroom.models import ClassificationRecord, Evidence

CRAWL_DATE = date(2026, 8, 7)


def classification(**overrides) -> ClassificationRecord:
    payload = {
        "domain": "acme.com",
        "cleanroom_status": "no_cleanroom_needed",
        "confidence": 0.7,
        "reasoning": "reasoning",
        "evidence": [],
        "pages_analyzed": 10,
        "needs_manual_review": False,
        "flag_reasons": [],
        "run_id": "run-1",
        "pack_version": "medical_v1",
        "source_crawl_date": CRAWL_DATE,
    }
    payload.update(overrides)
    return ClassificationRecord(**payload)


def test_a_non_flagged_row_scores_zero():
    assert review_priority(classification(needs_manual_review=False)) == 0.0


def test_thin_render_and_site_moved_score_zero_regardless_of_other_flags():
    """These need a re-crawl or an identity decision, not closer reading —
    review_priority() reports them separately, not ranked among real
    evidence disagreements."""
    for status in ("thin_render", "site_moved"):
        record = classification(
            needs_manual_review=True,
            run_status=status,
            flag_reasons=["ungrounded_negative"],
        )
        assert review_priority(record) == 0.0


def test_no_cleanroom_needed_with_ungrounded_negative_scores_highest():
    """The pipeline's worst failure mode — a real lead silently deleted —
    should float to the top of the queue."""
    worst = classification(
        cleanroom_status="no_cleanroom_needed",
        needs_manual_review=True,
        flag_reasons=["ungrounded_negative"],
        confidence=0.6,
    )
    milder = classification(
        cleanroom_status="in_house",
        needs_manual_review=True,
        flag_reasons=["llm_flagged"],
        confidence=0.6,
    )
    assert review_priority(worst) > review_priority(milder)


def test_matcher_disagreement_toward_a_better_lead_outweighs_the_other_direction():
    up = classification(
        cleanroom_status="no_cleanroom_needed",
        needs_manual_review=True,
        flag_reasons=["matcher_disagreement"],
        matcher_verdict={
            "cleanroom_status": "likely_in_house", "confidence": 0.55,
            "entity_type": "manufacturer", "run_status": "ok",
            "iso_class": None, "needs_manual_review": True, "reasons": [],
        },
    )
    down = classification(
        cleanroom_status="likely_in_house",
        needs_manual_review=True,
        flag_reasons=["matcher_disagreement"],
        matcher_verdict={
            "cleanroom_status": "unknown", "confidence": 0.40,
            "entity_type": "manufacturer", "run_status": "ok",
            "iso_class": None, "needs_manual_review": True, "reasons": [],
        },
    )
    assert review_priority(up) > review_priority(down)


def test_confidence_and_evidence_count_move_the_score_as_expected():
    low_confidence = classification(needs_manual_review=True, confidence=0.2)
    high_confidence = classification(needs_manual_review=True, confidence=0.9)
    assert review_priority(low_confidence) > review_priority(high_confidence)

    thin_evidence = classification(needs_manual_review=True, evidence=[])
    thick_evidence = classification(
        needs_manual_review=True,
        evidence=[Evidence(quote="q", source_url="https://acme.com/x")] * 3,
    )
    assert review_priority(thick_evidence) > review_priority(thin_evidence)
