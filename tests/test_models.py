"""The two rules that carry the extraction design, enforced at the schema."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from cleanroom.models import (
    CLEANROOM_STATUSES,
    ClassificationRecord,
    Evidence,
    ExtractionRecord,
    Finding,
)


def _extraction(**overrides) -> ExtractionRecord:
    base = dict(
        domain="example.com",
        url="https://example.com/",
        page_type="homepage",
        crawl_date=date(2026, 8, 6),
        pack_version="medical_v1",
        page_word_count=120,
    )
    return ExtractionRecord(**{**base, **overrides})


def _classification(**overrides) -> ClassificationRecord:
    base = dict(
        domain="example.com",
        cleanroom_status="unknown",
        confidence=0.4,
        reasoning="Two pages retrieved, no facility language.",
        pages_analyzed=2,
        run_id="run-1",
        pack_version="medical_v1",
        source_crawl_date=date(2026, 8, 6),
    )
    return ClassificationRecord(**{**base, **overrides})


# ---------------------------------------------------------------------------
# Rule 1: every claim carries a verbatim quote
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quote", ["", "   ", "\n\t"])
def test_finding_requires_a_quote(quote):
    with pytest.raises(ValidationError, match="quote"):
        Finding(claim="Operates a cleanroom", quote=quote, confidence=0.9)


def test_finding_keeps_the_quote_verbatim():
    quote = "Class 7  &  8   cleanroom assembly"
    assert Finding(claim="c", quote=f"  {quote}  ", confidence=0.5).quote == quote


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_confidence_is_bounded(confidence):
    with pytest.raises(ValidationError):
        Finding(claim="c", quote="q", confidence=confidence)


# ---------------------------------------------------------------------------
# Rule 2: required identity fields are always populated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["domain", "url", "page_type", "pack_version"])
def test_required_extraction_fields_reject_blanks(field):
    with pytest.raises(ValidationError, match=field):
        _extraction(**{field: "  "})


@pytest.mark.parametrize("field", ["domain", "url", "page_type", "crawl_date", "pack_version"])
def test_required_extraction_fields_reject_missing(field):
    payload = _extraction().model_dump()
    payload.pop(field)
    with pytest.raises(ValidationError, match=field):
        ExtractionRecord.model_validate(payload)


def test_list_fields_default_empty():
    """Most pages have nothing to say about sterilisation. That is not an error."""
    record = _extraction()
    assert record.cleanroom_terms == []
    assert record.products_named == []
    assert record.trap_terms_present == []


def test_negative_word_count_rejected():
    with pytest.raises(ValidationError):
        _extraction(page_word_count=-1)


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        _extraction(sentiment="positive")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", CLEANROOM_STATUSES)
def test_all_four_labels_are_valid(status):
    assert _classification(cleanroom_status=status).cleanroom_status == status


def test_a_fifth_label_is_rejected():
    with pytest.raises(ValidationError):
        _classification(cleanroom_status="maybe")


def test_controlled_but_unclassified_is_in_house_with_null_iso_class():
    record = _classification(cleanroom_status="in_house", iso_class=None)
    assert record.cleanroom_status == "in_house"
    assert record.iso_class is None


@pytest.mark.parametrize("value", ["", "  ", "null", "None", "N/A"])
def test_stringly_typed_nulls_collapse_to_none(value):
    """Three spellings of empty downstream is three bugs. Normalise at the edge."""
    assert _classification(iso_class=value).iso_class is None


def test_iso_class_kept_when_stated():
    assert _classification(iso_class="ISO 7").iso_class == "ISO 7"


def test_evidence_requires_both_quote_and_source():
    with pytest.raises(ValidationError):
        Evidence(quote="a cleanroom", source_url="")
    with pytest.raises(ValidationError):
        Evidence(quote="", source_url="https://example.com/")


def test_needs_manual_review_defaults_false_and_is_settable():
    assert _classification().needs_manual_review is False
    assert _classification(needs_manual_review=True).needs_manual_review is True


def test_json_round_trip_preserves_dates_and_nulls():
    record = _classification(iso_class=None, cleanroom_status="none")
    assert ClassificationRecord.model_validate_json(record.model_dump_json()) == record
