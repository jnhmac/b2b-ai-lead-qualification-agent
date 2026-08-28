"""The Phase 1 gate, stated as two tests.

    Done when the goldset domains normalize cleanly and a JSONL file
    round-trips through both store backends.

Everything else in `tests/` supports these two. Run with `-s` to see the
printed detail.
"""

from __future__ import annotations

from datetime import date

import pytest

from cleanroom.models import ClassificationRecord, Evidence, ExtractionRecord, Finding
from cleanroom.normalize import normalize_all, normalize_domain
from cleanroom.store import classifications_path, extractions_path

# 46 as provided, plus the 10 thin sites Phase 7a harvested from the batch 1
# crawl on 2026-08-11. Both numbers are asserted rather than derived, because
# this guards the fixture: a gate failure must never turn out to be a
# half-written label file.
EXPECTED_GOLDSET_SIZE = 56
EXPECTED_GROUPS = {
    "Confirmed yes": 19,
    "Confirmed no": 14,
    "Hard negatives": 13,
    "Thin sites": 10,
}


# ---------------------------------------------------------------------------
# Gate 1: the goldset domains normalize cleanly
# ---------------------------------------------------------------------------


def test_goldset_shape(goldset_rows):
    """Guard the fixture itself, so a gate failure is never a bad-file failure."""
    assert len(goldset_rows) == EXPECTED_GOLDSET_SIZE

    groups: dict[str, int] = {}
    for row in goldset_rows:
        groups[row["group"]] = groups.get(row["group"], 0) + 1
    assert groups == EXPECTED_GROUPS

    labels = {row["expected_label"] for row in goldset_rows}
    assert labels <= {"in_house", "likely_in_house", "outsourced", "no_cleanroom_needed", "unknown"}


def test_goldset_domains_normalize_cleanly(goldset_rows, capsys):
    raws = [row["domain"] for row in goldset_rows]
    result = normalize_all(raws)

    with capsys.disabled():
        print(f"\n  goldset rows in           : {len(raws)}")
        print(f"  unique domains out        : {len(result.entries)}")
        print(f"  unparseable               : {len(result.invalid)}")
        print(f"  collisions                : {len(result.collisions)}")
        print(f"  parent/subsidiary merges  : {len(result.subdomain_collapses)}")
        for i, entry in enumerate(result.entries, 1):
            marker = " " if entry.raw == entry.domain else "*"
            print(f"    {i:>2}. {entry.domain:<28}{marker} <- {entry.raw}")

    assert result.invalid == [], f"unparseable goldset domains: {result.invalid}"
    assert result.collisions == [], f"unexpected collisions: {result.collisions}"
    assert len(result.entries) == EXPECTED_GOLDSET_SIZE
    assert len(set(result.domains)) == EXPECTED_GOLDSET_SIZE


def test_goldset_domains_are_already_canonical(goldset_rows):
    """Each label keys on its registrable domain, so it must survive a round trip."""
    for row in goldset_rows:
        raw = row["domain"]
        assert normalize_domain(raw).domain == raw, f"{raw} is not canonical"


def test_goldset_domains_survive_url_decoration(goldset_rows):
    """The real input list arrives as URLs. Same domain out, either way."""
    for row in goldset_rows:
        raw = row["domain"]
        assert normalize_domain(f"https://www.{raw}/about-us?ref=list#top").domain == raw
        assert normalize_domain(f"  HTTP://{raw.upper()}  ").domain == raw


# ---------------------------------------------------------------------------
# Gate 2: a JSONL file round-trips through both store backends
# ---------------------------------------------------------------------------


@pytest.fixture
def gate_records(goldset_rows) -> list[ExtractionRecord]:
    """One extraction record per goldset domain: a realistic part file."""
    return [
        ExtractionRecord(
            domain=row["domain"],
            url=f"https://{row['domain']}/capabilities",
            page_type="capabilities",
            crawl_date=date(2026, 8, 6),
            pack_version="medical_v1",
            page_word_count=400 + index,
            cleanroom_terms=[
                Finding(
                    claim="Mentions a controlled environment",
                    quote=row["rationale"],
                    confidence=0.75,
                )
            ],
            products_named=[row["company"].strip()],
            trap_terms_present=["ISO 13485"],
        )
        for index, row in enumerate(goldset_rows)
    ]


def test_jsonl_round_trips_through_both_backends(local_store, gcs_store, gate_records, capsys):
    path = extractions_path("medical_v1", "run-phase1-gate")

    for store in (local_store, gcs_store):
        assert store.exists(path) is False
        store.write_jsonl(path, gate_records)
        assert store.exists(path) is True
        assert store.read_jsonl(path, ExtractionRecord) == gate_records

    local_bytes = local_store.read_text(path)
    gcs_bytes = gcs_store.read_text(path)

    with capsys.disabled():
        print(f"\n  records written           : {len(gate_records)}")
        print(f"  path                      : {path}")
        print(f"  local  -> {local_store.uri(path)}  ({len(local_bytes):,} bytes)")
        print(f"  gcs    -> {gcs_store.uri(path)}  ({len(gcs_bytes):,} bytes)")
        print(f"  byte-identical across backends: {local_bytes == gcs_bytes}")

    assert local_bytes == gcs_bytes
    assert local_bytes.count("\n") == EXPECTED_GOLDSET_SIZE


def test_classification_jsonl_round_trips_through_both_backends(
    local_store, gcs_store, goldset_rows
):
    """The same for the other record type, including nullable `iso_class`."""
    records = [
        ClassificationRecord(
            domain=row["domain"],
            cleanroom_status=row["expected_label"],
            iso_class="ISO 7" if row["expected_label"] == "in_house" else None,
            confidence=0.8,
            reasoning=row["rationale"],
            evidence=[
                Evidence(quote=row["rationale"], source_url=f"https://{row['domain']}/")
            ],
            pages_analyzed=8,
            page_types_found=["homepage", "capabilities"],
            needs_manual_review=row["group"] == "Hard negatives",
            run_id="run-phase1-gate",
            pack_version="medical_v1",
            source_crawl_date=date(2026, 8, 6),
        )
        for row in goldset_rows
    ]
    path = classifications_path("medical_v1", "run-phase1-gate")

    for store in (local_store, gcs_store):
        store.write_jsonl(path, records)
        assert store.read_jsonl(path, ClassificationRecord) == records

    assert local_store.read_text(path) == gcs_store.read_text(path)
