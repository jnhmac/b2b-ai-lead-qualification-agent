"""Store behaviour. Every functional test runs against both backends."""

from __future__ import annotations

from datetime import date

import pytest

from cleanroom.models import ClassificationRecord, Evidence, ExtractionRecord, Finding
from cleanroom.store import (
    GCSStore,
    LocalStore,
    ObjectNotFoundError,
    StoreConflictError,
    StoreError,
    classifications_path,
    decode_jsonl,
    encode_jsonl,
    extractions_path,
    get_store,
    parse_gs_uri,
    raw_manifest_path,
    raw_page_path,
    url_hash,
)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_raw_paths_are_keyed_by_crawl_date_only():
    """No pack version anywhere under raw/. Scraping and prompting version apart."""
    path = raw_page_path("example.com", date(2026, 8, 6), "https://example.com/about")
    assert path.startswith("raw/example.com/2026-08-06/")
    assert path.endswith(".md")
    assert "pack" not in path
    assert raw_manifest_path("example.com", "2026-08-06") == (
        "raw/example.com/2026-08-06/manifest.json"
    )


def test_derived_paths_carry_pack_and_run():
    assert extractions_path("medical_v1", "run-2026-08-06a") == (
        "extractions/pack=medical_v1/run=run-2026-08-06a/part-00000.jsonl"
    )
    assert classifications_path("medical_v1", "run-x", part=12) == (
        "classifications/pack=medical_v1/run=run-x/part-00012.jsonl"
    )


def test_url_hash_is_stable_and_distinct():
    assert url_hash("https://example.com/a") == url_hash("  https://example.com/a  ")
    assert url_hash("https://example.com/a") != url_hash("https://example.com/b")
    assert len(url_hash("https://example.com/a")) == 16


@pytest.mark.parametrize("bad", ["../etc", "a/../../b", "/absolute/path"])
def test_path_traversal_is_rejected(store, bad):
    with pytest.raises(ValueError):
        store.write_text(bad, "nope")


@pytest.mark.parametrize("bad", ["", "has space", "slash/inside", ".."])
def test_unsafe_path_segments_are_rejected(bad):
    with pytest.raises(ValueError):
        extractions_path(bad, "run-1")


def test_parse_gs_uri():
    assert parse_gs_uri("gs://bucket/prefix/deeper") == ("bucket", "prefix/deeper")
    assert parse_gs_uri("gs://bucket") == ("bucket", "")
    assert parse_gs_uri("gs://bucket/") == ("bucket", "")
    with pytest.raises(ValueError):
        parse_gs_uri("./data")


# ---------------------------------------------------------------------------
# Round trips, both backends
# ---------------------------------------------------------------------------


def _extraction(url: str = "https://example.com/capabilities") -> ExtractionRecord:
    return ExtractionRecord(
        domain="example.com",
        url=url,
        page_type="capabilities",
        crawl_date=date(2026, 8, 6),
        pack_version="medical_v1",
        page_word_count=812,
        cleanroom_terms=[
            Finding(
                claim="Operates a Class 7 cleanroom",
                quote="Class 7 & 8 cleanroom assembly, test, and packaging",
                confidence=0.92,
            )
        ],
        products_named=["catheters", "guidewires"],
        trap_terms_present=["ISO 13485"],
    )


def _classification() -> ClassificationRecord:
    return ClassificationRecord(
        domain="example.com",
        cleanroom_status="in_house",
        iso_class="ISO 7",
        confidence=0.88,
        reasoning="Capabilities page states cleanroom assembly with an ISO class.",
        evidence=[
            Evidence(
                quote="Class 7 & 8 cleanroom assembly, test, and packaging",
                source_url="https://example.com/capabilities",
            )
        ],
        pages_analyzed=7,
        page_types_found=["homepage", "capabilities", "quality"],
        needs_manual_review=False,
        run_id="run-2026-08-06a",
        pack_version="medical_v1",
        source_crawl_date=date(2026, 8, 6),
    )


def test_jsonl_round_trip_extractions(store):
    path = extractions_path("medical_v1", "run-2026-08-06a")
    records = [_extraction(), _extraction("https://example.com/quality")]

    assert store.exists(path) is False
    uri = store.write_jsonl(path, records)
    assert store.exists(path) is True
    assert path in uri

    loaded = store.read_jsonl(path, ExtractionRecord)
    assert loaded == records
    assert loaded[0].crawl_date == date(2026, 8, 6)
    assert loaded[0].cleanroom_terms[0].quote == (
        "Class 7 & 8 cleanroom assembly, test, and packaging"
    )


def test_jsonl_round_trip_classifications(store):
    path = classifications_path("medical_v1", "run-2026-08-06a")
    record = _classification()
    store.write_jsonl(path, [record])
    assert store.read_jsonl(path, ClassificationRecord) == [record]


def test_jsonl_round_trip_plain_dicts(store):
    path = "extractions/pack=medical_v1/run=r1/part-00000.jsonl"
    rows = [{"domain": "a.com", "crawl_date": date(2026, 8, 6)}, {"domain": "b.com"}]
    store.write_jsonl(path, rows)
    assert store.read_jsonl(path) == [
        {"domain": "a.com", "crawl_date": "2026-08-06"},
        {"domain": "b.com"},
    ]


def test_markdown_round_trip(store):
    path = raw_page_path("example.com", date(2026, 8, 6), "https://example.com/about")
    body = "# About\n\nWe operate an ISO 7 cleanroom.\n\n- Ünïcode ✓\n"
    store.write_markdown(path, body)
    assert store.read_markdown(path) == body


def test_json_round_trip(store):
    path = raw_manifest_path("example.com", date(2026, 8, 6))
    manifest = {
        "domain": "example.com",
        "crawl_date": date(2026, 8, 6),
        "attempted": ["https://example.com/", "https://example.com/facility"],
        "not_found": ["https://example.com/facility"],
        "empty_page_types": ["careers"],
    }
    store.write_json(path, manifest)
    assert store.read_json(path)["not_found"] == ["https://example.com/facility"]


def test_empty_jsonl_round_trips_as_empty_list(store):
    path = extractions_path("medical_v1", "run-empty")
    store.write_jsonl(path, [])
    assert store.exists(path) is True
    assert store.read_jsonl(path, ExtractionRecord) == []


# ---------------------------------------------------------------------------
# Append-only and error surfaces
# ---------------------------------------------------------------------------


def test_writes_are_append_only(store):
    path = extractions_path("medical_v1", "run-1")
    store.write_jsonl(path, [_extraction()])

    with pytest.raises(StoreConflictError):
        store.write_jsonl(path, [_extraction()])

    # The escape hatch exists, but has to be asked for.
    store.write_jsonl(path, [_extraction("https://example.com/other")], overwrite=True)
    assert store.read_jsonl(path, ExtractionRecord)[0].url == "https://example.com/other"


def test_reading_a_missing_object_raises(store):
    with pytest.raises(ObjectNotFoundError):
        store.read_text("extractions/pack=medical_v1/run=nope/part-00000.jsonl")


def test_list_paths(store):
    store.write_markdown(raw_page_path("a.com", "2026-08-06", "https://a.com/"), "a")
    store.write_markdown(raw_page_path("b.com", "2026-08-06", "https://b.com/"), "b")
    store.write_jsonl(extractions_path("medical_v1", "run-1"), [_extraction()])

    assert len(store.list_paths("raw")) == 2
    assert store.list_paths("raw/a.com") == [
        raw_page_path("a.com", "2026-08-06", "https://a.com/")
    ]
    assert len(store.list_paths()) == 3
    assert store.list_paths("nothing/here") == []


def test_malformed_jsonl_reports_the_line(store):
    path = "extractions/pack=medical_v1/run=broken/part-00000.jsonl"
    store.write_text(path, '{"domain": "a.com"}\nnot json\n')
    with pytest.raises(StoreError, match="line 2"):
        store.read_jsonl(path)


def test_encode_decode_without_a_backend():
    text = encode_jsonl([_extraction()])
    assert text.endswith("\n")
    assert decode_jsonl(text, ExtractionRecord) == [_extraction()]


def test_decode_jsonl_survives_a_unicode_line_separator_in_a_quote():
    """A verbatim quote can legally contain U+2028/U+2029 with no JSON escaping
    required — the JSON spec only mandates escaping `"`, `\\` and control
    characters below U+0020. Real quotes hit this: scraped markdown renders a
    soft line break as U+2028, so `matched_terms[]`'s verbatim quotes carry it
    straight through `model_dump_json()`.

    `str.splitlines()` treats U+2028/U+2029 (and a handful of other Unicode
    line-boundary characters) as line breaks too, not just `\\n` — so a naive
    `text.splitlines()` over JSONL silently shreds one valid record into two
    fragments, one of which fails to parse as "unterminated string". JSONL's
    own delimiter is `\\n` and nothing else; splitting has to match that.
    """
    quote_with_separator = "# Accelerate  your impact\n\nFrom prototype to production"
    record = _extraction()
    record = record.model_copy(
        update={"manufacturing_language": [
            Finding(claim="x", quote=quote_with_separator, confidence=0.5)
        ]}
    )
    other = _extraction("https://example.com/quality")

    text = encode_jsonl([record, other])
    decoded = decode_jsonl(text, ExtractionRecord)

    assert len(decoded) == 2, f"expected 2 records, got {len(decoded)} — a line was split apart"
    assert decoded[0].manufacturing_language[0].quote == quote_with_separator


def test_encode_rejects_unsupported_records():
    with pytest.raises(TypeError):
        encode_jsonl(["just a string"])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_get_store_returns_local_for_local_settings(local_settings):
    assert isinstance(get_store(local_settings), LocalStore)


def test_get_store_returns_gcs_for_gs_root(local_settings):
    from tests.conftest import FakeGCSClient

    settings = type(local_settings)(
        store_backend="gcs",
        store_root="gs://cleanroom-prod/v1",
        pack_dir=local_settings.pack_dir,
    )
    store = get_store(settings, client=FakeGCSClient())
    assert isinstance(store, GCSStore)
    assert store.bucket_name == "cleanroom-prod"
    assert store.prefix == "v1"


def test_backends_agree_on_paths_and_content(local_store, gcs_store):
    """The point of the interface: identical layout, identical bytes."""
    path = extractions_path("medical_v1", "run-parity")
    records = [_extraction()]

    local_store.write_jsonl(path, records)
    gcs_store.write_jsonl(path, records)

    assert local_store.read_text(path) == gcs_store.read_text(path)
    assert local_store.list_paths() == gcs_store.list_paths()
    assert local_store.read_jsonl(path, ExtractionRecord) == gcs_store.read_jsonl(
        path, ExtractionRecord
    )
    assert local_store.uri(path).endswith(path)
    assert gcs_store.uri(path) == f"gs://cleanroom-test/dev/{path}"
