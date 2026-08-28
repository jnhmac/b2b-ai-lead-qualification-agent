"""The Phase 5 gate: verbatim quote present on every claim.

    Done when a verbatim quote is present on every claim.

Stated as a test over the store rather than over the provider. It re-derives
the check from scratch — reading each record's quote and each page's markdown
off disk and matching them itself — so it is an independent audit of the run
rather than a re-reading of the run's own report. If `extract.py`'s verifier
were wrong, this would still catch it.

Costs nothing and calls nothing. On a checkout with no extraction run every
test here skips, so the suite stays green before the gate is reached and turns
into a real assertion the moment it is.

Run it against a specific run with `EXTRACT_RUN_ID=... pytest
tests/test_phase5_gate.py -s`; with nothing set it takes the newest run in the
store.
"""

from __future__ import annotations

import os
from collections import Counter
import pytest

from cleanroom.config import load_settings
from cleanroom.extract import FINDING_FIELDS, _loose, canonical, is_verbatim, manifest_pages
from cleanroom.models import ExtractionRecord
from cleanroom.store import Store, get_store


@pytest.fixture(scope="module")
def store() -> Store:
    return get_store(load_settings())


@pytest.fixture(scope="module")
def run(store: Store) -> tuple[str, str, list[ExtractionRecord]]:
    """The extraction run under test, as (pack, run_id, records)."""
    parts = [
        path
        for path in store.list_paths("extractions")
        if path.endswith(".jsonl") and "/run=" in path
    ]
    if not parts:
        pytest.skip("no extraction run in the store yet; run `cleanroom extract` first")

    wanted = os.environ.get("EXTRACT_RUN_ID")
    runs = sorted({(p.split("pack=")[1].split("/")[0], p.split("run=")[1].split("/")[0]) for p in parts})

    def started(item: tuple[str, str]) -> str:
        """When the run began, from its own report.

        Sorting run ids as strings is wrong the moment one is hand-named: a
        `--run-id smoke-01` sorts after every timestamp and quietly becomes
        "the latest run". The report knows when it actually ran.
        """
        pack_version, run_id = item
        path = f"extractions/pack={pack_version}/run={run_id}/report.json"
        if not store.exists(path):
            return ""
        return str(store.read_json(path).get("started_at") or "")

    pack_version, run_id = next(
        ((pack, rid) for pack, rid in runs if rid == wanted),
        max(runs, key=started),
    )

    records: list[ExtractionRecord] = []
    for path in sorted(p for p in parts if f"pack={pack_version}/run={run_id}/" in p):
        records.extend(store.read_jsonl(path, ExtractionRecord))
    return pack_version, run_id, records


@pytest.fixture(scope="module")
def pages(store: Store, run) -> dict[tuple[str, str], str]:
    """The stored markdown for every page the run produced a record for."""
    _, _, records = run
    text: dict[tuple[str, str], str] = {}
    for record in records:
        key = (record.domain, record.url)
        if key in text:
            continue
        for page in manifest_pages(store, record.domain, record.crawl_date):
            if page.url == record.url:
                text[key] = store.read_text(page.stored_path)
                break
    return text


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_every_claim_carries_a_verbatim_quote(run, pages, capsys):
    """The gate. Every quote on every finding must be findable on its page.

    Checked against the full stored markdown, which is a superset of the text
    the extractor was shown when a long page was truncated. A quote that
    verified at run time therefore cannot fail here for that reason, and a
    failure means a quote reached the store that the page does not contain.
    """
    pack_version, run_id, records = run
    violations: list[str] = []
    checked = 0

    for record in records:
        page = pages.get((record.domain, record.url))
        if page is None:
            violations.append(f"{record.domain} {record.url}: record has no stored page")
            continue
        canonical_page = canonical(page)
        for field_name in FINDING_FIELDS:
            for item in getattr(record, field_name):
                checked += 1
                assert item.quote.strip(), f"{record.url}: {field_name} has a blank quote"
                if not is_verbatim(item.quote, canonical_page):
                    violations.append(
                        f"{record.domain} {field_name}: {item.quote[:90]!r} is not on {record.url}"
                    )

    with capsys.disabled():
        print(f"\n  pack / run  : {pack_version} / {run_id}")
        print(f"  records     : {len(records)}")
        print(f"  claims      : {checked}, all re-verified against the stored markdown")
        for line in violations[:10]:
            print(f"  VIOLATION   : {line}")

    assert not violations, f"{len(violations)} claims are not verbatim"
    assert checked, "no claims at all; the extractor is returning nothing to verify"


def test_products_and_trap_terms_are_grounded(run, pages):
    """The two string lists carry no quote, so the value itself has to appear.

    Neither is matched against the pack. Both are uninterpreted at this stage
    and matched at classify time, which is what lets the product taxonomy and
    the trap list change without re-extracting a page. The guard that survives
    here is the one that matters: a certification a model supplied from sector
    knowledge rather than from the page does not appear on the page.
    """
    _, _, records = run

    for record in records:
        page = pages.get((record.domain, record.url))
        if page is None:
            continue
        canonical_page = canonical(page)
        for name in record.products_named:
            assert canonical(name) in canonical_page, (
                f"{record.domain}: product {name!r} is not on {record.url}"
            )
        for term in record.trap_terms_present:
            assert _loose(canonical(term)) in _loose(canonical_page), (
                f"{record.domain}: trap term {term!r} is not on {record.url}"
            )


def test_the_identity_fields_come_from_the_manifest(run, store):
    """Nothing the model generates is allowed into a record's identity.

    A hallucinated URL in a record whose whole job is to be auditable is worse
    than a missing record, which is why these six fields are assembled locally
    and never asked for.
    """
    pack_version, _, records = run
    by_domain: dict[str, set[str]] = {}

    for record in records:
        assert record.domain and record.url and record.page_type
        assert record.pack_version == pack_version
        assert record.page_word_count >= 0
        if record.domain not in by_domain:
            by_domain[record.domain] = {
                page.url for page in manifest_pages(store, record.domain, record.crawl_date)
            }
        assert record.url in by_domain[record.domain], (
            f"{record.domain}: {record.url} is not in the crawl manifest"
        )


def test_every_stored_page_produced_a_record(run, store, capsys):
    """Coverage. A page in the manifest with no record is a silent gap.

    Not fatal on its own — a page can legitimately fail and be retried under
    the same run id — but it has to be visible, because Phase 6 counts pages
    and a domain that lost half its pages to failures reads as a thin site.
    """
    pack_version, run_id, records = run
    extracted = {(record.domain, record.url) for record in records}
    domains = sorted({record.domain for record in records})

    gaps: list[str] = []
    total = 0
    for domain in domains:
        for page in manifest_pages(store, domain, next(
            r.crawl_date for r in records if r.domain == domain
        )):
            total += 1
            if (domain, page.url) not in extracted:
                gaps.append(f"{domain}: {page.url}")

    with capsys.disabled():
        print(f"\n  coverage    : {len(extracted)} of {total} stored pages across {len(domains)} domains")
        for line in gaps[:10]:
            print(f"  no record   : {line}")
        if len(gaps) > 10:
            print(f"  ... and {len(gaps) - 10} more")

    assert len(gaps) / total < 0.05 if total else True, (
        f"{len(gaps)} of {total} stored pages produced no record; retry with "
        f"--run-id {run_id}"
    )


def test_what_the_run_found(run, capsys):
    """Not an assertion so much as the handover to Phase 6.

    What matters going into classification is that the fields are not all
    empty and not all full: a field that never fires is a prompt that cannot
    see it, and a field that fires on every page is a prompt inventing it.
    """
    _, _, records = run
    counts = Counter()
    pages_with = Counter()
    for record in records:
        for field_name in FINDING_FIELDS:
            found = len(getattr(record, field_name))
            counts[field_name] += found
            pages_with[field_name] += int(found > 0)
        counts["products_named"] += len(record.products_named)
        pages_with["products_named"] += int(bool(record.products_named))
        counts["trap_terms_present"] += len(record.trap_terms_present)
        pages_with["trap_terms_present"] += int(bool(record.trap_terms_present))

    total = len(records)
    evidence_domains = {
        record.domain for record in records if record.iso_class_mentions or record.cleanroom_terms
    }

    with capsys.disabled():
        print(f"\n  {'field':<24}{'items':>8}{'pages':>8}{'% of pages':>12}")
        print(f"  {'-' * 24}{'-' * 8}{'-' * 8}{'-' * 12}")
        for field_name in (*FINDING_FIELDS, "products_named", "trap_terms_present"):
            share = pages_with[field_name] / total if total else 0
            print(
                f"  {field_name:<24}{counts[field_name]:>8}"
                f"{pages_with[field_name]:>8}{share:>11.0%}"
            )
        print(f"\n  domains with cleanroom or ISO-class evidence: {len(evidence_domains)}")

    assert counts["cleanroom_terms"], "no page mentioned a cleanroom; the prompt is not seeing them"
    assert counts["products_named"], "no page named a product"
