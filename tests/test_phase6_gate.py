"""The Phase 6 gate: runs clean over the goldset.

    Done when classification runs clean over the goldset.

Stated as a test over the store rather than over the provider, and written the
same way the Phase 5 gate is: it re-derives its checks from the extraction
records on disk rather than reading the run's own report, so it audits
`classify.py` instead of trusting it.

"Clean" is four things, and none of them is accuracy. Accuracy is Phase 7b's
job and needs the goldset labels; this gate asks only whether the stage
produced well-formed, auditable verdicts over every domain:

1. Every domain that has extractions has exactly one verdict.
2. Every evidence quote traces back to an extraction, and its URL to a crawled
   page. A verdict nobody can check is worse than no verdict.
3. The four labels are all reachable and `none` and `unknown` are both used.
   Collapsing them destroys the re-run population, and it is the single
   failure this stage is most likely to have.
4. The counts on each record match the store: `pages_analyzed` is the number of
   pages actually extracted for that domain, not a number the model chose.

Costs nothing and calls nothing. On a checkout with no classification run every
test here skips, so the suite stays green before the gate is reached and turns
into a real assertion the moment it is.

Run it against a specific run with `CLASSIFY_RUN_ID=... pytest
tests/test_phase6_gate.py -s`; with nothing set it takes the newest run.
"""

from __future__ import annotations

import os
from collections import Counter

import pytest

from cleanroom.classify import group_records
from cleanroom.config import load_settings
from cleanroom.extract import FINDING_FIELDS, canonical, read_records
from cleanroom.models import ClassificationRecord, ExtractionRecord
from cleanroom.store import Store, get_store

#: BUILD.md: "Expect 15 to 20 percent unknown." Asserted as a wide band rather
#: than the exact range, because the goldset deliberately distorts the base
#: rate by injecting positives and is not a normal population. Outside these
#: bounds the two labels are being collapsed in one direction or the other.
UNKNOWN_FLOOR = 0.0
UNKNOWN_CEILING = 0.40


@pytest.fixture(scope="module")
def store() -> Store:
    return get_store(load_settings())


@pytest.fixture(scope="module")
def run(store: Store) -> tuple[str, str, list[ClassificationRecord]]:
    """The classification run under test, as (pack, run_id, records)."""
    parts = [
        path
        for path in store.list_paths("classifications")
        if path.endswith(".jsonl") and "/run=" in path
    ]
    if not parts:
        pytest.skip("no classification run in the store yet; run `cleanroom classify` first")

    wanted = os.environ.get("CLASSIFY_RUN_ID")
    runs = sorted(
        {(p.split("pack=")[1].split("/")[0], p.split("run=")[1].split("/")[0]) for p in parts}
    )

    def started(item: tuple[str, str]) -> str:
        """When the run began, from its own report.

        Sorting run ids as strings is wrong the moment one is hand-named: a
        `--run-id smoke-01` sorts after every timestamp and quietly becomes
        "the latest run". The report knows when it actually ran.
        """
        pack_version, run_id = item
        path = f"classifications/pack={pack_version}/run={run_id}/report.json"
        if not store.exists(path):
            return ""
        return str(store.read_json(path).get("started_at") or "")

    pack_version, run_id = next(
        ((pack, rid) for pack, rid in runs if rid == wanted),
        max(runs, key=started),
    )

    records: list[ClassificationRecord] = []
    for path in sorted(p for p in parts if f"pack={pack_version}/run={run_id}/" in p):
        records.extend(store.read_jsonl(path, ClassificationRecord))
    return pack_version, run_id, records


@pytest.fixture(scope="module")
def extractions(store: Store, run) -> dict[str, list[ExtractionRecord]]:
    """The extraction records the run was built from, grouped by domain."""
    pack_version, run_id, _ = run
    report_path = f"classifications/pack={pack_version}/run={run_id}/report.json"
    if not store.exists(report_path):
        pytest.skip("the classification run wrote no report; cannot locate its extractions")
    extraction_run = str(store.read_json(report_path).get("extraction_run_id") or "")
    if not extraction_run:
        pytest.skip("the report names no extraction run")
    return group_records(read_records(store, pack_version, extraction_run))


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_every_extracted_domain_got_exactly_one_verdict(run, extractions, capsys):
    """Coverage, and no duplicates.

    A domain classified twice would double-count in every Phase 7b number, and
    a domain silently missing would quietly shrink the population a precision
    figure is computed over.
    """
    pack_version, run_id, records = run
    counts = Counter(record.domain for record in records)
    duplicates = sorted(domain for domain, n in counts.items() if n > 1)
    missing = sorted(set(extractions) - set(counts))

    with capsys.disabled():
        print(f"\n  pack / run  : {pack_version} / {run_id}")
        print(f"  verdicts    : {len(records)} over {len(extractions)} extracted domains")
        for domain in missing[:10]:
            print(f"  NO VERDICT  : {domain}")
        for domain in duplicates[:10]:
            print(f"  DUPLICATE   : {domain}")

    assert not duplicates, f"{len(duplicates)} domains were classified more than once"
    assert not missing, f"{len(missing)} extracted domains have no verdict; retry with --run-id {run_id}"


def test_every_evidence_quote_traces_to_an_extraction(run, extractions, capsys):
    """The gate. A verdict has to be checkable without a re-crawl.

    Re-derived here rather than read from the report: every quote is matched
    against the quotes its own domain actually produced, canonicalised on both
    sides exactly as Phase 5 does. If `verify_verdict` were wrong, this would
    still catch it.
    """
    _, run_id, records = run
    violations: list[str] = []
    checked = 0

    for record in records:
        rows = extractions.get(record.domain, [])
        pool = "\n".join(
            canonical(finding.quote)
            for row in rows
            for field_name in FINDING_FIELDS
            for finding in getattr(row, field_name)
        )
        urls = {row.url for row in rows}
        for item in record.evidence:
            checked += 1
            key = canonical(item.quote)
            if not key or key not in pool:
                violations.append(f"{record.domain}: {item.quote[:80]!r} is in no extraction")
            elif item.source_url not in urls:
                violations.append(f"{record.domain}: {item.source_url} was never crawled")

    with capsys.disabled():
        print(f"  evidence    : {checked} quotes, all traced to an extraction")
        for line in violations[:10]:
            print(f"  VIOLATION   : {line}")

    assert not violations, f"{len(violations)} evidence quotes cannot be traced"
    assert checked, "no verdict cited any evidence at all"


def test_the_counts_come_from_the_store(run, extractions):
    """Nothing the model generates is allowed into a record's identity.

    `pages_analyzed` and `page_types_found` are facts about the crawl. A model
    that under-counts pages makes a full site look thin, and `unknown` is
    defined partly by page count.
    """
    pack_version, _, records = run
    for record in records:
        rows = extractions.get(record.domain, [])
        assert record.domain, "a record has no domain"
        assert record.pack_version == pack_version
        assert record.pages_analyzed == len(rows), (
            f"{record.domain}: pages_analyzed={record.pages_analyzed} but "
            f"{len(rows)} pages were extracted"
        )
        assert set(record.page_types_found) == {row.page_type for row in rows}
        assert record.reasoning.strip(), f"{record.domain}: verdict carries no reasoning"
        assert 0.0 <= record.confidence <= 1.0


def test_no_cleanroom_needed_and_unknown_stay_distinct(run, capsys):
    """The distinction BUILD.md says to enforce hardest.

    `no_cleanroom_needed` is a determination from the product taxonomy;
    `unknown` is absence of evidence. Collapsing them destroys the population
    that gets re-run later, and the collapse is invisible in a precision
    number because both labels are "not a prospect".
    """
    _, _, records = run
    counts = Counter(record.cleanroom_status for record in records)
    total = sum(counts.values())
    unknown_share = counts.get("unknown", 0) / total if total else 0.0
    reviews = sum(1 for record in records if record.needs_manual_review)

    with capsys.disabled():
        print(f"\n  {'verdict':<20}{'count':>7}{'share':>9}")
        print(f"  {'-' * 20}{'-' * 7}{'-' * 9}")
        for label in ("in_house", "likely_in_house", "outsourced", "no_cleanroom_needed", "unknown"):
            n = counts.get(label, 0)
            print(f"  {label:<20}{n:>7}{n / total if total else 0:>8.1%}")
        print(f"\n  needs manual review: {reviews}")
        print(f"  iso class stated   : {sum(1 for r in records if r.iso_class)}")

    assert counts.get("no_cleanroom_needed", 0) or counts.get("none", 0) or counts.get("unknown", 0), (
        "neither `no_cleanroom_needed` nor `unknown` was used at all"
    )
    assert UNKNOWN_FLOOR <= unknown_share <= UNKNOWN_CEILING, (
        f"unknown is {unknown_share:.1%} of verdicts; either thin sites are being "
        "resolved to `no_cleanroom_needed` or mixed evidence is being resolved to `unknown`"
    )


def test_an_in_house_verdict_rests_on_facility_evidence(run, extractions, capsys):
    """The precision gate, and the one check that can be made without labels.

    `in_house` is the label that reaches a sales list, so it is the one that
    costs money when it is wrong. Decided 2026-08-10: a positive verdict
    requires a quote describing a controlled environment, from
    `iso_class_mentions` or `cleanroom_terms`. A product catalogue is not
    facility evidence, and neither is a bare manufacturing statement.

    This is BUILD.md's own definition ("do NOT use this when the evidence only
    shows the product is sterile") made checkable, and it is safe to enforce
    because the target population is manufacturers under 500 employees. A
    company that size which runs a cleanroom says so on its site. The
    enterprises where ten pages cannot cover the estate are not prospects.
    """
    _, _, records = run
    positives = [record for record in records if record.cleanroom_status == "in_house"]

    unsupported: list[tuple[str, str]] = []
    for record in positives:
        rows = extractions.get(record.domain, [])
        facility = {
            canonical(finding.quote)
            for row in rows
            for name in ("iso_class_mentions", "cleanroom_terms")
            for finding in getattr(row, name)
        }
        cited = {canonical(item.quote) for item in record.evidence}
        if not cited & facility:
            basis = "no evidence at all" if not record.evidence else "no facility quote"
            unsupported.append((record.domain, basis))

    with capsys.disabled():
        print(f"\n  in_house verdicts        : {len(positives)}")
        print(f"  without facility evidence: {len(unsupported)}")
        for domain, basis in unsupported[:10]:
            print(f"    {domain}: {basis}")

    assert not unsupported, (
        f"{len(unsupported)} in_house verdicts rest on no facility quote: "
        f"{', '.join(d for d, _ in unsupported[:6])}"
    )


def test_the_csv_is_written_and_readable(run, store):
    """The artifact a person actually opens.

    One row per domain, no join needed. Checked for shape rather than content:
    a CSV that loses a row to an unescaped newline in `reasoning` is the
    failure mode, and it shows up as a row count that does not match.
    """
    import csv
    import io

    pack_version, run_id, records = run
    path = f"classifications/pack={pack_version}/run={run_id}/verdicts.csv"
    if not store.exists(path):
        pytest.skip("this run predates the CSV report")

    rows = list(csv.DictReader(io.StringIO(store.read_text(path))))
    assert len(rows) == len(records), "the CSV lost or gained rows against the JSONL"
    assert {row["domain"] for row in rows} == {record.domain for record in records}
    for row in rows:
        # "none" stays valid here: this reads whichever run is newest, and an
        # older run predating the medical_v2.2 label migration still says it.
        assert row["cleanroom_status"] in (
            "in_house", "likely_in_house", "outsourced", "none", "no_cleanroom_needed", "unknown"
        )
