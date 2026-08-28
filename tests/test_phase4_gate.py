"""The Phase 4 gate, stated as a test over the store instead of over the network.

    Done when all target domains are crawled and stored, with a per-domain
    credit log.

This reads what the crawl wrote. It costs nothing and hits nothing: the crawl
is the expensive, unrepeatable part, and by the time this runs it has already
happened. On a checkout with no crawl in the store every test here skips, so
the suite stays green before the gate is reached and turns into a real
assertion the moment it is.

It also prints the thin-site ranking, which is the Phase 7a seam: manifests
sorted by pages stored and word count ascending, with the discovery-flagged
domains marked so they can be excluded from the `unknown` population before
anyone starts reading markdown.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cleanroom.config import load_pack, load_settings
from cleanroom.store import Store, get_store

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = REPO_ROOT / "packs"
GOLDSET_PATH = REPO_ROOT / "eval" / "goldset.jsonl"

#: A crawled domain under this many stored pages is a fetch or discovery
#: problem, not a thin site. The two look identical in a page count, which is
#: the whole reason `discovery_failed` exists.
MIN_PAGES = 3


@pytest.fixture(scope="module")
def store() -> Store:
    return get_store(load_settings())


@pytest.fixture(scope="module")
def crawl(store: Store, goldset_domains: list[str]) -> tuple[list[str], dict[str, dict]]:
    """The goldset crawl, as {domain: manifest}, newest manifest per domain.

    Not "the most recent crawl", and not "the newest crawl date holding goldset
    domains" either. Both worked while the goldset sat inside one crawl and
    both broke as soon as it did not: the first the moment batch 1 landed on
    2026-08-11, the second when Phase 7a added ten thin sites harvested *from*
    batch 1, so the goldset now legitimately spans two crawl dates and will span
    more as batches get labelled.

    Resolving each domain to its own newest crawl is the shape that survives
    that, and it keeps the supersede rule: a re-crawl of a domain wins.
    """
    paths = [path for path in store.list_paths("raw") if path.endswith("/manifest.json")]
    if not paths:
        pytest.skip("no crawl in the store yet; run `cleanroom crawl` first")

    wanted = set(goldset_domains)
    newest: dict[str, tuple[str, str]] = {}
    for path in paths:
        _, domain, crawl_date, _ = path.split("/")
        if domain not in wanted:
            continue
        if domain not in newest or crawl_date > newest[domain][0]:
            newest[domain] = (crawl_date, path)

    if not newest:
        pytest.skip("no goldset domain has been crawled yet")

    return (
        sorted({crawl_date for crawl_date, _ in newest.values()}),
        {domain: store.read_json(path) for domain, (_, path) in newest.items()},
    )


@pytest.fixture(scope="module")
def goldset_domains() -> list[str]:
    rows = [
        json.loads(line)
        for line in GOLDSET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row["domain"] for row in rows]


def test_every_goldset_domain_was_crawled(crawl, goldset_domains, capsys):
    crawl_dates, manifests = crawl
    missing = [domain for domain in goldset_domains if domain not in manifests]

    with capsys.disabled():
        print(f"\n  crawl dates: {', '.join(crawl_dates)}")
        print(f"  domains    : {len(manifests)} stored, {len(goldset_domains)} in the goldset")

    assert not missing, f"never crawled: {missing}"


def test_the_credit_log_is_complete_and_per_domain(crawl, capsys):
    """The gate's second half: every domain accounts for what it spent."""
    _, manifests = crawl
    total = 0

    with capsys.disabled():
        print(f"\n  {'domain':<28}{'pages':>6}{'words':>8}{'credits':>9}  notes")
        print(f"  {'-' * 28}{'-' * 6}{'-' * 8}{'-' * 9}  {'-' * 24}")
        for domain in sorted(manifests):
            fetch = manifests[domain].get("fetch") or {}
            total += fetch.get("credits_billed", 0)
            notes = []
            if fetch.get("rescued"):
                notes.append(f"rescued {fetch['rescue_pages']}")
            if fetch.get("stealth_pages"):
                notes.append(f"STEALTH x{fetch['stealth_pages']}")
            if fetch.get("pages_failed"):
                notes.append(f"{fetch['pages_failed']} failed")
            if fetch.get("discovery_failed"):
                notes.append("discovery flagged")
            print(
                f"  {domain:<28}{fetch.get('pages_stored', 0):>6}"
                f"{fetch.get('word_count', 0):>8}{fetch.get('credits_billed', 0):>9}"
                f"  {', '.join(notes)}"
            )
        print(f"\n  credits billed : {total}")

    for domain, manifest in manifests.items():
        fetch = manifest.get("fetch")
        assert fetch is not None, f"{domain}: manifest has no fetch block"
        assert fetch["credits_billed"] >= fetch["pages_stored"], (
            f"{domain}: {fetch['pages_stored']} pages stored but only "
            f"{fetch['credits_billed']} credits recorded"
        )


def test_every_recorded_page_is_actually_in_the_store(crawl, store):
    _, manifests = crawl
    for domain, manifest in manifests.items():
        for page in manifest["fetch"]["pages"]:
            if not page["stored_path"]:
                assert not page["ok"], f"{domain}: {page['url']} is ok but was never stored"
                continue
            assert store.exists(page["stored_path"]), f"{domain}: missing {page['stored_path']}"
            assert page["word_count"] > 0, f"{domain}: {page['url']} stored with no words"


def test_no_domain_exceeded_the_page_cap(crawl):
    pack = load_pack("medical_v1", pack_dir=PACK_DIR)
    _, manifests = crawl
    for domain, manifest in manifests.items():
        stored = manifest["fetch"]["pages_stored"]
        assert stored <= pack.total_page_cap, f"{domain} stored {stored} pages"


def test_the_thin_site_population_is_visible(crawl, capsys):
    """Phase 7a's input, printed rather than asserted.

    There is no correct number of thin sites to expect. What the gate has to
    guarantee is that the two populations can be told apart: a domain discovery
    flagged is a Phase 3 bug and belongs in the bug list, and only the rest are
    candidates for the `unknown` label.
    """
    _, manifests = crawl
    ranked = sorted(
        ((domain, manifest["fetch"]) for domain, manifest in manifests.items()),
        key=lambda item: (item[1]["pages_stored"], item[1]["word_count"]),
    )

    with capsys.disabled():
        print(f"\n  thin end of the crawl, {'flagged' :>28}")
        print(f"  {'domain':<28}{'pages':>6}{'words':>8}  verdict")
        print(f"  {'-' * 28}{'-' * 6}{'-' * 8}  {'-' * 34}")
        for domain, fetch in ranked[:15]:
            verdict = (
                "discovery failure, NOT a thin site"
                if fetch["discovery_failed"] or fetch["pages_stored"] < MIN_PAGES
                else "candidate for the unknown population"
            )
            print(f"  {domain:<28}{fetch['pages_stored']:>6}{fetch['word_count']:>8}  {verdict}")

    flagged = [domain for domain, fetch in ranked if fetch["discovery_failed"]]
    thin = [
        domain
        for domain, fetch in ranked
        if not fetch["discovery_failed"] and fetch["pages_stored"] >= MIN_PAGES
    ]
    assert thin, "no domain came back clean enough to judge; that is a fetch failure"
    with capsys.disabled():
        print(f"\n  flagged (Phase 3 bug list) : {len(flagged)}")
        print(f"  eligible for Phase 7a      : {len(thin)}")
