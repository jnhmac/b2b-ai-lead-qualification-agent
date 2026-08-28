"""The Phase 3 gate, stated as one live test.

    Done when discovery over 20 sample domains returns sensible page mixes with
    no single type dominating.

This one hits the network, so it is marked `live` and deselected by default.
Run it with `-m live -s` to see the per-domain mix. It costs no Firecrawl
credits: sitemaps, robots.txt and the homepage are plain HTTP, and `map_urls`
is left unwired here so the paid fallback cannot fire.

Discovery over real sites is not deterministic the way a unit test is. Sites go
down, redirect, and rewrite their navigation. The assertions are therefore about
the *shape* of the result across the sample rather than exact URLs, and the
printed table is the part a human reads.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from cleanroom.config import load_pack
from cleanroom.discover import MIN_PAGES, discover_all

pytestmark = pytest.mark.live

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = REPO_ROOT / "packs"
GOLDSET_PATH = REPO_ROOT / "eval" / "goldset.jsonl"

SAMPLE_SIZE = 20

#: A single type owning more than this share of all selected non-homepage pages
#: means the scorer or the caps are not working. A catalogue is the usual
#: culprit; the pack's per-type caps exist to prevent exactly this.
MAX_TYPE_SHARE = 0.45

#: Discovery failures across the sample. Some of these domains really are dead,
#: and a hard zero would make the gate a test of the internet rather than of
#: this module.
MAX_FAILURE_RATE = 0.25


def sample_domains(rows: list[dict], size: int = SAMPLE_SIZE) -> list[str]:
    """A deterministic spread across the goldset's three groups.

    Taking the first 20 rows would take 19 confirmed positives and one other:
    contract manufacturers have unusually tidy `/capabilities` pages, so the
    page mix would look far better than it is.
    """
    by_group: dict[str, list[str]] = {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(row["domain"])

    picked: list[str] = []
    for group in sorted(by_group):
        members = by_group[group]
        take = round(size * len(members) / len(rows))
        step = max(len(members) // take, 1) if take else 1
        picked.extend(members[::step][:take])
    return picked[:size]


@pytest.fixture(scope="module")
def pack():
    return load_pack("medical_v1", pack_dir=PACK_DIR)


@pytest.fixture(scope="module")
def results(pack):
    """Crawled once for the whole module. Twenty live sites is slow enough."""
    rows = [
        json.loads(line)
        for line in GOLDSET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    domains = sample_domains(rows)
    assert len(domains) == SAMPLE_SIZE
    return discover_all(domains, pack, max_workers=6, timeout=20.0)


def test_page_mix_is_sensible(results, pack, capsys):
    type_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    failures = [r for r in results if r.discovery_failed]

    with capsys.disabled():
        print(f"\n  {'domain':<28}{'pages':>6}  {'sources':<18}mix")
        print(f"  {'-' * 28}{'-' * 6}  {'-' * 18}{'-' * 28}")
        for result in results:
            counts = result.page_type_counts
            for page in result.pages:
                if page.page_type != pack.always_fetch_type:
                    type_counter[page.page_type] += 1
            source_counter.update(result.sources_used)

            mix = " ".join(f"{name}:{n}" for name, n in sorted(counts.items()))
            flag = "  FAIL" if result.discovery_failed else ""
            print(
                f"  {result.domain:<28}{len(result.pages):>6}  "
                f"{','.join(result.sources_used):<18}{mix}{flag}"
            )
            if result.discovery_failed:
                print(f"      -> {result.failure_reason}")

        total = sum(type_counter.values())
        print(f"\n  domains                   : {len(results)}")
        print(f"  discovery failures        : {len(failures)}")
        print(f"  pages selected            : {sum(len(r.pages) for r in results)}")
        print(f"  link sources              : {dict(source_counter)}")
        print("  page types (excl. homepage):")
        for name, count in type_counter.most_common():
            print(f"    {name:<16}{count:>4}  {count / total:>6.1%}")

    assert results, "no domains discovered"

    failure_rate = len(failures) / len(results)
    assert failure_rate <= MAX_FAILURE_RATE, (
        f"{len(failures)}/{len(results)} domains failed discovery: "
        f"{[(r.domain, r.failure_reason) for r in failures]}"
    )

    assert type_counter, "no typed pages selected on any domain"
    dominant, dominant_count = type_counter.most_common(1)[0]
    share = dominant_count / sum(type_counter.values())
    assert share <= MAX_TYPE_SHARE, f"{dominant} took {share:.0%} of all selected pages"

    assert len(type_counter) >= 4, f"only {len(type_counter)} page types ever selected"


def test_healthy_domains_clear_the_minimum(results, pack):
    """Every domain that discovery did not flag has at least the minimum pages."""
    for result in results:
        if result.discovery_failed:
            continue
        assert len(result.pages) >= MIN_PAGES, result.summary()
        assert len(result.pages) <= pack.total_page_cap, result.summary()


def test_homepage_is_always_selected(results, pack):
    for result in results:
        assert result.pages, f"{result.domain} selected nothing at all"
        assert result.pages[0].page_type == pack.always_fetch_type


def test_caps_hold_on_real_sites(results, pack):
    """The catalogue case, on sites that actually have catalogues."""
    for result in results:
        for page_type, count in result.page_type_counts.items():
            if page_type == "other":
                continue
            assert count <= pack.page_types[page_type].max_pages, (
                f"{result.domain}: {page_type} took {count} slots"
            )
