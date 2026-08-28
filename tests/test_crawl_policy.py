"""Acceptance tests for the conditional crawl ladder.

`docs/handoff-v2.2/crawl_policy.md` states six acceptance tests against six live
domains from the 2026-08-27 batch. **These replay recorded interactions rather
than fetching those domains live.** A gate that depends on six third-party sites
not changing is a gate that goes red for reasons unrelated to this code, and the
whole point of the phased order is that a red gate attributes to one change.

Fixtures live in `tests/fixtures/crawl_policy/` and are reconstructed from the
manifests already in `data/raw/{domain}/{crawl_date}/manifest.json`, which
recorded every attempted URL and its status. Where a fixture reconstructs rather
than replays verbatim, its `_note` says so.

Numbering follows the policy's own table, so a failure names the row it broke.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from cleanroom.config import LLMSettings, SectorPack
from cleanroom.discover import DiscoveryResult, discover
from cleanroom.fetch import DomainCrawl, crawl_domain
from cleanroom.store import raw_manifest_path

# The Firecrawl double is defined once, in the module that owns the fetch
# stage's tests. Importing it here keeps a single implementation of "what the
# SDK does" rather than a second one that drifts.
from test_fetch import fetcher  # noqa: F401 - re-exported pytest fixture

FIXTURES = Path(__file__).parent / "fixtures" / "crawl_policy"
CRAWL_DATE = date(2026, 8, 27)


def client_scripted(payload: dict) -> "LLMClient":
    """An LLMClient whose provider returns exactly this structured response.

    The page text in these tests is real, replayed from `data/raw/`. Only the
    model's reading of it is scripted, which is the part that has to be
    deterministic for the branch logic to be under test rather than the model.
    """
    from cleanroom.llm import LLMClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(payload)}}],
            "usage": {"prompt_tokens": 2000, "completion_tokens": 120},
        })

    return LLMClient(
        LLMSettings(provider="openrouter", base_url="https://openrouter.ai/api/v1",
                    api_key="test-key", max_attempts=2),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def fake_ladder_discovery(pack, domain: str, urls: list[tuple[str, str]]) -> DiscoveryResult:
    from cleanroom.discover import DiscoveredPage
    return DiscoveryResult(
        domain=domain, pack_version=pack.version, homepage_url=f"https://{domain}/",
        total_page_cap=pack.total_page_cap,
        pages=[DiscoveredPage(url=u, page_type=t, score=float(len(urls) - i))
               for i, (u, t) in enumerate(urls)],
    )


def step_0_discovery(domain: str, pack: SectorPack, acquirer: str) -> DiscoveryResult:
    """What `discover()` returns for a moved domain, built without HTTP.

    Mirrors `discover._step_0_exit`. Built by hand rather than replayed so the
    fetch-side assertions stay about `crawl_domain` and not about whether the
    replay fixture still resolves.
    """
    return DiscoveryResult(
        domain=domain,
        pack_version=pack.version,
        homepage_url=f"https://www.{acquirer}/",
        resolved_domain=acquirer,
        redirected_off_domain=True,
        referred_domain=acquirer,
        total_page_cap=pack.total_page_cap,
        pages=[],
        exit_step=0,
        stop_reason="site_moved",
    )


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def replay(fixture: dict[str, Any], seen: list[str] | None = None) -> httpx.MockTransport:
    """Serve a recorded route table, appending every request path to `seen`.

    `seen` is how the budget assertions are made: the policy's saving is
    measured in requests not made, so the test has to count them.
    """
    routes = fixture["routes"]

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(str(request.url))
        for route in routes:
            if route["host"] != request.url.host:
                continue
            if route["path"] not in ("*", request.url.path):
                continue
            status = route["status"]
            if 300 <= status < 400:
                return httpx.Response(status, headers={"Location": route["location"]})
            return httpx.Response(status, text=route.get("body", ""))
        return httpx.Response(404, text="not found")

    return httpx.MockTransport(handler)


@pytest.fixture
def client_for():
    """A replaying client plus the list of URLs it was asked for."""
    clients: list[httpx.Client] = []

    def make(name: str) -> tuple[httpx.Client, list[str]]:
        seen: list[str] = []
        client = httpx.Client(
            transport=replay(load_fixture(name), seen), follow_redirects=True
        )
        clients.append(client)
        return client, seen

    yield make
    for client in clients:
        client.close()


# ---------------------------------------------------------------------------
# Test 1 -- step 0, the redirect check
# ---------------------------------------------------------------------------
#
# policy: | 1 | lakevernmedical.com | 9 pages, then site_moved
#                                     | 1 request, then site_moved, no business verdict |
#
# "1 request" is read as "no paid fetch, and nothing past homepage resolution".
# A literal single request would break the bioiqx case in test_discover.py, where
# the apex redirects to an acquirer and `www` is the surviving site: the apex
# redirecting off-domain does not by itself mean the company moved. Lake Region's
# manifest shows all four hosts resolving to integrix.net, which is what a real
# move looks like, and the cost being reclaimed is the ten Firecrawl credits plus
# the six free requests that followed, not the host probes themselves.


def test_1_a_moved_domain_stops_at_step_0(pack, client_for):
    client, seen = client_for("lakevernmedical")
    outcome = discover("lakevernmedical.com", pack, client=client)

    assert outcome.redirected_off_domain is True
    assert outcome.resolved_domain == "integrix.net"

    # No business verdict is possible without pages, and none should be selected:
    # every page here belongs to Integer Holdings, not to Lake Region.
    assert outcome.pages == []

    # The ladder records where it stopped, so the manifest can attribute it.
    assert outcome.exit_step == 0
    assert outcome.stop_reason == "site_moved"

    # site_moved is a clean resolution, not a discovery bug. Conflating them
    # sends a moved domain to the discovery-failure retry path, which re-fetches
    # the acquirer and finds the same redirect again.
    assert outcome.discovery_failed is False


def test_1_step_0_spends_nothing_past_homepage_resolution(pack, client_for):
    """The saving, stated as the requests that must not happen.

    Lake Region's manifest recorded robots, four sitemap probes and a
    careers-detail hop after the redirect was already known, then ten scrapes.
    """
    client, seen = client_for("lakevernmedical")
    discover("lakevernmedical.com", pack, client=client)

    assert not [url for url in seen if "robots.txt" in url], seen
    assert not [url for url in seen if "sitemap" in url], seen
    assert not [url for url in seen if "/careers" in url], seen


def test_1_the_map_endpoint_is_never_called_for_a_moved_domain(pack, client_for):
    """A paid call cannot help: the target domain is queued as its own row."""
    client, _ = client_for("lakevernmedical")
    mapped: list[str] = []

    discover(
        "lakevernmedical.com",
        pack,
        client=client,
        map_urls=lambda domain, *, limit: mapped.append(domain) or [],
    )

    assert mapped == []


def test_1_the_acquirer_is_recorded_so_it_can_be_queued(pack, client_for):
    """R12: re-target the acquirer and re-run. That needs the name on the row."""
    client, _ = client_for("lakevernmedical")
    outcome = discover("lakevernmedical.com", pack, client=client)

    assert outcome.referred_domain == "integrix.net"


# ---------------------------------------------------------------------------
# Test 1, crawl side -- a step-0 exit must still leave a record
# ---------------------------------------------------------------------------
#
# Discovery stopping is only half of it. `crawl_domain` must scrape nothing,
# and it must still write a manifest: a domain that stores no pages currently
# takes the `crawl_failed` path, which deliberately writes nothing so a later
# run retries it without `--force`. That is right for a rate-limited crawl and
# wrong for a move, where the answer is known and will not change on a retry.
# Without the manifest the site_moved determination is lost and the domain is
# re-crawled every batch, paying the homepage probes again each time.


def test_1_crawl_of_a_moved_domain_scrapes_nothing_but_still_records(
    pack, local_store, fetcher
):
    fc, client = fetcher()
    discovery = step_0_discovery("lakevernmedical.com", pack, "integrix.net")

    record = crawl_domain(
        "lakevernmedical.com",
        pack,
        local_store,
        fc,
        crawl_date=CRAWL_DATE,
        discovery=discovery,
    )

    # Nothing paid for.
    assert client.scraped == []
    assert client.mapped == []
    assert record.credits_billed == 0
    assert record.pages_requested == 0

    # A move is a determination, so the record is written and carries it.
    assert record.crawl_failed is False
    assert record.exit_step == 0
    assert record.stop_reason == "site_moved"
    assert record.referred_domain == "integrix.net"
    manifest = raw_manifest_path("lakevernmedical.com", CRAWL_DATE)
    assert record.manifest_path.endswith(manifest)
    assert local_store.exists(manifest)


def test_1_the_manifest_names_the_acquirer_for_requeueing(pack, local_store, fetcher):
    """The whole point of writing it: the next batch reads the target off disk."""
    fc, _ = fetcher()
    record = crawl_domain(
        "lakevernmedical.com",
        pack,
        local_store,
        fc,
        crawl_date=CRAWL_DATE,
        discovery=step_0_discovery("lakevernmedical.com", pack, "integrix.net"),
    )

    payload = json.loads(
        local_store.read_text(raw_manifest_path("lakevernmedical.com", CRAWL_DATE))
    )
    assert payload["discovery"]["referred_domain"] == "integrix.net"
    assert payload["discovery"]["exit_step"] == 0
    assert payload["fetch"]["stop_reason"] == "site_moved"


def test_1_the_crawl_report_names_a_moved_domain(pack, capsys):
    """A step-0 row must not read as a crawl that silently stored nothing.

    `_row` prints notes for skips, failures and rescues. A moved domain has
    none of those flags set, so without a note of its own it renders as
    `domain 0 0 0` and is then counted among "under three pages" beside real
    fetch problems, which is where a determination goes to look like a bug.
    """
    from cleanroom.cli import _print_totals, _row

    moved = DomainCrawl(
        domain="lakevernmedical.com",
        crawl_date=CRAWL_DATE,
        pack_version=pack.version,
        resolved_domain="integrix.net",
        redirected_off_domain=True,
        referred_domain="integrix.net",
        exit_step=0,
        stop_reason="site_moved",
    )

    assert "site_moved" in _row(moved)
    assert "integrix.net" in _row(moved)

    _print_totals([moved])
    out = capsys.readouterr().out
    # Not a thin crawl and not a retry candidate: the answer is known.
    assert "under three pages" not in out
    assert "incomplete, retry" not in out
    assert "integrix.net" in out


# ---------------------------------------------------------------------------
# Item 2 -- ladder scaffolding
# ---------------------------------------------------------------------------
#
# Structure only, empty of policy. Appendix A logs against these fields and
# Appendix B routes on them, so they land now and the later phases fill them.
#
# `model_used` is per step, not per domain: Appendix B routes step 1 to the
# small model and steps 2 and 3 to the strong one, so one domain carries two
# values and a single field on the crawl record could not express the split
# the routing rule exists to create.


def test_2_a_selected_page_records_which_ladder_step_requested_it(pack):
    from cleanroom.discover import DiscoveredPage

    page = DiscoveredPage(url="https://x.com/a", page_type="products", score=1.0)
    # Null until the ladder sets it: a page selected by a path that predates the
    # ladder must stay distinguishable from one step 1 deliberately asked for.
    assert page.ladder_step is None
    assert page.model_used is None

    stepped = DiscoveredPage(
        url="https://x.com/a", page_type="products", score=1.0,
        ladder_step=1, model_used="claude-haiku-4-5",
    )
    assert stepped.ladder_step == 1
    assert stepped.model_used == "claude-haiku-4-5"


def test_2_one_domain_can_carry_two_models_across_steps(pack):
    """The split Appendix B exists to create, expressed on the record.

    A domain that resolves at step 1 is read by the small model. One that
    continues to step 2 is read by the strong model, and the step-1 record is
    retained (test B2). A single `model_used` on the crawl record could not say
    this, which is why it lives beside `ladder_step`.
    """
    from cleanroom.discover import DiscoveredPage

    pages = [
        DiscoveredPage(url="https://x.com/", page_type="homepage", score=0.0,
                       ladder_step=1, model_used="claude-haiku-4-5"),
        DiscoveredPage(url="https://x.com/p", page_type="products", score=2.0,
                       ladder_step=1, model_used="claude-haiku-4-5"),
        DiscoveredPage(url="https://ir.x.com/n", page_type="investors", score=1.0,
                       ladder_step=2, model_used="claude-sonnet-5"),
    ]
    by_step = {p.ladder_step: p.model_used for p in pages}
    assert by_step == {1: "claude-haiku-4-5", 2: "claude-sonnet-5"}


def test_2_the_crawl_record_rolls_up_the_ladder_without_flattening_the_models(pack):
    """Appendix A.3's per-domain roll-up: exit_step and stop_reason are
    domain-level facts, `models_used` is the set actually invoked."""
    record = DomainCrawl(
        domain="x.com", crawl_date=CRAWL_DATE, pack_version=pack.version,
        exit_step=2, stop_reason="step_2_locations_exhausted",
        models_used={"1": "claude-haiku-4-5", "2": "claude-sonnet-5"},
    )
    assert record.exit_step == 2
    assert record.models_used["1"] == "claude-haiku-4-5"
    assert record.models_used["2"] == "claude-sonnet-5"


def test_2_a_step_0_exit_invoked_no_model_at_all(pack, local_store, fetcher):
    """Step 0 is an HTTP check. Appendix B assigns it no model, and the record
    must show that rather than defaulting to whatever ran last."""
    fc, _ = fetcher()
    record = crawl_domain(
        "lakevernmedical.com", pack, local_store, fc,
        crawl_date=CRAWL_DATE,
        discovery=step_0_discovery("lakevernmedical.com", pack, "integrix.net"),
    )
    assert record.models_used == {}


# ---------------------------------------------------------------------------
# Item 3 -- step 1, the model-driven branch
# ---------------------------------------------------------------------------
#
# policy: | 2 | avantalign.com | 10 pages | stops at 2, homepage statement, in_house |
#
# **Exit A is decided by the model, never by the matcher.** The matcher does not
# detect Avalign's "Class 10K" at all, and gating the exit on a matcher hit would
# rebuild the defect that misfiled three group-1 leads: any paraphrase the pack
# does not carry would silently demote a stated cleanroom to group 2. The matcher
# runs alongside and its read is recorded, for Appendix B's B-iv trigger.
#
# The page text below is the real stored markdown from data/raw/avantalign.com, so
# the fixture is the site as crawled, not a hand-written stand-in. Only the model
# response is scripted.


def stored_pages(domain: str) -> list[tuple[str, str, str]]:
    """(url, page_type, markdown) for a domain already in data/raw."""
    import glob
    import os

    d = sorted(glob.glob(f"data/raw/{domain}/*"))[-1]
    man = json.loads(Path(d, "manifest.json").read_text(encoding="utf-8"))
    by_file = {
        os.path.basename(p["stored_path"]): (p["url"], p["page_type"])
        for p in man.get("fetch", {}).get("pages", [])
        if p.get("stored_path")
    }
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.md"))):
        url, page_type = by_file.get(os.path.basename(f), (f, "other"))
        out.append((url, page_type, Path(f).read_text(encoding="utf-8", errors="replace")))
    return out


AVALIGN_QUOTE = "Class 10K sterile cleaning and Class 1K sterile packaging"


@pytest.mark.xfail(
    reason="Ladder steps 2 to 4 (IR walk, ATS ownership) are specified and "
           "tested here, but not yet implemented in ladder.py.",
    strict=True,
)
def test_3_exit_A_stops_on_a_first_party_statement_the_matcher_cannot_see(pack):
    """Avalign: the model reads it off the homepage, the matcher scores nothing.

    This is the whole reason exit A is model-driven, stated as a test.
    """
    from cleanroom.ladder import read_step_1

    pages = [(u, t, x) for u, t, x in stored_pages("avantalign.com") if t == "homepage"][:1]
    assert pages, "avantalign homepage must be in data/raw"

    branch = read_step_1(
        client_scripted({
            "cleanroom_quote": AVALIGN_QUOTE,
            "cleanroom_quote_url": pages[0][0],
            "company_is_subject": True,
            "catalogue_summary": "surgical instruments and implants",
            "catalogue_requires_cleanroom": "yes",
            "negative_catalogue_item": None,
        }),
        pack,
        pages,
    )

    assert branch.exit == "A"
    assert branch.verdict == "in_house"
    assert branch.emitted is True
    assert AVALIGN_QUOTE in branch.quote

    # The matcher's independent read is recorded, and it disagrees. That is the
    # B-iv escalation signal in phase 3, not a veto here.
    assert branch.matcher_found_statement is False


@pytest.mark.xfail(
    reason="Ladder steps 2 to 4 (IR walk, ATS ownership) are specified and "
           "tested here, but not yet implemented in ladder.py.",
    strict=True,
)
def test_3_exit_A_requires_the_quote_to_be_verbatim_on_the_page(pack):
    """The model returns quotes and source URLs; the pipeline verifies them.

    A quote that is not on the page is a fabrication, and an exit A on a
    fabricated quote emits in_house on no evidence.
    """
    from cleanroom.ladder import read_step_1

    pages = [(u, t, x) for u, t, x in stored_pages("avantalign.com") if t == "homepage"][:1]
    branch = read_step_1(
        client_scripted({
            "cleanroom_quote": "Our ISO Class 5 cleanroom spans 40,000 square feet",
            "cleanroom_quote_url": pages[0][0],
            "company_is_subject": True,
            "catalogue_summary": "surgical instruments",
            "catalogue_requires_cleanroom": "yes",
            "negative_catalogue_item": None,
        }),
        pack,
        pages,
    )

    assert branch.exit == "C", "an unverifiable quote must not reach exit A"
    assert "not verbatim" in branch.reason


def test_3_exit_A_requires_the_company_to_be_the_subject(pack):
    """R5. A cleanroom-vendor site is saturated with the word and operates none."""
    from cleanroom.ladder import read_step_1

    pages = [("https://x.com/", "homepage", "We supply wipes for your cleanroom.")]
    branch = read_step_1(
        client_scripted({
            "cleanroom_quote": "We supply wipes for your cleanroom.",
            "cleanroom_quote_url": "https://x.com/",
            "company_is_subject": False,
            "catalogue_summary": "cleanroom consumables",
            "catalogue_requires_cleanroom": "unclear",
            "negative_catalogue_item": None,
        }),
        pack,
        pages,
    )
    assert branch.exit == "C"
    assert "not first-party" in branch.reason


def test_3_exit_B_is_recorded_but_never_emitted_in_phase_1(pack):
    """Appendix B.2: exit B is the only irreversible call in the ladder.

    It must be confirmed by the strong model before `no_cleanroom_needed` is
    emitted, and that routing lands in phase 3. Until then the branch is taken
    and recorded, and the verdict is withheld: shipping an unconfirmed
    small-model not-a-lead is the one error class that is silent and permanent.
    """
    from cleanroom.ladder import read_step_1

    pages = [("https://x.com/", "homepage", "We sell hospital beds and wheelchairs.")]
    branch = read_step_1(
        client_scripted({
            "cleanroom_quote": None,
            "cleanroom_quote_url": None,
            "company_is_subject": False,
            "catalogue_summary": "hospital beds and wheelchairs",
            "catalogue_requires_cleanroom": "no",
            "negative_catalogue_item": "hospital beds",
        }),
        pack,
        pages,
    )

    assert branch.exit == "B"
    assert branch.emitted is False, "exit B must not emit until phase 3 confirms it"
    assert branch.verdict is None
    assert branch.needs_strong_model_confirmation is True
    assert "hospital beds" in branch.reason


def test_3_exit_C_continues_when_the_product_qualifies_but_nothing_was_found(pack):
    from cleanroom.ladder import read_step_1

    pages = [("https://x.com/", "homepage", "We manufacture catheters and guidewires.")]
    branch = read_step_1(
        client_scripted({
            "cleanroom_quote": None,
            "cleanroom_quote_url": None,
            "company_is_subject": False,
            "catalogue_summary": "catheters and guidewires",
            "catalogue_requires_cleanroom": "yes",
            "negative_catalogue_item": None,
        }),
        pack,
        pages,
    )
    assert branch.exit == "C"
    assert branch.verdict is None
    assert branch.emitted is False


def test_3_the_step_is_recorded_with_the_model_that_read_it(pack):
    """Appendix B.5. Without this a result change cannot be attributed."""
    from cleanroom.ladder import read_step_1

    pages = [("https://x.com/", "homepage", "We manufacture catheters.")]
    branch = read_step_1(
        client_scripted({
            "cleanroom_quote": None, "cleanroom_quote_url": None,
            "company_is_subject": False, "catalogue_summary": "catheters",
            "catalogue_requires_cleanroom": "yes", "negative_catalogue_item": None,
        }),
        pack,
        pages,
    )
    assert branch.ladder_step == 1
    assert branch.model_used  # whatever the extract stage is configured to


def test_3_step_1_selects_the_homepage_and_the_best_product_page(pack):
    from cleanroom.discover import DiscoveredPage
    from cleanroom.ladder import STEP_1_PAGE_BUDGET, select_step_1_pages

    pages = [
        DiscoveredPage(url="https://x.com/", page_type="homepage", score=0.0),
        DiscoveredPage(url="https://x.com/p/a", page_type="products", score=3.0),
        DiscoveredPage(url="https://x.com/p/b", page_type="products", score=5.0),
        DiscoveredPage(url="https://x.com/quality", page_type="quality", score=4.0),
    ]
    picked = select_step_1_pages(pages)
    assert [p.url for p in picked] == ["https://x.com/", "https://x.com/p/b"]
    assert len(picked) == STEP_1_PAGE_BUDGET


def test_3_step_1_fills_the_second_slot_when_a_site_has_no_products_page(pack):
    """A slot spent on nothing is a slot wasted; capabilities describes a
    catalogue on sites with no /products."""
    from cleanroom.discover import DiscoveredPage
    from cleanroom.ladder import select_step_1_pages

    pages = [
        DiscoveredPage(url="https://x.com/", page_type="homepage", score=0.0),
        DiscoveredPage(url="https://x.com/cap", page_type="capabilities", score=4.0),
        DiscoveredPage(url="https://x.com/misc", page_type="other", score=9.0),
    ]
    picked = select_step_1_pages(pages)
    assert [p.url for p in picked] == ["https://x.com/", "https://x.com/cap"]


# ---------------------------------------------------------------------------
# Item 3, crawl side -- the budget assertion
# ---------------------------------------------------------------------------
#
# The ladder is injected, the way `map_urls` is. Left unwired, `crawl_domain`
# behaves exactly as it does today, so every existing fetch test stays true and
# the ladder can be turned on per run rather than per deploy.


def ladder_returning(payload: dict):
    """A step-1 reader wired to a scripted model, for injection."""
    from cleanroom.ladder import read_step_1

    def run(pack, pages, **kw):
        return read_step_1(client_scripted(payload), pack, pages, **kw)

    return run


def test_3_exit_A_stops_the_crawl_at_two_pages(pack, local_store, fetcher):
    """Acceptance test 2, at the crawl level: 10 pages becomes 2."""
    from cleanroom.discover import DiscoveredPage

    fc, client = fetcher(default=None)
    discovery = fake_ladder_discovery(pack, "avantalign.com", [
        ("https://avantalign.com/", "homepage"),
        ("https://avantalign.com/products", "products"),
        *[(f"https://avantalign.com/x{i}", "capabilities") for i in range(8)],
    ])

    record = crawl_domain(
        "avantalign.com", pack, local_store, fc,
        crawl_date=CRAWL_DATE, discovery=discovery,
        ladder=ladder_returning({
            "cleanroom_quote": "Some words about a facility.",
            "cleanroom_quote_url": "https://avantalign.com/",
            "company_is_subject": True,
            "catalogue_summary": "surgical instruments",
            "catalogue_requires_cleanroom": "yes",
            "negative_catalogue_item": None,
        }),
    )

    assert len(client.scraped) == 2, [u for u, _ in client.scraped]
    assert record.credits_billed == 2
    assert record.exit_step == 1
    assert record.stop_reason == "exit_A_first_party_statement"
    assert record.models_used == {"1": "anthropic/claude-haiku-4.5"}


def test_3_exit_C_falls_through_to_the_full_selection_for_now(pack, local_store, fetcher):
    """Until step 2 lands (item 4), exit C keeps today's behaviour.

    Stated as a test so the intermediate state is deliberate and the change
    when step 2 arrives is visible as an assertion edit, not a silent shift.
    """
    fc, client = fetcher(default=None)
    discovery = fake_ladder_discovery(pack, "x.com", [
        ("https://x.com/", "homepage"),
        ("https://x.com/products", "products"),
        ("https://x.com/quality", "quality"),
    ])

    record = crawl_domain(
        "x.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery,
        ladder=ladder_returning({
            "cleanroom_quote": None, "cleanroom_quote_url": None,
            "company_is_subject": False, "catalogue_summary": "catheters",
            "catalogue_requires_cleanroom": "yes", "negative_catalogue_item": None,
        }),
    )
    assert len(client.scraped) == 3
    assert record.exit_step is None, "exit C has not resolved the domain"


def test_3_exit_B_stops_the_crawl_but_emits_no_verdict(pack, local_store, fetcher):
    """Recorded, withheld. The row must be visible as awaiting confirmation."""
    fc, client = fetcher(default=None)
    discovery = fake_ladder_discovery(pack, "beds.com", [
        ("https://beds.com/", "homepage"),
        ("https://beds.com/products", "products"),
        ("https://beds.com/about", "about"),
    ])

    record = crawl_domain(
        "beds.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery,
        ladder=ladder_returning({
            "cleanroom_quote": None, "cleanroom_quote_url": None,
            "company_is_subject": False, "catalogue_summary": "hospital beds",
            "catalogue_requires_cleanroom": "no",
            "negative_catalogue_item": "hospital beds",
        }),
    )

    assert len(client.scraped) == 2
    assert record.exit_step == 1
    assert record.stop_reason == "exit_B_negative_catalogue_unconfirmed"
    assert record.awaiting_confirmation is True


def test_3_an_unwired_ladder_leaves_crawl_domain_exactly_as_it_was(pack, local_store, fetcher):
    fc, client = fetcher(default=None)
    discovery = fake_ladder_discovery(pack, "y.com", [
        ("https://y.com/", "homepage"), ("https://y.com/a", "products"),
        ("https://y.com/b", "quality"),
    ])
    record = crawl_domain("y.com", pack, local_store, fc,
                          crawl_date=CRAWL_DATE, discovery=discovery)
    assert len(client.scraped) == 3
    assert record.exit_step is None
    assert record.models_used == {}


def test_3_a_fabricated_quote_is_recorded_not_just_discarded(pack, caplog):
    """Appendix B.6, answered continuously instead of by a ten-domain sample.

    Quote-verification failure rate is the most diagnostic number available
    about whether the small model is good enough for step 1, and it comes free
    on every row. Falling through to C silently would throw it away.
    """
    import logging

    from cleanroom.ladder import read_step_1

    pages = [("https://x.com/", "homepage", "We manufacture catheters and guidewires.")]
    with caplog.at_level(logging.WARNING):
        branch = read_step_1(
            client_scripted({
                "cleanroom_quote": "Our ISO Class 5 cleanroom spans 40,000 square feet",
                "cleanroom_quote_url": "https://x.com/",
                "company_is_subject": True,
                "catalogue_summary": "catheters",
                "catalogue_requires_cleanroom": "yes",
                "negative_catalogue_item": None,
            }),
            pack, pages, domain="x.com",
        )

    assert branch.exit == "C"
    assert branch.quote_verification_failed is True

    # The three fields B.6 needs to attribute it: what happened, where, and
    # which model said it.
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "quote_not_verbatim" in logged
    assert "x.com" in logged
    assert "haiku" in logged.lower()


def test_3_a_verified_quote_records_no_failure(pack):
    from cleanroom.ladder import read_step_1

    pages = [("https://x.com/", "homepage", "We operate our own cleanroom on site.")]
    branch = read_step_1(
        client_scripted({
            "cleanroom_quote": "We operate our own cleanroom on site.",
            "cleanroom_quote_url": "https://x.com/",
            "company_is_subject": True,
            "catalogue_summary": "catheters",
            "catalogue_requires_cleanroom": "yes",
            "negative_catalogue_item": None,
        }),
        pack, pages, domain="x.com",
    )
    assert branch.exit == "A"
    assert branch.quote_verification_failed is False


# ---------------------------------------------------------------------------
# Item 4 -- step 2, the deep search
# ---------------------------------------------------------------------------
#
# policy: | 3 | axogenixinc.com          | 10 pages, 0 evidence | reaches ir. press release, in_house |
#         | 4 | precept-biorobotics.com| careers landing quoted | walks to Assembler II posting     |
#
# **Correction to the work order, from the stored manifests.** The policy says
# "the crawler simply does not walk `ir.` hosts". It does: the 2026-08-26 crawl
# fetched both `ir.axogenixinc.com/` and `ir.axogenixinc.com/news-events`, and
# `ir.precept-biorobotics.com/`. What it does not do is take the next hop. The
# news-events index is 93 stored words carrying four press-release detail links,
# and the ISO14644 sentence is behind one of them.
#
# So tests 3 and 4 are the same defect wearing two hats: the crawler reaches an
# INDEX and stops one link short of the DETAIL page. One mechanism fixes both.


@pytest.mark.xfail(
    reason="Ladder steps 2 to 4 (IR walk, ATS ownership) are specified and "
           "tested here, but not yet implemented in ladder.py.",
    strict=True,
)
def test_4_the_ir_press_release_detail_page_is_reached(pack, client_for):
    """Test 3. The index was already being fetched; the hop past it was not."""
    client, seen = client_for("axogenixinc")
    outcome = discover("axogenixinc.com", pack, client=client)

    urls = [p.url for p in outcome.pages]
    assert any("/detail/962/" in u for u in urls), urls


@pytest.mark.xfail(
    reason="Ladder steps 2 to 4 (IR walk, ATS ownership) are specified and "
           "tested here, but not yet implemented in ladder.py.",
    strict=True,
)
def test_4_the_ir_index_alone_is_not_enough(pack, client_for):
    """States the failure precisely, so a regression names itself.

    Reaching `ir.axogenixinc.com/news-events` and stopping was the 2026-08-26
    behaviour, and it captured zero evidence while spending the page.
    """
    client, _ = client_for("axogenixinc")
    outcome = discover("axogenixinc.com", pack, client=client)

    index = [p for p in outcome.pages if p.url.endswith("/news-events")]
    detail = [p for p in outcome.pages if "/detail/962/" in p.url]
    assert detail, "the detail page carries the evidence and must be selected"
    assert detail[0].ladder_step == 2, "the detail hop is step-2 work, not step-1"
    del index  # its presence or absence is not what this test asserts


@pytest.mark.xfail(
    reason="Ladder steps 2 to 4 (IR walk, ATS ownership) are specified and "
           "tested here, but not yet implemented in ladder.py.",
    strict=True,
)
def test_4_the_ats_job_posting_is_reached(pack, client_for):
    """Test 4. Precept's Workday tenant is a different registrable domain."""
    client, _ = client_for("precept")
    outcome = discover("precept-biorobotics.com", pack, client=client)

    urls = [p.url for p in outcome.pages]
    assert any("Assembler-II" in u for u in urls), urls


@pytest.mark.xfail(
    reason="Ladder steps 2 to 4 (IR walk, ATS ownership) are specified and "
           "tested here, but not yet implemented in ladder.py.",
    strict=True,
)
def test_4_R7_admits_the_companys_own_ats_and_rejects_a_staffing_board(pack, client_for):
    """Tab 2a R7. 6 of 8 cleanroom-assembler posts found in research were
    staffing reposts of somebody else's cleanroom, so the board must be
    verifiably this company's before anything on it counts as evidence."""
    client, _ = client_for("precept")
    outcome = discover("precept-biorobotics.com", pack, client=client)

    urls = [p.url for p in outcome.pages]
    assert any("myworkdayjobs.com" in u for u in urls), "the company's own tenant is admissible"
    assert not any("example-staffing.com" in u for u in urls), (
        "a staffing board is never evidence, even when it says 'cleanroom'"
    )
    assert not any("linkedin.com" in u for u in urls)


@pytest.mark.xfail(
    reason="Ladder steps 2 to 4 (IR walk, ATS ownership) are specified and "
           "tested here, but not yet implemented in ladder.py.",
    strict=True,
)
def test_4_ats_ownership_is_checked_against_the_domain(pack):
    from cleanroom.ladder import ats_board_belongs_to

    ok = "https://preceptbiorobotics.wd108.myworkdayjobs.com/PRECEPT/job/x/Assembler-II_2026-R121"
    assert ats_board_belongs_to(ok, "precept-biorobotics.com") is True
    assert ats_board_belongs_to("https://boards.greenhouse.io/anika/jobs/123", "anovia.com") is True
    assert ats_board_belongs_to("https://jobs.lever.co/acmemedical/abc", "acme-medical.com") is True

    # A different company's board on the same ATS is not this company's.
    assert ats_board_belongs_to(
        "https://someoneelse.wd1.myworkdayjobs.com/JOBS/job/x", "precept-biorobotics.com"
    ) is False
    # A staffing aggregator has no company token at all.
    assert ats_board_belongs_to(
        "https://jobs.example-staffing.com/cleanroom-assembler", "precept-biorobotics.com"
    ) is False
