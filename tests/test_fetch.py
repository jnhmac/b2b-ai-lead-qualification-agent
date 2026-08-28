"""Phase 4 offline: the whole fetch stage against a Firecrawl double.

Every test here runs with no key, no network and no spend. `FakeFirecrawl`
implements the three calls `fetch.py` makes and raises on `extract`, which is
the endpoint the spec forbids: billed separately, and it re-couples scraping to
prompting.

What these cannot cover, and what only a real crawl can: whether the map
endpoint actually renders JavaScript on a given site, and whether a WAF serves
Firecrawl's proxies. Those are properties of the internet, not of this module.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest

from cleanroom.config import ConfigError
from cleanroom.discover import DiscoveredPage, DiscoveryResult
from cleanroom.fetch import (
    DEFAULT_RATE_LIMIT_WAIT,
    MAP_CREDITS,
    PAGE_CREDITS,
    STEALTH_CREDITS,
    DomainCrawl,
    FetchError,
    Fetcher,
    PageFetch,
    RateLimiter,
    crawl_all,
    crawl_domain,
    credit_estimate,
    is_fatal,
    rate_limit_wait,
)
from cleanroom.store import raw_manifest_path, raw_page_path

CRAWL_DATE = date(2026, 8, 7)


# ---------------------------------------------------------------------------
# Firecrawl double
# ---------------------------------------------------------------------------


class FakeMetadata:
    def __init__(self, **fields: Any):
        self.status_code = fields.get("status_code", 200)
        self.credits_used = fields.get("credits_used", 1)
        self.proxy_used = fields.get("proxy_used", "basic")
        self.cache_state = fields.get("cache_state", "miss")
        self.error = fields.get("error")
        self.source_url = fields.get("source_url")


class FakeDocument:
    def __init__(self, markdown: str = "", links: list[str] | None = None, **metadata: Any):
        self.markdown = markdown
        self.links = links or []
        self.metadata = FakeMetadata(**metadata)


class FakeLink:
    def __init__(self, url: str):
        self.url = url


class FakeMapData:
    def __init__(self, urls: list[str]):
        self.links = [FakeLink(url) for url in urls]


class FakeCreditUsage:
    def __init__(self, remaining: int):
        self.remaining_credits = remaining


class FirecrawlDown(Exception):
    def __init__(self, message: str = "boom", status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class FakeFirecrawl:
    """The slice of the SDK `fetch.py` uses, and nothing else.

    `pages` maps a URL to a `FakeDocument` or an exception to raise. Anything
    not listed comes back as a plain one-credit page, which keeps the tests
    about the case under test rather than about fixture bookkeeping.
    """

    def __init__(
        self,
        pages: dict[str, Any] | None = None,
        map_urls: list[str] | None = None,
        credits: int = 1000,
        default: Any | None = None,
    ):
        self.pages = pages or {}
        self.map_urls = map_urls
        self.credits = credits
        self.default = default
        self.scraped: list[tuple[str, dict[str, Any]]] = []
        self.mapped: list[tuple[str, dict[str, Any]]] = []

    def scrape(self, url: str, **options: Any) -> FakeDocument:
        self.scraped.append((url, options))
        page = self.pages.get(url, self.default)
        if page is None:
            page = FakeDocument(markdown=f"# {url}\n\nSome words about a facility.")
        if isinstance(page, Exception):
            raise page
        return page

    def map(self, url: str, **options: Any) -> FakeMapData:
        self.mapped.append((url, options))
        if self.map_urls is None:
            raise FirecrawlDown("no map configured")
        return FakeMapData(self.map_urls)

    def get_credit_usage(self) -> FakeCreditUsage:
        return FakeCreditUsage(self.credits)

    def extract(self, *args: Any, **kwargs: Any):
        raise AssertionError(
            "Firecrawl's /extract must never be called: it is billed separately "
            "and it re-couples scraping to prompting"
        )


def fake_discovery(domain: str, pack, urls: list[tuple[str, str]], **overrides: Any) -> DiscoveryResult:
    """A DiscoveryResult built by hand, so fetch tests never touch the network."""
    return DiscoveryResult(
        domain=domain,
        pack_version=pack.version,
        homepage_url=f"https://{domain}/",
        total_page_cap=pack.total_page_cap,
        pages=[
            DiscoveredPage(url=url, page_type=page_type, score=1.0)
            for url, page_type in urls
        ],
        sources_used=["nav", "sitemap"],
        candidates_considered=len(urls),
        **overrides,
    )


@pytest.fixture
def fetcher():
    """A fetcher over the double, with the rate limiter off.

    `rate_limit=0` disables the spacing. Left at its default these tests would
    sleep six seconds between calls to obey a limit the double does not have,
    which turns a one-second suite into a two-minute one. The limiter has its
    own tests.
    """

    def build(**kwargs: Any) -> tuple[Fetcher, FakeFirecrawl]:
        client = FakeFirecrawl(**kwargs)
        return Fetcher(client=client, rate_limit=0), client

    return build


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def test_pages_land_at_the_documented_path_with_the_manifest_alongside(pack, local_store, fetcher):
    fc, _ = fetcher()
    discovery = fake_discovery(
        "example.com", pack,
        [("https://example.com/", "homepage"), ("https://example.com/capabilities", "capabilities")],
    )

    record = crawl_domain(
        "example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery
    )

    assert record.pages_stored == 2
    for page in record.pages:
        assert page.stored_path == raw_page_path("example.com", CRAWL_DATE, page.url)
        assert local_store.exists(page.stored_path)

    manifest = local_store.read_json(raw_manifest_path("example.com", CRAWL_DATE))
    assert manifest["crawl_date"] == "2026-08-07"
    assert manifest["discovery"]["domain"] == "example.com"
    assert manifest["fetch"]["pages_stored"] == 2


def test_raw_markdown_is_stored_verbatim(pack, local_store, fetcher):
    body = "# Acme\n\nISO 8 cleanroom, 4,000 sq ft.\n"
    fc, _ = fetcher(pages={"https://example.com/": FakeDocument(markdown=body)})
    discovery = fake_discovery("example.com", pack, [("https://example.com/", "homepage")])

    record = crawl_domain(
        "example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery
    )

    stored = local_store.read_markdown(record.pages[0].stored_path)
    assert stored == body, "the raw store holds what the page said, not a processed form"
    assert record.pages[0].word_count == len(body.split())
    assert record.word_count == record.pages[0].word_count


def test_an_empty_page_is_recorded_but_not_stored(pack, local_store, fetcher):
    """A zero-byte file is indistinguishable from a page with nothing on it."""
    fc, _ = fetcher(
        pages={
            "https://example.com/": FakeDocument(markdown="# Acme\n\nWords."),
            "https://example.com/empty": FakeDocument(markdown="   "),
        }
    )
    discovery = fake_discovery(
        "example.com", pack,
        [("https://example.com/", "homepage"), ("https://example.com/empty", "about")],
    )

    record = crawl_domain(
        "example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery
    )

    empty = next(page for page in record.pages if page.url.endswith("/empty"))
    assert not empty.ok
    assert empty.stored_path is None
    assert "no markdown" in empty.error
    assert record.pages_stored == 1
    assert record.pages_failed == 1


def test_a_failed_page_does_not_lose_the_domain(pack, local_store, fetcher):
    fc, _ = fetcher(
        pages={
            "https://example.com/": FakeDocument(markdown="# Acme"),
            "https://example.com/gone": FirecrawlDown("404 not found", status_code=404),
        }
    )
    discovery = fake_discovery(
        "example.com", pack,
        [("https://example.com/", "homepage"), ("https://example.com/gone", "about")],
    )

    record = crawl_domain(
        "example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery
    )

    gone = next(page for page in record.pages if page.url.endswith("/gone"))
    assert not gone.ok
    assert gone.status == 404
    assert "404" in gone.error
    assert record.pages_stored == 1


# ---------------------------------------------------------------------------
# Credits
# ---------------------------------------------------------------------------


def test_credits_are_summed_per_domain_from_response_metadata(pack, local_store, fetcher):
    fc, _ = fetcher(
        pages={
            "https://example.com/": FakeDocument(markdown="# Acme", credits_used=1),
            "https://example.com/a": FakeDocument(markdown="A page", credits_used=2),
        }
    )
    discovery = fake_discovery(
        "example.com", pack,
        [("https://example.com/", "homepage"), ("https://example.com/a", "about")],
    )

    record = crawl_domain(
        "example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery
    )

    assert record.credits_billed == 3
    assert [page.credits for page in record.pages] == [1, 2]


def test_an_unreported_credit_counts_as_a_page(pack, local_store, fetcher):
    """Under-reporting would quietly disarm the budget guard."""
    fc, _ = fetcher(pages={"https://example.com/": FakeDocument(markdown="# Acme", credits_used=None)})
    discovery = fake_discovery("example.com", pack, [("https://example.com/", "homepage")])

    record = crawl_domain(
        "example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery
    )

    assert record.credits_billed == PAGE_CREDITS


def test_stealth_mode_is_flagged(pack, local_store, fetcher, caplog):
    fc, _ = fetcher(
        pages={
            "https://example.com/": FakeDocument(
                markdown="# Acme", credits_used=STEALTH_CREDITS, proxy_used="stealth"
            )
        }
    )
    discovery = fake_discovery("example.com", pack, [("https://example.com/", "homepage")])

    with caplog.at_level("WARNING"):
        record = crawl_domain(
            "example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery
        )

    assert record.stealth_pages == 1
    assert record.credits_billed == STEALTH_CREDITS
    assert any("stealth" in message for message in caplog.messages)


def test_credit_balance_survives_a_failing_check(fetcher):
    fc, client = fetcher(credits=1025)
    assert fc.credit_balance() == 1025

    def boom() -> None:
        raise FirecrawlDown("service unavailable", status_code=503)

    client.get_credit_usage = boom
    assert fc.credit_balance() is None


def test_credit_estimate_matches_the_dry_run_arithmetic(pack):
    results = [
        fake_discovery("a.com", pack, [("https://a.com/", "homepage")] * 1),
        fake_discovery("b.com", pack, [("https://b.com/", "homepage")], discovery_failed=True),
    ]
    estimate = credit_estimate(results)
    assert estimate == {
        "domains": 2,
        "pages": 2,
        "map_calls": 1,
        "credits": 2 * PAGE_CREDITS + 1 * MAP_CREDITS,
    }


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_a_domain_crawled_today_is_skipped(pack, local_store, fetcher):
    fc, client = fetcher()
    discovery = fake_discovery("example.com", pack, [("https://example.com/", "homepage")])
    crawl_domain("example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery)

    again = crawl_domain(
        "example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery
    )

    assert again.skipped
    assert "already crawled" in again.skip_reason
    assert again.credits_billed == 0
    assert len(client.scraped) == 1, "the skip must happen before any billed call"


def test_force_re_crawls_and_overwrites(pack, local_store, fetcher):
    fc, client = fetcher(pages={"https://example.com/": FakeDocument(markdown="# v1")})
    discovery = fake_discovery("example.com", pack, [("https://example.com/", "homepage")])
    crawl_domain("example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery)

    client.pages["https://example.com/"] = FakeDocument(markdown="# v2")
    record = crawl_domain(
        "example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery, force=True
    )

    assert not record.skipped
    assert local_store.read_markdown(record.pages[0].stored_path) == "# v2"


def test_a_different_crawl_date_is_a_different_crawl(pack, local_store, fetcher):
    fc, _ = fetcher()
    discovery = fake_discovery("example.com", pack, [("https://example.com/", "homepage")])
    crawl_domain("example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery)

    later = crawl_domain(
        "example.com", pack, local_store, fc, crawl_date=date(2026, 9, 1), discovery=discovery
    )

    assert not later.skipped
    assert local_store.exists(raw_manifest_path("example.com", date(2026, 9, 1)))
    assert local_store.exists(raw_manifest_path("example.com", CRAWL_DATE))


# ---------------------------------------------------------------------------
# The paid discovery fallback
# ---------------------------------------------------------------------------


def test_map_returns_urls_and_bills_one_credit(fetcher):
    fc, client = fetcher(map_urls=["https://example.com/", "https://example.com/a", "https://example.com/b"])
    session = fc.session("example.com")

    urls = session.map_urls("example.com", limit=100)

    assert len(urls) == 3
    assert session.map_calls == 1
    assert session.credits == MAP_CREDITS
    assert client.mapped[0][1]["limit"] == 100
    assert not client.scraped, "the rendered fallback must not fire when map is enough"


def test_a_thin_map_falls_through_to_the_rendered_homepage(fetcher):
    """The bot-wall case: map sees nothing, only a full browser fetch does."""
    fc, client = fetcher(
        map_urls=[],
        pages={
            "https://example.com/": FakeDocument(
                markdown="# Acme", links=["https://example.com/capabilities"]
            )
        },
    )
    session = fc.session("example.com")

    urls = session.map_urls("example.com", limit=100)

    assert urls == ["https://example.com/capabilities"]
    assert client.scraped[0][1]["only_main_content"] is False, (
        "the nav is what is missing, and main-content extraction is what removes it"
    )
    assert "links" in client.scraped[0][1]["formats"]


def test_the_prefetched_homepage_is_not_billed_twice(pack, local_store, fetcher):
    fc, client = fetcher(
        map_urls=[],
        pages={
            "https://example.com/": FakeDocument(
                markdown="# Acme rendered", links=["https://example.com/capabilities"], credits_used=1
            )
        },
    )
    session = fc.session("example.com")
    session.map_urls("example.com", limit=10)
    assert session.credits == MAP_CREDITS + 1, "one map call plus the rendered homepage"

    scraped = session.scrape("https://example.com", "homepage")

    assert scraped.markdown == "# Acme rendered"
    assert session.credits == MAP_CREDITS + 1, "a cache hit re-bills nothing"
    assert len(client.scraped) == 1


# ---------------------------------------------------------------------------
# The rescue pass
# ---------------------------------------------------------------------------


def test_a_flagged_domain_is_rescued_from_the_rendered_nav(pack, local_store, fetcher):
    fc, client = fetcher(
        pages={
            "https://example.com/": FakeDocument(
                markdown="# Acme",
                links=[
                    "https://example.com/capabilities",
                    "https://example.com/about-us",
                    "https://example.com/contact",
                ],
            )
        }
    )
    discovery = fake_discovery(
        "example.com", pack, [("https://example.com/", "homepage")],
        discovery_failed=True,
        failure_reason="client-rendered navigation",
    )

    record = crawl_domain(
        "example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery
    )

    assert record.rescued
    assert record.rescue_pages >= 2
    assert record.pages_stored >= 3
    assert {page.origin for page in record.pages} == {"discovery", "rescue"}
    assert client.scraped[0][1]["only_main_content"] is False, (
        "a flagged domain gets the whole page, chrome included"
    )
    # The rescue still runs through the pack: caps, priorities and the
    # editorial rules are not bypassed by a different candidate source.
    assert all(page.page_type != "homepage" for page in record.pages if page.origin == "rescue")


def test_a_healthy_domain_is_never_rescued(pack, local_store, fetcher):
    fc, client = fetcher(
        pages={
            "https://example.com/": FakeDocument(
                markdown="# Acme", links=["https://example.com/elsewhere"]
            )
        }
    )
    discovery = fake_discovery(
        "example.com", pack,
        [
            ("https://example.com/", "homepage"),
            ("https://example.com/a", "capabilities"),
            ("https://example.com/b", "about"),
        ],
    )

    record = crawl_domain(
        "example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery
    )

    assert not record.rescued
    assert record.pages_stored == 3
    assert client.scraped[0][1]["only_main_content"] is True


def test_a_site_that_refuses_everything_is_not_rescued(pack, local_store, fetcher):
    """`bionostix.com`, 2026-08-26: 20 credits for one usable page.

    Nine 403s left a single `ok` page, which is under the minimum, so the
    rescue fired and bought nine more 403s off the same wall. The trigger
    counts how many pages succeeded; it has to look at why the rest failed.
    """
    fc, client = fetcher(
        pages={
            "https://example.com/": FakeDocument(
                markdown="# Acme",
                links=[
                    "https://example.com/capabilities",
                    "https://example.com/about-us",
                    "https://example.com/contact",
                ],
            ),
            "https://example.com/a": FakeDocument(markdown="Forbidden", status_code=403),
            "https://example.com/b": FakeDocument(markdown="Forbidden", status_code=403),
        }
    )
    discovery = fake_discovery(
        "example.com", pack,
        [
            ("https://example.com/", "homepage"),
            ("https://example.com/a", "capabilities"),
            ("https://example.com/b", "about"),
        ],
        discovery_failed=True,
        failure_reason="client-rendered navigation",
    )

    record = crawl_domain(
        "example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery
    )

    assert not record.rescued
    assert record.rescue_pages == 0
    assert [url for url, _ in client.scraped] == [
        "https://example.com/",
        "https://example.com/a",
        "https://example.com/b",
    ], "nothing beyond the original selection may be fetched from a refusing site"


def test_one_refusal_among_real_failures_still_allows_the_rescue(pack, local_store, fetcher):
    """A single gated page is not a wall.

    The guard is a ratio, not a tripwire: a login-gated careers page 403s on
    plenty of healthy sites, and that must not cost the domain its rescue.
    """
    fc, _ = fetcher(
        pages={
            "https://example.com/": FakeDocument(
                markdown="# Acme",
                links=[
                    "https://example.com/capabilities",
                    "https://example.com/about-us",
                    "https://example.com/contact",
                ],
            ),
            "https://example.com/a": FakeDocument(markdown="Forbidden", status_code=403),
            "https://example.com/b": FakeDocument(markdown="Gone", status_code=404),
            "https://example.com/c": FakeDocument(markdown="Oops", status_code=500),
        }
    )
    discovery = fake_discovery(
        "example.com", pack,
        [
            ("https://example.com/", "homepage"),
            ("https://example.com/a", "capabilities"),
            ("https://example.com/b", "about"),
            ("https://example.com/c", "quality"),
        ],
        discovery_failed=True,
        failure_reason="client-rendered navigation",
    )

    record = crawl_domain(
        "example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery
    )

    assert record.rescued, "1 refusal in 3 failures is not a site-wide refusal"
    assert record.rescue_pages >= 1


def test_a_refused_page_is_not_stored_as_site_text(pack, local_store, fetcher):
    """A 403 body is valid markdown, and it is not site content.

    `bionostix.com` left 17 files reading "Forbidden" under `data/raw/`.
    Extraction reads the manifest and was never fooled, but Phase 7a's
    thin-site harvest and any grep over the stored markdown would be.
    """
    fc, _ = fetcher(
        pages={"https://example.com/a": FakeDocument(markdown="Forbidden", status_code=403)}
    )
    discovery = fake_discovery(
        "example.com", pack,
        [("https://example.com/", "homepage"), ("https://example.com/a", "capabilities")],
    )

    record = crawl_domain(
        "example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery
    )

    refused = next(page for page in record.pages if page.url.endswith("/a"))
    assert not refused.ok
    assert refused.stored_path is None, "a refusal must not land in the raw store"
    assert refused.word_count == 0
    assert record.pages_stored == 1, "only the homepage is real text"
    # The failure is still on the record: a page that was refused and a page
    # that was never tried are different facts.
    assert refused.status == 403
    assert record.pages_failed == 1


# ---------------------------------------------------------------------------
# The batch
# ---------------------------------------------------------------------------


def one_page_each(domains: list[str], pack) -> dict[str, DiscoveryResult]:
    """A homepage-only selection per domain, so the batch never hits the network."""
    return {
        domain: fake_discovery(domain, pack, [(f"https://{domain}/", "homepage")])
        for domain in domains
    }


def test_crawl_all_preserves_input_order(pack, local_store, fetcher):
    fc, _ = fetcher()
    domains = ["a.com", "b.com", "c.com"]

    records = crawl_all(
        domains, pack, local_store, fc,
        crawl_date=CRAWL_DATE,
        concurrency=2,
        discoveries=one_page_each(domains, pack),
    )

    assert [record.domain for record in records] == domains
    assert all(record.pages_stored == 1 for record in records)


def test_a_discovery_result_for_another_domain_is_refused(pack, local_store, fetcher):
    fc, _ = fetcher()
    with pytest.raises(FetchError, match="not 'example.com'"):
        crawl_domain(
            "example.com", pack, local_store, fc,
            crawl_date=CRAWL_DATE,
            discovery=fake_discovery("other.com", pack, [("https://other.com/", "homepage")]),
        )


def test_the_budget_stops_the_batch(pack, local_store, fetcher):
    fc, _ = fetcher(default=FakeDocument(markdown="# page", credits_used=10))
    domains = [f"d{index}.com" for index in range(6)]

    records = crawl_all(
        domains, pack, local_store, fc,
        crawl_date=CRAWL_DATE,
        concurrency=1,
        budget=25,
        discoveries=one_page_each(domains, pack),
    )

    spent = sum(record.credits_billed for record in records)
    stopped = [record for record in records if record.skipped]
    assert stopped, "the batch must stop rather than spend past the budget"
    assert spent <= 25 + 10, "overshoot is bounded by one domain in flight"
    assert any("budget" in (record.skip_reason or "") for record in stopped)


def test_a_402_stops_the_whole_batch(pack, local_store, fetcher):
    """Out of credits fails identically on every later domain."""
    fc, _ = fetcher(default=FirecrawlDown("payment required", status_code=402))
    domains = [f"d{index}.com" for index in range(4)]

    records = crawl_all(
        domains, pack, local_store, fc,
        crawl_date=CRAWL_DATE,
        concurrency=1,
        discoveries=one_page_each(domains, pack),
    )

    assert all(record.skipped for record in records)
    assert "402" in str(records[0].skip_reason) or "payment" in str(records[0].skip_reason).lower()
    assert not local_store.list_paths("raw"), "a fatal batch must not leave manifests behind"


def test_is_fatal_only_covers_the_key_and_the_balance():
    assert is_fatal(FirecrawlDown(status_code=401))
    assert is_fatal(FirecrawlDown(status_code=402))
    # An unsupported site is one domain's problem, not the batch's.
    assert not is_fatal(FirecrawlDown(status_code=403))
    assert not is_fatal(FirecrawlDown(status_code=500))
    assert not is_fatal(FirecrawlDown())


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
#
# The limit that bit on the first real run. The plan caps requests per minute
# as well as credits, the SDK retries only 502s, and a refused page costs no
# credits but loses the page. Ten domains lost every page that way and each one
# wrote a manifest that read like a dead site.


def rate_limited(seconds: int | None = 48) -> FirecrawlDown:
    hint = f" please retry after {seconds}s," if seconds else ""
    return FirecrawlDown(
        f"Rate Limit Exceeded: Failed to scrape. Consumed (req/min): 14, "
        f"Remaining (req/min): 0.{hint} resets at Fri Aug 07 2026",
        status_code=429,
    )


def test_the_retry_after_hint_is_read_from_the_message():
    assert rate_limit_wait(rate_limited(48)) == 49.0
    assert rate_limit_wait(rate_limited(None)) == DEFAULT_RATE_LIMIT_WAIT
    assert rate_limit_wait(FirecrawlDown("404 not found", status_code=404)) is None


def test_calls_are_spaced_to_the_configured_rate(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("cleanroom.fetch.time.sleep", slept.append)

    limiter = RateLimiter(per_minute=60)
    for _ in range(3):
        limiter.acquire()

    # Slots are handed out ahead of the clock, so with no real time passing
    # between calls each one waits a full interval longer than the last. The
    # first goes straight through.
    assert len(slept) == 2
    assert abs(slept[0] - 1.0) < 0.05
    assert abs(slept[1] - 2.0) < 0.05


def test_a_rate_limit_of_zero_disables_the_spacing(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("cleanroom.fetch.time.sleep", slept.append)

    limiter = RateLimiter(per_minute=0)
    for _ in range(5):
        limiter.acquire()

    assert not slept


def test_a_rate_limited_page_is_retried_not_recorded(monkeypatch, fetcher):
    monkeypatch.setattr("cleanroom.fetch.time.sleep", lambda _: None)
    calls: list[str] = []
    fc, client = fetcher()

    def flaky(url: str, **options: Any) -> FakeDocument:
        calls.append(url)
        if len(calls) == 1:
            raise rate_limited()
        return FakeDocument(markdown="# Acme")

    client.scrape = flaky
    session = fc.session("example.com")

    scraped = session.scrape("https://example.com/", "homepage")

    assert scraped.page.ok
    assert not scraped.page.rate_limited
    assert len(calls) == 2, "the page is retried once the window resets"


def test_a_throttled_crawl_writes_no_manifest(monkeypatch, pack, local_store, fetcher):
    """A manifest asserts the domain was crawled. An account limit is not that."""
    monkeypatch.setattr("cleanroom.fetch.time.sleep", lambda _: None)
    fc, _ = fetcher(
        pages={
            "https://example.com/": FakeDocument(markdown="# Acme"),
            "https://example.com/a": rate_limited(),
        }
    )
    discovery = fake_discovery(
        "example.com", pack,
        [("https://example.com/", "homepage"), ("https://example.com/a", "capabilities")],
    )

    record = crawl_domain(
        "example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery
    )

    assert record.crawl_failed
    assert record.rate_limited_pages == 1
    assert "refused for rate" in record.crawl_error
    assert not local_store.exists(raw_manifest_path("example.com", CRAWL_DATE))


def test_a_domain_with_no_manifest_is_retried_without_force(monkeypatch, pack, local_store, fetcher):
    """The whole reason the manifest is withheld."""
    monkeypatch.setattr("cleanroom.fetch.time.sleep", lambda _: None)
    fc, client = fetcher(pages={"https://example.com/": rate_limited()})
    discovery = fake_discovery("example.com", pack, [("https://example.com/", "homepage")])

    first = crawl_domain(
        "example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery
    )
    assert first.crawl_failed

    client.pages["https://example.com/"] = FakeDocument(markdown="# Acme")
    second = crawl_domain(
        "example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery
    )

    assert not second.skipped, "no manifest means nothing to skip"
    assert not second.crawl_failed
    assert local_store.exists(raw_manifest_path("example.com", CRAWL_DATE))


def test_a_domain_that_stored_nothing_writes_no_manifest(pack, local_store, fetcher):
    fc, _ = fetcher(pages={"https://example.com/": FakeDocument(markdown="")})
    discovery = fake_discovery("example.com", pack, [("https://example.com/", "homepage")])

    record = crawl_domain(
        "example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery
    )

    assert record.crawl_failed
    assert "nothing stored" in record.crawl_error
    assert not local_store.exists(raw_manifest_path("example.com", CRAWL_DATE))


# ---------------------------------------------------------------------------
# The endpoint that must never be called
# ---------------------------------------------------------------------------


def test_extract_is_never_called(pack, local_store, fetcher):
    fc, client = fetcher()
    discovery = fake_discovery(
        "example.com", pack,
        [("https://example.com/", "homepage"), ("https://example.com/a", "capabilities")],
    )

    crawl_domain("example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery)

    with pytest.raises(AssertionError):
        client.extract("https://example.com/")


def test_a_fetcher_without_a_key_fails_loudly():
    with pytest.raises(ConfigError, match="FIRECRAWL_API_KEY"):
        Fetcher()


# ---------------------------------------------------------------------------
# The manifest as Phase 5 and Phase 7a will read it
# ---------------------------------------------------------------------------


def test_the_manifest_carries_everything_the_next_phases_need(pack, local_store, fetcher):
    fc, _ = fetcher()
    discovery = fake_discovery(
        "example.com", pack,
        [("https://example.com/", "homepage"), ("https://example.com/a", "capabilities")],
    )

    crawl_domain("example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery)
    manifest = local_store.read_json(raw_manifest_path("example.com", CRAWL_DATE))
    fetch = manifest["fetch"]

    # Phase 7a sorts manifests by these two without loading any class.
    assert fetch["pages_stored"] == 2
    assert fetch["word_count"] > 0

    # Phase 5 needs url, page_type and the stored path for every page.
    for page in fetch["pages"]:
        assert page["url"] and page["page_type"]
        assert page["stored_path"].startswith("raw/example.com/2026-08-07/")

    # And the whole thing has to survive a JSON round trip unchanged.
    assert json.loads(json.dumps(manifest)) == manifest


def test_the_manifest_carries_the_redirect_fact(pack, local_store, fetcher):
    """`resolved_domain`/`redirected_off_domain` are discovery's own fact
    (an acquired company's site redirecting wholesale to its new owner) and
    used to die at this boundary. Phase 5 needs them in the manifest to carry
    the fact through to classify time.
    """
    fc, _ = fetcher()
    discovery = fake_discovery(
        "example.com", pack,
        [("https://acquirer.com/", "homepage")],
        resolved_domain="acquirer.com",
        redirected_off_domain=True,
    )

    crawl_domain("example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery)
    manifest = local_store.read_json(raw_manifest_path("example.com", CRAWL_DATE))
    fetch = manifest["fetch"]

    assert fetch["resolved_domain"] == "acquirer.com"
    assert fetch["redirected_off_domain"] is True


def test_the_redirect_fact_defaults_false_and_none(pack, local_store, fetcher):
    fc, _ = fetcher()
    discovery = fake_discovery(
        "example.com", pack, [("https://example.com/", "homepage")]
    )
    crawl_domain("example.com", pack, local_store, fc, crawl_date=CRAWL_DATE, discovery=discovery)
    fetch = local_store.read_json(raw_manifest_path("example.com", CRAWL_DATE))["fetch"]
    assert fetch["resolved_domain"] is None
    assert fetch["redirected_off_domain"] is False


def test_records_round_trip_through_their_own_schema():
    page = PageFetch(url="https://example.com/", page_type="homepage", ok=True, credits=1)
    record = DomainCrawl(
        domain="example.com", crawl_date=CRAWL_DATE, pack_version="medical_v1", pages=[page]
    )
    assert DomainCrawl.model_validate(record.model_dump(mode="json")) == record
