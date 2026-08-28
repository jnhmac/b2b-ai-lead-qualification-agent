"""Phase 4: scrape the pages discovery selected, store the markdown, bill it.

This is the first stage that spends real money and the only one that cannot be
re-run for free, which sets every design decision in the module.

**Everything above the raw store runs once; everything below it runs fifty
times.** So this module writes plain markdown keyed by crawl date and nothing
else. No pack version in the path, no extraction, no interpretation. A prompt
revision must never imply a re-crawl.

**Credits are counted per domain, not per process.** The batch runs two domains
at a time against a plan whose concurrency limit is 2, and a single counter on
the client attributes a domain's spend to whichever one happened to finish
last. Each domain gets its own `DomainSession` holding its own tally, which is
what makes the per-domain credit log the gate asks for trustworthy.

**The manifest's fetch block is the authoritative list of what was stored.**
Discovery's half of the manifest records what was *selected*; this half records
what actually came back, with the store path, the word count and the credits
for each page. Phase 5 reads the fetch block and needs nothing else; Phase 7a
sorts manifests by `pages_stored` and `word_count` to find the thin-site
population.

**Firecrawl's `/extract` is never called.** It is billed separately and it
re-couples scraping to prompting, which is the one coupling this pipeline is
built to avoid. `scrape` and `map` are the only two endpoints used, plus the
free credit-balance check.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .config import ConfigError, SectorPack, Settings
from .discover import (
    MIN_PAGES,
    Candidate,
    DiscoveredPage,
    DiscoveryResult,
    canonical_url,
    dedupe_key,
    discover,
    is_crawlable,
    locale_free_key,
    select_pages,
    write_manifest,
)
from .ladder import Step1Branch, select_step_1_pages
from .store import Store, StoreConflictError, raw_manifest_path, raw_page_path

log = logging.getLogger(__name__)

PageOrigin = Literal["discovery", "rescue"]

#: Firecrawl bills one credit per scraped page and one per map call. The
#: numbers are assumed rather than derived because the map response carries no
#: usage block; a scrape's `credits_used` is read from metadata when present
#: and falls back to this.
PAGE_CREDITS = 1
MAP_CREDITS = 1

#: A stealth-proxy page bills five credits instead of one. Rare on these sites,
#: and rare is exactly why it has to be alerted on: five times the price on a
#: handful of pages is invisible in a total and obvious in a per-page log.
STEALTH_CREDITS = 5

#: Statuses that mean "this site is refusing us", as opposed to "this page is
#: not there". The distinction decides whether the rescue pass is worth its
#: credits: a 404 says nothing about the next URL, a 403 says the next URL will
#: be refused too. Measured on `bionostix.com` 2026-08-26, which billed 20
#: credits for one usable page by fetching nine 403s and then rescuing nine
#: more from the same wall.
BLOCKED_STATUSES = frozenset({401, 403, 429})

#: The plan's concurrency limit as measured on 2026-08-06. Exceeding it does
#: not fail, it queues, and a queued request holds a worker doing nothing.
DEFAULT_CONCURRENCY = 2

#: Requests per minute. The plan caps rate as well as credits, and the two fail
#: differently: credits run out once, a rate limit runs out every sixty
#: seconds. Measured on 2026-08-07, the first refusal landed at 14 requests
#: inside one window and a 10-request minute went through clean, so this is set
#: to the free tier's documented 10/min. Raise it with `--rate-limit` on a plan
#: that allows more; the ceiling is the reason a 428-page crawl takes the best
#: part of an hour.
DEFAULT_RATE_LIMIT_PER_MINUTE = 10

#: A rate-limited call costs no credits and loses the page, so it is worth
#: waiting out rather than giving up on.
MAX_RATE_LIMIT_RETRIES = 4
#: Used only when the provider does not say how long to wait. The window is
#: rolling, so this is a whole window rather than a guess at the remainder.
DEFAULT_RATE_LIMIT_WAIT = 60.0

#: How many URLs to ask the map endpoint for, as a multiple of the page cap.
#: Wide enough that scoring has something to choose between, since the call
#: costs the same whether it returns ten URLs or a hundred.
MAP_LIMIT_FACTOR = 10

#: HTTP statuses where retrying is pointless and costs money or trust: a bad
#: key and an exhausted balance both fail identically on every later domain, so
#: the whole batch stops rather than burning through 500 domains of failures.
#: 403 is deliberately absent: Firecrawl returns it for an unsupported site,
#: which is one domain's problem and not the batch's.
FATAL_STATUS = frozenset({401, 402})


class LadderStep1(Protocol):
    """Step 1's model-driven branch, injected rather than imported.

    `fetch.py` owns pages and credits; the ladder owns the reading. Keeping it
    a callable means the fetch stage has no dependency on `llm.py`, the paid
    path stays visible at the call site, and an unwired ladder leaves
    `crawl_domain` behaving exactly as it did before the ladder existed.

    Phase 3 swaps in a version that routes the step to its assigned model tier
    (Appendix B.1) without this signature changing.
    """

    def __call__(
        self, pack: SectorPack, pages: Sequence[tuple[str, str, str]]
    ) -> Step1Branch: ...


class FetchError(RuntimeError):
    """The fetch stage could not run, as opposed to a page that did not come back."""


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def _status_code(exc: BaseException) -> int | None:
    return getattr(exc, "status_code", None)


def is_fatal(exc: BaseException) -> bool:
    """True when continuing the batch would fail identically on every domain."""
    return _status_code(exc) in FATAL_STATUS


_RETRY_AFTER_RE = re.compile(r"retry after (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def rate_limit_wait(exc: BaseException) -> float | None:
    """Seconds to wait before retrying, or None when this is not a rate limit.

    Firecrawl puts the answer in the message ("please retry after 48s") and it
    is worth parsing rather than backing off exponentially: the window is
    rolling, so a doubling backoff either sleeps through minutes it did not
    need or retries straight back into the wall.
    """
    if _status_code(exc) != 429 and "rate limit" not in str(exc).lower():
        return None
    match = _RETRY_AFTER_RE.search(str(exc))
    # One second of margin: the server's own clock decides when the window
    # resets, and landing exactly on it fails again for free.
    return float(match.group(1)) + 1.0 if match else DEFAULT_RATE_LIMIT_WAIT


class RateLimiter:
    """A shared gate in front of every billed call.

    Spacing beats retrying here. A rate-limited call costs no credits but loses
    the page, and a domain that loses all ten of its pages is written to the
    manifest looking exactly like a dead site. One limiter is shared across the
    whole batch because the limit is per account, not per domain.
    """

    def __init__(self, per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE):
        self.per_minute = per_minute
        self.interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            wait = max(self._next - now, 0.0)
            self._next = max(now, self._next) + self.interval
        # Slept outside the lock: two workers hold two different slots and
        # should wait in parallel, not one behind the other.
        if wait:
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Records: the fetch half of the crawl manifest
# ---------------------------------------------------------------------------


class PageFetch(BaseModel):
    """One scrape attempt: what it cost, what came back, where it was stored.

    Written for failures too. A page that 404s and a page that was never tried
    are different facts, and only the record distinguishes them after the fact.
    """

    model_config = ConfigDict(extra="forbid")

    url: str
    page_type: str
    origin: PageOrigin = "discovery"

    ok: bool = False
    status: int | None = None
    credits: int = 0

    #: 'basic' or 'stealth'. Stealth is the five-credit path.
    proxy: str | None = None
    #: 'hit' or 'miss' on Firecrawl's own page cache. A hit still bills.
    cache_state: str | None = None

    word_count: int = 0
    bytes_stored: int = 0
    stored_path: str | None = None
    error: str | None = None

    #: The page was refused for rate, not for content. It costs no credits and
    #: says nothing about the site, so a domain carrying one of these is not a
    #: crawl result and no manifest is written for it.
    rate_limited: bool = False

    @property
    def stealth(self) -> bool:
        return self.proxy == "stealth" or self.credits >= STEALTH_CREDITS


class DomainCrawl(BaseModel):
    """What the crawl of one domain fetched, stored and spent.

    The totals are stored fields rather than derived properties on purpose:
    Phase 7a sorts the written manifests by `pages_stored` and `word_count`
    without loading this class, so those numbers have to be in the JSON.
    """

    model_config = ConfigDict(extra="forbid")

    domain: str
    crawl_date: date
    pack_version: str

    pages: list[PageFetch] = Field(default_factory=list)

    pages_requested: int = 0
    pages_stored: int = 0
    pages_failed: int = 0
    word_count: int = 0
    bytes_stored: int = 0

    credits_billed: int = 0
    map_calls: int = 0
    stealth_pages: int = 0

    #: True when discovery came back unusable and the rendered homepage was
    #: harvested for links to fill the crawl out. Phase 7a needs to see this:
    #: a rescued domain is not a thin site, it is a Phase 3 bug that Phase 4
    #: worked around.
    rescued: bool = False
    rescue_pages: int = 0

    discovery_failed: bool = False
    failure_reason: str | None = None

    #: Carried straight from `DiscoveryResult`: the registrable domain the
    #: crawl actually landed on, and whether that is off `domain` (an
    #: acquired company's site redirecting wholesale to its new owner). Used
    #: to die at this Phase 3/4 boundary; Phase 5/6 need it in the manifest.
    resolved_domain: str | None = None
    redirected_off_domain: bool = False
    #: The domain this row should be re-run against (tab 2a R12). Carried from
    #: `DiscoveryResult` so the requeue list can be built from the manifests
    #: alone, without re-resolving every moved domain.
    referred_domain: str | None = None

    #: Which ladder step resolved this domain and which stopping rule fired.
    #: `crawl_policy.md` Appendix A rolls these up per domain; they are set
    #: here so a manifest written today is already readable by that report.
    exit_step: int | None = None
    stop_reason: str | None = None

    #: Ladder step (as a string key, so it survives a JSON round trip) to the
    #: model that ran it. A roll-up of the per-page `model_used`, NOT a
    #: replacement for it: Appendix B.1 routes by step, so a domain that
    #: reaches step 2 carries the small model's step-1 entry and the strong
    #: model's step-2 entry at the same time, and B2 requires the step-1
    #: record be retained. Empty when no model ran, which is the correct and
    #: distinct answer for a step-0 exit rather than a stale default.
    models_used: dict[str, str] = Field(default_factory=dict)

    #: The ladder reached a branch whose verdict is withheld pending a stronger
    #: model (Appendix B.2, step-1 exit B). The crawl has resolved the domain
    #: and the verdict must still not be written, so the row has to be visible
    #: as awaiting something rather than silently absent from every queue.
    awaiting_confirmation: bool = False

    #: The crawl itself did not produce a usable result, as opposed to
    #: discovery not finding much. Nothing is written to the store for these,
    #: so a later run retries them without needing `--force`.
    crawl_failed: bool = False
    crawl_error: str | None = None
    rate_limited_pages: int = 0

    skipped: bool = False
    skip_reason: str | None = None

    manifest_path: str | None = None
    started_at: datetime | None = None
    duration_s: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "stored": self.pages_stored,
            "failed": self.pages_failed,
            "credits": self.credits_billed,
            "words": self.word_count,
            "skipped": self.skipped,
        }


@dataclass(frozen=True)
class Scraped:
    """A page fetch and the markdown it produced, before anything is stored."""

    page: PageFetch
    markdown: str = ""
    links: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The Firecrawl client
# ---------------------------------------------------------------------------


class ScrapeAPI(Protocol):
    """The three Firecrawl calls this pipeline makes.

    Typed as a protocol so the whole stage runs against a double with no key
    and no spend. `/extract` is absent by design and not by omission.
    """

    def scrape(self, url: str, **options: Any) -> Any: ...

    def map(self, url: str, **options: Any) -> Any: ...

    def get_credit_usage(self) -> Any: ...


class Fetcher:
    """Firecrawl with this pipeline's options pinned in one place.

    Holds no per-domain state. Everything mutable lives on the `DomainSession`
    it hands out, so two domains can be in flight without their credit tallies
    crossing.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: ScrapeAPI | None = None,
        timeout_ms: int | None = None,
        only_main_content: bool = True,
        proxy: str | None = None,
        block_ads: bool = True,
        rate_limit: int = DEFAULT_RATE_LIMIT_PER_MINUTE,
    ):
        if client is None:
            if not api_key:
                raise ConfigError("FIRECRAWL_API_KEY is not set, and the fetch stage needs it")
            try:
                from firecrawl import Firecrawl  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - depends on install
                raise FetchError("the fetch stage needs firecrawl-py installed") from exc
            client = Firecrawl(api_key=api_key)

        self.client = client
        self.timeout_ms = timeout_ms
        self.only_main_content = only_main_content
        self.proxy = proxy
        self.block_ads = block_ads
        self.limiter = RateLimiter(rate_limit)

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> Fetcher:
        return cls(settings.require_firecrawl_api_key(), **kwargs)

    def credit_balance(self) -> int | None:
        """Remaining credits, or None if the check itself failed.

        Free, and worth calling before a run: the store is append-only and a
        batch that dies halfway through leaves a half-crawled population that
        needs `--force` to finish.
        """
        try:
            usage = self.client.get_credit_usage()
        except Exception as exc:  # noqa: BLE001 - a balance check must not abort a run
            log.warning("could not read the Firecrawl credit balance: %s", exc)
            return None
        remaining = getattr(usage, "remaining_credits", None)
        return int(remaining) if remaining is not None else None

    def session(self, domain: str) -> DomainSession:
        return DomainSession(self, domain)

    # -- the two endpoints ---------------------------------------------------

    def call(self, what: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Every billed call goes through here: spaced, and retried on a 429.

        The limiter should make a rate-limit refusal rare. It is retried rather
        than recorded because the alternative is a manifest that says a domain
        has nothing to say when all that happened is that the minute was full.
        """
        for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
            self.limiter.acquire()
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - re-raised unless it is a rate limit
                wait = rate_limit_wait(exc)
                if wait is None or attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                log.warning(
                    "rate limited on %s, waiting %.0fs (attempt %d of %d)",
                    what, wait, attempt, MAX_RATE_LIMIT_RETRIES,
                )
                time.sleep(wait)
        raise FetchError(f"unreachable: {what} exhausted its rate-limit retries")

    def _scrape_options(self, *, want_links: bool, main_content: bool) -> dict[str, Any]:
        formats = ["markdown", "links"] if want_links else ["markdown"]
        options: dict[str, Any] = {
            "formats": formats,
            "only_main_content": main_content,
            "block_ads": self.block_ads,
        }
        if self.timeout_ms is not None:
            options["timeout"] = self.timeout_ms
        if self.proxy is not None:
            options["proxy"] = self.proxy
        return options


class DomainSession:
    """Every Firecrawl call for one domain, with that domain's credit tally.

    Also owns the prefetch cache. When the map fallback has to render the
    homepage to find links, that render is a page fetch this crawl was going to
    make anyway, so the document is kept and the later fetch reuses it instead
    of billing a second credit for the same page.
    """

    def __init__(self, fetcher: Fetcher, domain: str):
        self.fetcher = fetcher
        self.domain = domain
        self.credits = 0
        self.map_calls = 0
        self._prefetched: dict[str, Scraped] = {}

    # -- the paid discovery fallback ----------------------------------------

    def map_urls(self, domain: str, *, limit: int) -> list[str]:
        """Firecrawl's rendered link discovery. Matches `discover.MapUrls`.

        Called only when the free sources (sitemap, robots.txt, homepage nav)
        come back below `min_pages`, which is the client-rendered-nav and
        bot-wall case.

        Two steps, in cost order. The map endpoint first: it reads the sitemap
        and the rendered link graph for one credit. If that is still too thin,
        the homepage is scraped with the `links` format, which is the only way
        to get a full browser fetch behind Firecrawl's proxies. That scrape is
        kept in the prefetch cache, so the credit buys a page this crawl needed
        regardless.

        `only_main_content` is off for that scrape, and off deliberately: the
        nav is what is missing, and the nav is exactly what main-content
        extraction removes.
        """
        urls: list[str] = []
        try:
            data = self.fetcher.call(
                f"map {domain}",
                self.fetcher.client.map,
                f"https://{domain}",
                limit=limit,
                sitemap="include",
                ignore_query_parameters=True,
            )
        except Exception as exc:  # noqa: BLE001 - a paid dependency must not abort a batch
            log.warning("%s: map endpoint failed: %s", domain, exc)
            if is_fatal(exc):
                raise
        else:
            self.map_calls += 1
            self.credits += MAP_CREDITS
            urls = _map_links(data)
            log.info("%s: map endpoint returned %d URLs", domain, len(urls))

        usable = [url for url in urls if is_crawlable(url, domain)]
        if len(usable) >= MIN_PAGES:
            return urls

        log.info(
            "%s: map returned %d usable URL(s), rendering the homepage for its nav",
            domain, len(usable),
        )
        rendered = self._prefetch_homepage(domain)
        return [*urls, *rendered.links]

    def _prefetch_homepage(self, domain: str) -> Scraped:
        url = canonical_url(f"https://{domain}/")
        scraped = self._scrape(url, "homepage", want_links=True, main_content=False)
        self._prefetched[dedupe_key(url)] = scraped
        return scraped

    # -- page fetches --------------------------------------------------------

    def scrape(
        self,
        url: str,
        page_type: str,
        *,
        origin: PageOrigin = "discovery",
        want_links: bool = False,
        main_content: bool | None = None,
    ) -> Scraped:
        """Fetch one page, or hand back the prefetched copy of it.

        A cache hit re-labels the record with the page type and origin the
        caller asked for but keeps the credits it was actually billed, so the
        domain total stays equal to what Firecrawl charged.
        """
        cached = self._prefetched.pop(dedupe_key(url), None)
        if cached is not None:
            log.debug("%s: reusing the prefetched fetch of %s", self.domain, url)
            return Scraped(
                page=cached.page.model_copy(update={"page_type": page_type, "origin": origin}),
                markdown=cached.markdown,
                links=cached.links,
            )

        content = self.fetcher.only_main_content if main_content is None else main_content
        scraped = self._scrape(url, page_type, want_links=want_links, main_content=content)
        return Scraped(
            page=scraped.page.model_copy(update={"origin": origin}),
            markdown=scraped.markdown,
            links=scraped.links,
        )

    def _scrape(self, url: str, page_type: str, *, want_links: bool, main_content: bool) -> Scraped:
        options = self.fetcher._scrape_options(want_links=want_links, main_content=main_content)
        try:
            document = self.fetcher.call(
                f"scrape {url}", self.fetcher.client.scrape, url, **options
            )
        except Exception as exc:  # noqa: BLE001 - one dead page must not abort a domain
            log.warning("%s: scrape failed on %s: %s", self.domain, url, exc)
            if is_fatal(exc):
                raise
            return Scraped(
                page=PageFetch(
                    url=url,
                    page_type=page_type,
                    ok=False,
                    status=_status_code(exc),
                    rate_limited=rate_limit_wait(exc) is not None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        return self._from_document(document, url, page_type)

    def _from_document(self, document: Any, url: str, page_type: str) -> Scraped:
        metadata = getattr(document, "metadata", None)
        markdown = getattr(document, "markdown", None) or ""
        links = [str(link) for link in (getattr(document, "links", None) or [])]

        reported = getattr(metadata, "credits_used", None) if metadata else None
        # An absent `credits_used` is not a free page. Under-reporting here
        # would quietly disarm the budget guard, so an unreported scrape counts
        # as the standard page price.
        credits = int(reported) if reported is not None else PAGE_CREDITS
        self.credits += credits

        status = getattr(metadata, "status_code", None) if metadata else None
        proxy = getattr(metadata, "proxy_used", None) if metadata else None
        error = getattr(metadata, "error", None) if metadata else None

        page = PageFetch(
            url=url,
            page_type=page_type,
            ok=bool(markdown.strip()) and (status is None or status < 400),
            status=status,
            credits=credits,
            proxy=proxy,
            cache_state=getattr(metadata, "cache_state", None) if metadata else None,
            error=error or (None if markdown.strip() else "scrape returned no markdown"),
        )

        if page.stealth:
            log.warning(
                "%s: %s was fetched through the %s proxy at %d credits",
                self.domain, url, proxy or "stealth", credits,
            )
        return Scraped(page=page, markdown=markdown, links=links)


def _map_links(data: Any) -> list[str]:
    """Pull URLs out of a map response, whichever shape it arrives in."""
    links = getattr(data, "links", None)
    if links is None and isinstance(data, Mapping):
        links = data.get("links")
    out: list[str] = []
    for link in links or []:
        url = getattr(link, "url", None)
        if url is None and isinstance(link, Mapping):
            url = link.get("url")
        if url is None and isinstance(link, str):
            url = link
        if url:
            out.append(str(url))
    return out


# ---------------------------------------------------------------------------
# Crawling one domain
# ---------------------------------------------------------------------------


def crawl_domain(
    domain: str,
    pack: SectorPack,
    store: Store,
    fetcher: Fetcher,
    *,
    crawl_date: date | None = None,
    discovery: DiscoveryResult | None = None,
    force: bool = False,
    rescue: bool = True,
    min_pages: int = MIN_PAGES,
    ladder: LadderStep1 | None = None,
) -> DomainCrawl:
    """Discover, scrape, store, and write the manifest for one domain.

    Idempotent by crawl date: a domain whose manifest already exists for this
    date is skipped, because re-running a batch after a partial failure must
    not re-bill the domains that already succeeded. `force` overrides it and
    overwrites, which is the only sanctioned way to rewrite a crawl.

    `discovery` is injectable so a dry run can compute the selection for free
    and the crawl can reuse it without a second round of HTTP.
    """
    if discovery is not None and discovery.domain != domain:
        # The manifest path comes from the discovery record and the skip check
        # from this argument. Letting them disagree writes one domain's pages
        # under another domain's key.
        raise FetchError(
            f"discovery result is for {discovery.domain!r}, not {domain!r}"
        )

    crawl_date = crawl_date or date.today()
    started = datetime.now(timezone.utc)
    manifest_path = raw_manifest_path(domain, crawl_date)

    if not force and store.exists(manifest_path):
        log.info("%s: already crawled on %s, skipping", domain, crawl_date.isoformat())
        return DomainCrawl(
            domain=domain,
            crawl_date=crawl_date,
            pack_version=pack.version,
            skipped=True,
            skip_reason=f"already crawled on {crawl_date.isoformat()}; pass force to re-crawl",
            manifest_path=manifest_path,
            started_at=started,
        )

    session = fetcher.session(domain)
    if discovery is None:
        discovery = discover(domain, pack, map_urls=session.map_urls)

    if discovery.exit_step == 0:
        return _step_0_record(
            domain=domain,
            crawl_date=crawl_date,
            pack=pack,
            store=store,
            discovery=discovery,
            started=started,
            force=force,
        )

    homepage_type = pack.always_fetch_type
    # A flagged domain gets the whole page rather than the main content: what
    # its discovery was missing is the navigation, and main-content extraction
    # is what removes navigation.
    full_page = discovery.discovery_failed

    pages: list[PageFetch] = []
    seen: set[str] = set()
    homepage_links: list[str] = []

    # ---- STEP 1 of the ladder, when it is wired.
    #
    # Injected rather than imported, the same shape as `map_urls`: an unwired
    # ladder leaves this function behaving exactly as it did, so the change can
    # be turned on per run and every existing test stays true.
    #
    # Only exits A and B stop the crawl. Exit C falls through to the full
    # selection below, which is today's behaviour and is what step 2 (the
    # four-location search) will replace.
    if ladder is not None:
        selection = select_step_1_pages(discovery.pages)
        for selected in selection:
            key = dedupe_key(selected.url)
            if key in seen:
                continue
            seen.add(key)
            is_homepage = selected.page_type == homepage_type
            scraped = session.scrape(
                selected.url, selected.page_type, want_links=is_homepage,
                main_content=not full_page if is_homepage else None,
            )
            if is_homepage:
                homepage_links = scraped.links
            pages.append(_store_page(store, domain, crawl_date, scraped, force=force))

        readable = [
            (p.url, p.page_type, store.read_text(p.stored_path))
            for p in pages if p.stored_path
        ]
        if readable:
            branch = ladder(pack, readable)
            if branch.exit in ("A", "B"):
                return _step_1_record(
                    domain=domain, crawl_date=crawl_date, pack=pack, store=store,
                    discovery=discovery, pages=pages, session=session,
                    branch=branch, started=started, force=force,
                )
            log.info("%s: step 1 exit C, continuing (%s)", domain, branch.reason)

    for selected in discovery.pages:
        key = dedupe_key(selected.url)
        if key in seen:
            continue
        seen.add(key)

        is_homepage = selected.page_type == homepage_type
        scraped = session.scrape(
            selected.url,
            selected.page_type,
            want_links=is_homepage,
            main_content=not full_page if is_homepage else None,
        )
        if is_homepage:
            homepage_links = scraped.links
        pages.append(_store_page(store, domain, crawl_date, scraped, force=force))

    rescue_pages: list[DiscoveredPage] = []
    if rescue and homepage_links and sum(page.ok for page in pages) < min_pages:
        refused, failed = _refusals(pages)
        # `failed` guards the ordinary case this rule must not touch: discovery
        # selected only the homepage, nothing failed at all, and a rendered nav
        # is exactly what the rescue is there to find.
        if failed and refused * 2 >= failed:
            # The rescue recovers a nav the free path could not see. It cannot
            # recover a site that answers 403 to everything, and the trigger
            # above cannot tell the two apart: both leave too few `ok` pages.
            # Fetching more URLs off the same host just buys the same refusal
            # again, at one credit each.
            log.warning(
                "%s: %d of %d failed pages were refused by the site, skipping the "
                "rescue rather than paying to hit the same wall",
                domain, refused, failed,
            )
        else:
            rescue_pages = _rescue_selection(domain, pack, discovery, homepage_links, seen)
            for selected in rescue_pages:
                scraped = session.scrape(selected.url, selected.page_type, origin="rescue")
                pages.append(_store_page(store, domain, crawl_date, scraped, force=force))
            log.info(
                "%s: discovery returned too little, rescued %d page(s) from the rendered homepage",
                domain, len(rescue_pages),
            )

    record = _summarize(
        domain=domain,
        crawl_date=crawl_date,
        pack=pack,
        discovery=discovery,
        pages=pages,
        session=session,
        rescue_pages=len(rescue_pages),
        started=started,
    )

    # A manifest asserts that this domain was crawled, and every later phase
    # reads it that way: Phase 5 extracts from it, Phase 7a mines it for thin
    # sites, and the idempotency check above skips anything that has one. A
    # crawl that stored nothing, or one the rate limiter could not keep inside
    # the window, is not that. Writing it anyway would record an account limit
    # as a fact about the company, and it would take a `--force` to undo.
    if record.crawl_failed:
        log.error("%s: not writing a manifest, %s", domain, record.crawl_error)
        return record

    record.manifest_path = write_manifest(
        store,
        discovery,
        crawl_date,
        extra={"fetch": record.model_dump(mode="json")},
        overwrite=force,
    )
    log.info("crawled %s", record.summary())
    return record


def _step_0_record(
    *,
    domain: str,
    crawl_date: date,
    pack: SectorPack,
    store: Store,
    discovery: DiscoveryResult,
    started: datetime,
    force: bool,
) -> DomainCrawl:
    """Write the manifest for a domain the ladder stopped at step 0.

    This deliberately does not go through `_summarize`. That function reads
    "nothing stored" as a crawl failure and the caller then writes no manifest,
    so a later run retries without `--force`. That rule is right for a crawl
    the rate limiter cut short and wrong here: the domain moved, the answer is
    known, and a retry re-resolves the same redirect and reaches the same
    conclusion. Writing the manifest is what stops the requeue list having to
    be rebuilt by re-crawling every moved domain in the batch.

    Zero credits, zero pages, and `referred_domain` carried through so the
    acquirer can be queued from the manifest alone.
    """
    record = DomainCrawl(
        domain=domain,
        crawl_date=crawl_date,
        pack_version=pack.version,
        pages=[],
        pages_requested=0,
        pages_stored=0,
        pages_failed=0,
        credits_billed=0,
        map_calls=0,
        resolved_domain=discovery.resolved_domain,
        redirected_off_domain=True,
        referred_domain=discovery.referred_domain,
        discovery_failed=False,
        crawl_failed=False,
        exit_step=0,
        stop_reason="site_moved",
        started_at=started,
        duration_s=round((datetime.now(timezone.utc) - started).total_seconds(), 2),
    )
    record.manifest_path = write_manifest(
        store,
        discovery,
        crawl_date,
        extra={"fetch": record.model_dump(mode="json")},
        overwrite=force,
    )
    log.info(
        "%s: stopped at ladder step 0, redirects to %s; 0 credits spent",
        domain, discovery.referred_domain,
    )
    return record


def _step_1_record(
    *,
    domain: str,
    crawl_date: date,
    pack: SectorPack,
    store: Store,
    discovery: DiscoveryResult,
    pages: Sequence[PageFetch],
    session: DomainSession,
    branch: Step1Branch,
    started: datetime,
    force: bool,
) -> DomainCrawl:
    """The crawl stopped at step 1. Write what it read and why it stopped.

    Exit A and exit B both land here and are recorded identically except for
    `awaiting_confirmation`. That flag is the whole of Appendix B.2 at this
    stage: exit B has taken its branch and spent its pages, and its verdict is
    withheld until the strong model confirms it in phase 3. A withheld verdict
    that is also invisible would be worse than emitting one, so the row is
    written and flagged rather than dropped.
    """
    stop_reason = {
        "A": "exit_A_first_party_statement",
        "B": "exit_B_negative_catalogue_unconfirmed",
    }[branch.exit]

    record = _summarize(
        domain=domain, crawl_date=crawl_date, pack=pack, discovery=discovery,
        pages=pages, session=session, rescue_pages=0, started=started,
    )
    record.exit_step = 1
    record.stop_reason = stop_reason
    record.models_used = {"1": branch.model_used}
    record.awaiting_confirmation = branch.exit == "B"

    if record.crawl_failed:
        log.error("%s: not writing a manifest, %s", domain, record.crawl_error)
        return record

    record.manifest_path = write_manifest(
        store, discovery, crawl_date,
        extra={"fetch": record.model_dump(mode="json"),
               "ladder": {
                   "step": 1,
                   "exit": branch.exit,
                   "reason": branch.reason,
                   "quote": branch.quote,
                   "quote_url": branch.quote_url,
                   "verdict": branch.verdict,
                   "emitted": branch.emitted,
                   "needs_strong_model_confirmation": branch.needs_strong_model_confirmation,
                   "matcher_found_statement": branch.matcher_found_statement,
                   "model_used": branch.model_used,
                   "pages_read": branch.pages_read,
               }},
        overwrite=force,
    )
    log.info(
        "%s: stopped at ladder step 1, exit %s after %d page(s): %s",
        domain, branch.exit, len(pages), branch.reason,
    )
    return record


def _refusals(pages: Sequence[PageFetch]) -> tuple[int, int]:
    """How many failed pages were refusals, and how many failed at all.

    Both numbers, because the ratio is what matters: one 403 on a login-gated
    careers page is not a wall, and nine 403s out of nine is.
    """
    failed = [page for page in pages if not page.ok]
    return sum(1 for page in failed if page.status in BLOCKED_STATUSES), len(failed)


def _store_page(
    store: Store, domain: str, crawl_date: date, scraped: Scraped, *, force: bool
) -> PageFetch:
    """Write one page's markdown, and record where it landed.

    An empty page is not stored. Phase 7a separates genuinely thin sites from
    discovery failures by reading the stored markdown, and a zero-byte file
    that exists is indistinguishable from a page with nothing on it.

    Neither is a *failed* page, for the same reason. A refusal still carries a
    body: `bionostix.com`'s 403 renders as the word "Forbidden" and is valid
    markdown, so the emptiness check above waves it through. Storing it puts
    text in `raw/` that reads as site content to anything walking the directory
    rather than the manifest -- Phase 7a's thin-site harvest, and every ad-hoc
    grep over `data/raw/*/*/*.md`, including the one behind the 2026-08-26
    design review's facility-language ceiling.
    """
    page = scraped.page
    text = scraped.markdown
    if not page.ok:
        return page
    if not text.strip():
        return page.model_copy(
            update={"ok": False, "error": page.error or "scrape returned no markdown"}
        )

    path = raw_page_path(domain, crawl_date, page.url)
    update: dict[str, Any] = {
        "word_count": len(text.split()),
        "bytes_stored": len(text.encode("utf-8")),
        "stored_path": path,
    }
    try:
        store.write_markdown(path, text, overwrite=force)
    except StoreConflictError:
        # The page file survives from an earlier partial run whose manifest was
        # never written. Keeping the stored bytes is the append-only answer;
        # the credit for the re-fetch is spent either way and is still logged.
        log.warning("%s: %s already exists, keeping the stored copy", domain, path)
        update["error"] = page.error or "a page already existed at this path; kept the stored copy"
    return page.model_copy(update=update)


def _rescue_selection(
    domain: str,
    pack: SectorPack,
    discovery: DiscoveryResult,
    links: Sequence[str],
    already: set[str],
) -> list[DiscoveredPage]:
    """Re-select from links harvested off the rendered homepage.

    Discovery's free path reads the homepage with plain HTTP, so a
    client-rendered nav or a bot wall leaves it with nothing to score. The
    Firecrawl fetch of that same homepage renders it, and its link list is the
    nav the free path could not see. Scoring runs through `select_pages`
    unchanged, so the caps, the type priorities and the editorial rules all
    still apply; only the source of the candidates differs.

    Anchor text is unavailable here — the links format returns URLs alone — so
    these pages match on path only.
    """
    # The same domain the free path settled on, which is not always the input:
    # a site that redirects to its acquirer serves a rendered homepage whose
    # every link is on the new registrable domain, and filtering those against
    # the old one throws the whole rescue away.
    site_domain = discovery.resolved_domain or domain

    candidates: dict[str, Candidate] = {}
    for link in links:
        if not is_crawlable(link, site_domain):
            continue
        key = dedupe_key(link)
        if key in already or key in candidates:
            continue
        candidates[key] = Candidate(url=canonical_url(link), sources={"rendered-nav"})

    if not candidates:
        return []

    selected = select_pages(discovery.homepage_url, candidates, pack)
    taken = set(already)
    taken_pages = {locale_free_key(url) for url in already}
    out: list[DiscoveredPage] = []
    for page in selected:
        key, page_key = dedupe_key(page.url), locale_free_key(page.url)
        if key in taken or page_key in taken_pages:
            continue
        taken.add(key)
        taken_pages.add(page_key)
        out.append(page)
    return out


def _summarize(
    *,
    domain: str,
    crawl_date: date,
    pack: SectorPack,
    discovery: DiscoveryResult,
    pages: Sequence[PageFetch],
    session: DomainSession,
    rescue_pages: int,
    started: datetime,
) -> DomainCrawl:
    stored = [page for page in pages if page.stored_path]
    throttled = [page for page in pages if page.rate_limited]

    crawl_error: str | None = None
    if not stored:
        crawl_error = f"nothing stored from {len(pages)} attempted page(s)"
    elif throttled:
        crawl_error = (
            f"{len(throttled)} of {len(pages)} pages were refused for rate, "
            "so this crawl is incomplete rather than thin"
        )

    return DomainCrawl(
        domain=domain,
        crawl_date=crawl_date,
        pack_version=pack.version,
        pages=list(pages),
        pages_requested=len(pages),
        pages_stored=len(stored),
        pages_failed=sum(1 for page in pages if not page.ok),
        word_count=sum(page.word_count for page in stored),
        bytes_stored=sum(page.bytes_stored for page in stored),
        credits_billed=session.credits,
        map_calls=session.map_calls,
        stealth_pages=sum(1 for page in pages if page.stealth),
        rescued=rescue_pages > 0,
        rescue_pages=rescue_pages,
        resolved_domain=discovery.resolved_domain,
        redirected_off_domain=discovery.redirected_off_domain,
        discovery_failed=discovery.discovery_failed,
        failure_reason=discovery.failure_reason,
        crawl_failed=crawl_error is not None,
        crawl_error=crawl_error,
        rate_limited_pages=len(throttled),
        started_at=started,
        duration_s=round((datetime.now(timezone.utc) - started).total_seconds(), 2),
    )


# ---------------------------------------------------------------------------
# Crawling a batch
# ---------------------------------------------------------------------------


def crawl_all(
    domains: Sequence[str],
    pack: SectorPack,
    store: Store,
    fetcher: Fetcher,
    *,
    crawl_date: date | None = None,
    force: bool = False,
    concurrency: int = DEFAULT_CONCURRENCY,
    budget: int | None = None,
    discoveries: Mapping[str, DiscoveryResult] | None = None,
    on_result: Callable[[DomainCrawl], None] | None = None,
    **kwargs: Any,
) -> list[DomainCrawl]:
    """Crawl a batch, in input order, stopping if the budget or the key runs out.

    `budget` is a stop line, not a ceiling: the domains already in flight when
    it is crossed finish and can overshoot by their own cost. It exists so a
    run against a 1,025-credit balance cannot silently spend 6,000.

    `discoveries` lets a caller supply selections computed earlier — a dry run
    prices exactly the pages a crawl then fetches, with no second round of
    HTTP. Domains absent from it are discovered as they are reached.

    A 401 or a 402 stops the batch outright. Every later domain would fail the
    same way, and the failures are not free of consequence: each one writes a
    manifest that then needs `--force` to redo.
    """
    if not domains:
        return []

    crawl_date = crawl_date or date.today()
    results: list[DomainCrawl | None] = [None] * len(domains)
    spent = 0
    lock = threading.Lock()
    stop = threading.Event()
    stop_reason: list[str] = []

    def halt(reason: str) -> None:
        if not stop.is_set():
            stop.set()
            stop_reason.append(reason)
            log.error("stopping the batch: %s", reason)

    def run(item: tuple[int, str]) -> None:
        nonlocal spent
        index, domain = item
        if stop.is_set():
            results[index] = _not_run(domain, crawl_date, pack, stop_reason[0] if stop_reason else "batch stopped")
            return

        with lock:
            if budget is not None and spent >= budget:
                halt(f"credit budget of {budget} reached after {spent} credits")
                results[index] = _not_run(domain, crawl_date, pack, stop_reason[0])
                return

        try:
            record = crawl_domain(
                domain, pack, store, fetcher,
                crawl_date=crawl_date,
                force=force,
                discovery=discoveries.get(domain) if discoveries else None,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - one domain must not lose the batch
            if is_fatal(exc):
                halt(f"{type(exc).__name__}: {exc}")
            else:
                log.exception("%s: crawl failed", domain)
            results[index] = _not_run(domain, crawl_date, pack, f"{type(exc).__name__}: {exc}")
            return

        with lock:
            spent += record.credits_billed
            results[index] = record
            if on_result is not None:
                on_result(record)

    with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as pool:
        list(pool.map(run, enumerate(domains)))

    return [
        record or _not_run(domain, crawl_date, pack, "not run")
        for domain, record in zip(domains, results)
    ]


def _not_run(domain: str, crawl_date: date, pack: SectorPack, reason: str) -> DomainCrawl:
    return DomainCrawl(
        domain=domain,
        crawl_date=crawl_date,
        pack_version=pack.version,
        skipped=True,
        skip_reason=reason,
    )


def credit_estimate(results: Sequence[DiscoveryResult]) -> dict[str, int]:
    """What a crawl of these discovery results would cost, before spending it.

    Map calls are estimated from the flagged domains: the endpoint fires only
    when the free sources come back below the minimum, which is the same
    condition that sets `discovery_failed`.
    """
    pages = sum(len(result.pages) for result in results)
    maps = sum(1 for result in results if result.discovery_failed)
    return {
        "domains": len(results),
        "pages": pages,
        "map_calls": maps,
        "credits": pages * PAGE_CREDITS + maps * MAP_CREDITS,
    }
