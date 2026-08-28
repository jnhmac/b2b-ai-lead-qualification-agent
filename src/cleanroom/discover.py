"""Page discovery: from a bare domain to a ranked, capped list of URLs to crawl.

Discovery is the cheap stage that decides how the expensive stage spends its
money. Every URL selected here becomes a Firecrawl credit in Phase 4 and an LLM
call in Phase 5, so the job is not "find pages" but "find the ten pages most
likely to say something about a facility."

Three design points worth knowing before editing.

**Anchor text is a first-class signal, not a fallback.** `/company/heritage`
matches no path pattern in the pack, but a nav link reading "Who We Are" catches
it. So the homepage nav is harvested even when a sitemap exists: the sitemap
supplies breadth and the nav supplies the labels, and a candidate found by both
scores higher than one found by either. A literal reading of the spec makes nav
harvesting a fallback for a missing sitemap; that reading throws away the anchor
text on every site that has a sitemap, which is most of them.

**Per-type caps are what stop a deep catalogue eating every slot.** A medical
device catalogue can publish four hundred URLs under `/products/`. Without the
cap they win on volume and the crawl comes back with ten SKU pages and nothing
about the facility. The pack's `max_pages` per type sums to 18 against a total
cap of 10 on purpose, so the scorer can fill slots from whichever types a given
site actually has.

**A thin result and a broken discovery are different failures.** Fewer than
three pages sets `discovery_failed`, which is a bug report against this module.
It is not a thin site, and it must never reach the classifier as a
`unknown` verdict: Phase 7a harvests the real thin-site population from crawl
manifests, and a discovery bug mixed into it corrupts that population.

Nothing here costs a Firecrawl credit. Sitemaps, robots.txt and the homepage are
plain HTTP. The one paid path is the injected `map_urls` callable, wired to
Firecrawl's map endpoint in Phase 4 and called only when the free sources come
back too thin to work with, which is the JS-rendered-nav case.
"""

from __future__ import annotations

import gzip
import logging
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from html.parser import HTMLParser
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .config import SectorPack
from .normalize import InvalidDomainError, normalize_domain
from .store import Store, raw_manifest_path

log = logging.getLogger(__name__)

#: Browser-shaped, and still self-identifying. Measured over the goldset sample:
#: a bare `cleanroom-classifier/0.1` token is refused by two of twenty homepages
#: (403 and 406 from the WAF, before any content is served), and both serve the
#: page normally once the string carries the usual browser prefix. Discovery
#: reads public marketing pages a handful at a time, so the block is a UA regex
#: rather than a rate limit, and the tool name stays in the string so a site
#: owner reading their logs can see who called.
#:
#: The sites that refuse this too are refusing on a full browser fingerprint,
#: not a header. That is the `map_urls` fallback's job, not this module's.
DISCOVERY_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 cleanroom-classifier/0.1"
)
DEFAULT_TIMEOUT = 15.0

#: Below this many selected pages the domain is a discovery failure, not a thin
#: site. The two look identical in a page count and have opposite fixes.
MIN_PAGES = 3

MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_SITEMAP_URLS = 5_000
MAX_NESTED_SITEMAPS = 12
#: Nav labels are short. A paragraph link slugs into a string containing half
#: the pattern list, so long anchors are kept for the record but not matched on.
MAX_ANCHOR_MATCH_CHARS = 80
MAX_ANCHORS_KEPT = 3

# -- scoring ----------------------------------------------------------------
# Direction and magnitude both matter here, so the constants are named rather
# than inlined: tuning discovery means editing these five numbers.

PATH_MATCH_SCORE = 3.0
ANCHOR_MATCH_SCORE = 2.0
PRIORITY_STEP = 0.25
DEPTH_PENALTY = 0.5
QUERY_PENALTY = 0.5
#: The target market is the US. Sized larger than any structural term (depth,
#: priority, query) so an English page wins whenever one exists, but smaller
#: than a path match so the only facilities page on a site can still get in
#: when it happens to be German. Preference, not a filter.
NON_ENGLISH_PENALTY = 2.0


class DiscoveryError(RuntimeError):
    """Discovery could not run at all, as opposed to running and finding little."""


# ---------------------------------------------------------------------------
# URL hygiene
# ---------------------------------------------------------------------------
#
# This is not the exclusion filtering the spec forbids. That rule is about
# *companies*: enterprise multi-division and non-English sites are crawled and
# classified like everything else. These lists are about *URLs* — a PDF, a
# shopping cart and a cookie policy are not company content in any language, and
# spending a credit on one is a credit not spent on the capabilities page.

_NON_PAGE_SUFFIXES = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".csv",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".bmp", ".tif", ".tiff",
    ".mp3", ".mp4", ".mov", ".avi", ".wmv", ".webm",
    ".zip", ".gz", ".tar", ".rar", ".7z", ".dmg", ".exe",
    ".css", ".js", ".json", ".xml", ".rss", ".atom",
    ".woff", ".woff2", ".ttf", ".eot",
    # Geo and calendar payloads. `locations.kml` was being scraped as a
    # facilities page: a credit and an LLM call spent on XML coordinates, and a
    # facilities page reported where the site has none.
    ".kml", ".kmz", ".gpx", ".ics", ".vcf",
)

#: Path segments that are never company content. Matched segment-exact.
_EXCLUDED_SEGMENTS = frozenset(
    {
        "wp-content", "wp-admin", "wp-json", "wp-includes", "cdn-cgi",
        "feed", "rss", "atom", "sitemap", "sitemap_index", "xmlrpc.php",
        "cart", "checkout", "basket", "my-account", "account", "login", "logout",
        "signin", "sign-in", "signup", "sign-up", "register", "password",
        "search", "tag", "tags", "author", "404", "error",
        "disclaimer", "accessibility", "unsubscribe", "thank-you", "thanks",
    }
)

#: The boilerplate footer family, which every site spells differently:
#: `cookie-usage`, `privacy-policy`, `terms-of-service`, `impressum`. An exact
#: list never keeps up, and each one that slips through costs a crawl slot.
_BOILERPLATE_SEGMENT_RE = re.compile(
    r"^(cookies?|privacy|terms|gdpr|legal|imprint|impressum|disclaimer)([-_]|$)",
    re.IGNORECASE,
)

#: Handled in two halves, because the index and the posts under it are not the
#: same kind of thing.
#:
#: The index (`/blog`, `/news`) is dropped by `is_section_index`: titles and
#: dates, nothing to quote.
#:
#: The posts are barred from the *typed* pass by `is_low_value` but stay
#: eligible for the unmatched fill. A post titled "Medical Balloon Product
#: Development and Technology Transfer" slugs into `product-development` and
#: would take a products slot, displacing the actual catalogue and then
#: reporting itself as `products` in `page_types_found`. It is still content
#: though, and "we opened a Class 7 cleanroom" is exactly the evidence this
#: pipeline is looking for, so on a site with nothing better it gets crawled
#: as `other`.
_LOW_VALUE_SEGMENTS = frozenset(
    {
        "blog", "news", "press", "press-release", "press-releases", "newsroom",
        "media", "events", "event", "webinar", "webinars", "podcast",
        "resources", "downloads", "library", "gallery", "archive", "archives",
    }
)

_INDEX_FILE_RE = re.compile(r"/(index|default|home)\.(html?|php|aspx?|jsp)$", re.IGNORECASE)
#: `en`, `en-us`, `de_DE`. Stripped before counting depth so a localised site is
#: not systematically pushed down the ranking; non-English sites stay in the
#: population and must compete on equal terms.
_LOCALE_SEGMENT_RE = re.compile(r"^[a-z]{2}([-_][a-z]{2})?$", re.IGNORECASE)
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_UNSUPPORTED_SCHEME_RE = re.compile(r"^(mailto|tel|javascript|data|ftp|sms|file):", re.IGNORECASE)


def _slug(text: str) -> str:
    """Lowercase, hyphen-joined form used for every pattern comparison.

    "Who We Are" and `/who-we-are/` and `/Who_We_Are` all reduce to the same
    string, which is what lets one pattern list match paths and anchor text.
    """
    return _SLUG_RE.sub("-", text.lower()).strip("-")


def _contains_slug(haystack: str, needle: str) -> bool:
    """Token-boundary containment, so `about` does not match `aboutique`."""
    return f"-{needle}-" in f"-{haystack}-"


def _path_segments(path: str) -> list[str]:
    return [segment for segment in path.split("/") if segment]


def canonical_url(url: str) -> str:
    """Absolute URL reduced to a comparable form.

    Fragments dropped, trailing slash and `index.html` normalised away, host
    lowercased. The query is kept: `?page_id=12` is a different page, while
    `/about` and `/about/` are not.
    """
    parts = urlsplit(url.strip())
    netloc = parts.netloc.lower()
    if netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif netloc.endswith(":443"):
        netloc = netloc[:-4]

    path = _INDEX_FILE_RE.sub("/", parts.path or "/")
    path = path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower() or "https", netloc, path, parts.query, ""))


def dedupe_key(url: str) -> str:
    """Canonical form with scheme and `www` folded, for identity comparisons.

    `http://example.com/about` and `https://www.example.com/about/` are one
    page. They are crawled once and one of them is arbitrary, so the key
    ignores both differences while `canonical_url` preserves them.

    The query is dropped at the site root, where it is a language switcher or a
    campaign parameter rather than a page: `/?lang=en-US` took a crawl slot to
    re-fetch the homepage until this was added. Everywhere else the query is
    kept, because `?page_id=12` really is a different page.
    """
    parts = urlsplit(canonical_url(url))
    host = parts.netloc[4:] if parts.netloc.startswith("www.") else parts.netloc
    query = "" if parts.path in ("", "/") else parts.query.lower()
    return urlunsplit(("https", host, parts.path.lower(), query, ""))


def is_section_index(url: str) -> bool:
    """True for `/blog`, `/news`, `/resources/downloads`: a listing, not a page.

    A section index is site navigation plus post titles and dates. There is no
    prose on it to extract a quote from, so it is the one part of a blog worth
    dropping outright rather than demoting. Measured on the goldset sample it
    was 4 of 182 selected pages, all of them indexes and no actual posts.
    """
    segments = _path_segments(urlsplit(url).path.lower())
    return bool(segments) and segments[-1] in _LOW_VALUE_SEGMENTS


#: A company page is named, not sentenced: `/capabilities`,
#: `/contract-manufacturing`, `/about-us`. An article is a headline in a URL.
#: Measured over the goldset sample, the split is clean at five words: below it
#: sits `/medical-balloon-development-and-manufacturing` and
#: `/current-good-manufacturing-practice-cgmp`; above it sits
#: `/how-when-and-why-to-leverage-the-fdas-safer-technologies-program-step`.
MAX_SLUG_WORDS = 5

_YEAR_SEGMENT_RE = re.compile(r"^(19|20)\d{2}$")


def is_article(url: str) -> bool:
    """True for editorial content published outside any `/blog/` section.

    Two signals, both structural. A long final slug, because a headline in a URL
    is a headline; and a four-digit year anywhere in the path, which no company
    page carries and every dated post does.

    This exists because the section check cannot see articles published at the
    site root. It also removes a class of *wrong* type assignments rather than
    merely low-value ones: `/plant-cell-model-magnified-500k-to-100m-times`
    matched `plant` and was filed as a **facilities** page, which tells the
    classifier a facility page exists when the site has none.
    """
    segments = _path_segments(urlsplit(url).path.lower())
    if not segments:
        return False
    if any(_YEAR_SEGMENT_RE.match(segment) for segment in segments):
        return True
    stem = segments[-1].rsplit(".", 1)[0]
    return len([word for word in stem.replace("_", "-").split("-") if word]) > MAX_SLUG_WORDS


def is_low_value(url: str) -> bool:
    """True for a page *inside* a blog, news or press section.

    Crawlable, but never a typed page: a post titled "Medical Balloon Product
    Development" slugs into `product-development` and would take a products
    slot. It stays eligible for the unmatched fill, because "we opened a Class 7
    cleanroom" is real evidence and a press release is where a company says it.
    """
    return bool(_LOW_VALUE_SEGMENTS.intersection(_path_segments(urlsplit(url).path.lower())))


#: Locale segments that mean "this is the page a US reader gets".
_ENGLISH_LOCALES = frozenset({"en", "us", "en-us", "en_us", "en-gb", "en_gb"})


def locale_rank(url: str) -> int:
    """Preference among locale twins: unprefixed, then English, then the rest.

    The market is the US, so when a manufacturer publishes one page a dozen
    times the English copy is the one to spend the credit on. This only ever
    picks *between duplicates* — it is a tiebreaker inside `select_pages`, not
    a filter. A site published solely in German still gets crawled and
    classified, then flagged for manual review, because dropping it would
    delete a real supplier from the population.
    """
    segments = _path_segments(urlsplit(url).path.lower())
    prefix = []
    while segments and len(prefix) < 2 and _LOCALE_SEGMENT_RE.match(segments[0]):
        prefix.append(segments.pop(0))

    if not prefix:
        return 0
    return 1 if _ENGLISH_LOCALES.intersection(prefix) else 2


def locale_free_key(url: str) -> str:
    """`dedupe_key` with leading locale segments removed.

    `/ch/en/about.html` and `/it/en/about.html` are one page published twice.
    Enterprise sites publish a dozen such twins and they score identically, so
    without this they take every slot of their type and the crawl comes back
    with the same page in four languages. Two segments are stripped at most,
    which is what a `country/language` prefix needs.
    """
    parts = urlsplit(dedupe_key(url))
    segments = _path_segments(parts.path)
    stripped = 0
    while segments and stripped < 2 and _LOCALE_SEGMENT_RE.match(segments[0]):
        segments.pop(0)
        stripped += 1
    return urlunsplit(("https", parts.netloc, "/" + "/".join(segments), parts.query, ""))


def is_crawlable(url: str, domain: str, *, allow_pdf: bool = False) -> bool:
    """Internal, http(s), and plausibly a page rather than an asset.

    `allow_pdf` is a narrow, opt-in exception to the `.pdf` skip: a public
    filer's 10-K can carry the one decisive facility sentence a marketing
    site never states (HANDOFF.md item 7), but every other PDF on these sites
    really is noise, so callers only pass this when the URL has already
    matched the pack's `investors` page type — see `discover()`.
    """
    if _UNSUPPORTED_SCHEME_RE.match(url.strip()):
        return False

    parts = urlsplit(url)
    if parts.scheme and parts.scheme.lower() not in ("http", "https"):
        return False
    if not parts.netloc:
        return False

    try:
        if normalize_domain(url).domain != domain:
            return False
    except InvalidDomainError:
        return False

    path = parts.path.lower()
    if path.endswith(_NON_PAGE_SUFFIXES) and not (allow_pdf and path.endswith(".pdf")):
        return False

    segments = _path_segments(path)
    if _EXCLUDED_SEGMENTS.intersection(segments):
        return False
    if any(_BOILERPLATE_SEGMENT_RE.match(segment) for segment in segments):
        return False
    return not is_section_index(url)


# ---------------------------------------------------------------------------
# Records: the discovery half of the crawl manifest
# ---------------------------------------------------------------------------


class FetchAttempt(BaseModel):
    """One HTTP request discovery made, and what came back.

    Recorded for every attempt including the failures. A 404 on
    `/sitemap.xml` is normal; a 404 on the homepage is the whole story, and
    without the record the two are indistinguishable after the fact.
    """

    model_config = ConfigDict(extra="forbid")

    url: str
    kind: str
    status: int | None = None
    ok: bool = False
    urls_found: int = 0
    bytes_read: int = 0
    error: str | None = None
    #: True when the request only succeeded with certificate verification off.
    #: Recorded rather than silent, because "we chose not to check the
    #: certificate" is a fact about our client, not about the site.
    tls_unverified: bool = False


class DiscoveredPage(BaseModel):
    """One selected URL, with the reasoning that selected it.

    `matched_pattern` and `matched_on` exist so a bad page mix can be diagnosed
    from the manifest instead of by re-running discovery with a debugger.
    """

    model_config = ConfigDict(extra="forbid")

    url: str
    page_type: str
    score: float
    depth: int = 0
    matched_pattern: str | None = None
    matched_on: list[str] = Field(default_factory=list)
    anchor_texts: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)

    #: Which step of the crawl ladder asked for this URL (`crawl_policy.md`
    #: Appendix A: `ladder_step`). Null means the selection predates the ladder
    #: or was made by a path the ladder does not drive, which must stay
    #: distinguishable from a page step 1 deliberately requested.
    ladder_step: int | None = None

    #: The model that read this page, recorded per page because the ladder step
    #: decides the model (Appendix B.1): step 1 runs small, steps 2 and 3 run
    #: strong, so one domain routinely carries two values. A single field on
    #: the domain record could not express the split the routing rule exists to
    #: create, and B.5 needs `model_used` per step to attribute a change to the
    #: model rather than to the pack or the ladder.
    model_used: str | None = None
    #: Set when Appendix B.3 escalation fired: the model it escalated from, and
    #: which of the four triggers fired. Null on the ordinary path.
    escalated_from: str | None = None
    escalation_trigger: str | None = None


class DiscoveryResult(BaseModel):
    """What discovery found for one domain. Serialises straight into the manifest."""

    model_config = ConfigDict(extra="forbid")

    domain: str
    pack_version: str
    homepage_url: str
    total_page_cap: int
    min_pages: int = MIN_PAGES

    #: The registrable domain the homepage actually resolved to. Equal to
    #: `domain` on all but the acquired-and-rebranded cases, where every page
    #: crawled is on the acquirer's site and evidence found there is evidence
    #: about the acquirer. Kept separate from `domain` so the store stays keyed
    #: by the domain the input list carries.
    resolved_domain: str | None = None
    redirected_off_domain: bool = False
    #: The domain this row should be re-targeted at, per tab 2a R12. Set only
    #: on a step-0 exit, where the answer a human needs is "go look at who owns
    #: this now" rather than a verdict about the wrong company.
    referred_domain: str | None = None
    #: Hosts that only answered with certificate verification turned off.
    tls_unverified_hosts: list[str] = Field(default_factory=list)

    #: Which step of the crawl ladder resolved this domain, and which stopping
    #: rule fired. Both are null until the ladder sets them, so a result
    #: produced by a path that predates the ladder is distinguishable from one
    #: the ladder deliberately stopped. `crawl_policy.md` Appendix A rolls these
    #: up per domain as `exit_step` / `stop_reason`.
    exit_step: int | None = None
    stop_reason: str | None = None

    pages: list[DiscoveredPage] = Field(default_factory=list)

    sources_used: list[str] = Field(default_factory=list)
    candidates_considered: int = 0
    attempts: list[FetchAttempt] = Field(default_factory=list)
    not_found: list[str] = Field(default_factory=list)

    page_types_found: list[str] = Field(default_factory=list)
    page_types_empty: list[str] = Field(default_factory=list)

    discovery_failed: bool = False
    failure_reason: str | None = None

    @property
    def urls(self) -> list[str]:
        return [page.url for page in self.pages]

    @property
    def page_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for page in self.pages:
            counts[page.page_type] += 1
        return dict(counts)

    def summary(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "pages": len(self.pages),
            "types": self.page_type_counts,
            "sources": self.sources_used,
            "failed": self.discovery_failed,
        }


# ---------------------------------------------------------------------------
# Pattern matching against the pack
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PagePattern:
    """One pack pattern, pre-slugged, tagged with the type that owns it."""

    page_type: str
    priority: int
    pattern: str
    slug: str


@dataclass(frozen=True)
class PatternMatch:
    page_type: str
    priority: int
    pattern: str
    matched_on: tuple[str, ...]


def build_patterns(pack: SectorPack) -> list[PagePattern]:
    """Every pattern in the pack, most specific first.

    Sorting by slug length is how `manufacturing-services` (a `capabilities`
    page) wins over `manufacturing` on a URL that contains both. Ties fall back
    to page-type priority so ordering is total and discovery is deterministic.
    """
    patterns = [
        PagePattern(page_type=name, priority=spec.priority, pattern=pattern, slug=_slug(pattern))
        for name, spec in pack.page_types.items()
        for pattern in spec.patterns
        if _slug(pattern)
    ]
    patterns.sort(key=lambda p: (-len(p.slug), p.priority, p.slug))
    return patterns


def match_url(url: str, anchors: Sequence[str], patterns: Sequence[PagePattern]) -> PatternMatch | None:
    """Assign a page type by matching patterns against path *and* anchor text.

    The winning type comes from the most specific matching pattern. `matched_on`
    then unions across every pattern of that same type, so a page caught by its
    path under one pattern and by its nav label under another is recorded as
    both and scores accordingly.
    """
    path_slug = _slug(urlsplit(url).path)
    anchor_slugs = [
        _slug(anchor) for anchor in anchors if len(anchor) <= MAX_ANCHOR_MATCH_CHARS
    ]

    hits: list[tuple[PagePattern, set[str]]] = []
    for pattern in patterns:
        matched_on: set[str] = set()
        if _contains_slug(path_slug, pattern.slug):
            matched_on.add("path")
        if any(_contains_slug(anchor, pattern.slug) for anchor in anchor_slugs):
            matched_on.add("anchor")
        if matched_on:
            hits.append((pattern, matched_on))

    if not hits:
        return None

    winner = hits[0][0]
    combined: set[str] = set()
    for pattern, matched_on in hits:
        if pattern.page_type == winner.page_type:
            combined |= matched_on

    return PatternMatch(
        page_type=winner.page_type,
        priority=winner.priority,
        pattern=winner.pattern,
        matched_on=tuple(sorted(combined)),
    )


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """One internal URL under consideration, merged across every source that saw it."""

    url: str
    anchors: list[str] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)

    def add_anchor(self, text: str) -> None:
        cleaned = " ".join(text.split())
        if cleaned and cleaned not in self.anchors and len(self.anchors) < MAX_ANCHORS_KEPT:
            self.anchors.append(cleaned)


def url_depth(url: str) -> int:
    """Path segments, ignoring a leading locale segment."""
    segments = _path_segments(urlsplit(url).path)
    if segments and _LOCALE_SEGMENT_RE.match(segments[0]):
        segments = segments[1:]
    return len(segments)


def score_candidate(candidate: Candidate, match: PatternMatch, max_priority: int) -> float:
    """Rank a matched candidate.

    Four terms, in descending order of influence:

    * how it matched — path and anchor together beat either alone
    * a demotion for non-English copies, since the target market is the US
    * page-type priority, so `capabilities` outranks `about` at equal evidence
    * depth, which is the second defence against catalogue pages after the caps

    Blog and press URLs are not scored down here; `is_low_value` keeps them out
    of the typed pass entirely, which is a cleaner cut than a penalty large
    enough to matter but small enough to be overtaken on a quiet site.
    """
    score = 0.0
    if "path" in match.matched_on:
        score += PATH_MATCH_SCORE
    if "anchor" in match.matched_on:
        score += ANCHOR_MATCH_SCORE

    score += PRIORITY_STEP * (max_priority - match.priority)
    score -= DEPTH_PENALTY * max(url_depth(candidate.url) - 1, 0)

    if urlsplit(candidate.url).query:
        score -= QUERY_PENALTY
    if locale_rank(candidate.url) == 2:
        score -= NON_ENGLISH_PENALTY

    return round(score, 4)


def _best_of_patterns(
    candidates: Mapping[str, Candidate], patterns: Sequence[PagePattern]
) -> str | None:
    """URL of the single highest-scoring candidate matching `patterns`.

    Scoped to one page type, unlike `select_pages`'s full pass over every
    type at once. Used for a bounded, one-shot second-hop fetch (see
    `discover()`'s careers-detail pass): fetch only the best candidate of a
    type rather than every one, so the extra request stays free and singular
    instead of an open-ended crawl.
    """
    max_priority = max((p.priority for p in patterns), default=0)
    best_url: str | None = None
    best_score = float("-inf")
    for candidate in candidates.values():
        match = match_url(candidate.url, candidate.anchors, patterns)
        if match is None:
            continue
        score = score_candidate(candidate, match, max_priority)
        if score > best_score:
            best_score = score
            best_url = candidate.url
    return best_url


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_pages(
    homepage_url: str,
    candidates: Mapping[str, Candidate],
    pack: SectorPack,
    *,
    fill_unmatched: bool = True,
) -> list[DiscoveredPage]:
    """Homepage always, then top scorers respecting per-type caps and the total cap.

    The unmatched fill is the last pass. A URL the pack has no pattern for is
    worth less than one it does, but it is worth more than an empty slot: sites
    with idiosyncratic navigation ("/what-we-make") are exactly the ones at risk
    of a false `unknown`, and their real content lives in unmatched URLs.
    Shallow-first ordering keeps the fill away from catalogue depths.
    """
    patterns = build_patterns(pack)
    max_priority = max(spec.priority for spec in pack.page_types.values())
    homepage_type = pack.always_fetch_type

    homepage = canonical_url(homepage_url)
    selected: list[DiscoveredPage] = [
        DiscoveredPage(
            url=homepage,
            page_type=homepage_type,
            score=0.0,
            depth=0,
            sources=["homepage"],
        )
    ]
    taken = {dedupe_key(homepage)}
    taken_pages = {locale_free_key(homepage)}
    per_type: dict[str, int] = defaultdict(int)
    per_type[homepage_type] = 1

    matched: list[tuple[DiscoveredPage, PatternMatch]] = []
    unmatched: list[Candidate] = []

    for key, candidate in candidates.items():
        if key in taken:
            continue
        # Editorial content never takes a typed slot, whether it announces
        # itself with a `/blog/` prefix or hides at the site root behind a
        # headline-shaped slug. It stays in `unmatched` and remains crawlable.
        editorial = is_low_value(candidate.url) or is_article(candidate.url)
        match = None if editorial else match_url(candidate.url, candidate.anchors, patterns)
        if match is None:
            unmatched.append(candidate)
            continue
        matched.append(
            (
                DiscoveredPage(
                    url=candidate.url,
                    page_type=match.page_type,
                    score=score_candidate(candidate, match, max_priority),
                    depth=url_depth(candidate.url),
                    matched_pattern=match.pattern,
                    matched_on=list(match.matched_on),
                    anchor_texts=list(candidate.anchors),
                    sources=sorted(candidate.sources),
                ),
                match,
            )
        )

    # Deterministic: score, then the pack's own type ordering, then shallowest,
    # then the English copy, then the URL itself. Two runs over the same site
    # must select the same ten pages or run-to-run diffs stop meaning anything.
    matched.sort(
        key=lambda item: (
            -item[0].score,
            item[1].priority,
            item[0].depth,
            locale_rank(item[0].url),
            item[0].url,
        )
    )

    for page, _ in matched:
        if len(selected) >= pack.total_page_cap:
            break
        cap = pack.page_types[page.page_type].max_pages
        if per_type[page.page_type] >= cap:
            continue
        key, page_key = dedupe_key(page.url), locale_free_key(page.url)
        if key in taken or page_key in taken_pages:
            continue
        selected.append(page)
        taken.add(key)
        taken_pages.add(page_key)
        per_type[page.page_type] += 1

    if fill_unmatched and len(selected) < pack.total_page_cap:
        unmatched.sort(
            key=lambda c: (url_depth(c.url), 0 if c.anchors else 1, locale_rank(c.url), c.url)
        )
        for candidate in unmatched:
            if len(selected) >= pack.total_page_cap:
                break
            key, page_key = dedupe_key(candidate.url), locale_free_key(candidate.url)
            if key in taken or page_key in taken_pages:
                continue
            selected.append(
                DiscoveredPage(
                    url=candidate.url,
                    page_type="other",
                    score=0.0,
                    depth=url_depth(candidate.url),
                    anchor_texts=list(candidate.anchors),
                    sources=sorted(candidate.sources),
                )
            )
            taken.add(key)
            taken_pages.add(page_key)

    return selected


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fetched:
    final_url: str
    content: bytes
    status: int


#: A certificate that fails verification on a site that is otherwise serving
#: normally. The usual cause is a server that omits the intermediate
#: certificate: a browser repairs the chain by fetching the issuer named in the
#: certificate's AIA extension, and httpx does not, so an ordinary marketing
#: site becomes unreachable over https and the domain is written off as a bot
#: wall. Measured on lensara.com, which serves 200 with 58 links once the check
#: is skipped.
_TLS_VERIFY_RE = re.compile(
    r"CERTIFICATE_VERIFY_FAILED|SSLCertVerificationError|certificate verify failed",
    re.IGNORECASE,
)

#: Set on a response that only arrived because verification was skipped.
TLS_UNVERIFIED_EXTENSION = "cleanroom_tls_unverified"


def _is_tls_verify_error(exc: BaseException) -> bool:
    return bool(_TLS_VERIFY_RE.search(f"{type(exc).__name__}: {exc}"))


class TlsFallbackTransport(httpx.BaseTransport):
    """Retry a certificate verification failure once with the check off.

    This lives in the transport rather than in `_fetch` for two reasons, and
    the second one was found the hard way. First, httpx resolves redirects at
    the client level by calling the transport per hop, so a retry written above
    the client only covers the first hop. Second, a retry that builds its own
    client escapes any transport the caller injected — which in the test suite
    means an offline test quietly reaching the real network, and in
    `discover_all` means a second connection pool per bad certificate.

    The narrowing of trust is deliberate and worth being explicit about.
    Discovery reads public marketing pages, sends no credentials, and uses the
    text only to *choose* URLs. Firecrawl re-fetches every page that is
    actually crawled under its own policy, so nothing read unverified here ever
    reaches the raw store. Only a verification failure is retried: a timeout or
    a refused connection is passed straight through, or a dead host would look
    like a certificate problem.
    """

    def __init__(self, verified: httpx.BaseTransport, unverified: httpx.BaseTransport) -> None:
        self._verified = verified
        self._unverified = unverified

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        try:
            return self._verified.handle_request(request)
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is the one case
            if not _is_tls_verify_error(exc):
                raise
            log.warning(
                "%s: certificate verification failed, retrying unverified (%s)", request.url, exc
            )
        response = self._unverified.handle_request(request)
        response.extensions = {**response.extensions, TLS_UNVERIFIED_EXTENSION: True}
        return response

    def close(self) -> None:
        self._verified.close()
        self._unverified.close()


def new_client(
    timeout: float | httpx.Timeout = DEFAULT_TIMEOUT, *, tls_fallback: bool = True
) -> httpx.Client:
    return httpx.Client(
        transport=(
            TlsFallbackTransport(httpx.HTTPTransport(), httpx.HTTPTransport(verify=False))
            if tls_fallback
            else None
        ),
        follow_redirects=True,
        timeout=timeout,
        headers={
            "User-Agent": DISCOVERY_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            # No Accept-Encoding: httpx advertises exactly the codecs it can
            # decode, and hardcoding `br` without brotli installed hands back
            # compressed bytes that the HTML parser silently finds no links in.
            "Accept-Language": "en-US,en;q=0.9,*;q=0.5",
        },
    )


def _fetch(
    client: httpx.Client,
    url: str,
    kind: str,
    attempts: list[FetchAttempt],
    *,
    max_bytes: int = MAX_RESPONSE_BYTES,
) -> Fetched | None:
    """GET with a byte cap, recording the attempt either way.

    Streamed so a sitemap that turns out to be a 400 MB dump cannot take the
    process with it. A failure returns `None`: one dead host must not abort a
    batch of five hundred.

    A certificate that only verified with the check off is recorded as such,
    from the extension `TlsFallbackTransport` sets. That is a fact about our
    client rather than about the site, so it belongs in the manifest.
    """
    try:
        with client.stream("GET", url) as response:
            status = response.status_code
            unverified = bool(response.extensions.get(TLS_UNVERIFIED_EXTENSION, False))
            if status >= 400:
                attempts.append(
                    FetchAttempt(
                        url=url, kind=kind, status=status, ok=False, tls_unverified=unverified
                    )
                )
                return None

            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                chunks.append(chunk)
                total += len(chunk)
                if total >= max_bytes:
                    log.warning("%s at %s exceeded %d bytes, truncating", kind, url, max_bytes)
                    break

            body = b"".join(chunks)
            attempts.append(
                FetchAttempt(
                    url=url, kind=kind, status=status, ok=True, bytes_read=len(body),
                    tls_unverified=unverified,
                )
            )
            return Fetched(final_url=str(response.url), content=body, status=status)
    except httpx.HTTPError as exc:
        attempts.append(
            FetchAttempt(url=url, kind=kind, error=f"{type(exc).__name__}: {exc}", ok=False)
        )
        return None


def _decode(content: bytes) -> str:
    if content[:2] == b"\x1f\x8b":
        try:
            content = gzip.decompress(content)
        except OSError as exc:
            log.debug("gzip decompression failed: %s", exc)
    return content.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Source 1: sitemaps
# ---------------------------------------------------------------------------

_DOCTYPE_RE = re.compile(r"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)

#: The whole `<!DOCTYPE ...>` declaration, including an internal subset in
#: square brackets, which is where entity definitions live.
_DOCTYPE_BLOCK_RE = re.compile(r"<!\s*DOCTYPE[^>\[]*(\[[^\]]*\])?[^>]*>", re.IGNORECASE | re.DOTALL)
_SITEMAP_DIRECTIVE_RE = re.compile(r"^\s*sitemap\s*:\s*(\S+)", re.IGNORECASE | re.MULTILINE)
#: Nested sitemaps that never contain page URLs. Fetching them costs a request
#: and returns nothing this pipeline can crawl.
_SKIP_SITEMAP_HINTS = ("image", "video", "news", "author", "tag", "category")


def parse_sitemap(text: str) -> tuple[list[str], list[str]]:
    """Split a sitemap into (page URLs, nested sitemap URLs).

    `xml.etree` expands internal entities, so a hostile or merely broken sitemap
    can blow memory up from a few KB of input ("billion laughs"). Internal
    entities can only be *defined* inside a DOCTYPE, so removing the DOCTYPE
    removes the bomb: any surviving `&entity;` reference is then undefined and
    fails as a parse error, which is caught below.

    **Revised 2026-08-11.** This used to refuse the whole document, on the
    reasoning that a sitemap has no legitimate reason to declare a DOCTYPE and
    so refusing cost nothing. Measured: it cost two of 46 domains their only
    free source of URLs. meditronex.com and masimor.com both serve a valid
    sitemap behind a DOCTYPE declaration, and meditronex's homepage returns no
    links at all, so discarding its sitemap forced the paid map endpoint for
    nothing. Strip the declaration, keep the protection.
    """
    stripped = _DOCTYPE_BLOCK_RE.sub("", text, count=1)
    if _DOCTYPE_RE.search(stripped[:4096]):
        # A bare `<!ENTITY` outside a DOCTYPE is malformed XML and never a real
        # sitemap. Refuse rather than guess at what it meant.
        raise DiscoveryError("sitemap declares an ENTITY outside a DOCTYPE; refusing to parse")

    try:
        root = ET.fromstring(stripped.strip())
    except ET.ParseError as exc:
        raise DiscoveryError(f"sitemap is not valid XML: {exc}") from exc

    def localname(tag: str) -> str:
        return tag.rsplit("}", 1)[-1].lower()

    # A `<loc>` means a page inside a `<urlset>` and another sitemap inside a
    # `<sitemapindex>`. The root element is the only thing that distinguishes
    # them, and the two must not be merged: a sitemap URL is not crawlable.
    is_index = localname(root.tag) == "sitemapindex"
    locations = [
        element.text.strip()
        for element in root.iter()
        if localname(element.tag) == "loc" and (element.text or "").strip()
    ]
    return ([], locations) if is_index else (locations, [])


def sitemap_urls(
    domain: str,
    client: httpx.Client,
    attempts: list[FetchAttempt],
    *,
    max_urls: int = MAX_SITEMAP_URLS,
) -> list[str]:
    """Collect page URLs from the sitemap, wherever the site keeps it.

    Order is the spec's: `/sitemap.xml`, `/sitemap_index.xml`, then whatever
    `robots.txt` points at. Nested indexes are followed one level, skipping the
    image and news sitemaps that carry nothing crawlable.
    """
    # Both hosts, for the same reason `homepage_links` tries both: a
    # certificate that covers www only takes out every https request to the
    # bare domain, sitemaps included.
    seeds = [
        f"https://{host}/{name}"
        for host in hosts_for(domain)
        for name in ("sitemap.xml", "sitemap_index.xml")
    ]
    seeds.extend(_robots_sitemaps(domain, client, attempts))

    found: list[str] = []
    seen_sitemaps: set[str] = set()
    queue = list(dict.fromkeys(seeds))
    nested_budget = MAX_NESTED_SITEMAPS

    while queue and len(found) < max_urls:
        target = queue.pop(0)
        # Deduped on the URL itself, NOT `dedupe_key`, which folds `www.` away.
        # That is the right identity for a *page* (both hosts serve the same
        # article) and the wrong one for a *fetch target*: sagerly.com fails
        # certificate verification while www.sagerly.com serves fine, and folding
        # the two together threw the reachable host out of the queue unfetched.
        key = target.rstrip("/")
        if key in seen_sitemaps:
            continue
        seen_sitemaps.add(key)

        fetched = _fetch(client, target, "sitemap", attempts)
        if fetched is None:
            continue

        try:
            pages, nested = parse_sitemap(_decode(fetched.content))
        except DiscoveryError as exc:
            log.info("%s: %s (%s)", domain, exc, target)
            attempts[-1] = attempts[-1].model_copy(update={"ok": False, "error": str(exc)})
            continue

        attempts[-1] = attempts[-1].model_copy(update={"urls_found": len(pages) + len(nested)})
        found.extend(pages)

        for child in nested:
            if nested_budget <= 0:
                break
            if any(hint in child.lower() for hint in _SKIP_SITEMAP_HINTS):
                continue
            queue.append(child)
            nested_budget -= 1

    return found[:max_urls]


def _robots_sitemaps(
    domain: str, client: httpx.Client, attempts: list[FetchAttempt]
) -> list[str]:
    fetched = _fetch(client, f"https://{domain}/robots.txt", "robots", attempts, max_bytes=512_000)
    if fetched is None:
        return []
    directives = _SITEMAP_DIRECTIVE_RE.findall(_decode(fetched.content))
    resolved = [urljoin(fetched.final_url, value) for value in directives]
    attempts[-1] = attempts[-1].model_copy(update={"urls_found": len(resolved)})
    return resolved


# ---------------------------------------------------------------------------
# Source 2: homepage nav harvest
# ---------------------------------------------------------------------------

_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input",
     "link", "meta", "param", "source", "track", "wbr"}
)
_CHROME_TAGS = frozenset({"header", "nav", "footer"})
_CHROME_HINT_RE = re.compile(r"(^|[\s_-])(nav|navbar|menu|header|footer|masthead|topbar)([\s_-]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class HarvestedLink:
    href: str
    text: str
    in_chrome: bool


class LinkHarvester(HTMLParser):
    """Pull `<a>` hrefs and their labels out of a homepage, noting nav regions.

    Chrome detection covers `<header>/<nav>/<footer>` and the class/id/role
    conventions sites used before those elements existed. It is a hint, not a
    filter: links outside the chrome are still harvested, because a homepage
    body often links to the capabilities page more usefully than the menu does.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[HarvestedLink] = []
        self._stack: list[tuple[str, bool]] = []
        self._chrome_depth = 0
        self._href: str | None = None
        self._label: str | None = None
        self._chunks: list[str] = []
        self._anchor_in_chrome = False

    # -- region tracking ----------------------------------------------------

    @staticmethod
    def _is_chrome(tag: str, attrs: dict[str, str]) -> bool:
        if tag in _CHROME_TAGS:
            return True
        if attrs.get("role", "").lower() in ("navigation", "banner", "contentinfo"):
            return True
        return any(
            _CHROME_HINT_RE.search(attrs.get(name, "")) for name in ("class", "id")
        )

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = {name.lower(): (value or "") for name, value in attrs}

        if tag == "a":
            self._open_anchor(attributes)

        if tag in _VOID_TAGS:
            return

        chrome = self._is_chrome(tag, attributes)
        self._stack.append((tag, chrome))
        if chrome:
            self._chrome_depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            self._open_anchor({name.lower(): (value or "") for name, value in attrs})
            self._close_anchor()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "a":
            self._close_anchor()

        # Unwind to the matching open tag. Real-world HTML leaves tags unclosed,
        # and a stack that never pops leaves every later link marked as chrome.
        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index][0] == tag:
                for _, chrome in self._stack[index:]:
                    if chrome:
                        self._chrome_depth -= 1
                del self._stack[index:]
                return

    # -- anchors ------------------------------------------------------------

    def _open_anchor(self, attributes: Mapping[str, str]) -> None:
        href = (attributes.get("href") or "").strip()
        if not href:
            return
        self._href = href
        self._label = (attributes.get("aria-label") or attributes.get("title") or "").strip()
        self._chunks = []
        self._anchor_in_chrome = self._chrome_depth > 0

    def _close_anchor(self) -> None:
        if self._href is None:
            return
        text = " ".join("".join(self._chunks).split()) or self._label or ""
        self.links.append(
            HarvestedLink(href=self._href, text=text, in_chrome=self._anchor_in_chrome)
        )
        self._href = None
        self._label = None
        self._chunks = []

    def handle_data(self, data: str) -> None:
        if self._href is not None and self._stack and self._stack[-1][0] not in ("script", "style"):
            self._chunks.append(data)

    def close(self) -> None:
        """Flush a trailing `<a>` that the document never closed."""
        super().close()
        self._close_anchor()


def harvest_links(html: str) -> list[HarvestedLink]:
    parser = LinkHarvester()
    try:
        parser.feed(html)
        parser.close()
    except (AssertionError, ValueError) as exc:  # malformed markup, keep what we have
        log.debug("HTML parsing stopped early: %s", exc)
    return parser.links


def hosts_for(domain: str) -> list[str]:
    """The hosts worth trying for a registrable domain, in order.

    Added 2026-08-11 after measuring what "the site blocked us" actually was.
    Normalisation strips to the registrable domain, and a surprising number of
    sites serve nothing useful there: nextragroup.com answers 404 on the bare
    domain over both schemes while www.nextragroup.com is the real site, and
    sagerly.com fails certificate verification because its certificate covers www
    only. Two of five discovery failures on the goldset were this, and both
    were being written off as bot walls.
    """
    if domain.startswith("www."):
        return [domain, domain[4:]]
    return [domain, f"www.{domain}"]


@dataclass(frozen=True)
class Homepage:
    """The homepage discovery settled on, and where it actually landed.

    `domain` is the registrable domain of the *final* URL, which is not always
    the one asked for: an acquired company's site redirects wholesale to its
    new owner, and every link on the page then belongs to a domain the input
    list has never heard of.
    """

    url: str
    domain: str
    links: list[HarvestedLink]
    #: The homepage resolved to a different registrable domain than requested.
    redirected_off_domain: bool = False
    #: Internal links under `domain`, i.e. what survives `is_crawlable`.
    internal_links: int = 0


def _homepage_candidate(domain: str, fetched: Fetched, links: list[HarvestedLink]) -> Homepage:
    # Canonicalised because the final URL is whatever the redirect chain ended
    # on, and `https://host///` is a real thing servers return.
    url = canonical_url(fetched.final_url)
    try:
        landed = normalize_domain(url).domain
    except InvalidDomainError:
        landed = domain
    internal = sum(1 for link in links if is_crawlable(urljoin(url, link.href), landed))
    return Homepage(
        url=url,
        domain=landed,
        links=links,
        redirected_off_domain=landed != domain,
        internal_links=internal,
    )


def homepage_links(
    domain: str, client: httpx.Client, attempts: list[FetchAttempt]
) -> Homepage:
    """Fetch the homepage and harvest its links.

    Tries https then http on the bare domain, then the same on `www.`, because
    a handful of these sites still redirect the other way and losing a domain
    to a scheme or a missing subdomain is a poor trade for one request.

    **Which host wins is decided on internal links, not on any links at all.**
    That distinction is the whole of this function. bioiqx.com redirects the
    bare host to its acquirer getcheckednow.com, whose homepage harvests 96
    links; counting those as success stopped the search and never reached
    www.bioiqx.com, which is the surviving site. So every host is tried and
    ranked, and a page whose links all point somewhere else counts as having
    found nothing here.

    An off-domain landing is still accepted when *no* host stays put, because
    that is a real and common thing for this list — memriq.com is now
    resonetix.com and the pages worth reading are all on the new domain.
    It is recorded rather than silently followed, since evidence found there is
    evidence about the acquirer.
    """
    best: Homepage | None = None

    def better(candidate: Homepage) -> bool:
        if best is None:
            return True
        # On-domain beats off-domain, and having internal links beats not,
        # in that order. A reachable page with nothing on it is still worth
        # keeping over no page at all.
        return (not candidate.redirected_off_domain, candidate.internal_links > 0) > (
            not best.redirected_off_domain, best.internal_links > 0
        )

    for host in hosts_for(domain):
        for scheme in ("https", "http"):
            fetched = _fetch(client, f"{scheme}://{host}/", "homepage", attempts)
            if fetched is None:
                continue
            links = harvest_links(_decode(fetched.content))
            candidate = _homepage_candidate(domain, fetched, links)
            attempts[-1] = attempts[-1].model_copy(update={"urls_found": candidate.internal_links})
            if better(candidate):
                best = candidate
            if not candidate.redirected_off_domain and candidate.internal_links:
                # The ordinary case: the site is where it said it would be and
                # it has a nav. Nothing further can outrank this, so stop
                # rather than spend a request proving it.
                return candidate

    return best or Homepage(url=f"https://{domain}/", domain=domain, links=[])


# ---------------------------------------------------------------------------
# Source 3: the paid fallback
# ---------------------------------------------------------------------------


class MapUrls(Protocol):
    """Firecrawl's map endpoint, injected rather than imported.

    Phase 4 owns the Firecrawl client; discovery only needs "give me the URLs
    for this domain". Keeping it a callable means Phase 3 has no dependency on
    Phase 4, and the paid path stays visible at the call site instead of hiding
    inside a helper.

    The map endpoint renders JavaScript, which is the entire reason it is here:
    a site whose nav is built client-side harvests zero links and would
    otherwise be a discovery failure. Confirm rendering is on when wiring it.
    """

    def __call__(self, domain: str, *, limit: int) -> Sequence[str]: ...


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def discover(
    domain: str,
    pack: SectorPack,
    *,
    client: httpx.Client | None = None,
    map_urls: MapUrls | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    min_pages: int = MIN_PAGES,
    fill_unmatched: bool = True,
    map_threshold: int | None = None,
) -> DiscoveryResult:
    """Rank and cap the URLs to crawl for one domain.

    Sitemap and homepage nav both run: the first supplies breadth, the second
    supplies the anchor text that catches pages no path pattern would. The
    injected `map_urls` runs only if those two leave fewer candidates than
    `map_threshold`, which is the JS-rendered-nav case and the only paid step.
    """
    owns_client = client is None
    client = client or new_client(timeout)
    threshold = min_pages if map_threshold is None else map_threshold

    attempts: list[FetchAttempt] = []
    candidates: dict[str, Candidate] = {}

    # Only non-empty for a pack that names an `investors` type. A PDF is
    # otherwise always noise; this is the one shape worth the skip-list
    # exception, and only when the URL or its anchor actually matches.
    _investor_patterns = [p for p in build_patterns(pack) if p.page_type == "investors"]

    # Only non-empty for a pack that names a `careers` type. Backs the
    # one-hop-deeper careers fetch below: individual job postings are where
    # facility evidence often lives (a "Manufacturing Technician" role
    # listing "Manufacturing of Antibodies" duties), and the homepage nav
    # only ever links the careers index, never the postings under it.
    _careers_patterns = [p for p in build_patterns(pack) if p.page_type == "careers"]

    # Set once the homepage is fetched. Everything that filters URLs keys on
    # this rather than on `domain`, because a site that redirects to its
    # acquirer publishes every page it has under the new registrable domain,
    # and filtering those against the old one discards the entire site.
    site_domain = domain

    def add(url: str, source: str, anchor: str = "") -> None:
        absolute = urljoin(f"https://{site_domain}/", url.strip())
        allow_pdf = bool(
            _investor_patterns
            and urlsplit(absolute).path.lower().endswith(".pdf")
            and match_url(absolute, [anchor] if anchor else [], _investor_patterns)
        )
        if not is_crawlable(absolute, site_domain, allow_pdf=allow_pdf):
            return
        key = dedupe_key(absolute)
        candidate = candidates.get(key)
        if candidate is None:
            candidate = Candidate(url=canonical_url(absolute))
            candidates[key] = candidate
        candidate.sources.add(source)
        if anchor:
            candidate.add_anchor(anchor)

    try:
        homepage = homepage_links(domain, client, attempts)
        homepage_url = homepage.url
        site_domain = homepage.domain
        if homepage.redirected_off_domain:
            # STEP 0 of the crawl ladder. Every host of the requested domain
            # landed on a different registrable domain, so the company that
            # answered is not the company being asked about. Stop here: emit no
            # verdict, name the acquirer, and let it be queued as its own row
            # (tab 2a R12).
            #
            # Note this is decided *after* `homepage_links` has tried every
            # host, not on the first redirect. bioiqx.com's apex redirects to its
            # acquirer while www.bioiqx.com is the surviving site, so one
            # off-domain hop is not a move. What is being reclaimed is
            # everything below: robots, the sitemap probes, the careers hop, the
            # paid map call, and then ten Firecrawl credits spent scraping the
            # acquirer's pages into this domain's key. Measured on
            # lakevernmedical.com, 2026-08-26: six free requests and ten
            # credits, for a fact already known here.
            log.info(
                "%s redirects to %s; stopping at ladder step 0 (site_moved)",
                domain, site_domain,
            )
            return _step_0_exit(
                domain=domain,
                pack=pack,
                homepage_url=homepage_url,
                site_domain=site_domain,
                attempts=attempts,
                min_pages=min_pages,
            )
        for link in homepage.links:
            add(urljoin(homepage_url, link.href), "nav" if link.in_chrome else "body", link.text)

        for url in sitemap_urls(site_domain, client, attempts):
            add(url, "sitemap")

        # One bounded, free hop deeper: fetch the single best careers-type
        # candidate found so far and harvest ITS links. A third-party ATS
        # (Greenhouse, Workday, ...) is cross-domain, so `is_crawlable`
        # inside `add()` filters those links out same as anywhere else —
        # this degrades to a silent no-op on those sites, not an error.
        # Unconditional (not failure-gated like the paid map fallback):
        # every site with a careers page benefits, not just ones already
        # in trouble.
        if _careers_patterns:
            careers_url = _best_of_patterns(candidates, _careers_patterns)
            if careers_url is not None:
                detail = _fetch(client, careers_url, "careers-detail", attempts)
                if detail is not None:
                    for link in harvest_links(_decode(detail.content)):
                        add(urljoin(careers_url, link.href), "careers-detail", link.text)

        # Paid, and deliberately last. Everything above is free HTTP.
        if map_urls is not None and len(candidates) < threshold:
            log.info(
                "%s: %d free candidates, below the %d threshold; calling the map endpoint",
                domain, len(candidates), threshold,
            )
            try:
                # `site_domain`, not `domain`: mapping the old domain of an
                # acquired company returns the redirect and nothing else, so
                # the credit buys nothing.
                mapped = list(map_urls(site_domain, limit=pack.total_page_cap * 10))
            except Exception as exc:  # noqa: BLE001 - a paid dependency must not abort a batch
                attempts.append(
                    FetchAttempt(
                        url=f"map:{site_domain}", kind="map", ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            else:
                attempts.append(
                    FetchAttempt(
                        url=f"map:{site_domain}", kind="map", ok=True, urls_found=len(mapped)
                    )
                )
                for url in mapped:
                    add(url, "map")
    finally:
        if owns_client:
            client.close()

    # `body` links are nav links by another name for reporting purposes: both
    # come from the homepage harvest, and the distinction that matters
    # downstream is which *fetch* produced the candidate, not which region.
    raw_sources = {source for candidate in candidates.values() for source in candidate.sources}
    sources_used = sorted({"nav" if source == "body" else source for source in raw_sources})

    pages = select_pages(homepage_url, candidates, pack, fill_unmatched=fill_unmatched)

    found_types = sorted({page.page_type for page in pages})
    empty_types = [
        name for name, _ in pack.page_types_by_priority if name not in set(found_types)
    ]
    not_found = [a.url for a in attempts if a.status == 404]

    failed, reason = _failure(
        pages, min_pages, attempts, len(candidates), sources_used,
        redirected_to=site_domain if site_domain != domain else None,
    )

    tls_hosts = sorted(
        {urlsplit(a.url).netloc for a in attempts if a.tls_unverified and urlsplit(a.url).netloc}
    )

    result = DiscoveryResult(
        domain=domain,
        pack_version=pack.version,
        homepage_url=homepage_url,
        resolved_domain=site_domain,
        redirected_off_domain=site_domain != domain,
        tls_unverified_hosts=tls_hosts,
        total_page_cap=pack.total_page_cap,
        min_pages=min_pages,
        pages=pages,
        sources_used=sources_used,
        candidates_considered=len(candidates),
        attempts=attempts,
        not_found=not_found,
        page_types_found=found_types,
        page_types_empty=empty_types,
        discovery_failed=failed,
        failure_reason=reason,
    )

    if failed:
        log.warning("discovery failure on %s: %s", domain, reason)
    else:
        log.info("discovered %s", result.summary())
    return result


def _step_0_exit(
    *,
    domain: str,
    pack: SectorPack,
    homepage_url: str,
    site_domain: str,
    attempts: list[FetchAttempt],
    min_pages: int,
) -> DiscoveryResult:
    """The ladder's step-0 result: resolved, not failed, and carrying no pages.

    `discovery_failed` stays False deliberately. A moved domain is a clean
    determination, not a bug in this module, and the two have opposite
    handling: a discovery failure is retried with rendering and a map call,
    which here would buy the same redirect a second time. Tab 2's `run_status`
    vocabulary already separates `site_moved` from `discovery_failed`; this
    keeps that separation true at the point the fact is discovered rather than
    reconstructing it downstream.
    """
    return DiscoveryResult(
        domain=domain,
        pack_version=pack.version,
        homepage_url=homepage_url,
        resolved_domain=site_domain,
        redirected_off_domain=True,
        referred_domain=site_domain,
        total_page_cap=pack.total_page_cap,
        min_pages=min_pages,
        pages=[],
        sources_used=[],
        candidates_considered=0,
        attempts=attempts,
        not_found=[a.url for a in attempts if a.status == 404],
        page_types_found=[],
        page_types_empty=[name for name, _ in pack.page_types_by_priority],
        tls_unverified_hosts=sorted(
            {urlsplit(a.url).netloc for a in attempts if a.tls_unverified and urlsplit(a.url).netloc}
        ),
        discovery_failed=False,
        failure_reason=None,
        exit_step=0,
        stop_reason="site_moved",
    )


#: A homepage smaller than this that yielded no links is not a homepage. It is
#: a bot wall or a client-rendered shell, both of which answer 200.
INTERSTITIAL_BYTES = 4_000


def _failure(
    pages: Sequence[DiscoveredPage],
    min_pages: int,
    attempts: Sequence[FetchAttempt],
    candidates: int,
    sources_used: Sequence[str],
    redirected_to: str | None = None,
) -> tuple[bool, str | None]:
    """Name the failure precisely enough to fix it.

    "Fewer than three pages" is the trigger, but the useful part is why. Five
    causes look identical in a page count and have five different fixes: an
    unreachable homepage, a bot wall answering 200 with an interstitial, a
    client-rendered nav, a site that has moved to another domain, and links
    that all failed the URL filters. Only the last is a bug in this module;
    the middle two are what `map_urls` is for, and the moved site needs a
    decision about identity rather than another request.

    Getting this attribution right is the point. memriq.com spent a run reported
    as a client-rendered nav when it had in fact redirected wholesale to
    resonetix.com and served 133 perfectly ordinary links.
    """
    homepage = next((a for a in attempts if a.kind == "homepage" and a.ok), None)
    if len(pages) >= min_pages and homepage is not None:
        return False, None

    reasons: list[str] = []
    if len(pages) < min_pages:
        reasons.append(f"only {len(pages)} page(s) selected, minimum is {min_pages}")

    if homepage is None:
        reasons.append("homepage did not return a usable response")
    elif redirected_to:
        reasons.append(
            f"the site redirects to {redirected_to}, a different registrable domain, "
            f"and discovery against it produced {candidates} candidate URL(s)"
        )
    elif candidates == 0 and homepage.bytes_read < INTERSTITIAL_BYTES:
        reasons.append(
            f"homepage answered {homepage.status} with {homepage.bytes_read} bytes and "
            "no links, which is a bot wall or a holding page rather than a site"
        )
    elif candidates == 0:
        reasons.append(
            "no internal links found from sitemap or homepage nav "
            "(client-rendered navigation is the usual cause)"
        )
    else:
        reasons.append(f"{candidates} candidate URLs reduced to {len(pages)} selected pages")

    if not sources_used:
        reasons.append("no link source produced anything")

    return True, "; ".join(reasons) or "discovery produced an unusable result"


def discover_all(
    domains: Sequence[str],
    pack: SectorPack,
    *,
    max_workers: int = 8,
    timeout: float = DEFAULT_TIMEOUT,
    map_urls: MapUrls | None = None,
    **kwargs: Any,
) -> list[DiscoveryResult]:
    """Discover a batch over a shared connection pool, preserving input order."""
    if not domains:
        return []

    with new_client(timeout) as client:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(
                pool.map(
                    lambda domain: discover(
                        domain, pack, client=client, map_urls=map_urls, **kwargs
                    ),
                    domains,
                )
            )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest(
    store: Store,
    result: DiscoveryResult,
    crawl_date: date,
    *,
    extra: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> str:
    """Write the manifest to `raw/{domain}/{crawl_date}/manifest.json`.

    `extra` is how Phase 4 adds the fetch side — credits billed, bytes stored,
    which pages actually came back — so the crawl writes one manifest rather
    than discovery writing one and fetch overwriting it.
    """
    payload: dict[str, Any] = {
        "crawl_date": crawl_date.isoformat(),
        "discovery": result.model_dump(mode="json"),
    }
    if extra:
        payload.update(extra)
    return store.write_json(
        raw_manifest_path(result.domain, crawl_date), payload, overwrite=overwrite
    )
