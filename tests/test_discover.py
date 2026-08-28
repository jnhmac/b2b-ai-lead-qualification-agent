"""Discovery, exercised offline against a scripted site.

Every HTTP response here is served by an `httpx.MockTransport`, so the whole
path — robots, sitemap index, nested sitemaps, homepage harvest, scoring,
per-type caps, the manifest — runs with no network and no credits. The live
run over real domains is the Phase 3 gate and lives in `test_phase3_gate.py`.
"""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from cleanroom.discover import (
    MIN_PAGES,
    Candidate,
    DiscoveryError,
    DiscoveryResult,
    TlsFallbackTransport,
    _best_of_patterns,
    _is_tls_verify_error,
    build_patterns,
    canonical_url,
    dedupe_key,
    discover,
    harvest_links,
    hosts_for,
    is_article,
    is_crawlable,
    match_url,
    parse_sitemap,
    score_candidate,
    sitemap_urls,
    select_pages,
    url_depth,
    write_manifest,
)

# ---------------------------------------------------------------------------
# A scripted site
# ---------------------------------------------------------------------------

HOMEPAGE = """
<html><body>
  <header class="site-header">
    <nav>
      <a href="/">Home</a>
      <a href="/capabilities/">Capabilities</a>
      <a href="/company/heritage">Who We Are</a>
      <a href="https://twitter.com/acme">Follow us</a>
      <a href="mailto:hi@acme-medical.com">Email</a>
    </nav>
  </header>
  <main>
    <p>We build <a href="/products/catheters/">catheters</a> and more.</p>
    <a href="/brochure.pdf">Download the brochure</a>
    <a href="/cart">Cart</a>
  </main>
  <footer>
    <a href="/quality/certifications">Quality &amp; Certifications</a>
    <a href="/privacy-policy">Privacy</a>
  </footer>
</body></html>
"""

SITEMAP_INDEX = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://acme-medical.com/page-sitemap.xml</loc></sitemap>
  <sitemap><loc>https://acme-medical.com/image-sitemap.xml</loc></sitemap>
</sitemapindex>
"""

PAGE_SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://acme-medical.com/</loc></url>
  <url><loc>https://acme-medical.com/capabilities/</loc></url>
  <url><loc>https://acme-medical.com/manufacturing/</loc></url>
  <url><loc>https://acme-medical.com/manufacturing-services/</loc></url>
  <url><loc>https://acme-medical.com/facilities/cleanroom</loc></url>
  <url><loc>https://acme-medical.com/products/catheters/</loc></url>
  <url><loc>https://acme-medical.com/products/guidewires/</loc></url>
  <url><loc>https://acme-medical.com/products/balloons/</loc></url>
  <url><loc>https://acme-medical.com/products/balloons/sku-1001</loc></url>
  <url><loc>https://acme-medical.com/products/balloons/sku-1002</loc></url>
  <url><loc>https://acme-medical.com/products/balloons/sku-1003</loc></url>
  <url><loc>https://acme-medical.com/careers/</loc></url>
  <url><loc>https://acme-medical.com/company/heritage</loc></url>
  <url><loc>https://acme-medical.com/blog/2025/new-press-release</loc></url>
  <url><loc>https://acme-medical.com/contact</loc></url>
</urlset>
"""

ROBOTS = "User-agent: *\nDisallow: /admin\nSitemap: https://acme-medical.com/sitemap_index.xml\n"

ROUTES: dict[str, tuple[int, str]] = {
    "/": (200, HOMEPAGE),
    "/robots.txt": (200, ROBOTS),
    "/sitemap.xml": (404, "not found"),
    "/sitemap_index.xml": (200, SITEMAP_INDEX),
    "/page-sitemap.xml": (200, PAGE_SITEMAP),
}


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.host != "acme-medical.com":
        return httpx.Response(404, text="wrong host")
    status, body = ROUTES.get(request.url.path, (404, "not found"))
    return httpx.Response(status, text=body)


@pytest.fixture
def site_client() -> httpx.Client:
    with httpx.Client(transport=httpx.MockTransport(_handler), follow_redirects=True) as client:
        yield client


@pytest.fixture
def result(pack, site_client) -> DiscoveryResult:
    return discover("acme-medical.com", pack, client=site_client)


# ---------------------------------------------------------------------------
# URL hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://a-med.com/about/", "https://a-med.com/about"),
        ("https://a-med.com/about/index.html", "https://a-med.com/about"),
        ("https://A-MED.com/About#team", "https://a-med.com/About"),
        ("https://a-med.com:443/x", "https://a-med.com/x"),
        ("https://a-med.com", "https://a-med.com/"),
    ],
)
def test_canonical_url(raw, expected):
    assert canonical_url(raw) == expected


def test_dedupe_key_folds_scheme_and_www():
    assert dedupe_key("http://example.com/about") == dedupe_key("https://www.example.com/about/")
    assert dedupe_key("https://example.com/a") != dedupe_key("https://example.com/b")


def test_dedupe_key_keeps_subdomains_distinct():
    """`www` is a convention; `shop` is a different site section."""
    assert dedupe_key("https://shop.example.com/a") != dedupe_key("https://example.com/a")


def test_dedupe_key_drops_the_query_at_the_site_root_only():
    """A language switcher pointing at `/?lang=en-US` is not a second page."""
    assert dedupe_key("https://example.com/?lang=en-US") == dedupe_key("https://example.com/")
    assert dedupe_key("https://example.com/p?id=1") != dedupe_key("https://example.com/p")


@pytest.mark.parametrize(
    "path",
    ["/cookie-usage", "/cookies", "/privacy-policy", "/terms-of-service", "/impressum", "/legal"],
)
def test_boilerplate_footer_pages_are_dropped(path):
    """Every site spells these differently and each one costs a crawl slot."""
    assert is_crawlable(f"https://acme-medical.com{path}", "acme-medical.com") is False


@pytest.mark.parametrize("path", ["/termination-kits", "/legalizer-products", "/cookware"])
def test_boilerplate_matching_stops_at_segment_boundaries(path):
    assert is_crawlable(f"https://acme-medical.com{path}", "acme-medical.com") is True


@pytest.mark.parametrize(
    "url",
    [
        "https://acme-medical.com/brochure.pdf",
        "https://acme-medical.com/wp-content/uploads/x",
        "https://acme-medical.com/cart",
        "https://acme-medical.com/privacy-policy",
        "https://other.com/capabilities",
        "mailto:hi@acme-medical.com",
        "javascript:void(0)",
        "https://acme-medical.com/logo.svg",
        # Seen live: scraped as a facilities page, spending a credit and an LLM
        # call on XML coordinates and reporting a facility the site has none of.
        "https://acme-medical.com/locations.kml",
    ],
)
def test_is_crawlable_rejects(url):
    assert is_crawlable(url, "acme-medical.com") is False


@pytest.mark.parametrize(
    "url",
    [
        "https://acme-medical.com/capabilities/",
        "https://acme-medical.com/products/catheters",
        "https://shop.acme-medical.com/products",
        "http://acme-medical.com/about-us",
    ],
)
def test_is_crawlable_accepts(url):
    assert is_crawlable(url, "acme-medical.com") is True


def test_is_crawlable_does_not_filter_by_language():
    """Non-English sites stay in the population. Only asset URLs are dropped."""
    assert is_crawlable("https://acme-medical.com/de/unternehmen", "acme-medical.com") is True
    assert is_crawlable("https://acme-medical.com/产品", "acme-medical.com") is True


def test_allow_pdf_lets_a_pdf_through():
    """HANDOFF.md item 7: a 10-K PDF can carry the decisive quote, but every
    other PDF on these sites really is noise. The exception has to be opt-in
    per call, not a change to the default.
    """
    url = "https://nvea.com/investors/sec-filings/10-k.pdf"
    assert is_crawlable(url, "nvea.com") is False
    assert is_crawlable(url, "nvea.com", allow_pdf=True) is True


def test_allow_pdf_does_not_reopen_other_asset_suffixes():
    """The exception is `.pdf`-shaped, not "ignore the skip-list"."""
    assert is_crawlable("https://acme-medical.com/brochure.docx", "acme-medical.com", allow_pdf=True) is False
    assert is_crawlable("https://acme-medical.com/logo.svg", "acme-medical.com", allow_pdf=True) is False


def test_url_depth_ignores_locale_prefix():
    assert url_depth("https://a-med.com/products") == 1
    assert url_depth("https://a-med.com/de/products") == 1
    assert url_depth("https://a-med.com/en-us/products/catheters") == 2
    assert url_depth("https://a-med.com/") == 0


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------


def test_patterns_are_most_specific_first(pack):
    patterns = build_patterns(pack)
    lengths = [len(p.slug) for p in patterns]
    assert lengths == sorted(lengths, reverse=True)


def test_specific_pattern_wins_over_shorter_one(pack):
    """`/manufacturing-services/` belongs to capabilities, not manufacturing."""
    patterns = build_patterns(pack)
    assert match_url("https://a-med.com/manufacturing-services/", [], patterns).page_type == (
        "capabilities"
    )
    assert match_url("https://a-med.com/manufacturing/", [], patterns).page_type == "manufacturing"


def test_anchor_text_catches_what_the_path_misses(pack):
    """The case the spec calls out: /company/heritage plus a "Who We Are" link."""
    patterns = build_patterns(pack)
    path_only = match_url("https://a-med.com/company/heritage", [], patterns)
    with_anchor = match_url("https://a-med.com/company/heritage", ["Who We Are"], patterns)

    assert path_only.matched_on == ("path",)
    assert with_anchor.page_type == "about"
    assert with_anchor.matched_on == ("anchor", "path")


def test_anchor_only_match_is_selectable(pack):
    """A URL no pattern touches is still caught by its nav label."""
    patterns = build_patterns(pack)
    match = match_url("https://a-med.com/x7", ["Our Facilities"], patterns)
    assert match is not None
    assert match.page_type == "facilities"
    assert match.matched_on == ("anchor",)


def test_pattern_matching_respects_token_boundaries(pack):
    patterns = build_patterns(pack)
    assert match_url("https://a-med.com/aboutique", [], patterns) is None
    assert match_url("https://a-med.com/about", [], patterns).page_type == "about"


def test_long_anchor_text_does_not_match(pack):
    """A paragraph link slugs into a string containing half the pattern list."""
    patterns = build_patterns(pack)
    essay = (
        "Read more about our company and the manufacturing services and quality "
        "certifications that support our products across every facility"
    )
    assert match_url("https://a-med.com/x7", [essay], patterns) is None


def test_both_matches_outscore_one(pack):
    patterns = build_patterns(pack)
    max_priority = max(spec.priority for spec in pack.page_types.values())

    path_only = Candidate(url="https://a-med.com/capabilities")
    both = Candidate(url="https://a-med.com/capabilities", anchors=["Capabilities"])

    score_one = score_candidate(path_only, match_url(path_only.url, [], patterns), max_priority)
    score_two = score_candidate(both, match_url(both.url, both.anchors, patterns), max_priority)
    assert score_two > score_one


def test_deeper_and_parameterised_pages_score_lower(pack):
    patterns = build_patterns(pack)
    max_priority = max(spec.priority for spec in pack.page_types.values())

    def score(url: str) -> float:
        return score_candidate(Candidate(url=url), match_url(url, [], patterns), max_priority)

    assert score("https://a-med.com/products") > score("https://a-med.com/a/b/products")
    assert score("https://a-med.com/products") > score("https://a-med.com/products?page=4")


def test_low_value_sections_never_take_a_typed_slot(pack):
    """A blog post whose title slugs into `product-development` is not a product page.

    Seen live on a contract manufacturer: three blog posts filled the entire
    products cap, displaced the catalogue, and reported themselves as
    `products` in the manifest.
    """
    post = "https://a-med.com/blog/medical-balloon-product-development-and-transfer"
    catalogue = "https://a-med.com/products/catheters"
    candidates = {dedupe_key(u): Candidate(url=u) for u in (post, catalogue)}

    types = {p.url: p.page_type for p in select_pages("https://a-med.com/", candidates, pack)}
    assert types[catalogue] == "products"
    assert types[post] == "other"


def test_low_value_posts_are_demoted_not_excluded(pack):
    """On a quiet site the newsroom is the only content there is.

    "We opened a Class 7 cleanroom" is exactly the evidence this pipeline
    looks for, and a press release is where a company says it.
    """
    post = "https://a-med.com/news/we-opened-a-cleanroom"
    candidates = {dedupe_key(post): Candidate(url=post)}
    assert [p.url for p in select_pages("https://a-med.com/", candidates, pack)][1:] == [post]


@pytest.mark.parametrize(
    "path",
    [
        "/how-when-and-why-to-leverage-the-fdas-safer-technologies-program",
        "/supply-chain-woes-covid-and-medical-device-manufacturing",
        "/2024/07/09/exergen-receives-regulatory-approval",
        "/plant-cell-model-magnified-500k-to-100m-times.html",
    ],
)
def test_root_level_articles_never_take_a_typed_slot(path, pack):
    """Articles published outside any `/blog/` section still have to be caught.

    The last case is the one that matters most: `plant-cell-model...` matched
    the `plant` pattern and was filed as a **facilities** page, which reports a
    facility the site does not have.
    """
    url = f"https://a-med.com{path}"
    candidates = {dedupe_key(url): Candidate(url=url)}
    assert [p.page_type for p in select_pages("https://a-med.com/", candidates, pack)] == [
        "homepage",
        "other",
    ]


@pytest.mark.parametrize(
    "path",
    [
        "/capabilities",
        "/about-us",
        "/contract-manufacturing",
        "/medical-balloon-development-and-manufacturing",
        "/current-good-manufacturing-practice-cgmp",
        "/acute-care/products/isotour.html",
    ],
)
def test_real_company_pages_survive_the_slug_rule(path):
    """The five-word cut, checked against the pages it must not touch."""
    assert is_article(f"https://a-med.com{path}") is False


@pytest.mark.parametrize(
    "path", ["/blog", "/news/", "/press-releases", "/resources/downloads", "/media"]
)
def test_section_indexes_are_dropped(path):
    """A listing page is nav plus titles and dates. There is nothing to quote.

    Measured on the goldset sample before this rule: 4 of 182 selected pages
    were blog indexes and not one was an actual post.
    """
    assert is_crawlable(f"https://acme-medical.com{path}", "acme-medical.com") is False


@pytest.mark.parametrize(
    "path",
    ["/blog/we-opened-a-cleanroom", "/news/2025/iso-7-certification", "/resources/iso-guide"],
)
def test_pages_inside_those_sections_survive(path):
    assert is_crawlable(f"https://acme-medical.com{path}", "acme-medical.com") is True


def test_locale_twins_take_one_slot(pack):
    """`/ch/en/about` and `/it/en/about` are one page published twice."""
    urls = [
        "https://a-med.com/ch/en/about.html",
        "https://a-med.com/it/en/about.html",
        "https://a-med.com/de/de/about.html",
        "https://a-med.com/ch/en/quality.html",
    ]
    candidates = {dedupe_key(u): Candidate(url=u) for u in urls}
    pages = select_pages("https://a-med.com/", candidates, pack)

    assert sorted(p.page_type for p in pages) == ["about", "homepage", "quality"]


def test_english_copy_wins_among_locale_twins(pack):
    """The market is the US, so spend the credit on the English page.

    `/de/` sorts before `/uk/en/` alphabetically, so without a locale rank the
    German copy would win by accident.
    """
    urls = [
        "https://a-med.com/de/de/about.html",
        "https://a-med.com/uk/en/about.html",
        "https://a-med.com/fr/fr/about.html",
    ]
    candidates = {dedupe_key(u): Candidate(url=u) for u in urls}
    pages = select_pages("https://a-med.com/", candidates, pack)

    assert [p.url for p in pages][1:] == ["https://a-med.com/uk/en/about.html"]


def test_english_beats_a_structurally_better_non_english_page(pack):
    """The preference has to outweigh depth, or a shallow Dutch page wins.

    Seen live on strykon.com: `/nl/nl/emergency-care/products.html` outscored
    every English product page, so the tiebreaker never got a say.
    """
    patterns = build_patterns(pack)
    max_priority = max(spec.priority for spec in pack.page_types.values())

    def score(url: str) -> float:
        return score_candidate(Candidate(url=url), match_url(url, [], patterns), max_priority)

    assert score("https://a-med.com/ch/en/acute-care/products/x.html") > score(
        "https://a-med.com/nl/nl/products.html"
    )


def test_a_lone_non_english_page_can_still_be_selected(pack):
    """Sized below a path match, so the only facilities page on a site gets in."""
    patterns = build_patterns(pack)
    max_priority = max(spec.priority for spec in pack.page_types.values())
    url = "https://a-med.com/de/de/quality.html"
    assert score_candidate(Candidate(url=url), match_url(url, [], patterns), max_priority) > 0


def test_unprefixed_page_beats_every_locale_copy(pack):
    urls = ["https://a-med.com/ch/en/about.html", "https://a-med.com/about"]
    candidates = {dedupe_key(u): Candidate(url=u) for u in urls}
    assert [p.url for p in select_pages("https://a-med.com/", candidates, pack)][1:] == [
        "https://a-med.com/about"
    ]


def test_a_site_with_no_english_copy_is_still_crawled(pack):
    """Preference among duplicates, never a filter.

    BUILD.md locks this: non-English sites are crawled and classified like
    everything else and flagged for review. A German-only supplier selling into
    the US must not vanish from the population.
    """
    urls = [
        "https://a-med.com/de/de/uber-uns.html",
        "https://a-med.com/de/de/qualitat.html",
    ]
    candidates = {dedupe_key(u): Candidate(url=u) for u in urls}
    pages = select_pages("https://a-med.com/", candidates, pack)

    assert len(pages) == 3
    assert all(url_depth(p.url) <= 2 for p in pages)


def test_locale_dedupe_leaves_ordinary_paths_alone(pack):
    urls = ["https://a-med.com/about", "https://a-med.com/quality", "https://a-med.com/careers"]
    candidates = {dedupe_key(u): Candidate(url=u) for u in urls}
    assert len(select_pages("https://a-med.com/", candidates, pack)) == 4


def test_homepage_url_is_canonicalised(pack):
    """Servers really do redirect to `https://host///`."""
    def sloppy(request: httpx.Request) -> httpx.Response:
        if request.url.path == "///":
            return httpx.Response(200, text=HOMEPAGE)
        if request.url.path == "/":
            return httpx.Response(301, headers={"Location": "https://sloppy-med.com///"})
        return httpx.Response(404, text="not found")

    with httpx.Client(transport=httpx.MockTransport(sloppy), follow_redirects=True) as client:
        outcome = discover("sloppy-med.com", pack, client=client)

    assert outcome.homepage_url == "https://sloppy-med.com/"
    assert outcome.pages[0].url == "https://sloppy-med.com/"


# ---------------------------------------------------------------------------
# Sitemaps
# ---------------------------------------------------------------------------


def test_parse_sitemap_urlset():
    pages, nested = parse_sitemap(PAGE_SITEMAP)
    assert nested == []
    assert "https://acme-medical.com/capabilities/" in pages


def test_parse_sitemap_index_returns_nested_only():
    pages, nested = parse_sitemap(SITEMAP_INDEX)
    assert pages == []
    assert nested == [
        "https://acme-medical.com/page-sitemap.xml",
        "https://acme-medical.com/image-sitemap.xml",
    ]


def test_parse_sitemap_defuses_an_entity_bomb():
    """Entity expansion turns a few KB of XML into gigabytes of memory.

    Revised 2026-08-11: the DOCTYPE is stripped rather than the document
    refused, so the bomb dies as an undefined entity instead of a rejection.
    Same protection, and it no longer takes valid sitemaps down with it.
    """
    bomb = (
        '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">]>'
        "<urlset><url><loc>&lol;</loc></url></urlset>"
    )
    with pytest.raises(DiscoveryError, match="undefined entity"):
        parse_sitemap(bomb)


def test_parse_sitemap_reads_a_sitemap_behind_a_doctype():
    """A real sitemap that declares a DOCTYPE must still be read.

    Refusing the whole document cost meditronex.com and masimor.com their only
    free source of URLs on the 2026-08-07 crawl. Meditronex's homepage returns
    no links at all, so discarding its sitemap forced the paid map endpoint.
    """
    sitemap = (
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE urlset PUBLIC "-//W3C//DTD" "http://www.w3.org/TR/x.dtd">\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<url><loc>https://acme-medical.com/capabilities</loc></url>"
        "<url><loc>https://acme-medical.com/quality</loc></url></urlset>"
    )
    pages, nested = parse_sitemap(sitemap)
    assert pages == [
        "https://acme-medical.com/capabilities",
        "https://acme-medical.com/quality",
    ]
    assert nested == []


def test_parse_sitemap_refuses_a_bare_entity_declaration():
    """An `<!ENTITY` outside a DOCTYPE is malformed, never a real sitemap."""
    with pytest.raises(DiscoveryError, match="ENTITY outside a DOCTYPE"):
        parse_sitemap('<!ENTITY lol "lol"><urlset></urlset>')


def test_hosts_for_tries_www_as_well():
    """Two of five goldset discovery failures were a missing `www.`."""
    assert hosts_for("nextragroup.com") == ["nextragroup.com", "www.nextragroup.com"]
    assert hosts_for("www.sagerly.com") == ["www.sagerly.com", "sagerly.com"]


def test_www_and_bare_sitemaps_are_both_fetched():
    """`dedupe_key` folds `www.` away, which is wrong for a fetch target.

    Right identity for a page, since both hosts serve the same article. Wrong
    identity for a URL we are about to request: sagerly.com fails certificate
    verification while www.sagerly.com serves a 191-URL sitemap, and folding them
    together dropped the reachable host from the queue unfetched.
    """
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if request.url.host == "sager.test":
            raise httpx.ConnectError("certificate verify failed")
        if request.url.path == "/sitemap.xml":
            return httpx.Response(200, text=PAGE_SITEMAP)
        return httpx.Response(404)

    attempts: list = []
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        urls = sitemap_urls("sager.test", client, attempts)

    assert any(url.startswith("https://www.sager.test/") for url in requested)
    assert urls, "the reachable www host produced no URLs"


def test_parse_sitemap_rejects_non_xml():
    with pytest.raises(DiscoveryError, match="not valid XML"):
        parse_sitemap("404 Not Found")


# ---------------------------------------------------------------------------
# Homepage harvesting
# ---------------------------------------------------------------------------


def test_harvest_links_captures_text_and_region():
    links = {link.href: link for link in harvest_links(HOMEPAGE)}

    assert links["/capabilities/"].text == "Capabilities"
    assert links["/capabilities/"].in_chrome is True
    assert links["/company/heritage"].text == "Who We Are"
    assert links["/quality/certifications"].text == "Quality & Certifications"
    assert links["/quality/certifications"].in_chrome is True
    # Body links are harvested too: a homepage often links the capabilities
    # page more usefully than its own menu does.
    assert links["/products/catheters/"].in_chrome is False


def test_harvest_links_recovers_from_unclosed_tags():
    """Real markup leaves tags open. A stack that never pops marks everything nav."""
    html = "<header><nav><a href='/a'>A</a></header><main><a href='/b'>B</a></main>"
    links = {link.href: link.in_chrome for link in harvest_links(html)}
    assert links == {"/a": True, "/b": False}


def test_harvest_links_falls_back_to_aria_label():
    html = "<a href='/facilities' aria-label='Our Facilities'><img src='x.png'></a>"
    assert harvest_links(html)[0].text == "Our Facilities"


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_homepage_is_always_first(result, pack):
    assert result.pages[0].url == "https://acme-medical.com/"
    assert result.pages[0].page_type == pack.always_fetch_type


def test_total_cap_is_respected(result, pack):
    assert len(result.pages) <= pack.total_page_cap
    assert len(result.pages) == pack.total_page_cap


def test_per_type_caps_stop_a_catalogue_taking_every_slot(result, pack):
    counts = result.page_type_counts
    for page_type, count in counts.items():
        if page_type == "other":
            continue
        assert count <= pack.page_types[page_type].max_pages, f"{page_type} over cap"

    # Six product URLs were on offer, three of them SKU pages.
    assert counts.get("products", 0) == pack.page_types["products"].max_pages
    assert not any("sku-" in page.url for page in result.pages)


def test_page_mix_is_not_dominated_by_one_type(result):
    counts = result.page_type_counts
    assert len(counts) >= 4
    assert max(counts.values()) <= 3


def test_selection_is_deterministic(pack, site_client):
    first = discover("acme-medical.com", pack, client=site_client)
    second = discover("acme-medical.com", pack, client=site_client)
    assert first.urls == second.urls


def test_duplicate_urls_across_sources_are_selected_once(result):
    keys = [dedupe_key(url) for url in result.urls]
    assert len(keys) == len(set(keys))


def test_an_ordinary_brochure_pdf_stays_rejected(result):
    """Adding the `investors` page type must not reopen the general PDF
    skip-list. `/brochure.pdf` is linked from the homepage nav and matches no
    `investors` pattern, so it must still be absent from what got selected.
    """
    assert not any(page.url.endswith(".pdf") for page in result.pages)


INVESTOR_HOMEPAGE = """
<html><body>
  <nav>
    <a href="/">Home</a>
    <a href="/capabilities/">Capabilities</a>
    <a href="/investors/sec-filings/10-k.pdf">Annual Report (10-K)</a>
    <a href="/brochure.pdf">Download the brochure</a>
  </nav>
</body></html>
"""

INVESTOR_ROBOTS = "User-agent: *\nDisallow: /admin\n"


def _investor_handler(request: httpx.Request) -> httpx.Response:
    if request.url.host != "sec-filer.com":
        return httpx.Response(404, text="wrong host")
    routes = {
        "/": (200, INVESTOR_HOMEPAGE),
        "/robots.txt": (200, INVESTOR_ROBOTS),
        "/sitemap.xml": (404, "not found"),
    }
    status, body = routes.get(request.url.path, (404, "not found"))
    return httpx.Response(status, text=body)


def test_an_investor_relations_pdf_is_selected(pack):
    """HANDOFF.md item 7: this is the one PDF shape worth the skip-list
    exception, and only because the pack now names `investors` as a type.
    """
    with httpx.Client(transport=httpx.MockTransport(_investor_handler)) as client:
        result = discover("sec-filer.com", pack, client=client)

    pdf_pages = [page for page in result.pages if page.page_type == "investors"]
    assert len(pdf_pages) == 1
    assert pdf_pages[0].url.endswith("10-k.pdf")
    # The un-patterned PDF alongside it is still noise.
    assert not any(page.url.endswith("brochure.pdf") for page in result.pages)


def test_unmatched_fill_prefers_shallow_pages(pack):
    """`/contact` beats a deep unmatched URL when slots are left over."""
    candidates = {
        dedupe_key(url): Candidate(url=url)
        for url in (
            "https://a-med.com/contact",
            "https://a-med.com/x/y/z/deep",
        )
    }
    pages = select_pages("https://a-med.com/", candidates, pack)
    assert [p.page_type for p in pages] == ["homepage", "other", "other"]
    assert pages[1].url == "https://a-med.com/contact"


def test_fill_can_be_switched_off(pack):
    candidates = {dedupe_key("https://a-med.com/contact"): Candidate(url="https://a-med.com/contact")}
    assert len(select_pages("https://a-med.com/", candidates, pack, fill_unmatched=False)) == 1


# ---------------------------------------------------------------------------
# Careers-page enumeration: _best_of_patterns, the pure ranking helper behind
# discover()'s one-hop-deeper careers fetch (see below for the integration
# tests). Scoped to a single page_type, unlike select_pages's full pass.
# ---------------------------------------------------------------------------


def test_best_of_patterns_picks_the_highest_scoring_candidate(pack):
    patterns = [p for p in build_patterns(pack) if p.page_type == "careers"]
    candidates = {
        dedupe_key(url): Candidate(url=url)
        for url in (
            "https://a-med.com/company/join-us",  # matches, but a weaker pattern
            "https://a-med.com/careers/",  # matches "careers" directly
            "https://a-med.com/products/catheters/",  # does not match at all
        )
    }
    best = _best_of_patterns(candidates, patterns)
    assert best == "https://a-med.com/careers/"


def test_best_of_patterns_returns_none_with_no_matching_candidate(pack):
    patterns = [p for p in build_patterns(pack) if p.page_type == "careers"]
    candidates = {
        dedupe_key(url): Candidate(url=url)
        for url in ("https://a-med.com/products/", "https://a-med.com/about/")
    }
    assert _best_of_patterns(candidates, patterns) is None


# ---------------------------------------------------------------------------
# Careers-page enumeration: the real discover() integration.
#
# biocarta.net was misclassified no_cleanroom_needed because the actual
# evidence — a "Manufacturing of Antibodies" job posting — lived one hop past
# the /careers index page discover() selected, and was never a candidate at
# all. These exercise the one bounded, free, extra GET that fixes that: fetch
# the best careers-type candidate, harvest ITS links, and let the individual
# postings compete for a (now-larger) careers slot like any other candidate.
# ---------------------------------------------------------------------------

CAREERS_HOMEPAGE = """
<html><body>
  <nav>
    <a href="/">Home</a>
    <a href="/careers">Careers</a>
  </nav>
</body></html>
"""

#: Job slugs deliberately avoid every OTHER page_type's own pattern words
#: (manufacturing, quality, ...) — "Manufacturing Technician" and "Quality
#: Engineer" were the first slugs tried here and both lost their `careers`
#: page_type to a same-page longer-slug match (`match_url`'s "most specific
#: pattern wins" rule, discover.py:496-510, correctly prefers a real
#: `manufacturing`/`quality` pattern match over `careers` when a job title
#: happens to contain that word). Real career-page slugs collide with real
#: page types often enough that this is worth a comment, not a workaround.
CAREERS_INDEX = """
<html><body>
  <main>
    <a href="/careers/assembly-technician">Assembly Technician</a>
    <a href="/careers/machine-operator">Machine Operator</a>
  </main>
</body></html>
"""


def _careers_handler(request: httpx.Request) -> httpx.Response:
    if request.url.host != "careers-med.com":
        return httpx.Response(404, text="wrong host")
    routes = {
        "/": (200, CAREERS_HOMEPAGE),
        "/careers": (200, CAREERS_INDEX),
        "/robots.txt": (404, "not found"),
        "/sitemap.xml": (404, "not found"),
    }
    status, body = routes.get(request.url.path, (404, "not found"))
    return httpx.Response(status, text=body)


def test_careers_detail_fetch_surfaces_job_posting_candidates(pack):
    with httpx.Client(transport=httpx.MockTransport(_careers_handler)) as client:
        result = discover("careers-med.com", pack, client=client)

    assert any(a.kind == "careers-detail" and a.ok for a in result.attempts)
    urls = {page.url for page in result.pages}
    assert "https://careers-med.com/careers/assembly-technician" in urls


def test_a_job_posting_survives_selection_once_careers_max_pages_is_raised(pack):
    with httpx.Client(transport=httpx.MockTransport(_careers_handler)) as client:
        result = discover("careers-med.com", pack, client=client)

    careers_pages = [page for page in result.pages if page.page_type == "careers"]
    assert len(careers_pages) <= pack.page_types["careers"].max_pages
    assert any("assembly-technician" in page.url for page in careers_pages)
    # The index page still gets a slot too; enumeration adds to it rather
    # than displacing it.
    assert any(page.url == "https://careers-med.com/careers" for page in careers_pages)


NO_CAREERS_HOMEPAGE = """
<html><body>
  <nav>
    <a href="/">Home</a>
    <a href="/about">About</a>
    <a href="/products">Products</a>
  </nav>
</body></html>
"""


def _no_careers_handler(request: httpx.Request) -> httpx.Response:
    if request.url.host != "no-careers-med.com":
        return httpx.Response(404, text="wrong host")
    routes = {
        "/": (200, NO_CAREERS_HOMEPAGE),
        "/about": (200, "<html><body>About us.</body></html>"),
        "/products": (200, "<html><body>Our products.</body></html>"),
        "/robots.txt": (404, "not found"),
        "/sitemap.xml": (404, "not found"),
    }
    status, body = routes.get(request.url.path, (404, "not found"))
    return httpx.Response(status, text=body)


def test_no_careers_type_candidate_never_triggers_the_extra_fetch(pack):
    """A site with no careers link anywhere never triggers the extra fetch.

    (Not the shared `result` fixture: its sitemap already lists a `/careers/`
    URL, which is exactly the case this test needs to NOT have.)
    """
    with httpx.Client(transport=httpx.MockTransport(_no_careers_handler)) as client:
        result = discover("no-careers-med.com", pack, client=client)
    assert not any(a.kind == "careers-detail" for a in result.attempts)


def _off_domain_ats_handler(request: httpx.Request) -> httpx.Response:
    if request.url.host != "ats-med.com":
        return httpx.Response(404, text="wrong host")
    routes = {
        "/": (
            200,
            '<html><body><nav><a href="/">Home</a>'
            '<a href="/about">About</a><a href="/products">Products</a>'
            '<a href="https://boards.greenhouse.io/acme">Careers</a>'
            "</nav></body></html>",
        ),
        "/about": (200, "<html><body>About us.</body></html>"),
        "/products": (200, "<html><body>Our products.</body></html>"),
        "/robots.txt": (404, "not found"),
        "/sitemap.xml": (404, "not found"),
    }
    status, body = routes.get(request.url.path, (404, "not found"))
    return httpx.Response(status, text=body)


def test_an_off_domain_careers_link_never_becomes_a_candidate(pack):
    """A third-party ATS (Greenhouse, Workday, ...) is cross-domain, so
    is_crawlable's domain check filters it out before it can ever become a
    careers candidate — the extra fetch degrades to a silent no-op, not an
    error, and discovery completes normally on what's left."""
    with httpx.Client(transport=httpx.MockTransport(_off_domain_ats_handler)) as client:
        result = discover("ats-med.com", pack, client=client)

    assert not any("greenhouse" in url for url in result.urls)
    assert not any(a.kind == "careers-detail" for a in result.attempts)
    assert result.discovery_failed is False


# ---------------------------------------------------------------------------
# Manifest and failure reporting
# ---------------------------------------------------------------------------


def test_manifest_records_attempts_and_404s(result):
    kinds = {attempt.kind for attempt in result.attempts}
    assert {"homepage", "robots", "sitemap"} <= kinds
    assert "https://acme-medical.com/sitemap.xml" in result.not_found


def test_manifest_records_empty_page_types(result, pack):
    assert set(result.page_types_found) | set(result.page_types_empty) >= set(pack.page_types)
    assert not set(result.page_types_found) & set(result.page_types_empty)


def test_robots_directive_is_followed(result):
    """`/sitemap.xml` 404s here, so every sitemap URL came via robots.txt."""
    assert "sitemap" in result.sources_used
    assert any(a.url.endswith("page-sitemap.xml") and a.ok for a in result.attempts)


def test_image_sitemaps_are_not_fetched(result):
    assert not any("image-sitemap" in attempt.url for attempt in result.attempts)


def test_healthy_domain_is_not_a_discovery_failure(result):
    assert result.discovery_failed is False
    assert result.failure_reason is None


def test_dead_domain_is_a_discovery_failure(pack):
    def dead(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with httpx.Client(transport=httpx.MockTransport(dead)) as client:
        outcome = discover("dead-host.com", pack, client=client)

    assert outcome.discovery_failed is True
    assert "homepage" in outcome.failure_reason
    assert len(outcome.pages) < MIN_PAGES


def test_thin_site_is_a_discovery_failure_not_an_unknown(pack):
    """Under three pages is a bug in this module, never a thin-site verdict.

    Phase 7a harvests the real `unknown` population from crawl manifests. A
    discovery bug mixed into that population corrupts the only test coverage
    the `unknown` label has.
    """
    def one_page(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text="<html><body><p>Coming soon</p></body></html>")
        return httpx.Response(404, text="not found")

    with httpx.Client(transport=httpx.MockTransport(one_page)) as client:
        outcome = discover("thin-site.com", pack, client=client)

    assert len(outcome.pages) == 1
    assert outcome.discovery_failed is True
    assert "minimum is 3" in outcome.failure_reason
    assert "holding page" in outcome.failure_reason


def test_bot_wall_and_client_rendered_nav_are_named_apart(pack):
    """Both answer 200 with zero links, and they need different fixes.

    A 200 is not proof the homepage was served: enterprise sites front their
    marketing pages with a WAF that returns a one-kilobyte interstitial. Phase 4
    triage needs to tell that apart from a real page whose nav is built in the
    browser.
    """
    def wall(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><title>Incorrect Browser</title></html>")

    def spa(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/":
            return httpx.Response(404, text="not found")
        filler = "<p>Loading the experience.</p>" * 300
        return httpx.Response(200, text=f"<html><body><div id='root'>{filler}</div></body></html>")

    with httpx.Client(transport=httpx.MockTransport(wall)) as client:
        blocked = discover("walled.com", pack, client=client)
    with httpx.Client(transport=httpx.MockTransport(spa)) as client:
        rendered = discover("clientside.com", pack, client=client)

    assert "bot wall" in blocked.failure_reason
    assert "client-rendered navigation" in rendered.failure_reason


def test_map_endpoint_is_the_last_resort_not_the_first(pack, site_client):
    """A site with a working sitemap must never spend a Firecrawl credit."""
    calls: list[str] = []

    def fake_map(domain: str, *, limit: int) -> list[str]:
        calls.append(domain)
        return []

    discover("acme-medical.com", pack, client=site_client, map_urls=fake_map)
    assert calls == []


def test_map_endpoint_rescues_a_js_rendered_nav(pack):
    """Zero links from free sources is exactly the case the paid call exists for."""
    def js_only(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text="<html><body><div id='root'></div></body></html>")
        return httpx.Response(404, text="not found")

    def fake_map(domain: str, *, limit: int) -> list[str]:
        return [
            f"https://{domain}/capabilities",
            f"https://{domain}/manufacturing",
            f"https://{domain}/quality",
        ]

    with httpx.Client(transport=httpx.MockTransport(js_only)) as client:
        outcome = discover("spa-med.com", pack, client=client, map_urls=fake_map)

    assert "map" in outcome.sources_used
    assert outcome.discovery_failed is False
    assert len(outcome.pages) == 4


def test_map_failure_does_not_abort_discovery(pack, site_client):
    def broken_map(domain: str, *, limit: int) -> list[str]:
        raise RuntimeError("firecrawl is down")

    outcome = discover("acme-medical.com", pack, client=site_client, map_urls=broken_map, map_threshold=999)
    assert outcome.discovery_failed is False
    assert any(a.kind == "map" and a.error for a in outcome.attempts)


def test_manifest_round_trips_through_the_store(result, store):
    uri = write_manifest(store, result, date(2026, 8, 7))
    assert uri.endswith("raw/acme-medical.com/2026-08-07/manifest.json")

    payload = store.read_json("raw/acme-medical.com/2026-08-07/manifest.json")
    assert payload["crawl_date"] == "2026-08-07"
    assert DiscoveryResult.model_validate(payload["discovery"]) == result


def test_manifest_leaves_room_for_the_fetch_side(result, store):
    """Phase 4 adds credits and bytes to the same file rather than overwriting it."""
    write_manifest(store, result, date(2026, 8, 7), extra={"credits_billed": 10})
    payload = store.read_json("raw/acme-medical.com/2026-08-07/manifest.json")
    assert payload["credits_billed"] == 10
    assert payload["discovery"]["domain"] == "acme-medical.com"


# ---------------------------------------------------------------------------
# Three failures that looked like the web and were this module
#
# All three came out of the batch-1 dry run on 2026-08-11, before a credit was
# spent, by reading the attempt records rather than the failure reasons. Each
# had already been written down as a fact about the site.
# ---------------------------------------------------------------------------


def test_a_host_whose_links_all_point_elsewhere_does_not_end_the_search(pack):
    """bioiqx.com: the bare host redirects to its acquirer, `www` is the real site.

    The bare host landed on getcheckednow.com and harvested 96 links. Counting
    those as "this host produced links" returned the acquirer's homepage and
    never tried www.bioiqx.com, whose links are the ones worth crawling. What
    decides a host is *internal* links, not links.
    """
    def acquired(request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        if host == "bioiqx.com":
            return httpx.Response(301, headers={"Location": "https://newowner.com/for-business/"})
        if host == "newowner.com":
            links = "".join(
                '<a href="https://newowner.com/p{n}">Product {n}</a>'.format(n=n)
                for n in range(40)
            )
            return httpx.Response(200, text="<html><nav>" + links + "</nav></html>")
        if host == "www.bioiqx.com" and path == "/":
            return httpx.Response(
                200,
                text=(
                    "<html><nav>"
                    '<a href="/about-us">About Us</a>'
                    '<a href="/our-products">Products</a>'
                    '<a href="/quality-certifications">Quality</a>'
                    "</nav></html>"
                ),
            )
        return httpx.Response(404, text="not found")

    with httpx.Client(transport=httpx.MockTransport(acquired), follow_redirects=True) as client:
        outcome = discover("bioiqx.com", pack, client=client)

    assert outcome.resolved_domain == "bioiqx.com"
    assert outcome.redirected_off_domain is False
    assert outcome.homepage_url == "https://www.bioiqx.com/"
    assert len(outcome.pages) >= MIN_PAGES
    assert not any("newowner.com" in page.url for page in outcome.pages)


def test_a_site_that_moved_domain_stops_at_step_0(pack):
    """memriq.com: redirects wholesale to resonetix.com.

    **Assertion changed 2026-08-27, deliberately.** This test previously
    asserted that discovery continued against the destination and selected
    pages there. `crawl_policy.md` step 0 supersedes that: the company that
    answered is not the company being asked about, so no pages are selected,
    no verdict is emitted, and the acquirer is queued as its own row (tab 2a
    R12). Discovering at the destination spent ten Firecrawl credits filing
    the acquirer's pages under the old domain's key.

    What has not changed is the attribution the old test was written for: this
    is a move, not a client-rendered nav, and the record says so.
    """
    def moved(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host in ("memriq.com", "www.memriq.com"):
            return httpx.Response(301, headers={"Location": "https://acquirer.com/"})
        if host == "acquirer.com" and request.url.path == "/":
            return httpx.Response(
                200,
                text=(
                    "<html><nav>"
                    '<a href="/about">About</a>'
                    '<a href="/capabilities">Capabilities</a>'
                    '<a href="/facilities">Our Facility</a>'
                    '<a href="/quality">Quality</a>'
                    "</nav></html>"
                ),
            )
        return httpx.Response(404, text="not found")

    with httpx.Client(transport=httpx.MockTransport(moved), follow_redirects=True) as client:
        outcome = discover("memriq.com", pack, client=client)

    # The record stays keyed by the domain the input list carries, so the store
    # and the lead list still line up.
    assert outcome.domain == "memriq.com"
    assert outcome.resolved_domain == "acquirer.com"
    assert outcome.redirected_off_domain is True

    assert outcome.pages == []
    assert outcome.exit_step == 0
    assert outcome.stop_reason == "site_moved"
    assert outcome.referred_domain == "acquirer.com"
    # A move is a determination, not a discovery bug: the retry path for a
    # discovery failure would re-fetch the acquirer and find the same redirect.
    assert outcome.discovery_failed is False


def test_a_moved_site_stops_at_step_0_whatever_the_destination_looks_like(pack):
    """**Assertion changed 2026-08-27, deliberately.**

    This previously asserted that a moved site whose destination was also thin
    reported the move rather than blaming the nav. Under step 0 the question
    never arises: the ladder stops on the move and never looks at what the
    destination serves, so there is no thin-site judgement to get wrong. The
    original bug it guarded against (reporting a move as a client-rendered
    nav) is now impossible by construction rather than by wording.
    """
    def moved_and_thin(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host in ("gone.com", "www.gone.com"):
            return httpx.Response(301, headers={"Location": "https://elsewhere.com/"})
        if host == "elsewhere.com" and request.url.path == "/":
            return httpx.Response(200, text="<html><body><p>Nothing here</p></body></html>")
        return httpx.Response(404, text="not found")

    with httpx.Client(
        transport=httpx.MockTransport(moved_and_thin), follow_redirects=True
    ) as client:
        outcome = discover("gone.com", pack, client=client)

    assert outcome.exit_step == 0
    assert outcome.stop_reason == "site_moved"
    assert outcome.referred_domain == "elsewhere.com"
    assert outcome.discovery_failed is False
    assert outcome.failure_reason is None


def test_the_map_endpoint_is_never_called_for_a_moved_domain(pack):
    """**Assertion changed 2026-08-27, deliberately.**

    This previously asserted the map call was made against the destination
    rather than the old domain, which was the right fix for the question it
    was asking. Step 0 removes the question: a moved domain stops before any
    paid path, so the correct number of map calls is zero. Mapping the
    destination buys URLs for a company this row is not about.
    """
    def moved(request: httpx.Request) -> httpx.Response:
        if request.url.host in ("old.com", "www.old.com"):
            return httpx.Response(301, headers={"Location": "https://new.com/"})
        if request.url.host == "new.com" and request.url.path == "/":
            return httpx.Response(200, text="<html><body><p>hello</p></body></html>")
        return httpx.Response(404, text="not found")

    mapped: list[str] = []

    def fake_map(domain: str, *, limit: int) -> list[str]:
        mapped.append(domain)
        return []

    with httpx.Client(transport=httpx.MockTransport(moved), follow_redirects=True) as client:
        outcome = discover("old.com", pack, client=client, map_urls=fake_map)

    assert mapped == []
    assert outcome.exit_step == 0
    assert outcome.referred_domain == "new.com"


def test_an_incomplete_certificate_chain_is_retried_unverified(pack):
    """lensara.com: serves 200 with 58 links, refused on an incomplete chain.

    A server that omits its intermediate certificate is repaired by a browser
    fetching the issuer named in the AIA extension. httpx does not, so the
    domain is written off as unreachable and takes a paid map call with it.
    """
    def refuses_verification(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
            "unable to get local issuer certificate",
            request=request,
        )

    def serves_fine(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/":
            return httpx.Response(404, text="not found")
        return httpx.Response(
            200,
            text=(
                "<html><nav>"
                '<a href="/about">About</a>'
                '<a href="/products">Products</a>'
                '<a href="/quality">Quality</a>'
                "</nav></html>"
            ),
        )

    transport = TlsFallbackTransport(
        httpx.MockTransport(refuses_verification), httpx.MockTransport(serves_fine)
    )
    with httpx.Client(transport=transport, follow_redirects=True) as client:
        outcome = discover("badchain.com", pack, client=client)

    assert len(outcome.pages) >= MIN_PAGES
    assert outcome.discovery_failed is False
    # The site is reachable, but only because we stopped checking, and the
    # manifest has to say so.
    assert outcome.tls_unverified_hosts == ["badchain.com", "www.badchain.com"]
    assert any(a.tls_unverified for a in outcome.attempts if a.kind == "homepage")


def test_only_a_verification_failure_is_retried(pack):
    """A timeout must not silently downgrade into an unverified fetch."""
    request = httpx.Request("GET", "https://x.com/")
    assert _is_tls_verify_error(
        httpx.ConnectError(
            "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed", request=request
        )
    )
    assert not _is_tls_verify_error(httpx.ReadTimeout("timed out", request=request))
    assert not _is_tls_verify_error(httpx.ConnectError("connection refused", request=request))

    def times_out(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=req)

    reached = []

    def must_not_run(req: httpx.Request) -> httpx.Response:
        reached.append(str(req.url))
        return httpx.Response(200, text="<html></html>")

    transport = TlsFallbackTransport(
        httpx.MockTransport(times_out), httpx.MockTransport(must_not_run)
    )
    with httpx.Client(transport=transport, follow_redirects=True) as client:
        outcome = discover("slow.com", pack, client=client)

    assert reached == []
    assert outcome.discovery_failed is True
    assert not any(a.tls_unverified for a in outcome.attempts)


def test_a_healthy_site_is_never_marked_unverified(result):
    assert result.tls_unverified_hosts == []
    assert not any(attempt.tls_unverified for attempt in result.attempts)
