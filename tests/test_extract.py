"""Phase 5: the verbatim check, the assembly, and what happens when a page fails.

Every test here runs against a transport double, so the whole extraction path
is exercised with no key and no spend. The one thing a double cannot tell you
is whether a real model returns findings worth having; that is what the live
smoke test and the gate over the stored run are for.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from cleanroom.config import LLMSettings, SectorPack
from cleanroom.extract import (
    FINDING_FIELDS,
    MAX_PAGE_CHARS,
    MIN_QUOTE_CHARS,
    DomainExtraction,
    ManifestPage,
    PageFindings,
    PartWriter,
    build_system_prompt,
    build_user_prompt,
    canonical,
    cost_estimate,
    count_findings,
    crawl_dates,
    crawled_domains,
    default_run_id,
    existing_pages,
    extract_all,
    extract_page,
    has_evidence,
    is_verbatim,
    latest_crawl_date,
    manifest_pages,
    read_records,
    report_path,
    verify,
)
from cleanroom.llm import LLMClient
from cleanroom.models import ExtractionRecord, Finding
from cleanroom.store import LocalStore, extractions_path, raw_manifest_path

CRAWL_DATE = date(2026, 8, 7)


# ---------------------------------------------------------------------------
# Fixtures: a store holding one crawled domain, and a scripted provider
# ---------------------------------------------------------------------------


PAGE_MARKDOWN = """\
# Precision Molding

Acme Medical is an **ISO 13485** certified contract manufacturer.
Our Akron facility includes 12,000 square feet of ISO Class 7 cleanroom space,
with [gowning rooms](/facility/gowning) and HEPA-filtered laminar flow.

We produce balloon catheters and guidewires, and offer validated ethylene
oxide sterilisation.
"""

HOMEPAGE_MARKDOWN = "# Acme Medical\n\nWe make things well.\n"


def findings_payload(**overrides) -> dict:
    payload = {
        "iso_class_mentions": [],
        "cleanroom_terms": [],
        "sterilization_methods": [],
        "products_named": [],
        "manufacturing_language": [],
        "trap_terms_present": [],
    }
    payload.update(overrides)
    return payload


def finding(quote: str, claim: str = "a claim", confidence: float = 0.9) -> dict:
    return {"claim": claim, "quote": quote, "confidence": confidence}


@pytest.fixture
def crawled_store(tmp_path: Path) -> LocalStore:
    """A store with one domain, two stored pages and one failed page."""
    store = LocalStore(tmp_path / "data")
    store.write_markdown("raw/acme.com/2026-08-07/aaaa.md", PAGE_MARKDOWN)
    store.write_markdown("raw/acme.com/2026-08-07/bbbb.md", HOMEPAGE_MARKDOWN)
    store.write_json(
        raw_manifest_path("acme.com", CRAWL_DATE),
        {
            "crawl_date": "2026-08-07",
            "discovery": {"domain": "acme.com"},
            "fetch": {
                "domain": "acme.com",
                "pages": [
                    {
                        "url": "https://acme.com/",
                        "page_type": "homepage",
                        "ok": True,
                        "stored_path": "raw/acme.com/2026-08-07/bbbb.md",
                        "word_count": 6,
                    },
                    {
                        "url": "https://acme.com/capabilities",
                        "page_type": "capabilities",
                        "ok": True,
                        "stored_path": "raw/acme.com/2026-08-07/aaaa.md",
                        "word_count": 48,
                    },
                    {
                        "url": "https://acme.com/gone",
                        "page_type": "about",
                        "ok": False,
                        "stored_path": None,
                        "word_count": 0,
                    },
                ],
            },
        },
    )
    return store


class Provider:
    """Replays one response per call, cycling on the last.

    `by_url` keys responses off the URL in the user prompt instead, which is
    what a concurrent run needs: pages complete in whatever order the pool
    finishes them, so positional scripting tests the scheduler rather than the
    code under test.
    """

    def __init__(self, *payloads: dict | httpx.Response, by_url: dict[str, dict] | None = None):
        self.payloads = list(payloads)
        self.by_url = by_url or {}
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.by_url:
            prompt = json.loads(request.content)["messages"][1]["content"]
            item = next(
                (payload for url, payload in self.by_url.items() if f"URL: {url}\n" in prompt),
                findings_payload(),
            )
        else:
            item = self.payloads[min(len(self.requests) - 1, len(self.payloads) - 1)]
        if isinstance(item, httpx.Response):
            return item
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(item)}}],
                "usage": {
                    "prompt_tokens": 5000,
                    "completion_tokens": 200,
                    "prompt_tokens_details": {"cached_tokens": 4000},
                },
            },
        )

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(request.content) for request in self.requests]


def client_for(provider: Provider) -> LLMClient:
    settings = LLMSettings(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        max_attempts=2,
    )
    return LLMClient(settings, client=httpx.Client(transport=httpx.MockTransport(provider.handler)))


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------


class TestCanonical:
    def test_collapses_whitespace_and_case(self):
        assert canonical("  ISO   Class\n7  ") == "iso class 7"

    def test_strips_markdown_emphasis(self):
        assert canonical("**ISO Class 7** cleanroom") == "iso class 7 cleanroom"

    def test_keeps_link_text_and_drops_the_target(self):
        assert canonical("[gowning rooms](/facility/gowning)") == "gowning rooms"

    def test_keeps_image_alt_text_and_drops_the_url(self):
        """Alt text is authored page content, and sometimes the only copy of it.

        A certifications page that renders its certificates as logos states the
        certificate names nowhere else. arterismedical.com lost its whole
        certification profile, and a real "ISO Class 8 Clean Room Facility"
        headline, to an earlier version of this that deleted images whole.
        """
        assert canonical("before ![a cleanroom](/img/cr.png) after") == "before a cleanroom after"

    def test_collapses_a_thumbnail_wrapped_in_a_link(self):
        """`[![alt](img)](href)` has to collapse from the inside out."""
        markdown = "[![Opening of 9,000 Sq. Ft. ISO Class 8 Clean Room](/t.png)](/news/1)"
        assert canonical(markdown) == "opening of 9,000 sq. ft. iso class 8 clean room"

    def test_folds_typography(self):
        assert canonical("our cleanroom’s “class 7”") == 'our cleanroom\'s "class 7"'

    def test_strips_table_pipes_and_headings(self):
        assert canonical("| ## Class 7 | 12,000 sq ft |") == "class 7 12,000 sq ft"

    def test_strips_list_markers_so_a_flattened_list_matches(self):
        """A bulleted list quoted as the run of text a reader sees.

        Measured on the first goldset pass: a third of all dropped quotes were
        capability lists the model had flattened. The items are contiguous page
        content in the order the page has them, so the strictness was in the
        syntax and not in the question the check exists to answer.
        """
        page = canonical("Services:\n- shape setting\n- tipping\n- micro-molding\n")
        assert is_verbatim("shape setting tipping micro-molding", page)

    def test_strips_numbered_and_bulleted_markers_alike(self):
        page = canonical("1. gowning\n2. airlock\n* HEPA filtration\n+ laminar flow\n")
        assert is_verbatim("gowning airlock", page)
        assert is_verbatim("HEPA filtration laminar flow", page)

    def test_list_markers_do_not_let_items_be_reordered(self):
        """Contiguity still holds. Only the syntax is tolerated."""
        page = canonical("- shape setting\n- tipping\n- crimping\n")
        assert not is_verbatim("crimping shape setting", page)
        assert not is_verbatim("shape setting crimping", page)

    def test_mid_word_hyphens_are_untouched(self):
        page = canonical("we offer micro-molding and over-moulding")
        assert is_verbatim("micro-molding", page)
        assert not is_verbatim("micro molding", page)

    def test_strips_trademark_marks(self):
        page = canonical("the AVEIR™ DR Leadless Pacemaker and Navitor® TAVI")
        assert is_verbatim("AVEIR DR Leadless Pacemaker", page)
        assert is_verbatim("Navitor TAVI", page)

    def test_keeps_punctuation_that_carries_meaning(self):
        # Numbers and units are quoted verbatim by the prompt, so a canonical
        # form that ate commas or decimals would make "12,000" match "12000".
        assert canonical("12,000 sq ft") == "12,000 sq ft"
        assert "12000" not in canonical("12,000 sq ft")


class TestIsVerbatim:
    @pytest.fixture
    def page(self) -> str:
        return canonical(PAGE_MARKDOWN)

    def test_exact_span_matches(self, page):
        assert is_verbatim("12,000 square feet of ISO Class 7 cleanroom space", page)

    def test_case_and_whitespace_differences_are_tolerated(self, page):
        assert is_verbatim("iso class 7   CLEANROOM   space", page)

    def test_markdown_emphasis_is_tolerated(self, page):
        assert is_verbatim("ISO 13485 certified contract manufacturer", page)

    def test_link_text_is_tolerated(self, page):
        assert is_verbatim("gowning rooms and HEPA-filtered laminar flow", page)

    def test_paraphrase_is_rejected(self, page):
        assert not is_verbatim("twelve thousand square feet of ISO 7 cleanroom", page)

    def test_stitched_quote_is_rejected(self, page):
        assert not is_verbatim("Acme Medical produces balloon catheters", page)

    def test_invention_is_rejected(self, page):
        assert not is_verbatim("we operate three ISO 5 cleanrooms in Tijuana", page)

    def test_too_short_is_rejected(self, page):
        assert not is_verbatim(".", page)
        assert len(canonical("ab")) < MIN_QUOTE_CHARS


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class TestVerify:
    def test_keeps_findings_whose_quotes_are_on_the_page(self):
        findings = PageFindings(
            cleanroom_terms=[Finding(**finding("HEPA-filtered laminar flow"))]
        )
        kept, drops = verify(findings, PAGE_MARKDOWN)
        assert len(kept.cleanroom_terms) == 1
        assert drops.total == 0

    def test_drops_findings_whose_quotes_are_not(self):
        findings = PageFindings(
            cleanroom_terms=[
                Finding(**finding("HEPA-filtered laminar flow")),
                Finding(**finding("our ISO 5 sterile fill line")),
            ]
        )
        kept, drops = verify(findings, PAGE_MARKDOWN)
        assert [f.quote for f in kept.cleanroom_terms] == ["HEPA-filtered laminar flow"]
        assert drops.quotes == 1

    def test_every_finding_field_is_checked(self):
        invented = [Finding(**finding("nothing on this page says this"))]
        findings = PageFindings(**{name: invented for name in FINDING_FIELDS})
        kept, drops = verify(findings, PAGE_MARKDOWN)
        assert drops.quotes == len(FINDING_FIELDS)
        assert all(getattr(kept, name) == [] for name in FINDING_FIELDS)

    def test_products_must_appear_on_the_page(self):
        findings = PageFindings(products_named=["balloon catheters", "hip implants"])
        kept, drops = verify(findings, PAGE_MARKDOWN)
        assert kept.products_named == ["balloon catheters"]
        assert drops.products == 1

    def test_products_are_deduped_case_insensitively(self):
        findings = PageFindings(products_named=["guidewires", "Guidewires"])
        kept, _ = verify(findings, PAGE_MARKDOWN)
        assert kept.products_named == ["guidewires"]

    def test_products_are_not_matched_against_the_taxonomy(self):
        # The taxonomy is deliberately absent from this stage. A product the
        # pack has never heard of survives as long as the page names it.
        findings = PageFindings(products_named=["Precision Molding"])
        kept, drops = verify(findings, PAGE_MARKDOWN)
        assert kept.products_named == ["Precision Molding"]
        assert drops.products == 0

    def test_trap_terms_are_not_matched_against_the_pack(self):
        """Certifications are an unbounded tail, so the field stays open.

        A closed vocabulary threw away 88 real certifications that were plainly
        on their pages, for no safety benefit: the guard that catches an
        invented one is page containment, not pack membership. Matching moves
        to classify time, where changing the list costs a re-classify rather
        than a re-extract.
        """
        page = "We hold AS9100D and ITAR registration."
        kept, drops = verify(PageFindings(trap_terms_present=["AS9100D", "ITAR"]), page)
        assert kept.trap_terms_present == ["AS9100D", "ITAR"]
        assert drops.trap_terms == 0

    def test_trap_terms_absent_from_the_page_are_still_dropped(self):
        """The guard that matters is retained.

        "FDA registered" is the certification a model is most likely to supply
        from sector knowledge rather than from the page: 25 of the 125 drops on
        the first goldset pass were exactly this.
        """
        kept, drops = verify(PageFindings(trap_terms_present=["FDA registered"]), PAGE_MARKDOWN)
        assert kept.trap_terms_present == []
        assert drops.trap_terms == 1

    def test_trap_terms_ignore_spacing_and_hyphens(self):
        """`ISO 9001` and `ISO9001` are the same certification.

        Confined to this field. A quote is evidence and stays strict; a
        certification name is a token whose internal punctuation is set by
        whoever wrote the marketing copy.
        """
        page = "Certified to ISO9001 and FDA-registered since 1998."
        kept, drops = verify(
            PageFindings(trap_terms_present=["ISO 9001", "FDA registered"]), page
        )
        assert kept.trap_terms_present == ["ISO 9001", "FDA registered"]
        assert drops.trap_terms == 0

    def test_trap_terms_keep_the_reported_spelling(self):
        """Uninterpreted, exactly like `products_named`."""
        kept, _ = verify(PageFindings(trap_terms_present=["ISO 13485"]), PAGE_MARKDOWN)
        assert kept.trap_terms_present == ["ISO 13485"]

    def test_trap_terms_are_deduped(self):
        page = "ISO 13485 certified. ISO13485 since 2004."
        kept, _ = verify(PageFindings(trap_terms_present=["ISO 13485", "ISO13485"]), page)
        assert kept.trap_terms_present == ["ISO 13485"]

    def test_loose_matching_does_not_leak_into_quotes(self):
        """The tolerance is scoped to certification names and nowhere else."""
        page = "we operate a class 7 cleanroom"
        kept, drops = verify(
            PageFindings(cleanroom_terms=[Finding(**finding("weoperate a class7 cleanroom"))]),
            page,
        )
        assert kept.cleanroom_terms == []
        assert drops.quotes == 1


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_clears_the_caching_minimum(self, pack: SectorPack):
        """Below 4,096 tokens Haiku caches nothing and says nothing about it.

        `llm.py` estimates tokens as chars/4 and warns on that basis, so the
        assertion uses the same measure the warning does. This stage is 90% of
        all calls; a prompt that silently stops caching is a bill that silently
        quadruples.
        """
        spec_minimum = 4096
        assert len(build_system_prompt(pack)) / 4 > spec_minimum * 1.1

    def test_carries_the_evidence_and_trap_vocabulary(self, pack: SectorPack):
        prompt = build_system_prompt(pack)
        for term in pack.evidence_terms:
            assert term.term in prompt
        for term in pack.trap_terms:
            assert term.term in prompt
        for term in pack.outsourcing_cues:
            assert term.term in prompt

    def test_withholds_the_product_taxonomy(self, pack: SectorPack):
        """The taxonomy is never rendered into the prompt.

        This is what keeps `products_named` uninterpreted: a model shown a list
        of "products that indicate a cleanroom" starts reporting the ones on the
        list and dropping the ones that are not, and then changing the taxonomy
        means re-extracting every page instead of re-classifying.

        The check is against the vocabulary section, which is where the pack is
        rendered. Individual product names still appear in the worked examples,
        because an example of a medical device catalogue with no medical devices
        in it teaches nothing.
        """
        from cleanroom.extract import _vocabulary

        vocabulary = _vocabulary(pack).casefold()
        for term in [*pack.positive_products, *pack.negative_products]:
            for form in term.surface_forms:
                assert form.casefold() not in vocabulary, form

        prompt = build_system_prompt(pack)
        assert "positive_products" not in prompt
        assert "negative_products" not in prompt

    def test_the_examples_do_not_correlate_products_with_cleanrooms(self, pack: SectorPack):
        """Worked examples name products from both halves of the taxonomy.

        If every example product were a catheter, the prompt would be teaching
        the correlation it is supposed to stay out of. The wheelchair example
        exists to say "list these too, it is not your call".
        """
        prompt = build_system_prompt(pack).casefold()
        assert any(term.term.casefold() in prompt for term in pack.positive_products)
        assert any(term.term.casefold() in prompt for term in pack.negative_products)

    def test_tracks_pack_edits(self, pack: SectorPack):
        edited = pack.model_copy(deep=True)
        edited.evidence_terms[0].term = "ISO 14644-9000"
        assert "ISO 14644-9000" in build_system_prompt(edited)

    def test_user_prompt_carries_the_page_and_its_identity(self):
        page = ManifestPage("acme.com", "https://acme.com/x", "capabilities", "p.md", 10)
        prompt = build_user_prompt(page, PAGE_MARKDOWN)
        assert "https://acme.com/x" in prompt
        assert "capabilities" in prompt
        assert PAGE_MARKDOWN in prompt


# ---------------------------------------------------------------------------
# One page
# ---------------------------------------------------------------------------


class TestExtractPage:
    @pytest.fixture
    def page(self) -> ManifestPage:
        return ManifestPage(
            domain="acme.com",
            url="https://acme.com/capabilities",
            page_type="capabilities",
            stored_path="raw/acme.com/2026-08-07/aaaa.md",
            word_count=48,
        )

    def test_assembles_the_record_from_the_manifest(self, pack, page):
        provider = Provider(findings_payload(products_named=["balloon catheters"]))
        record, outcome, _ = extract_page(
            client_for(provider), pack, page, PAGE_MARKDOWN, crawl_date=CRAWL_DATE
        )
        assert record.domain == "acme.com"
        assert record.url == "https://acme.com/capabilities"
        assert record.page_type == "capabilities"
        assert record.crawl_date == CRAWL_DATE
        assert record.pack_version == pack.version
        assert record.page_word_count == 48
        assert outcome.ok

    def test_does_not_ask_the_model_for_the_fields_it_already_knows(self, pack, page):
        provider = Provider(findings_payload())
        extract_page(client_for(provider), pack, page, PAGE_MARKDOWN, crawl_date=CRAWL_DATE)
        schema = provider.bodies[0]["response_format"]["json_schema"]["schema"]
        assert set(schema["properties"]) == {
            "iso_class_mentions",
            "cleanroom_terms",
            "sterilization_methods",
            "products_named",
            "manufacturing_language",
            "trap_terms_present",
        }

    def test_marks_the_system_prompt_for_caching(self, pack, page):
        provider = Provider(findings_payload())
        extract_page(client_for(provider), pack, page, PAGE_MARKDOWN, crawl_date=CRAWL_DATE)
        system = provider.bodies[0]["messages"][0]["content"]
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_unverifiable_quotes_never_reach_the_record(self, pack, page):
        provider = Provider(
            findings_payload(
                cleanroom_terms=[
                    finding("HEPA-filtered laminar flow"),
                    finding("our four ISO 5 suites in Galway"),
                ]
            )
        )
        record, outcome, _ = extract_page(
            client_for(provider), pack, page, PAGE_MARKDOWN, crawl_date=CRAWL_DATE
        )
        assert [f.quote for f in record.cleanroom_terms] == ["HEPA-filtered laminar flow"]
        assert outcome.quotes_dropped == 1
        assert outcome.findings_kept == 1

    def test_long_pages_are_truncated_and_flagged(self, pack, page):
        provider = Provider(findings_payload())
        huge = "word " * (MAX_PAGE_CHARS // 2)
        _, outcome, _ = extract_page(
            client_for(provider), pack, page, huge, crawl_date=CRAWL_DATE
        )
        assert outcome.truncated
        assert outcome.chars_sent <= MAX_PAGE_CHARS + 32

    def test_quotes_are_verified_against_what_was_sent(self, pack, page):
        """A quote from beyond the truncation point cannot be verified,
        because the model was never shown it."""
        tail = "our ISO Class 5 aseptic fill suite"
        huge = ("word " * (MAX_PAGE_CHARS // 4)) + tail
        provider = Provider(findings_payload(cleanroom_terms=[finding(tail)]))
        record, outcome, _ = extract_page(
            client_for(provider), pack, page, huge, crawl_date=CRAWL_DATE
        )
        assert record.cleanroom_terms == []
        assert outcome.quotes_dropped == 1

    def test_usage_and_cost_are_recorded(self, pack, page):
        provider = Provider(findings_payload())
        _, outcome, _ = extract_page(
            client_for(provider), pack, page, PAGE_MARKDOWN, crawl_date=CRAWL_DATE
        )
        assert outcome.cache_read_tokens == 4000
        assert outcome.input_tokens == 1000  # prompt_tokens less the cached read
        assert outcome.cost_usd > 0

    def test_investors_pages_are_windowed_around_keyword_hits_not_head_truncated(self, pack):
        """HANDOFF.md item 7: a 10-K's decisive sentence can sit anywhere in a
        100+ page filing. Head-truncation (the default for every other page
        type) would keep the front matter and lose it; investors pages get
        excerpted around wherever the pack's own vocabulary actually hits.
        """
        investors_page = ManifestPage(
            domain="nvea.com",
            url="https://nvea.com/investors/sec-filings/10-k.pdf",
            page_type="investors",
            stored_path="raw/nvea.com/2026-08-07/aaaa.md",
            word_count=50_000,
        )
        tail = "Wafer production is a cleanroom area."
        huge = ("Item 1. Business. " * (MAX_PAGE_CHARS // 10)) + tail
        assert len(huge) > MAX_PAGE_CHARS, "fixture must actually exceed the truncation floor"

        provider = Provider(findings_payload(cleanroom_terms=[finding(tail)]))
        record, outcome, _ = extract_page(
            client_for(provider), pack, investors_page, huge, crawl_date=CRAWL_DATE
        )

        sent = provider.bodies[0]["messages"][1]["content"]
        assert tail in sent, "the decisive sentence must survive into what the model saw"
        assert [f.quote for f in record.cleanroom_terms] == [tail]
        assert outcome.cost_usd > 0

    def test_investors_pages_with_no_keyword_hits_skip_the_llm_call(self, pack):
        """A filing with nothing in it worth reading is not worth paying to
        have the model re-confirm silence. The local scan is a pre-filter,
        not a replacement for the model's judgment when something did hit.
        """
        investors_page = ManifestPage(
            domain="nvea.com",
            url="https://nvea.com/investors/sec-filings/10-k.pdf",
            page_type="investors",
            stored_path="raw/nvea.com/2026-08-07/aaaa.md",
            word_count=50_000,
        )
        huge = "Item 1. Business. Risk factors and forward-looking statements. " * (
            MAX_PAGE_CHARS // 30
        )
        assert len(huge) > MAX_PAGE_CHARS

        provider = Provider(findings_payload())
        record, outcome, completion = extract_page(
            client_for(provider), pack, investors_page, huge, crawl_date=CRAWL_DATE
        )

        assert provider.requests == []
        assert completion is None
        assert record.cleanroom_terms == []
        assert record.iso_class_mentions == []
        assert outcome.cost_usd == 0.0
        assert outcome.local_scan_only is True

    def test_carries_the_redirect_fact_onto_the_record(self, pack):
        page = ManifestPage(
            domain="acme.com",
            url="https://acquirer.com/",
            page_type="homepage",
            stored_path="raw/acme.com/2026-08-07/aaaa.md",
            word_count=6,
            resolved_domain="acquirer.com",
            redirected_off_domain=True,
        )
        provider = Provider(findings_payload())
        record, _, _ = extract_page(
            client_for(provider), pack, page, HOMEPAGE_MARKDOWN, crawl_date=CRAWL_DATE
        )
        assert record.resolved_domain == "acquirer.com"
        assert record.redirected_off_domain is True


# ---------------------------------------------------------------------------
# Reading the manifest
# ---------------------------------------------------------------------------


class TestManifest:
    def test_reads_the_fetch_block(self, crawled_store):
        pages = manifest_pages(crawled_store, "acme.com", CRAWL_DATE)
        assert [page.url for page in pages] == [
            "https://acme.com/",
            "https://acme.com/capabilities",
        ]
        assert pages[1].page_type == "capabilities"
        assert pages[1].stored_path == "raw/acme.com/2026-08-07/aaaa.md"

    def test_skips_pages_that_did_not_come_back(self, crawled_store):
        urls = [page.url for page in manifest_pages(crawled_store, "acme.com", CRAWL_DATE)]
        assert "https://acme.com/gone" not in urls

    def test_a_missing_manifest_raises(self, crawled_store):
        from cleanroom.store import ObjectNotFoundError

        with pytest.raises(ObjectNotFoundError):
            manifest_pages(crawled_store, "nowhere.com", CRAWL_DATE)

    def test_lists_crawl_dates_and_domains(self, crawled_store):
        assert crawl_dates(crawled_store) == ["2026-08-07"]
        assert latest_crawl_date(crawled_store) == "2026-08-07"
        assert crawled_domains(crawled_store, CRAWL_DATE) == ["acme.com"]

    def test_an_empty_store_has_no_crawl_date(self, tmp_path):
        assert latest_crawl_date(LocalStore(tmp_path / "empty")) is None

    def test_reads_the_redirect_fact(self, tmp_path):
        """`resolved_domain`/`redirected_off_domain` are Phase 4's fact
        (carried from discovery), read here the same way page.word_count is.
        """
        store = LocalStore(tmp_path / "data")
        store.write_markdown("raw/acme.com/2026-08-07/aaaa.md", "# Acquirer\n")
        store.write_json(
            raw_manifest_path("acme.com", CRAWL_DATE),
            {
                "fetch": {
                    "domain": "acme.com",
                    "resolved_domain": "acquirer.com",
                    "redirected_off_domain": True,
                    "pages": [
                        {
                            "url": "https://acquirer.com/",
                            "page_type": "homepage",
                            "ok": True,
                            "stored_path": "raw/acme.com/2026-08-07/aaaa.md",
                            "word_count": 2,
                        },
                    ],
                },
            },
        )
        pages = manifest_pages(store, "acme.com", CRAWL_DATE)
        assert pages[0].resolved_domain == "acquirer.com"
        assert pages[0].redirected_off_domain is True

    def test_the_redirect_fact_defaults_false_and_none(self, crawled_store):
        pages = manifest_pages(crawled_store, "acme.com", CRAWL_DATE)
        assert pages[0].resolved_domain is None
        assert pages[0].redirected_off_domain is False


# ---------------------------------------------------------------------------
# A batch
# ---------------------------------------------------------------------------


class TestExtractAll:
    def test_writes_one_record_per_stored_page(self, pack, crawled_store):
        provider = Provider(findings_payload())
        run = extract_all(
            ["acme.com"], pack, crawled_store, client_for(provider),
            crawl_date=CRAWL_DATE, run_id="testrun", concurrency=1,
        )
        assert run.pages_extracted == 2
        assert run.records_written == 2
        records = read_records(crawled_store, pack.version, "testrun")
        assert {record.url for record in records} == {
            "https://acme.com/",
            "https://acme.com/capabilities",
        }

    def test_records_round_trip_as_extraction_records(self, pack, crawled_store):
        provider = Provider(
            by_url={
                "https://acme.com/capabilities": findings_payload(
                    cleanroom_terms=[finding("HEPA-filtered laminar flow")],
                    products_named=["guidewires"],
                    trap_terms_present=["ISO 13485"],
                )
            }
        )
        extract_all(
            ["acme.com"], pack, crawled_store, client_for(provider),
            crawl_date=CRAWL_DATE, run_id="testrun", concurrency=1,
        )
        records = read_records(crawled_store, pack.version, "testrun")
        assert all(isinstance(record, ExtractionRecord) for record in records)
        rich = [record for record in records if record.cleanroom_terms]
        assert rich and rich[0].trap_terms_present == ["ISO 13485"]

    def test_writes_to_the_pack_and_run_keyed_path(self, pack, crawled_store):
        provider = Provider(findings_payload())
        run = extract_all(
            ["acme.com"], pack, crawled_store, client_for(provider),
            crawl_date=CRAWL_DATE, run_id="testrun", concurrency=1,
        )
        assert run.parts == [extractions_path(pack.version, "testrun", 0)]
        assert crawled_store.exists(report_path(pack.version, "testrun"))

    def test_a_failed_page_writes_no_record(self, pack, crawled_store):
        """An empty record and a page that said nothing are indistinguishable
        downstream, so a failure has to be absent rather than empty."""
        provider = Provider(httpx.Response(500, text="upstream is down"))
        run = extract_all(
            ["acme.com"], pack, crawled_store, client_for(provider),
            crawl_date=CRAWL_DATE, run_id="testrun", concurrency=1,
        )
        assert run.pages_failed == 2
        assert run.records_written == 0
        assert read_records(crawled_store, pack.version, "testrun") == []

    def test_a_malformed_request_halts_the_run(self, pack, crawled_store):
        """A 400 will fail identically on all 450 remaining pages."""
        provider = Provider(httpx.Response(400, text="bad request"))
        run = extract_all(
            ["acme.com"], pack, crawled_store, client_for(provider),
            crawl_date=CRAWL_DATE, run_id="testrun", concurrency=1,
        )
        assert run.stopped_reason is not None
        assert "400" in run.stopped_reason

    def test_a_domain_with_no_manifest_is_reported_not_raised(self, pack, crawled_store):
        provider = Provider(findings_payload())
        run = extract_all(
            ["acme.com", "nowhere.com"], pack, crawled_store, client_for(provider),
            crawl_date=CRAWL_DATE, run_id="testrun", concurrency=1,
        )
        missing = [d for d in run.domains if d.domain == "nowhere.com"]
        assert missing and "no manifest" in (missing[0].error or "")
        assert run.pages_extracted == 2

    def test_resuming_a_run_skips_what_is_already_stored(self, pack, crawled_store):
        provider = Provider(findings_payload())
        first = extract_all(
            ["acme.com"], pack, crawled_store, client_for(provider),
            crawl_date=CRAWL_DATE, run_id="testrun", concurrency=1,
        )
        assert first.records_written == 2

        second = extract_all(
            ["acme.com"], pack, crawled_store, client_for(provider),
            crawl_date=CRAWL_DATE, run_id="testrun", concurrency=1,
        )
        assert second.pages_skipped == 2
        assert second.records_written == 0
        assert len(read_records(crawled_store, pack.version, "testrun")) == 2

    def test_a_retry_fills_in_only_the_failed_pages(self, pack, crawled_store):
        down = Provider(httpx.Response(503, text="unavailable"))
        failed_run = extract_all(
            ["acme.com"], pack, crawled_store, client_for(down),
            crawl_date=CRAWL_DATE, run_id="testrun", concurrency=1,
        )
        assert failed_run.records_written == 0

        up = Provider(findings_payload())
        retry = extract_all(
            ["acme.com"], pack, crawled_store, client_for(up),
            crawl_date=CRAWL_DATE, run_id="testrun", concurrency=1,
        )
        assert retry.records_written == 2

    def test_the_budget_stops_the_run(self, pack, crawled_store):
        provider = Provider(findings_payload())
        run = extract_all(
            ["acme.com"], pack, crawled_store, client_for(provider),
            crawl_date=CRAWL_DATE, run_id="testrun", concurrency=1,
            budget_usd=0.0000001,
        )
        assert run.stopped_reason is not None
        assert run.pages_extracted < 2

    def test_the_run_report_carries_the_drop_counts(self, pack, crawled_store):
        provider = Provider(
            findings_payload(
                cleanroom_terms=[finding("we operate a Class 5 suite in Cork")],
                products_named=["hip implants"],
                trap_terms_present=["ISO 9001"],
            )
        )
        run = extract_all(
            ["acme.com"], pack, crawled_store, client_for(provider),
            crawl_date=CRAWL_DATE, run_id="testrun", concurrency=1,
        )
        assert run.quotes_dropped == 2
        assert run.products_dropped == 2
        assert run.trap_terms_dropped == 2
        assert run.quote_drop_rate == 1.0

        stored = crawled_store.read_json(report_path(pack.version, "testrun"))
        assert stored["quotes_dropped"] == 2

    def test_the_report_names_what_was_dropped(self, pack, crawled_store):
        """A count says the rate changed; only the text says which way to fix it.

        A dropped quote that is plainly on the page means the canonical form is
        too strict. One that reads like a summary means the prompt is not being
        obeyed. Opposite fixes, indistinguishable from a number.
        """
        provider = Provider(
            by_url={
                "https://acme.com/capabilities": findings_payload(
                    cleanroom_terms=[finding("we run four ISO 5 suites in Cork")],
                    products_named=["hip implants"],
                    trap_terms_present=["ISO 9001"],
                )
            }
        )
        run = extract_all(
            ["acme.com"], pack, crawled_store, client_for(provider),
            crawl_date=CRAWL_DATE, run_id="testrun", concurrency=1,
        )
        page = next(p for p in run.pages if p.quotes_dropped)
        assert page.dropped_quotes == ["cleanroom_terms: we run four ISO 5 suites in Cork"]
        assert page.dropped_products == ["hip implants"]
        assert page.dropped_trap_terms == ["ISO 9001"]

    def test_long_dropped_quotes_are_truncated_in_the_report(self, pack, crawled_store):
        from cleanroom.extract import DROP_SAMPLE_CHARS

        provider = Provider(
            by_url={
                "https://acme.com/capabilities": findings_payload(
                    cleanroom_terms=[finding("invented " * 60)]
                )
            }
        )
        run = extract_all(
            ["acme.com"], pack, crawled_store, client_for(provider),
            crawl_date=CRAWL_DATE, run_id="testrun", concurrency=1,
        )
        page = next(p for p in run.pages if p.quotes_dropped)
        assert len(page.dropped_quotes[0]) < DROP_SAMPLE_CHARS + 40

    def test_per_domain_rows_arrive_as_the_run_proceeds(self, pack, crawled_store):
        provider = Provider(findings_payload())
        seen: list[DomainExtraction] = []
        extract_all(
            ["acme.com"], pack, crawled_store, client_for(provider),
            crawl_date=CRAWL_DATE, run_id="testrun", concurrency=1,
            on_domain=seen.append,
        )
        assert [summary.domain for summary in seen] == ["acme.com"]
        assert seen[0].pages_extracted == 2

    def test_parts_are_flushed_as_the_run_proceeds(self, pack, crawled_store):
        provider = Provider(findings_payload())
        run = extract_all(
            ["acme.com"], pack, crawled_store, client_for(provider),
            crawl_date=CRAWL_DATE, run_id="testrun", concurrency=1,
            records_per_part=1,
        )
        assert len(run.parts) == 2


# ---------------------------------------------------------------------------
# Part writing
# ---------------------------------------------------------------------------


def a_record(url: str) -> ExtractionRecord:
    return ExtractionRecord(
        domain="acme.com",
        url=url,
        page_type="homepage",
        crawl_date=CRAWL_DATE,
        pack_version="medical_v1",
        page_word_count=10,
    )


class TestPartWriter:
    def test_numbering_continues_past_existing_parts(self, local_store):
        local_store.write_jsonl(extractions_path("medical_v1", "r1", 0), [a_record("https://a")])
        writer = PartWriter(local_store, "medical_v1", "r1")
        writer.add(a_record("https://b"))
        writer.flush()
        assert writer.written == [extractions_path("medical_v1", "r1", 1)]

    def test_never_overwrites(self, local_store):
        from cleanroom.store import StoreConflictError

        path = extractions_path("medical_v1", "r1", 0)
        local_store.write_jsonl(path, [a_record("https://a")])
        writer = PartWriter(local_store, "medical_v1", "r1")
        writer.index = 0  # force the collision the numbering normally avoids
        writer.add(a_record("https://b"))
        with pytest.raises(StoreConflictError):
            writer.flush()

    def test_flushing_an_empty_buffer_writes_nothing(self, local_store):
        writer = PartWriter(local_store, "medical_v1", "r1")
        writer.flush()
        assert writer.written == []

    def test_existing_pages_reads_every_part(self, local_store):
        local_store.write_jsonl(extractions_path("medical_v1", "r1", 0), [a_record("https://a")])
        local_store.write_jsonl(extractions_path("medical_v1", "r1", 1), [a_record("https://b")])
        assert existing_pages(local_store, "medical_v1", "r1") == {
            ("acme.com", "https://a"),
            ("acme.com", "https://b"),
        }


# ---------------------------------------------------------------------------
# Odds and ends
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_run_ids_are_path_safe(self):
        from cleanroom.store import extractions_path

        extractions_path("medical_v1", default_run_id())  # raises if not

    def test_count_findings_ignores_the_string_lists(self):
        record = a_record("https://a")
        record.products_named = ["a", "b"]
        record.trap_terms_present = ["ISO 13485"]
        record.cleanroom_terms = [Finding(**finding("HEPA-filtered laminar flow"))]
        assert count_findings(record) == 1

    def test_has_evidence_is_about_environments_not_products(self):
        record = a_record("https://a")
        record.products_named = ["catheters"]
        assert not has_evidence(record)
        record.iso_class_mentions = [Finding(**finding("ISO Class 7"))]
        assert has_evidence(record)

    def test_cost_estimate_prices_the_pages_on_disk(self, pack, crawled_store):
        provider = Provider(findings_payload())
        estimate = cost_estimate(
            crawled_store, ["acme.com"], pack, client_for(provider), crawl_date=CRAWL_DATE
        )
        assert estimate["pages"] == 2
        assert estimate["caching"] == "on"
        assert estimate["cost_usd"] > 0
        assert provider.requests == []  # an estimate calls nothing
