"""Phase 6: aggregation, the taxonomy match, and what reaches a verdict.

Every test here runs against a transport double, so the whole classification
path is exercised with no key and no spend. What a double cannot tell you is
whether a real model weighs the evidence well; that is what the goldset run and
Phase 7b are for.

The tests that matter most are the ones about evidence. A verdict is only worth
having if someone can check it, and the check is that every quote in a record
came out of the extractions rather than out of the model.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from cleanroom.classify import (
    CSV_COLUMNS,
    MAX_QUOTES_PER_DOMAIN,
    THIN_RENDER_WORD_THRESHOLD,
    THIN_SITE_PAGES,
    DomainEvidence,
    DomainVerdict,
    aggregate,
    build_csv,
    build_system_prompt,
    build_user_prompt,
    classify_all,
    classify_domain,
    cost_estimate,
    csv_path,
    existing_domains,
    extraction_runs,
    group_records,
    latest_extraction_run,
    match_products,
    match_trap_terms,
    read_classifications,
    report_path,
    verify_verdict,
)
from cleanroom.config import LLMSettings
from cleanroom.llm import LLMClient
from cleanroom.models import ClassificationRecord, Evidence, ExtractionRecord, Finding
from cleanroom.store import LocalStore, extractions_path

CRAWL_DATE = date(2026, 8, 7)
EXTRACT_RUN = "20260810-164543"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def finding(quote: str, claim: str = "a claim", confidence: float = 0.9) -> Finding:
    return Finding(claim=claim, quote=quote, confidence=confidence)


def record(url: str, page_type: str = "capabilities", **fields) -> ExtractionRecord:
    payload = {
        "domain": "acme.com",
        "url": url,
        "page_type": page_type,
        "crawl_date": CRAWL_DATE,
        "pack_version": "medical_v1",
        "page_word_count": 400,
    }
    payload.update(fields)
    return ExtractionRecord(**payload)


@pytest.fixture
def records() -> list[ExtractionRecord]:
    """One domain, three pages, with evidence spread across them."""
    return [
        record(
            "https://acme.com/",
            "homepage",
            products_named=["Balloon Catheters", "guidewires"],
            trap_terms_present=["ISO 13485"],
        ),
        record(
            "https://acme.com/capabilities",
            "capabilities",
            iso_class_mentions=[finding("12,000 square feet of ISO Class 7 cleanroom space", confidence=0.97)],
            cleanroom_terms=[finding("gowning rooms and HEPA-filtered laminar flow", confidence=0.9)],
            manufacturing_language=[finding("all assembly is performed in house", confidence=0.8)],
            products_named=["balloon catheters"],
            trap_terms_present=["ISO 13485:2016", "ISO 80369-7"],
        ),
        record(
            "https://acme.com/quality",
            "quality",
            sterilization_methods=[finding("validated ethylene oxide sterilisation", confidence=0.85)],
            trap_terms_present=["FDA registered"],
        ),
    ]


@pytest.fixture
def extracted_store(tmp_path: Path, records) -> LocalStore:
    store = LocalStore(tmp_path / "data")
    store.write_jsonl(extractions_path("medical_v1", EXTRACT_RUN, 0), records)
    return store


def verdict_payload(**overrides) -> dict:
    payload = {
        "cleanroom_status": "in_house",
        "iso_class": "ISO Class 7",
        "confidence": 0.95,
        "reasoning": "The capabilities page states the class and the gowning.",
        "evidence": [
            {
                "quote": "12,000 square feet of ISO Class 7 cleanroom space",
                "source_url": "https://acme.com/capabilities",
            }
        ],
        "needs_manual_review": False,
        "entity_type": "manufacturer",
        "referred_domain": None,
    }
    payload.update(overrides)
    return payload


class Provider:
    """Replays one response per call, cycling on the last.

    `by_domain` keys responses off the domain in the user prompt instead, which
    is what a concurrent run needs: domains complete in whatever order the pool
    finishes them, so positional scripting would test the scheduler rather than
    the code under test.
    """

    def __init__(self, *payloads: dict | httpx.Response, by_domain: dict[str, dict] | None = None):
        self.payloads = list(payloads)
        self.by_domain = by_domain or {}
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.by_domain:
            prompt = json.loads(request.content)["messages"][1]["content"]
            item = next(
                (payload for domain, payload in self.by_domain.items() if f"DOMAIN: {domain}\n" in prompt),
                verdict_payload(),
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
                    "prompt_tokens": 6000,
                    "completion_tokens": 400,
                    "prompt_tokens_details": {"cached_tokens": 3700},
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
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_folds_pages_into_one_bundle(self, records, pack):
        bundle = aggregate(records, pack)
        assert bundle.domain == "acme.com"
        assert bundle.pages == 3
        assert bundle.crawl_date == CRAWL_DATE
        assert bundle.page_types == ["homepage", "capabilities", "quality"]
        assert bundle.total_words == 1200
        assert len(bundle.quotes) == 4

    def test_computes_the_median_page_word_count(self, pack):
        records = [
            record("https://acme.com/a", page_word_count=10),
            record("https://acme.com/b", page_word_count=20),
            record("https://acme.com/c", page_word_count=400),
        ]
        bundle = aggregate(records, pack)
        assert bundle.median_page_word_count == 20

    def test_orders_quotes_by_field_priority_then_confidence(self, records, pack):
        """A stated ISO class outranks a cleanroom term outranks the rest.

        The order is what survives trimming on an enterprise domain, so it has
        to put the load-bearing evidence first rather than whichever page the
        crawl happened to store first.
        """
        bundle = aggregate(records, pack)
        assert [quote.field for quote in bundle.quotes] == [
            "iso_class_mentions",
            "cleanroom_terms",
            "sterilization_methods",
            "manufacturing_language",
        ]

    def test_dedupes_products_across_pages(self, records, pack):
        """`Balloon Catheters` and `balloon catheters` are one product."""
        bundle = aggregate(records, pack)
        assert len(bundle.products) == 2
        assert "Balloon Catheters" in bundle.products
        assert "guidewires" in bundle.products

    def test_collapses_certification_spellings_onto_one_pack_term(self, records, pack):
        """`ISO 13485` and `ISO 13485:2016` reach the prompt as one term.

        Two stages, not one. Aggregation dedupes exact repeats across pages;
        the pack match is what folds the version suffixes together. Keeping
        both spellings under the term is deliberate: the model is shown what
        the site actually said, not a tidied version of it.
        """
        bundle = aggregate(records, pack)
        assert set(bundle.trap_matched) == {"ISO 13485", "FDA registered"}
        assert bundle.trap_matched["ISO 13485"] == ["ISO 13485", "ISO 13485:2016"]
        assert bundle.trap_unmatched == ["ISO 80369-7"]

    def test_dedupes_a_certification_repeated_across_pages(self, pack):
        rows = [
            record("https://acme.com/a", trap_terms_present=["ISO 13485"]),
            record("https://acme.com/b", trap_terms_present=["ISO 13485"]),
        ]
        bundle = aggregate(rows, pack)
        assert bundle.trap_matched == {"ISO 13485": ["ISO 13485"]}

    def test_carries_every_source_url(self, records, pack):
        bundle = aggregate(records, pack)
        assert set(bundle.urls) == {
            "https://acme.com/",
            "https://acme.com/capabilities",
            "https://acme.com/quality",
        }

    def test_refuses_an_empty_record_list(self, pack):
        """A domain with no records was never extracted, which is not `unknown`.

        Same rule as Phase 4's withheld manifest: absence means run the stage,
        and a record asserting a verdict over zero pages would be a lie the
        store cannot distinguish from a genuinely thin site.
        """
        with pytest.raises(ValueError):
            aggregate([], pack)

    def test_carries_the_redirect_fact_through(self, pack):
        """A domain that redirected off its own registrable domain at crawl
        time is a fact discovery already knows. Losing it here is how it used
        to die at the Phase 3/4 boundary instead of reaching classify_domain.
        """
        rows = [
            record(
                "https://acquirer.com/",
                resolved_domain="acquirer.com",
                redirected_off_domain=True,
            )
        ]
        bundle = aggregate(rows, pack)
        assert bundle.resolved_domain == "acquirer.com"
        assert bundle.redirected_off_domain is True

    def test_redirect_fields_default_false_and_none(self, records, pack):
        bundle = aggregate(records, pack)
        assert bundle.resolved_domain is None
        assert bundle.redirected_off_domain is False


class TestGroupRecords:
    def test_groups_by_domain_in_first_seen_order(self, records):
        rows = [*records, record("https://other.com/", domain="other.com")]
        grouped = group_records(rows)
        assert list(grouped) == ["acme.com", "other.com"]
        assert len(grouped["acme.com"]) == 3


# ---------------------------------------------------------------------------
# The taxonomy match
# ---------------------------------------------------------------------------


class TestMatchProducts:
    def test_matches_the_term_inside_the_page_name(self, pack):
        """The pack says `catheters`, the page says `Balloon Catheters`."""
        match = match_products(["Balloon Catheters"], pack)
        assert "catheters" in match.positive
        assert match.positive["catheters"] == ["Balloon Catheters"]
        assert match.unmatched == 0

    def test_matches_aliases(self, pack):
        match = match_products(["mobility aids for home use"], pack)
        assert "wheelchairs" in match.negative

    def test_counts_names_that_match_nothing(self, pack):
        match = match_products(["quantum flux widget"], pack)
        assert match.unmatched == 1
        assert not match.positive and not match.negative

    def test_a_name_may_match_both_lists(self, pack):
        """Nothing here breaks the tie; direction is the classifier's problem.

        A company selling both catheters and wheelchairs is a real thing the
        verdict has to face, not a data error for this function to resolve.
        """
        match = match_products(["Balloon Catheters", "powered wheelchairs"], pack)
        assert match.positive and match.negative

    def test_ignores_markdown_and_case(self, pack):
        match = match_products(["**Balloon Catheters**"], pack)
        assert "catheters" in match.positive

    def test_a_singular_page_name_matches_a_plural_pack_term(self, pack):
        """The bug this function shipped with, found on 2026-08-11.

        The pack names a category in the plural and a page names one product in
        the singular, so `catheters` was never inside `peripheral catheter`.
        nuvellis.com listed a peripheral catheter, a dual lumen catheter and a
        blood circuit set, and the taxonomy reported no positive product at all.
        """
        match = match_products(
            ["peripheral catheter", "Dual lumen Extended Length Catheter"], pack
        )
        assert match.positive["catheters"] == [
            "peripheral catheter",
            "Dual lumen Extended Length Catheter",
        ]
        assert match.unmatched == 0

    def test_the_negative_list_singularises_too(self, pack):
        match = match_products(["wheelchair", "hospital bed"], pack)
        assert "wheelchairs" in match.negative
        assert "hospital beds" in match.negative

    def test_a_name_is_listed_once_per_term_it_matches(self, pack):
        """`coronary stent` hits the term `stents` and its own alias.

        The prompt shows this list to the model, so the same name twice under
        one term reads as two products.
        """
        match = match_products(["coronary stent"], pack)
        assert match.positive["stents"] == ["coronary stent"]


class TestMatchTrapTerms:
    def test_matches_ignoring_spaces_and_hyphens(self, pack):
        matched, unmatched = match_trap_terms(["ISO9001", "FDA-registered"], pack)
        assert set(matched) == {"ISO 9001", "FDA registered"}
        assert not unmatched

    def test_keeps_the_long_tail_separate(self, pack):
        """The unlisted certifications are neutral too, and still go to the model.

        The pack names 11 traps and the corpus produced 57. Dropping the tail
        would hide from the classifier exactly the certifications the shape
        rule exists to cover.
        """
        matched, unmatched = match_trap_terms(["ISO 13485", "ISO 14971"], pack)
        assert "ISO 13485" in matched
        assert unmatched == ["ISO 14971"]

    def test_matches_a_versioned_certification(self, pack):
        matched, _ = match_trap_terms(["ISO 13485:2016"], pack)
        assert "ISO 13485" in matched


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_carries_the_label_definitions_verbatim(self, pack):
        """BUILD.md's definitions are the specification of the output.

        Paraphrasing a specification into a prompt is how a pipeline quietly
        starts answering a different question than the one it was built for.
        """
        prompt = build_system_prompt(pack)
        assert "operates its own cleanroom" in prompt
        assert "sells sterile product, manufactures none of it" in prompt
        assert "the product line does not require a cleanroom" in prompt
        assert "the site is too thin to say what the company even does" in prompt

    def test_carries_the_fifth_label(self, pack):
        """likely_in_house: the inference band added for the medical_v2.2 handoff."""
        prompt = build_system_prompt(pack)
        assert "`likely_in_house`" in prompt
        assert "strongly imply a cleanroom" in prompt
        assert "## The five labels" in prompt

    def test_carries_the_edge_cases(self, pack):
        prompt = build_system_prompt(pack)
        assert "HYBRID" in prompt and "resolves to `in_house`" in prompt
        assert "CONTROLLED BUT UNCLASSIFIED" in prompt
        assert "DISTRIBUTOR ONLY resolves to `no_cleanroom_needed`" in prompt
        assert "needs_manual_review: true" in prompt

    def test_states_that_no_cleanroom_needed_is_a_determination(self, pack):
        prompt = build_system_prompt(pack)
        assert "POSITIVE DETERMINATION FROM THE PRODUCT TAXONOMY" in prompt
        assert "`no_cleanroom_needed` is a DETERMINATION." in prompt
        assert "`likely_in_house` is an INFERENCE." in prompt
        assert "`unknown` is ABSENCE OF EVIDENCE." in prompt

    def test_carries_the_standard_number_shape_rule(self, pack):
        """The one line that makes an unbounded trap list finite.

        The pack names 11 traps and 500 domains will produce hundreds of
        certifications. Enumerating them is hopeless, so the prompt generalises
        by shape and the pack list stays illustrative.
        """
        prompt = build_system_prompt(pack)
        assert "ISO 14644" in prompt
        assert "EU GMP Grade" in prompt
        assert "management-system, product-safety or testing standard" in prompt
        assert "ISO 14971" in prompt

    def test_says_traps_are_neutral_in_both_directions(self, pack):
        prompt = build_system_prompt(pack)
        assert "Never subtract points for a trap term. Never add them." in prompt

    def test_carries_the_attribution_rules(self, pack):
        """R20/R21/R22 from medical_v2.2.yaml: subject-level attribution,
        deliberately not implemented in matcher.py (see its decide() docstring)
        because it is grammar-level judgment, which is this prompt's job.
        """
        prompt = build_system_prompt(pack)
        assert "CONTRACT-MANUFACTURER POLARITY" in prompt
        assert "CUSTOMER-ENVIRONMENT SCOPING" in prompt
        assert "ENTITY-ESTABLISHING CUES ARE NOT CLEANROOM EVIDENCE" in prompt

    def test_ivd_negative_carves_out_microfluidic_cartridges(self, pack):
        """v1's blanket `IVD assays -> negative` and v2.2's `microfluidic
        cartridge -> STRONG_INFERRED positive` were two independently-built
        packs that never got reconciled — HANDOFF_v2.2.md never references
        medical_v1.yaml at all. Demonstrated live on `beebix.com`: its FINDER
        point-of-care cartridge is a sealed, precision-assembled device, not
        bulk reagent/kit production, and the two packs disagreed about which
        category it falls in. Fix is granularity, not a pack sync: bulk IVD
        reagents/kits stay negative; sealed cartridges become their own
        positive entry.
        """
        prompt = build_system_prompt(pack)
        assert "microfluidic cartridge" in prompt.lower()
        assert "sealed" in prompt.lower() or "cartridge" in prompt.lower()

    def test_carries_the_multinational_local_facility_test(self, pack):
        """pari.com: a 33-person US sales office's page said "PARI has been
        manufacturing... for 100 years" (the global brand's history) and the
        classifier credited the local entity with it, landing on
        no_cleanroom_needed/manufacturer instead of the adjudicated
        likely_in_house/distributor. Two questions, in order: is this a US
        location at all, and does *this* location — not the brand — run the
        cleanroom. "Office" with no plant/facility content of its own answers
        the second question no, regardless of brand-level claims.
        """
        prompt = build_system_prompt(pack)
        assert "MULTINATIONAL LOCAL-FACILITY TEST" in prompt
        assert "office" in prompt.lower()

    def test_ventilator_console_is_a_negative_product(self, pack):
        """batch-1's ventexlife.com: ventilator consoles are capital equipment
        and breathing circuits are typically bought in, not made in-house.
        Flagged in the 2026-08-19 batch-1 diagnosis (Section F) and never
        implemented — the domain still reaches `likely_in_house` today for
        lack of this term. `imaging hardware` already covers a related but
        distinct capital-equipment case at medium weight; this is its own row
        because ventilator consoles are not imaging devices.
        """
        prompt = build_system_prompt(pack)
        assert "ventilator" in prompt.lower()

    def test_non_sterile_single_patient_use_respiratory_consumables_are_negative(self, pack):
        """The open team question from the 2026-08-19 diagnosis, Section 1 row 33
        and recommended-work item 7, ruled by the user 2026-08-26:
        `mgxdiagnostics.com` is `no_cleanroom_needed`.

        The doc asked the general question, not the per-domain one -- pneumotach
        flow sensors, spirometry mouthpieces, nose clips and spirometry filters
        "are not on any current list in either direction". A single-patient-use
        item that is never labelled sterile is a hygiene article, not a sterile
        barrier product, so it belongs on the negative list.
        """
        prompt = build_system_prompt(pack)
        for term in ("pneumotach", "nose clip", "spirometry"):
            assert term in prompt.lower(), term

    def test_the_respiratory_consumable_rule_is_scoped_to_NON_sterile(self, pack):
        """The guard on the rule above, and the reason it is not written as
        "breathing circuit" or "patient circuit".

        `vaportheq.com` is a CONFIRMED `in_house` lead in Tier C whose product
        line is single-use disposable patient circuits with a STERILE FLUID
        PATH. The 2026-08-19 diagnosis drew the line at exactly that: MGC's
        respiratory consumables are "None are labelled sterile", Vaportheq's
        are. A blanket negative on respiratory disposables would attack a real
        lead, so the terms are narrow and the note carries the scope.
        """
        negatives = {row.term.lower() for row in pack.negative_products}
        for banned in ("breathing circuit", "patient circuit", "respiratory consumable"):
            assert banned not in negatives, (
                f"{banned!r} on the negative list would push vaportheq.com, a "
                f"confirmed in_house lead with sterile-fluid-path circuits, "
                f"toward no_cleanroom_needed"
            )
        prompt = build_system_prompt(pack)
        assert "sterile" in prompt.lower()

    def test_carries_r6_exec_bio_subject_scoping(self, pack):
        """R6 from HANDOFF.md: an outsourcing cue only counts in present tense,
        about the company itself. `infraredix.com` was flipped to `outsourced`
        in batch-1 by an executive biography describing a PRIOR employer; a
        human reviewer refused it, the classifier did not. matcher.py got its
        own fix for this (the suppress_if_preceded_by window on "contract
        manufacturer"), but that never reached the LLM prompt.
        """
        prompt = build_system_prompt(pack)
        assert "PRIOR EMPLOYER" in prompt or "PRIOR-EMPLOYER" in prompt

    def test_carries_all_eight_entity_types(self, pack):
        prompt = build_system_prompt(pack)
        for entity_type in (
            "manufacturer", "contract_manufacturer", "distributor", "design_house",
            "service_lab", "consultancy", "cleanroom_vendor", "acquired_stub",
        ):
            assert f"`{entity_type}`" in prompt, entity_type

    def test_requires_facility_evidence_for_a_positive(self, pack):
        """Decided 2026-08-10, and it is the precision rule the tool turns on.

        Safe only because the target population is manufacturers under 500
        employees, where a company that runs a cleanroom says so. On an
        enterprise, ten pages cannot cover the estate, which is why the rule
        sends those to a flagged `unknown` rather than guessing.
        """
        prompt = build_system_prompt(pack)
        assert "`in_house` REQUIRES FACILITY EVIDENCE" in prompt
        assert "A product catalogue is not facility evidence for `in_house`." in prompt
        assert "`likely_in_house` with\n`needs_manual_review: true`" in prompt

    def test_says_size_and_purpose_do_not_matter(self, pack):
        """The correction that matters most for this customer base.

        The buyer sells cleanroom consumables, so a lab-sized packaging room is
        as much a customer as a production hall. Every other worked example in
        the prompt is a large medical facility, and a model learns from its
        examples, so the rule is stated outright and given its own small case.
        """
        prompt = build_system_prompt(pack)
        assert "SIZE AND PURPOSE DO NOT MATTER" in prompt
        assert "There is no minimum size and no required use." in prompt
        assert "Packaging, kitting, labelling" in prompt
        assert "one small packaging room" in prompt

    def test_outsourced_requires_a_physical_product(self, pack):
        """A consultancy selling hours is `no_cleanroom_needed`, not `outsourced`.

        Both labels are non-leads so this does not change the sales list, but
        collapsing them muddies the population that gets re-run later.
        """
        prompt = build_system_prompt(pack)
        assert "**`outsourced` needs a product.**" in prompt
        assert "sells hours rather than goods is `no_cleanroom_needed`" in prompt

    def test_scopes_the_facility_requirement_to_in_house(self, pack):
        """Added after the first tightening over-corrected.

        Three domains collapsed from a determination into `unknown` because the
        model generalised "be strict about evidence" past the label it was
        written for. `no_cleanroom_needed` and `outsourced` are read off the
        catalogue.
        """
        prompt = build_system_prompt(pack)
        assert "applies to `in_house`\nONLY" in prompt
        assert "`no_cleanroom_needed` is a judgment about the product line, not a lookup." in prompt

    def test_says_an_empty_taxonomy_match_is_normal(self, pack):
        """2,973 product names on the goldset, most matching nothing.

        The model read "no taxonomy hit" as "I cannot tell what they sell" and
        abstained on a chemical blender and an electronics distributor.
        """
        prompt = build_system_prompt(pack)
        assert "Products matching no taxonomy term is the normal case" in prompt

    def test_does_not_let_the_taxonomy_alone_carry_a_positive(self, pack):
        """The worked example exists because the rule is counter-intuitive.

        A model knows catheters are made in cleanrooms. The example is what
        stops it treating that industry fact as a fact about this company.
        """
        prompt = build_system_prompt(pack)
        assert "strong products, no facility evidence anywhere" in prompt

    def test_includes_the_product_taxonomy_the_extractor_never_sees(self, pack):
        """Withheld at extract time, shown here. That is the whole design.

        Keeping it out of the extraction prompt stops the extractor filtering
        its catalogue to the list; showing it here is what lets the taxonomy
        change cost a re-classify instead of a re-extract.
        """
        prompt = build_system_prompt(pack)
        for term in pack.positive_products:
            assert term.term in prompt
        for term in pack.negative_products:
            assert term.term in prompt

    def test_is_long_enough_to_cache(self, pack):
        """Below the provider's threshold the prefix silently never caches."""
        from cleanroom.config import STAGE_MODELS

        assert len(build_system_prompt(pack)) // 4 >= STAGE_MODELS["classify"].cache_min_tokens


class TestUserPrompt:
    def test_shows_each_quote_with_its_source_url(self, records, pack):
        prompt = build_user_prompt(aggregate(records, pack))
        assert '"12,000 square feet of ISO Class 7 cleanroom space"' in prompt
        assert "source_url: https://acme.com/capabilities" in prompt

    def test_reports_the_taxonomy_match_rather_than_the_raw_list(self, records, pack):
        """3,114 names on the goldset, 294 on one domain. Match in code."""
        prompt = build_user_prompt(aggregate(records, pack))
        assert "POSITIVE taxonomy hits" in prompt
        assert "catheters: Balloon Catheters" in prompt

    def test_labels_certifications_neutral(self, records, pack):
        prompt = build_user_prompt(aggregate(records, pack))
        assert "Certifications reported (NEUTRAL)" in prompt
        assert "ISO 80369-7" in prompt

    def test_flags_a_thin_site_without_deciding_it(self, pack):
        bundle = aggregate([record("https://thin.com/", "homepage", domain="thin.com")], pack)
        prompt = build_user_prompt(bundle)
        assert f"fewer than {THIN_SITE_PAGES} pages" in prompt
        assert "do not treat it as automatic" in prompt

    def test_says_plainly_when_there_are_no_quotes(self, pack):
        bundle = aggregate([record("https://thin.com/", "homepage", domain="thin.com")], pack)
        assert "NONE. No page produced a verifiable statement" in build_user_prompt(bundle)

    def test_trims_an_enormous_domain(self, pack):
        rows = [
            record(
                f"https://acme.com/p{index}",
                manufacturing_language=[finding(f"we manufacture widget {index}")],
            )
            for index in range(MAX_QUOTES_PER_DOMAIN + 20)
        ]
        prompt = build_user_prompt(aggregate(rows, pack))
        assert "further quotes of lower priority were not shown" in prompt


# ---------------------------------------------------------------------------
# Verifying the verdict
# ---------------------------------------------------------------------------


class TestVerifyVerdict:
    def test_keeps_evidence_that_is_in_the_extractions(self, records, pack):
        bundle = aggregate(records, pack)
        verdict = DomainVerdict(**verdict_payload())
        kept, drops = verify_verdict(verdict, bundle)
        assert len(kept.evidence) == 1
        assert drops.evidence == 0

    def test_drops_an_invented_quote(self, records, pack):
        """The Phase 5 discipline, one stage down.

        There the question was whether the page said it; here it is whether the
        pipeline collected it. A verdict resting on a quote nobody can trace
        cannot be audited without a re-crawl, which is the cost this whole
        architecture exists to avoid.
        """
        bundle = aggregate(records, pack)
        verdict = DomainVerdict(
            **verdict_payload(
                evidence=[
                    {
                        "quote": "our state of the art ISO Class 5 sterile suite",
                        "source_url": "https://acme.com/capabilities",
                    }
                ]
            )
        )
        kept, drops = verify_verdict(verdict, bundle)
        assert kept.evidence == []
        assert drops.evidence == 1

    def test_repairs_a_mis_copied_source_url(self, records, pack):
        """A real quote with the wrong URL is repaired, not dropped.

        The quote identifies its own page unambiguously, and losing real
        evidence to a mis-copied URL would repeat the false-drop mistake that
        cost Phase 5 a whole certification profile.
        """
        bundle = aggregate(records, pack)
        verdict = DomainVerdict(
            **verdict_payload(
                evidence=[
                    {
                        "quote": "12,000 square feet of ISO Class 7 cleanroom space",
                        "source_url": "https://acme.com/facilities",
                    }
                ]
            )
        )
        kept, drops = verify_verdict(verdict, bundle)
        assert drops.evidence == 0
        assert kept.evidence[0].source_url == "https://acme.com/capabilities"

    def test_dedupes_repeated_evidence(self, records, pack):
        bundle = aggregate(records, pack)
        item = {
            "quote": "12,000 square feet of ISO Class 7 cleanroom space",
            "source_url": "https://acme.com/capabilities",
        }
        verdict = DomainVerdict(**verdict_payload(evidence=[item, item]))
        kept, _ = verify_verdict(verdict, bundle)
        assert len(kept.evidence) == 1

    def test_tolerates_markdown_differences(self, records, pack):
        bundle = aggregate(records, pack)
        verdict = DomainVerdict(
            **verdict_payload(
                evidence=[
                    {
                        "quote": "**12,000 square feet of ISO Class 7 cleanroom space**",
                        "source_url": "https://acme.com/capabilities",
                    }
                ]
            )
        )
        kept, drops = verify_verdict(verdict, bundle)
        assert drops.evidence == 0

    def test_an_empty_evidence_list_is_allowed(self, records, pack):
        """`unknown` over a thin site has nothing to quote, and that is honest."""
        bundle = aggregate(records, pack)
        verdict = DomainVerdict(**verdict_payload(cleanroom_status="unknown", evidence=[]))
        kept, drops = verify_verdict(verdict, bundle)
        assert kept.evidence == [] and drops.evidence == 0


# ---------------------------------------------------------------------------
# One domain end to end
# ---------------------------------------------------------------------------


class TestClassifyDomain:
    def test_assembles_the_identity_fields_locally(self, records, pack, extracted_store):
        """Nothing the model generates reaches a record's identity.

        Six of the twelve fields come from the store. A domain that is never
        generated is a domain that cannot be hallucinated onto a verdict.
        """
        provider = Provider(verdict_payload())
        bundle = aggregate(records, pack)
        with client_for(provider) as client:
            record_out, outcome, _ = classify_domain(
                client, pack, bundle, extracted_store, run_id="run-1"
            )

        assert record_out.domain == "acme.com"
        assert record_out.run_id == "run-1"
        assert record_out.pack_version == "medical_v1"
        assert record_out.source_crawl_date == CRAWL_DATE
        assert record_out.pages_analyzed == 3
        assert record_out.page_types_found == ["homepage", "capabilities", "quality"]
        assert outcome.ok and outcome.cost_usd > 0

    def test_carries_the_verdict_through(self, records, pack, extracted_store):
        provider = Provider(verdict_payload())
        with client_for(provider) as client:
            record_out, _, _ = classify_domain(
                client, pack, aggregate(records, pack), extracted_store, run_id="run-1"
            )
        assert record_out.cleanroom_status == "in_house"
        assert record_out.iso_class == "ISO Class 7"
        assert record_out.evidence[0].source_url == "https://acme.com/capabilities"

    def test_a_blank_iso_class_becomes_null(self, records, pack, extracted_store):
        """Controlled but unclassified is `in_house` with a null class."""
        provider = Provider(verdict_payload(iso_class="none"))
        with client_for(provider) as client:
            record_out, _, _ = classify_domain(
                client, pack, aggregate(records, pack), extracted_store, run_id="run-1"
            )
        assert record_out.iso_class is None
        assert record_out.cleanroom_status == "in_house"

    def test_a_fifth_label_is_a_parse_failure(self, records, pack, extracted_store):
        """The four labels reach the provider as a schema enum.

        A label nothing downstream accepts should fail loudly at the boundary,
        not travel as a plausible-looking record.
        """
        from cleanroom.llm import LLMResponseError

        provider = Provider(verdict_payload(cleanroom_status="probably"))
        with client_for(provider) as client:
            with pytest.raises(LLMResponseError):
                classify_domain(
                    client, pack, aggregate(records, pack), extracted_store, run_id="run-1"
                )

    def test_a_matcher_disagreement_is_recorded_but_does_not_force_review(
        self, records, pack, extracted_store, monkeypatch
    ):
        """CHANGED 2026-08-26. This test previously asserted that a matcher
        disagreement forces `needs_manual_review`. It no longer does, and the
        assertion changed on measurement, not on taste.

        On Tier B's 23 adjudicated rows, `matcher_disagreement` fires on 12 and
        catches 3 of the 5 wrong verdicts — worse than confidence at both jobs.
        Of the 5 high-confidence rows it disagrees with, all 5 are correct, so
        as a gate its whole measured contribution was 5 false alarms. The one
        error it would have caught alone (`biocarta.net`) sits at confidence 0.72
        and is caught by the confidence gate anyway, as are all three domains
        the original 2026-08-25 disagreement fix was built for (`beebix.com`
        0.55, `gracelynbio.com` 0.65, `apexomedical.com` 0.50).

        The signal is not discarded: it stays in `flag_reasons` and carries a
        direction-aware weight in `eval/review_priority.py`, which is where an
        informative-but-not-decisive signal belongs. See classify.REVIEW_GATES.
        """
        import cleanroom.classify as classify_module
        from cleanroom.match import MatcherOutcome
        from cleanroom.matcher import Result, Verdict

        disagreeing = MatcherOutcome(
            domain="acme.com",
            crawl_date=CRAWL_DATE,
            pack_version="medical_v2.2",
            pages_scored=3,
            result=Result(),
            verdict=Verdict(
                "likely_in_house", 0.55, "manufacturer", "ok", None, True, ["disagrees"]
            ),
        )
        monkeypatch.setattr(classify_module, "_matcher_pass", lambda *a, **k: disagreeing)

        provider = Provider(verdict_payload(cleanroom_status="in_house", needs_manual_review=False))
        with client_for(provider) as client:
            record_out, outcome, _ = classify_domain(
                client, pack, aggregate(records, pack), extracted_store, run_id="run-1"
            )

        assert record_out.cleanroom_status == "in_house"
        # verdict_payload's default confidence is 0.95, well above the gate.
        assert record_out.confidence >= classify_module.AUTO_ACCEPT_CONFIDENCE
        assert record_out.needs_manual_review is False
        assert outcome.needs_manual_review is False
        # Recorded, so review_priority can still rank on it.
        assert record_out.flag_reasons == ["matcher_disagreement"]

    def test_a_matcher_disagreement_below_the_confidence_gate_still_queues(
        self, records, pack, extracted_store, monkeypatch
    ):
        """The case the disagreement signal was actually built for. `beebix.com`
        shape: the LLM is unsure AND the second opinion reads it differently.
        Confidence gates it in; the disagreement reason explains why it ranks.
        """
        import cleanroom.classify as classify_module
        from cleanroom.match import MatcherOutcome
        from cleanroom.matcher import Result, Verdict

        disagreeing = MatcherOutcome(
            domain="acme.com",
            crawl_date=CRAWL_DATE,
            pack_version="medical_v2.2",
            pages_scored=3,
            result=Result(),
            verdict=Verdict(
                "likely_in_house", 0.55, "manufacturer", "ok", None, True, ["disagrees"]
            ),
        )
        monkeypatch.setattr(classify_module, "_matcher_pass", lambda *a, **k: disagreeing)

        provider = Provider(verdict_payload(
            cleanroom_status="in_house", needs_manual_review=False, confidence=0.55,
        ))
        with client_for(provider) as client:
            record_out, outcome, _ = classify_domain(
                client, pack, aggregate(records, pack), extracted_store, run_id="run-1"
            )

        assert record_out.needs_manual_review is True
        assert outcome.needs_manual_review is True
        assert record_out.flag_reasons == ["low_confidence", "matcher_disagreement"]

    def test_matcher_agreement_does_not_force_review(
        self, records, pack, extracted_store, monkeypatch
    ):
        import cleanroom.classify as classify_module
        from cleanroom.match import MatcherOutcome
        from cleanroom.matcher import Result, Verdict

        agreeing = MatcherOutcome(
            domain="acme.com",
            crawl_date=CRAWL_DATE,
            pack_version="medical_v2.2",
            pages_scored=3,
            result=Result(),
            verdict=Verdict("in_house", 0.9, "manufacturer", "ok", None, False, ["agrees"]),
        )
        monkeypatch.setattr(classify_module, "_matcher_pass", lambda *a, **k: agreeing)

        provider = Provider(verdict_payload(cleanroom_status="in_house", needs_manual_review=False))
        with client_for(provider) as client:
            record_out, _, _ = classify_domain(
                client, pack, aggregate(records, pack), extracted_store, run_id="run-1"
            )

        assert record_out.needs_manual_review is False

    def test_no_cleanroom_needed_with_no_negative_match_is_recorded_not_gated(
        self, records, pack, extracted_store, monkeypatch
    ):
        """CHANGED 2026-08-26, on measurement. This previously forced review.

        The reasoning it was built on still holds: a `no_cleanroom_needed`
        reached by product-line judgment rather than a literal negative-taxonomy
        match is a legitimate label — the LLM's own prompt instructs exactly
        this generalization — and it is the classifier's most dangerous failure
        mode when wrong, because a real lead is deleted silently and never
        revisited.

        What was wrong was making it a *gate*. On the 2026-08-25 holdout run it
        fires on 9 of the 11 `no_cleanroom_needed` rows and exactly 1
        (`biocarta.net`) is wrong, because the product taxonomy is near-silent on
        the target population's brand names — only 3 of 38 batch-1 domains match
        any positive term at all. "No taxonomy hit" is therefore the normal
        case, so gating on it queued nearly every negative verdict. `biocarta.net`
        is caught by the confidence gate anyway, at 0.72.

        It stays in `flag_reasons` and carries the heaviest single bonus in
        `eval/review_priority.py`, so a reviewer working the queue top-down still
        reaches these first. See classify.REVIEW_GATES.

        `records` here names "Balloon Catheters"/"guidewires" — both positive
        terms, no negative-list hit — so the taxonomy grounding is genuinely
        absent, same shape as `gcxo.com`/`sakuraeus.com`/`suttera.com` in the
        2026-08-24 holdout.
        """
        import cleanroom.classify as classify_module

        monkeypatch.setattr(classify_module, "_matcher_pass", lambda *a, **k: None)

        provider = Provider(verdict_payload(
            cleanroom_status="no_cleanroom_needed", needs_manual_review=False,
        ))
        with client_for(provider) as client:
            record_out, outcome, _ = classify_domain(
                client, pack, aggregate(records, pack), extracted_store, run_id="run-1"
            )

        assert record_out.cleanroom_status == "no_cleanroom_needed"
        assert record_out.needs_manual_review is False
        assert outcome.needs_manual_review is False
        assert record_out.flag_reasons == ["ungrounded_negative"]

    def test_an_ungrounded_negative_below_the_confidence_gate_still_queues(
        self, records, pack, extracted_store, monkeypatch
    ):
        """`biocarta.net`'s shape, and the one row this signal would have caught
        on its own: an ungrounded `no_cleanroom_needed` the model is unsure of.
        Confidence queues it; the ungrounded reason pushes it up the ranking.
        """
        import cleanroom.classify as classify_module

        monkeypatch.setattr(classify_module, "_matcher_pass", lambda *a, **k: None)

        provider = Provider(verdict_payload(
            cleanroom_status="no_cleanroom_needed", needs_manual_review=False,
            confidence=0.72,
        ))
        with client_for(provider) as client:
            record_out, _, _ = classify_domain(
                client, pack, aggregate(records, pack), extracted_store, run_id="run-1"
            )

        assert record_out.needs_manual_review is True
        assert record_out.flag_reasons == ["low_confidence", "ungrounded_negative"]

    def test_no_cleanroom_needed_with_a_real_negative_match_is_not_forced(
        self, pack, extracted_store, monkeypatch
    ):
        import cleanroom.classify as classify_module

        negative_records = [record(
            "https://acme.com/",
            "homepage",
            products_named=["wheelchairs"],
        )]
        monkeypatch.setattr(classify_module, "_matcher_pass", lambda *a, **k: None)

        provider = Provider(verdict_payload(
            cleanroom_status="no_cleanroom_needed", needs_manual_review=False,
        ))
        with client_for(provider) as client:
            record_out, _, _ = classify_domain(
                client, pack, aggregate(negative_records, pack), extracted_store,
                run_id="run-1",
            )

        assert record_out.needs_manual_review is False

    def test_llm_flagged_review_is_recorded_as_a_reason(
        self, pack, extracted_store, monkeypatch
    ):
        """Isolates the third OR branch: the LLM's own needs_manual_review,
        with no matcher disagreement (matcher pass patched out) and real
        negative-taxonomy grounding (so ungrounded_negative can't also fire).
        """
        import cleanroom.classify as classify_module

        negative_records = [record(
            "https://acme.com/", "homepage", products_named=["wheelchairs"],
        )]
        monkeypatch.setattr(classify_module, "_matcher_pass", lambda *a, **k: None)

        provider = Provider(verdict_payload(
            cleanroom_status="no_cleanroom_needed", needs_manual_review=True,
        ))
        with client_for(provider) as client:
            record_out, _, _ = classify_domain(
                client, pack, aggregate(negative_records, pack), extracted_store,
                run_id="run-1",
            )

        assert record_out.needs_manual_review is True
        assert record_out.flag_reasons == ["llm_flagged"]

    def test_llm_flagged_gates_even_at_full_confidence(
        self, pack, extracted_store, monkeypatch
    ):
        """BUILD.md: parent/subsidiary, recently acquired and non-English domains
        are flagged regardless. That is a caveat about which company was crawled,
        not a doubt about the label, so no confidence can clear it.
        """
        import cleanroom.classify as classify_module

        negative_records = [record(
            "https://acme.com/", "homepage", products_named=["wheelchairs"],
        )]
        monkeypatch.setattr(classify_module, "_matcher_pass", lambda *a, **k: None)

        provider = Provider(verdict_payload(
            cleanroom_status="no_cleanroom_needed", needs_manual_review=True,
            confidence=1.0,
        ))
        with client_for(provider) as client:
            record_out, _, _ = classify_domain(
                client, pack, aggregate(negative_records, pack), extracted_store,
                run_id="run-1",
            )

        assert record_out.needs_manual_review is True

    def test_confidence_below_the_gate_queues_a_row_with_no_other_signal(
        self, pack, extracted_store, monkeypatch
    ):
        """The gate standing on its own: real negative-taxonomy grounding, no
        matcher, LLM content with itself, and still queued because it is unsure.
        """
        import cleanroom.classify as classify_module

        negative_records = [record(
            "https://acme.com/", "homepage", products_named=["wheelchairs"],
        )]
        monkeypatch.setattr(classify_module, "_matcher_pass", lambda *a, **k: None)

        provider = Provider(verdict_payload(
            cleanroom_status="no_cleanroom_needed", needs_manual_review=False,
            confidence=classify_module.AUTO_ACCEPT_CONFIDENCE - 0.01,
        ))
        with client_for(provider) as client:
            record_out, _, _ = classify_domain(
                client, pack, aggregate(negative_records, pack), extracted_store,
                run_id="run-1",
            )

        assert record_out.needs_manual_review is True
        assert record_out.flag_reasons == ["low_confidence"]

    def test_the_gate_is_inclusive_at_the_threshold(
        self, pack, extracted_store, monkeypatch
    ):
        """Exactly AUTO_ACCEPT_CONFIDENCE auto-accepts. Three of Tier B's nine
        clean auto-accepts sit exactly on 0.75 (`wqmed.com`, `euroimmun.us`,
        `adeptmedical.com`), so an exclusive comparison would silently discard
        a third of the measured benefit.
        """
        import cleanroom.classify as classify_module

        negative_records = [record(
            "https://acme.com/", "homepage", products_named=["wheelchairs"],
        )]
        monkeypatch.setattr(classify_module, "_matcher_pass", lambda *a, **k: None)

        provider = Provider(verdict_payload(
            cleanroom_status="no_cleanroom_needed", needs_manual_review=False,
            confidence=classify_module.AUTO_ACCEPT_CONFIDENCE,
        ))
        with client_for(provider) as client:
            record_out, _, _ = classify_domain(
                client, pack, aggregate(negative_records, pack), extracted_store,
                run_id="run-1",
            )

        assert record_out.needs_manual_review is False
        assert record_out.flag_reasons == []

    def test_review_gates_are_a_strict_subset_of_the_recorded_reasons(self):
        """Guards the split itself. Every gate must be a reason the code can
        actually emit, and the advisory signals must stay out of the gate set --
        putting either back restores the 84% queue this change exists to fix.
        """
        import cleanroom.classify as classify_module

        emitted = {"llm_flagged", "low_confidence", "ungrounded_negative",
                   "matcher_disagreement"}
        assert classify_module.REVIEW_GATES < emitted
        assert "ungrounded_negative" not in classify_module.REVIEW_GATES
        assert "matcher_disagreement" not in classify_module.REVIEW_GATES

    def test_carries_entity_type_and_referred_domain_through(self, records, pack, extracted_store):
        provider = Provider(verdict_payload(
            cleanroom_status="outsourced", entity_type="distributor",
            referred_domain="acmecontract.com",
        ))
        with client_for(provider) as client:
            record_out, _, _ = classify_domain(
                client, pack, aggregate(records, pack), extracted_store, run_id="run-1"
            )
        assert record_out.entity_type == "distributor"
        assert record_out.referred_domain == "acmecontract.com"

    def test_sends_the_domain_in_the_user_prompt(self, records, pack, extracted_store):
        provider = Provider(verdict_payload())
        with client_for(provider) as client:
            classify_domain(
                client, pack, aggregate(records, pack), extracted_store, run_id="run-1"
            )
        prompt = provider.bodies[0]["messages"][1]["content"]
        assert prompt.startswith("DOMAIN: acme.com")

    def test_a_redirected_domain_short_circuits_before_the_llm_call(
        self, records, pack, extracted_store
    ):
        """An acquired/moved domain is a structural fact from discovery, not
        something to re-derive by asking the model to read whatever evidence
        the redirect happened to serve up. No call, no spend, no waiting on a
        verdict about a company that is not the one that was crawled.
        """
        provider = Provider(verdict_payload())
        bundle = aggregate(records, pack)
        bundle.redirected_off_domain = True
        bundle.resolved_domain = "acquirer.com"

        with client_for(provider) as client:
            record_out, outcome, completion = classify_domain(
                client, pack, bundle, extracted_store, run_id="run-1"
            )

        assert provider.requests == []
        assert completion is None
        assert record_out.entity_type == "acquired_stub"
        assert record_out.cleanroom_status == "unknown"
        assert record_out.needs_manual_review is True
        assert record_out.referred_domain == "acquirer.com"
        assert record_out.run_id == "run-1"
        assert outcome.ok is True
        assert outcome.cost_usd == 0.0
        assert outcome.entity_type == "acquired_stub"

    def test_a_redirected_domain_carries_site_moved_as_run_status(
        self, records, pack, extracted_store
    ):
        """`unknown` from a redirect and `unknown` from genuinely thin evidence
        must not collapse into the same fact. `run_status` is how a reader (or
        a future auto-write gate) tells them apart without re-deriving it from
        `entity_type`/`confidence` — decide()'s own convention already treats
        `run_status != "ok"` as pipeline health, never a business verdict; this
        run_status was just never populated with a real value before.
        """
        bundle = aggregate(records, pack)
        bundle.redirected_off_domain = True
        bundle.resolved_domain = "acquirer.com"

        with client_for(Provider(verdict_payload())) as client:
            record_out, _, _ = classify_domain(
                client, pack, bundle, extracted_store, run_id="run-1"
            )

        assert record_out.run_status == "site_moved"

    def test_a_thin_render_domain_short_circuits_before_the_llm_call(
        self, records, pack, extracted_store
    ):
        """A domain whose median page is functionally empty (a bot wall, an
        interstitial, a JS shell) has nothing for the LLM to read. Confirmed
        on frantellgroup.com: matcher.py's independent pass over the same raw
        pages also scored zero matched terms, so skipping that call too costs
        nothing real. Same discipline as the redirected-domain short-circuit
        above: no call, no spend.
        """
        provider = Provider(verdict_payload())
        bundle = aggregate(records, pack)
        bundle.median_page_word_count = 5

        with client_for(provider) as client:
            record_out, outcome, completion = classify_domain(
                client, pack, bundle, extracted_store, run_id="run-1"
            )

        assert provider.requests == []
        assert completion is None
        assert record_out.cleanroom_status == "unknown"
        assert record_out.needs_manual_review is True
        assert record_out.run_id == "run-1"
        assert outcome.ok is True
        assert outcome.cost_usd == 0.0

    def test_a_thin_render_domain_carries_thin_render_as_run_status(
        self, records, pack, extracted_store
    ):
        """`unknown` from a near-empty crawl and `unknown` from genuinely thin
        but real evidence must not collapse into the same fact, for the same
        reason `site_moved` doesn't collapse into a thin-evidence `unknown`.
        """
        bundle = aggregate(records, pack)
        bundle.median_page_word_count = 5

        with client_for(Provider(verdict_payload())) as client:
            record_out, _, _ = classify_domain(
                client, pack, bundle, extracted_store, run_id="run-1"
            )

        assert record_out.run_status == "thin_render"

    def test_a_domain_with_stub_pages_and_real_evidence_survives(self, pack, extracted_store):
        """The real near-miss found while validating THIN_RENDER_WORD_THRESHOLD:
        structmedical.com runs a genuine ISO Class 7 cleanroom, but 5 of its
        10 crawled pages are ~11-word WordPress media-attachment stubs. Real
        per-page word counts from the stored crawl, median 33 — just above the
        30-word cutoff. This must keep going to the LLM, not get swallowed by
        the thin-render short-circuit.
        """
        stub_records = [
            record(f"https://structmedical.com/p{i}", page_word_count=count)
            for i, count in enumerate([673, 261, 55, 184, 338, 11, 11, 11, 11, 11])
        ]
        bundle = aggregate(stub_records, pack)
        assert bundle.median_page_word_count == 33

        provider = Provider(verdict_payload())
        with client_for(provider) as client:
            classify_domain(client, pack, bundle, extracted_store, run_id="run-1")
        assert provider.requests != []

    def test_a_bot_walled_domain_with_healthy_discovery_still_triggers_thin_render(
        self, pack, extracted_store
    ):
        """frantellgroup.com: discovery itself was healthy (10 pages selected,
        not a discovery_failed domain) but the actual crawled content is a bot
        wall — real per-page word counts from the stored crawl, median 6. This
        is intentionally a POSITIVE test: it does not conflict with the
        separate, same-day decision not to build a JS renderer for this
        domain (outputs/2026-08-25_response-to-audit-rebuttal.md ss3) — that
        decision was about the fetch stage; this is a cheaper classify-stage
        short-circuit reaching the same eventual `unknown` outcome for free.
        """
        thin_records = [
            record(f"https://frantellgroup.com/p{i}", page_word_count=count)
            for i, count in enumerate([7, 6, 6, 6, 6, 6, 6, 533, 533])
        ]
        bundle = aggregate(thin_records, pack)
        assert bundle.median_page_word_count == 6

        provider = Provider(verdict_payload())
        with client_for(provider) as client:
            record_out, _, completion = classify_domain(
                client, pack, bundle, extracted_store, run_id="run-1"
            )
        assert provider.requests == []
        assert completion is None
        assert record_out.run_status == "thin_render"


# ---------------------------------------------------------------------------
# A batch
# ---------------------------------------------------------------------------


class TestClassifyAll:
    def test_writes_records_a_report_and_a_csv(self, extracted_store, pack):
        provider = Provider(verdict_payload())
        with client_for(provider) as client:
            run = classify_all(
                ["acme.com"], pack, extracted_store, client,
                extraction_run_id=EXTRACT_RUN,
                run_id="run-1",
            )

        assert run.records_written == 1
        assert run.domains_classified == 1
        assert extracted_store.exists(report_path("medical_v1", "run-1"))
        assert extracted_store.exists(csv_path("medical_v1", "run-1"))

        stored = read_classifications(extracted_store, "medical_v1", "run-1")
        assert [item.domain for item in stored] == ["acme.com"]

    def test_a_domain_with_no_extractions_writes_no_record(self, extracted_store, pack):
        """Absence means "extract this", never "this company said nothing"."""
        provider = Provider(verdict_payload())
        with client_for(provider) as client:
            run = classify_all(
                ["acme.com", "never-extracted.com"], pack, extracted_store, client,
                extraction_run_id=EXTRACT_RUN,
                run_id="run-1",
            )

        assert run.records_written == 1
        assert run.domains_failed == 1
        missing = next(row for row in run.domains if row.domain == "never-extracted.com")
        assert "no records in extraction run" in (missing.error or "")

    def test_a_failed_call_writes_no_record(self, extracted_store, pack):
        provider = Provider(httpx.Response(500, json={"error": "boom"}))
        with client_for(provider) as client:
            run = classify_all(
                ["acme.com"], pack, extracted_store, client,
                extraction_run_id=EXTRACT_RUN,
                run_id="run-1",
            )
        assert run.records_written == 0
        assert run.domains_failed == 1

    def test_resume_skips_domains_already_stored(self, extracted_store, pack):
        provider = Provider(verdict_payload())
        with client_for(provider) as client:
            classify_all(
                ["acme.com"], pack, extracted_store, client,
                extraction_run_id=EXTRACT_RUN, run_id="run-1",
            )
            second = classify_all(
                ["acme.com"], pack, extracted_store, client,
                extraction_run_id=EXTRACT_RUN, run_id="run-1",
            )

        assert second.domains_skipped == 1
        assert second.records_written == 0
        assert len(provider.requests) == 1
        assert existing_domains(extracted_store, "medical_v1", "run-1") == {"acme.com"}

    def test_the_budget_stops_the_run(self, extracted_store, pack, records):
        rows = [*records, record("https://other.com/", domain="other.com")]
        extracted_store.write_jsonl(extractions_path("medical_v1", "run-b", 0), rows)
        provider = Provider(verdict_payload())
        with client_for(provider) as client:
            run = classify_all(
                ["acme.com", "other.com"], pack, extracted_store, client,
                extraction_run_id="run-b",
                run_id="run-2",
                budget_usd=0.0001,
                concurrency=1,
            )
        assert run.stopped_reason is not None
        assert run.domains_skipped >= 1

    def test_counts_labels_and_the_unknown_share(self, extracted_store, pack, records):
        rows = [*records, record("https://other.com/", "homepage", domain="other.com")]
        extracted_store.write_jsonl(extractions_path("medical_v1", "run-c", 0), rows)
        provider = Provider(
            by_domain={
                "acme.com": verdict_payload(),
                "other.com": verdict_payload(
                    cleanroom_status="unknown", iso_class=None, evidence=[]
                ),
            }
        )
        with client_for(provider) as client:
            run = classify_all(
                ["acme.com", "other.com"], pack, extracted_store, client,
                extraction_run_id="run-c",
                run_id="run-3",
            )
        assert run.label_counts == {"in_house": 1, "unknown": 1}
        assert run.unknown_share == 0.5

    def test_records_dropped_evidence_with_its_text(self, extracted_store, pack):
        """A count says the rate moved; only the text says which way to fix it."""
        provider = Provider(
            verdict_payload(
                evidence=[
                    {"quote": "an ISO Class 5 suite we never mentioned",
                     "source_url": "https://acme.com/capabilities"}
                ]
            )
        )
        with client_for(provider) as client:
            run = classify_all(
                ["acme.com"], pack, extracted_store, client,
                extraction_run_id=EXTRACT_RUN, run_id="run-1",
            )
        assert run.evidence_dropped == 1
        assert "ISO Class 5 suite" in run.domains[0].dropped_evidence[0]


# ---------------------------------------------------------------------------
# Runs in the store
# ---------------------------------------------------------------------------


class TestRunDiscovery:
    def test_lists_runs_with_parts(self, extracted_store):
        assert extraction_runs(extracted_store, "medical_v1") == [EXTRACT_RUN]

    def test_latest_ignores_hand_named_runs(self, extracted_store, records):
        """`smoke-01` sorts after every digit and would win "latest" forever.

        Picking it up by default would classify a tenth of the goldset and look
        like it worked.
        """
        extracted_store.write_jsonl(extractions_path("medical_v1", "smoke-01", 0), records)
        assert "smoke-01" in extraction_runs(extracted_store, "medical_v1")
        assert latest_extraction_run(extracted_store, "medical_v1") == EXTRACT_RUN

    def test_latest_is_none_on_an_empty_store(self, tmp_path):
        assert latest_extraction_run(LocalStore(tmp_path / "data"), "medical_v1") is None


# ---------------------------------------------------------------------------
# The CSV
# ---------------------------------------------------------------------------


def classification(**overrides) -> ClassificationRecord:
    payload = {
        "domain": "acme.com",
        "cleanroom_status": "in_house",
        "iso_class": "ISO 7",
        "confidence": 0.95,
        "reasoning": "Line one.\nLine two.",
        "evidence": [Evidence(quote="a quote", source_url="https://acme.com/x")],
        "pages_analyzed": 10,
        "page_types_found": ["homepage", "quality"],
        "needs_manual_review": False,
        "run_id": "run-1",
        "pack_version": "medical_v1",
        "source_crawl_date": CRAWL_DATE,
    }
    payload.update(overrides)
    return ClassificationRecord(**payload)


class TestCsv:
    def test_has_one_row_per_domain(self):
        text = build_csv([classification(), classification(domain="beta.com")])
        assert len(text.strip().splitlines()) == 3

    def test_flattens_reasoning_onto_one_line(self):
        """A newline inside a cell is what breaks a CSV opened in Excel."""
        text = build_csv([classification()])
        assert "Line one. Line two." in text

    def test_pads_the_evidence_columns(self):
        text = build_csv([classification()])
        assert text.splitlines()[0].split(",")[:3] == list(CSV_COLUMNS[:3])
        assert text.count("\r\n") == 2

    def test_sorts_by_domain(self):
        text = build_csv([classification(domain="zeta.com"), classification(domain="alpha.com")])
        rows = text.strip().splitlines()
        assert rows[1].startswith("alpha.com")

    def test_marks_the_review_flag_readably(self):
        text = build_csv([classification(needs_manual_review=True)])
        assert ",yes," in text

    def test_carries_entity_type_and_referred_domain(self):
        text = build_csv([classification(entity_type="contract_manufacturer",
                                         referred_domain=None)])
        header, row = text.strip().splitlines()
        cols = header.split(",")
        idx = cols.index("entity_type")
        assert cols[idx:idx + 2] == ["entity_type", "referred_domain"]
        assert row.split(",")[idx] == "contract_manufacturer"

    def test_carries_run_status_and_matcher_disagreement(self):
        """A reviewer opening this CSV must not mistake pipeline health for a
        thin-evidence verdict, or miss that the QC layer disagreed.

        Both were real incidents: `airsepta.com`'s `run_status="site_moved"` was
        invisible in the exported review CSV and read as a bare `unknown`, and
        an LLM/matcher disagreement was only visible by opening the JSONL.
        """
        text = build_csv([classification(
            run_status="site_moved",
            matcher_verdict={
                "cleanroom_status": "likely_in_house", "confidence": 0.55,
                "entity_type": "manufacturer", "run_status": "ok",
                "iso_class": None, "needs_manual_review": True, "reasons": [],
            },
        )])
        header, row = text.strip().splitlines()[0], text.strip().splitlines()[1]
        cols = header.split(",")
        vals = row.split(",")
        by_col = dict(zip(cols, vals))
        assert by_col["run_status"] == "site_moved"
        assert by_col["matcher_cleanroom_status"] == "likely_in_house"
        assert by_col["matcher_agrees"] == "no"

    def test_matcher_agreement_reads_yes(self):
        text = build_csv([classification(
            cleanroom_status="in_house",
            matcher_verdict={
                "cleanroom_status": "in_house", "confidence": 0.90,
                "entity_type": "manufacturer", "run_status": "ok",
                "iso_class": None, "needs_manual_review": False, "reasons": [],
            },
        )])
        header, row = text.strip().splitlines()
        by_col = dict(zip(header.split(","), row.split(",")))
        assert by_col["matcher_agrees"] == "yes"

    def test_no_matcher_pass_reads_blank_not_no(self):
        """`matcher_verdict is None` (the matcher failed or was skipped) is a
        different fact from "the matcher looked and agreed" — collapsing them
        into the same "no" would misreport a QC gap as a confirmed check.
        """
        text = build_csv([classification(matcher_verdict=None)])
        header, row = text.strip().splitlines()
        by_col = dict(zip(header.split(","), row.split(",")))
        assert by_col["matcher_agrees"] == ""
        assert by_col["matcher_cleanroom_status"] == ""

    def test_carries_flag_reasons(self):
        """response-to-audit-rebuttal.md item 3: the review CSV needs WHY a
        row was flagged, not just that it was — this is that column.
        """
        text = build_csv([classification(
            needs_manual_review=True,
            flag_reasons=["ungrounded_negative", "matcher_disagreement"],
        )])
        header, row = text.strip().splitlines()
        by_col = dict(zip(header.split(","), row.split(",")))
        assert by_col["flag_reasons"] == "ungrounded_negative matcher_disagreement"


# ---------------------------------------------------------------------------
# Estimating
# ---------------------------------------------------------------------------


class TestCostEstimate:
    def test_prices_a_run_without_calling_anything(self, extracted_store, pack):
        provider = Provider(verdict_payload())
        with client_for(provider) as client:
            estimate = cost_estimate(
                extracted_store, ["acme.com"], pack, client, extraction_run_id=EXTRACT_RUN
            )
        assert estimate["domains"] == 1
        assert estimate["pages"] == 3
        assert estimate["cost_usd"] > 0
        assert estimate["caching"] == "on"
        assert provider.requests == []

    def test_reports_domains_with_no_records(self, extracted_store, pack):
        provider = Provider(verdict_payload())
        with client_for(provider) as client:
            estimate = cost_estimate(
                extracted_store, ["acme.com", "ghost.com"], pack, client,
                extraction_run_id=EXTRACT_RUN,
            )
        assert "ghost.com" in estimate["missing"]
