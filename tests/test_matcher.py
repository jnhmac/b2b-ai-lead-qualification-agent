"""
Regression suite for the cleanroom matcher.

Every test below is a bug that was actually shipped in medical_v2 or v2.1 and
found by running the spec against real text. Do not delete a test to make a
change pass — if behaviour must change, change the assertion deliberately and
say why in the commit message.

    pytest tests/test_matcher.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cleanroom.matcher import Pack, decide

PACK_DIR = Path(__file__).resolve().parents[1] / "packs"
PACK = PACK_DIR / "medical_v2.2.yaml"


@pytest.fixture(scope="module")
def pack():
    return Pack.load(str(PACK))


def terms(res):
    return {m.term for m in res.matches}


# ---------------------------------------------------------------- T1-T10: v2 baseline
def test_T1_possessive_iso_class(pack):
    r = pack.score("Our ISO Class 7 Clean Room is used for device assembly.")
    assert r.total >= 10 and r.by_tier("DIRECT")
    assert pack.iso_class("Our ISO Class 7 Clean Room") == "ISO 7"


def test_T2_hyphenated_spelling(pack):
    """clean-room / clean room / cleanroom must all reach the same rows."""
    a = terms(pack.score("clean-room manufacturing"))
    b = terms(pack.score("cleanroom manufacturing"))
    c = terms(pack.score("clean room manufacturing"))
    assert a == b == c and a


def test_T3_legacy_class_maps(pack):
    assert pack.iso_class("Class 100,000 environment") == "ISO 8"


def test_T4_eu_grade_maps(pack):
    assert pack.iso_class("Grade B aseptic filling suite") == "ISO 5"


def test_T5_seller_cue_present(pack):
    r = pack.score("We supply cleanroom-compatible wipes for your cleanroom.")
    assert r.verdict_inputs()["seller_cue"], "seller cue must be detected"


def test_T7_sqft_normalization(pack):
    assert terms(pack.score("120,000 sq. ft. of cleanroom")) == terms(pack.score("120000 sqft of cleanroom"))


def test_T9_iso13485_is_a_trap_not_evidence(pack):
    r = pack.score("ISO 13485:2016 certified")
    assert r.total == 0, f"trap term scored {r.total}: {[(m.term, m.weight) for m in r.matches]}"


# ---------------------------------------------------------------- T11-T14: the iso/class substring bug
def test_T11_iso1_does_not_match_inside_iso13485(pack):
    r = pack.score("ISO 13485:2016 certified quality management system")
    assert "iso 1" not in terms(r)
    assert not r.by_tier("DIRECT"), "no DIRECT evidence may come from ISO 13485"


def test_T12_iso9_does_not_match_inside_iso9001(pack):
    r = pack.score("We are ISO 9001:2015 registered.")
    assert "iso 9" not in terms(r)
    assert r.total == 0


def test_T13_class_100000_scores_once(pack):
    """v2 fired class 10 + class 100 + class 1000 + class 10000 + class 100000 = 42 pts."""
    r = pack.score("Our Class 100,000 environment is monitored.")
    cls = [m for m in r.matches if m.term.startswith("class ")]
    assert len(cls) <= 1, f"{len(cls)} class rows fired: {[m.term for m in cls]}"


def test_T14_iso1_does_not_match_inside_iso14644_1(pack):
    assert "iso 1" not in terms(pack.score("ISO 14644-1 classification report"))


def test_T14b_iso1_does_not_match_inside_iso10993(pack):
    r = pack.score("Our products are tested to ISO 10993 for biocompatibility.")
    assert "iso 1" not in terms(r) and not r.by_tier("DIRECT")


# ---------------------------------------------------------------- the residual bugs found in v2.1
def test_annex_i_is_not_cleanroom_evidence(pack):
    """EU MDR Annex I is the general safety clause cited by every CE-marked maker."""
    r = pack.score("Our devices comply with EU MDR Annex I.")
    assert not r.by_tier("DIRECT"), f"Annex I produced DIRECT evidence: {[m.term for m in r.matches]}"


def test_grade_a_stainless_is_not_a_cleanroom(pack):
    r = pack.score("All components are machined from Grade A stainless steel.")
    assert not r.by_tier("DIRECT"), f"scored: {[(m.term, m.weight) for m in r.matches]}"


def test_grade_a_environment_is_evidence(pack):
    """...but the qualified form must still fire."""
    assert pack.score("Aseptic filling in a Grade A environment.").total > 0


def test_temperature_controlled_environment_suppressed(pack):
    r = pack.score("Products ship from our temperature controlled environment.")
    assert not any(m.term == "controlled environment" for m in r.matches)


def test_controlled_environment_alone_still_scores(pack):
    r = pack.score("Assembly is performed in a controlled environment.")
    assert any(m.term == "controlled environment" for m in r.matches)


@pytest.mark.parametrize("word", ["tab", "sec", "psi", "mat", "dp", "bet", "sip", "cer", "aus", "des"])
def test_generic_english_words_never_score(pack, word):
    r = pack.score(f"Hydraulic systems operate and the {word} value is recorded.")
    assert r.total == 0, f"'{word}' scored {r.total}: {[(m.term, m.via) for m in r.matches]}"


@pytest.mark.parametrize("word", ["finance", "insurance"])
def test_holding_company_generic_variants_never_score(pack, word):
    """`finance` and `insurance` are bare English words listed as `holding
    company` variants — the exact 1c bug (generic words as literal variants)
    the v2.2 cleanup was supposed to have removed, found live in a CFO bio:
    'responsible for accounting and finance' scored -5 as a holding-company
    signal on a company with no holding-company relationship at all.
    """
    r = pack.score(f"Charles joined the company and is responsible for accounting and {word}.")
    assert r.total == 0, f"'{word}' scored {r.total}: {[(m.term, m.via) for m in r.matches]}"


def test_full_trap_paragraph_is_not_in_house(pack):
    """Boilerplate present on essentially every medical device site. v2.1 scored this 39 with 3 DIRECT."""
    text = (
        "Founded in 1998, we are an ISO 13485:2016 and ISO 9001:2015 certified, FDA-registered "
        "manufacturer of Class II medical devices. Our products are tested to ISO 10993 for "
        "biocompatibility and comply with EU MDR Annex I. Our quality management system covers "
        "design and manufacturing at our 45,000 square foot facility. All components are machined "
        "to Grade A stainless steel specifications. Hydraulic systems operate at 3000 psi. "
        "See our latest SEC filings for details."
    )
    r = pack.score(text)
    vi = r.verdict_inputs()
    assert not vi["has_direct"], f"DIRECT terms from boilerplate: {[m.term for m in r.by_tier('DIRECT')]}"
    assert vi["score"] < 6, f"boilerplate scored {vi['score']}: {[(m.term, m.weight) for m in r.matches]}"


# ---------------------------------------------------------------- span dedup (M5)
def test_span_dedup_one_fact_scores_once(pack):
    """v2.1 scored this 55 against an in_house floor of 10."""
    r = pack.score("Our facility includes an ISO 7 cleanroom for device assembly.")
    assert r.total <= 15, f"scored {r.total}: {[(m.term, m.weight) for m in r.matches]}"
    assert r.verdict_inputs()["has_direct"]
    spans = [(m.start, m.end) for m in r.matches]
    for i, (s1, e1) in enumerate(spans):
        for s2, e2 in spans[i + 1:]:
            assert not (s1 < e2 and s2 < e1), f"overlapping spans {(s1,e1)} {(s2,e2)}"


def test_cleanroom_token_does_not_fire_182_rows(pack):
    r = pack.score("We operate a cleanroom.")
    assert len(r.matches) <= 4, f"{len(r.matches)} rows fired: {[m.term for m in r.matches]}"


# ---------------------------------------------------------------- evidence quality
def test_quote_is_verbatim_from_original(pack):
    original = "Our ISO Class 7 Clean Room is certified annually."
    r = pack.score(original)
    assert r.matches
    for m in r.matches:
        assert m.quote in original, f"quote not verbatim: {m.quote!r}"


def test_matched_terms_emitted(pack):
    r = pack.score("Our ISO 7 cleanroom is validated.")
    mt = r.matched_terms()
    assert mt and all({"term", "weight", "tier", "span", "quote"} <= set(d) for d in mt)


# ---------------------------------------------------------------- real batch-1 rows
def test_vaportheq_bare_first_person_cleanroom(pack):
    """One bare cleanroom sentence. Must land in the 6-9 + DIRECT band, not fall through."""
    r = pack.score("Image of Vaportheq clean room team holding letters that spell out Vaportheq.")
    vi = r.verdict_inputs()
    assert vi["has_direct"] and 6 <= vi["score"] < 10


def test_unchartedlabs_customer_facility_is_not_a_lead(pack):
    """The GMP facility belongs to the customer. The only false lead a human produced in batch 1."""
    r = pack.score("Turn on the taps for your clinical trials in a GMP-compliant facility with Sunbather.")
    assert not r.by_tier("DIRECT")


def test_beebix_exec_bio_prior_employer_is_not_an_outsourcing_cue(pack):
    """HANDOFF's R6 (`infraredix.com` was flipped to `outsourced` by an
    executive biography describing a prior employer) is a documented, named
    failure mode — found live again here, in matcher.py's own independent
    scoring rather than the LLM's. The executive's prior company is not the one on this page.
    """
    t = (
        "Prior to Northgate Fluidics, the executive was Founder and CEO of Company 118, "
        "a venture-funded medical equipment contract manufacturer that he and his "
        "team grew to more than 500 employees."
    )
    r = pack.score(t)
    assert not any(m.term == "contract manufacturer" for m in r.matches), (
        f"scored: {[(m.term, m.weight, m.via) for m in r.matches]}"
    )


def test_a_companys_own_contract_manufacturing_still_scores(pack):
    """The R6 guard must not swallow the existing self-description path: the
    longer, more specific "we are a contract manufacturer" row (IN_HOUSE_CUE,
    +2) already wins this span over the bare OUTSOURCED_CUE row via M6's
    longest-span rule, independent of my new suppression.
    """
    t = "We are a contract manufacturer serving the world's leading medical device companies."
    r = pack.score(t)
    assert any(m.term == "we are a contract manufacturer" for m in r.matches)
    assert not any(m.term == "contract manufacturer" for m in r.matches)


def test_anvilsurgical_real_class_7(pack):
    r = pack.score("Anchor has multiple subassembly in a class 7 clean room with final assembly space.")
    assert r.verdict_inputs()["has_direct"] and r.total >= 10


# ---------------------------------------------------------------- verdict layer
def test_seller_cue_suppresses_cleanroom_evidence(pack):
    t = "We supply cleanroom-compatible wipes and gloves for your ISO 5 cleanroom."
    v = decide(pack.score(t), t)
    assert v.cleanroom_status == "no_cleanroom_needed"


def test_possessive_cleanroom_is_in_house(pack):
    t = "Our facility includes an ISO 7 cleanroom for device assembly."
    v = decide(pack.score(t), t, iso_class=pack.iso_class(t))
    assert v.cleanroom_status == "in_house" and v.iso_class == "ISO 7"


def test_vaportheq_now_reaches_in_house_via_band_extension(pack):
    t = "Image of our Vaportheq clean room team holding letters."
    v = decide(pack.score(t), t)
    assert v.cleanroom_status == "in_house" and v.needs_manual_review


def test_trap_boilerplate_is_unknown_not_in_house(pack):
    t = ("We are an ISO 13485:2016 and ISO 9001:2015 certified, FDA-registered manufacturer. "
         "Products tested to ISO 10993 and compliant with EU MDR Annex I. Machined from Grade A "
         "stainless steel at our facility. Systems operate at 3000 psi.")
    v = decide(pack.score(t), t)
    assert v.cleanroom_status != "in_house", v.reasons


def test_run_status_never_becomes_a_verdict(pack):
    v = decide(pack.score(""), "", run_status="discovery_failed")
    assert v.cleanroom_status == "unknown" and v.needs_manual_review


def test_service_lab_is_not_eligible_for_in_house(pack):
    t = "Our CLIA and CAP accredited laboratory operates an ISO 7 cleanroom."
    v = decide(pack.score(t), t, entity_type="service_lab")
    assert v.cleanroom_status != "in_house"


# ---------------------------------------------------------------- domain-level dedup (score_domain)
def test_footer_boilerplate_scores_once_across_pages(pack):
    """A phrase repeated in a site-wide template is one fact, not ten.

    The single-page score for this exact sentence is 17 (`iso 7` DIRECT=7 +
    `cleanroom manufacturing` DIRECT=10, both real, non-overlapping rows;
    verified with a bare `pack.score()` call, no aggregation involved). The
    bound below is that verified single-page baseline plus headroom, not the
    stacked total (which would be 170) — the property under test is that
    repeating the page 10x does not multiply the score, and page_count
    records the repetition as a confidence signal instead.
    """
    pages = [(f"/p{i}", "products", "ISO 7 Cleanroom Manufacturing | Contact")
             for i in range(10)]
    r = pack.score_domain(pages)
    assert r.total <= 20, f"{r.total} — boilerplate stacked"
    assert r.matched_terms()[0]["page_count"] == 10


def test_distinct_evidence_still_accumulates(pack):
    """Different facts on different pages must still add up."""
    pages = [("/a", "facilities", "Our ISO 7 cleanroom."),
             ("/b", "careers",   "Gowning qualification required."),
             ("/c", "quality",   "Environmental monitoring program.")]
    assert pack.score_domain(pages).total > pack.score("Our ISO 7 cleanroom.").total


# ---------------------------------------------------------------- decide(): R10 is a floor, not a mint
def test_R10_does_not_override_negative_evidence(pack):
    """A STRONG positive product must not mint likely_in_house over a heavily negative score."""
    t = (
        "We are a CLIA-certified clinical laboratory and a holding company for "
        "hospital IT services. Our regenerative medicine division refers all "
        "specimens to outside labs and never touches the product."
    )
    v = decide(pack.score(t), t)
    vi = pack.score(t).verdict_inputs()
    assert vi["strong_positive_product"], "test needs a STRONG positive product to exercise R10"
    assert vi["score"] < 0, "test needs a heavily negative score to exercise the override"
    assert v.cleanroom_status not in ("likely_in_house", "in_house"), v.reasons


def test_service_lab_never_reaches_likely_in_house(pack):
    """The mdxhealth shape: CLIA/CAP lab, entity gate wins regardless of score."""
    t = "Our CLIA and CAP accredited laboratory operates a regenerative medicine program."
    v = decide(pack.score(t), t, entity_type="service_lab")
    assert v.cleanroom_status not in ("likely_in_house", "in_house")
