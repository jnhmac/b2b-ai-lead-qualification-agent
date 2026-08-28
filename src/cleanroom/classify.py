"""Phase 6: one LLM call per domain, over that domain's aggregated extractions.

This stage reads the store and never the network. Everything it needs was
collected once in Phase 4 and verified once in Phase 5, which is what makes a
prompt revision a three-minute re-run instead of a four-hour re-crawl.

**All judgment lives here.** Phase 5 reports what pages say; this decides what a
company is. The split is the reason the extraction records carry no verdict and
no taxonomy match, and the reason this module carries no scraping.

**The pack is applied here, in code, not in the prompt.** `products_named` and
`trap_terms_present` arrive raw and unmatched. Matching them against the pack
happens below, which is what lets a taxonomy edit cost a re-classify instead of
a re-extract. It also has to happen in code for a duller reason: the goldset
carries 3,114 product names and up to 294 on a single enterprise domain, and
pasting that into a prompt buys nothing but input tokens.

**Evidence is quoted, never generated.** Every quote reaching this stage has
already been checked against the page it came from, twice. So the classifier is
asked to *select* from those quotes, and anything it returns that is not one of
them is dropped exactly as an unverifiable quote is dropped in Phase 5. A
verdict whose evidence cannot be traced to a page is a verdict nobody can audit
six months from now.

**A domain with no extractions writes no record.** Same rule as Phase 4's
withheld manifest and Phase 5's failed page: absence means "run the stage", and
a record asserting `unknown` over zero pages would be indistinguishable from a
site that genuinely had nothing to say.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import statistics
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .config import LLMStage, SectorPack, Term
from .extract import (
    FINDING_FIELDS,
    canonical,
    _loose,
    default_run_id,
    read_records,
)
from .llm import Completion, LLMClient, LLMError, LLMResponseError, LLMTransportError
from .match import MatcherOutcome, run_matcher
from .models import (
    ClassificationRecord,
    EntityType,
    Evidence,
    ExtractionRecord,
    NewCleanroomStatus,
)
from .store import Store, classifications_path

log = logging.getLogger(__name__)

STAGE: LLMStage = "classify"

#: One call per domain rather than per page, so a goldset pass is 46 calls and
#: not 448. Lower than extraction's concurrency because each call carries a
#: whole domain's evidence and the responses are long.
DEFAULT_CONCURRENCY = 3

#: Records per part file. A run is one record per domain, so at 500 domains the
#: whole run is a handful of parts.
DEFAULT_RECORDS_PER_PART = 200

#: Evidence quotes sent per domain. Enterprise domains produce hundreds of
#: findings and the tail is repetitive; the cap keeps a single domain from
#: costing ten times the median. Quotes are ordered by field priority and
#: confidence, so what gets cut is the least load-bearing.
MAX_QUOTES_PER_DOMAIN = 120

#: Product names sent per domain. Only the taxonomy match matters for the
#: verdict, and that is computed in code; the raw sample is context for the
#: model to sanity-check the match against.
MAX_PRODUCTS_PER_DOMAIN = 60

#: Rough output size per domain, used only by the pre-run estimate. Reasoning
#: plus three or four evidence quotes.
ESTIMATED_OUTPUT_TOKENS = 700

#: BUILD.md's threshold for a site being too thin to judge.
THIN_SITE_PAGES = 3

#: Below this, a domain's median page is functionally empty: a bot wall, a
#: consent/geo interstitial, or a JS shell the crawler never rendered through.
#: Reuses Phase 7a's own measured definition of a near-empty page rather than
#: inventing a new number ("35 of 792 usable pages (4.4%) came back under 30
#: words" — NEXT.md). Validated against every domain in the store as of
#: 2026-08-25: only cuteris.com (median 24) and frantellgroup.com (median 6) fall
#: under it, and structmedical.com — real ISO Class 7 evidence mixed with
#: WordPress attachment-page stubs — sits at 33, just above. That margin is
#: thin; see test_a_domain_with_stub_pages_and_real_evidence_survives before
#: ever moving this number.
THIN_RENDER_WORD_THRESHOLD = 30

#: At or above this confidence a verdict skips the review queue entirely.
#:
#: Measured 2026-08-26 against Tier B's 23 adjudicated rows
#: (`outputs/2026-08-26_design-review.md`). Every one of the 5 wrong verdicts in
#: run `20260825-regression` sits at 0.72 or below, and all 9 rows at 0.75 or
#: above are correct. Gating on confidence alone catches 5 of 5 errors while
#: queueing 14 of 23 rows; the previous rule queued 19 of 23 to catch the same
#: 5. The model's own confidence was already the best predictor available and
#: nothing was reading it.
#:
#: PROVISIONAL. n=23 with 5 errors, and the threshold was chosen on the same 23
#: rows it is justified by; "catches 5 of 5" has a 95% interval of [57%, 100%].
#: The experiment that settles it is the next fresh holdout: score it with
#: `eval/evaluate.py` and check whether any wrong verdict lands at or above
#: 0.75. One such observation should move this number to 0.85, which auto-
#: accepts 3 of 23 instead of 9 and is still no worse than the old rule.
AUTO_ACCEPT_CONFIDENCE = 0.75

#: Which `flag_reasons` force a row into the review queue, and which are only
#: recorded for `eval/review_priority.py` to rank on.
#:
#: The split matters and was measured. `matcher_disagreement` and
#: `ungrounded_negative` are both real signals and both poor gates:
#:
#: - `matcher_disagreement` fires on 12 of 23 Tier B rows and catches 3 of the
#:   5 errors, so it is strictly worse than confidence at both jobs. Of the 5
#:   high-confidence rows it disagrees with, all 5 are correct.
#: - `ungrounded_negative` fires on 9 of 11 `no_cleanroom_needed` rows and
#:   catches 1 error, because the product taxonomy is near-silent on SMB brand
#:   names (only 3 of 38 batch-1 domains match any positive term). "No taxonomy
#:   hit" is the normal case, so a gate on it flags almost every negative
#:   verdict.
#:
#: Both stay in `flag_reasons` and both carry weight in `review_priority()`,
#: which is where a signal that is informative but not decisive belongs. The one
#: error each would have caught alone (`biocarta.net`) is caught by confidence
#: anyway, at 0.72. So are all three domains the 2026-08-25 disagreement fix was
#: built for: `beebix.com` 0.55, `gracelynbio.com` 0.65, `apexomedical.com` 0.50.
#:
#: `llm_flagged` stays a gate because BUILD.md requires it: parent/subsidiary,
#: recently acquired and non-English domains are flagged regardless of how
#: confident the verdict is. That is a caveat about the company, not a doubt
#: about the label, and confidence cannot express it.
REVIEW_GATES = frozenset({"llm_flagged", "low_confidence"})

#: Field order used when evidence is trimmed. A stated ISO class outranks a
#: cleanroom term, which outranks a sterilisation method, which outranks
#: general manufacturing language.
_FIELD_PRIORITY: dict[str, int] = {name: index for index, name in enumerate(FINDING_FIELDS)}


# ---------------------------------------------------------------------------
# What the model is asked for
# ---------------------------------------------------------------------------


class DomainVerdict(BaseModel):
    """The eight fields the model produces. The other six are assembled locally.

    Deliberately not `ClassificationRecord`: `domain`, `run_id`, `pack_version`,
    `source_crawl_date`, `pages_analyzed` and `page_types_found` are all facts
    the store already knows. Asking for them spends output tokens re-stating
    them and invites a hallucinated domain into a record whose whole job is to
    be auditable.
    """

    model_config = ConfigDict(extra="forbid")

    #: The five current labels reach the provider as a JSON schema enum, "none"
    #: excluded on purpose (retired -> no_cleanroom_needed; see NewCleanroomStatus).
    #: A sixth or stale label is then a parse failure and gets a retry, rather
    #: than a plausible-looking record nothing downstream accepts.
    cleanroom_status: NewCleanroomStatus
    iso_class: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    evidence: list[Evidence] = Field(default_factory=list)
    needs_manual_review: bool = False
    #: medical_v2.2.yaml's labels.entity_type. Required, same reasoning as
    #: cleanroom_status: a fifth-option-shaped answer is a parse failure, not a
    #: plausible record. Every domain is some entity type; there is no "unknown"
    #: to fall back on here the way there is for cleanroom_status.
    entity_type: EntityType
    #: Only when cleanroom_status=outsourced and a specific contract
    #: manufacturer is named (medical_v2.2.yaml's label_rules).
    referred_domain: str | None = None


# ---------------------------------------------------------------------------
# Aggregating a domain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Quote:
    """One verified finding, with the page it came from."""

    field: str
    claim: str
    quote: str
    confidence: float
    source_url: str
    page_type: str

    @property
    def rank(self) -> tuple[int, float]:
        return (_FIELD_PRIORITY.get(self.field, len(_FIELD_PRIORITY)), -self.confidence)


@dataclass
class TaxonomyMatch:
    """What the pack's product lists found in one domain's product names.

    Kept as term -> the page's own spellings, rather than a count, because the
    prompt shows the model which of *its* names drove the match. A match on
    "catheters" via "Balloon Catheters" is checkable; a bare count is not.
    """

    positive: dict[str, list[str]] = field(default_factory=dict)
    negative: dict[str, list[str]] = field(default_factory=dict)
    unmatched: int = 0

    @property
    def positive_terms(self) -> list[str]:
        return sorted(self.positive)

    @property
    def negative_terms(self) -> list[str]:
        return sorted(self.negative)


@dataclass
class DomainEvidence:
    """Everything known about one domain, aggregated from its extractions."""

    domain: str
    crawl_date: date
    pack_version: str
    pages: int = 0
    page_types: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    total_words: int = 0
    #: The domain-level MEDIAN of each page's own word count, distinct from
    #: total_words (a sum). A single bot-walled page hides inside a healthy
    #: sum on a 10-page domain; it cannot hide from the median. Computed here
    #: rather than carried as a full per-page list, since DomainEvidence is
    #: rebuilt fresh from persisted ExtractionRecords on every call — nothing
    #: is lost by not keeping more than this needs today.
    median_page_word_count: float = 0.0
    quotes: list[Quote] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    taxonomy: TaxonomyMatch = field(default_factory=TaxonomyMatch)
    trap_matched: dict[str, list[str]] = field(default_factory=dict)
    trap_unmatched: list[str] = field(default_factory=list)
    resolved_domain: str | None = None
    redirected_off_domain: bool = False

    @property
    def evidence_quotes(self) -> list[Quote]:
        """Quotes that bear on a controlled environment specifically."""
        return [q for q in self.quotes if q.field in ("iso_class_mentions", "cleanroom_terms")]

    def quote_pool(self) -> str:
        """Every verified quote, canonicalised, for checking returned evidence."""
        return "\n".join(canonical(q.quote) for q in self.quotes)


def aggregate(records: Sequence[ExtractionRecord], pack: SectorPack) -> DomainEvidence:
    """Fold one domain's page records into a single evidence bundle.

    Raises `ValueError` on an empty list. A caller with no records has a domain
    that was never extracted, which is a different thing from a domain with
    nothing to say, and the two must not produce the same record.
    """
    if not records:
        raise ValueError("cannot aggregate an empty record list")

    domain = records[0].domain
    bundle = DomainEvidence(
        domain=domain,
        crawl_date=records[0].crawl_date,
        pack_version=records[0].pack_version,
        resolved_domain=records[0].resolved_domain,
        redirected_off_domain=records[0].redirected_off_domain,
    )

    seen_types: list[str] = []
    seen_products: dict[str, str] = {}
    trap_seen: dict[str, str] = {}
    page_word_counts: list[int] = []

    for record in records:
        bundle.pages += 1
        bundle.total_words += record.page_word_count
        page_word_counts.append(record.page_word_count)
        bundle.urls.append(record.url)
        if record.page_type not in seen_types:
            seen_types.append(record.page_type)

        for field_name in FINDING_FIELDS:
            for finding in getattr(record, field_name):
                bundle.quotes.append(
                    Quote(
                        field=field_name,
                        claim=finding.claim,
                        quote=finding.quote,
                        confidence=finding.confidence,
                        source_url=record.url,
                        page_type=record.page_type,
                    )
                )

        # Deduped across pages by canonical form: a product listed on the
        # catalogue and again on the homepage is one product, and an enterprise
        # site repeats itself relentlessly.
        for name in record.products_named:
            key = canonical(name)
            if key and key not in seen_products:
                seen_products[key] = name
        for value in record.trap_terms_present:
            key = _loose(canonical(value))
            if key and key not in trap_seen:
                trap_seen[key] = value

    bundle.page_types = seen_types
    bundle.median_page_word_count = statistics.median(page_word_counts)
    bundle.products = list(seen_products.values())
    bundle.quotes.sort(key=lambda q: q.rank)
    bundle.taxonomy = match_products(bundle.products, pack)
    bundle.trap_matched, bundle.trap_unmatched = match_trap_terms(
        list(trap_seen.values()), pack
    )
    return bundle


def match_products(names: Sequence[str], pack: SectorPack) -> TaxonomyMatch:
    """Match raw product names against the pack's taxonomy.

    Substring containment on the canonical form, term inside name: the pack
    says `catheters` and the page says `Balloon Catheters`, so the term is what
    has to appear in the name and not the other way round. Aliases count, which
    is what `surface_forms` is for.

    **Both sides are singularised first, and that is not cosmetic.** The pack
    names a category in the plural and a page names one product in the
    singular, so `catheters` was never inside `peripheral catheter` and the
    match silently never fired. Measured when it was found on 2026-08-11: 6
    domains across the two extraction runs carried a positive product the
    taxonomy could not see, nuvellis.com's catheters and blood circuit sets
    among them.

    A name may match both lists, and nothing here resolves that. Direction is
    the classifier's problem, and a company selling both catheters and
    wheelchairs is a real thing that the verdict has to face rather than a
    tie for this function to break.
    """
    match = TaxonomyMatch()
    positives = _surface_index(pack.positive_products, singular=True)
    negatives = _surface_index(pack.negative_products, singular=True)

    for name in names:
        key = _singular(canonical(name))
        if not key:
            continue
        hit = False
        for bucket, index in ((match.positive, positives), (match.negative, negatives)):
            for form, term in index:
                if form not in key:
                    continue
                hit = True
                # One name can match a term through several of its surface
                # forms ("coronary stent" hits both `stents` and its own alias),
                # and the prompt shows this list to the model. Listing the same
                # name twice under one term reads as two products.
                names = bucket.setdefault(term, [])
                if name not in names:
                    names.append(name)
        if not hit:
            match.unmatched += 1
    return match


def match_trap_terms(
    values: Sequence[str], pack: SectorPack
) -> tuple[dict[str, list[str]], list[str]]:
    """Split reported certifications into the pack's list and the long tail.

    Compared ignoring internal spaces and hyphens, exactly as Phase 5 does:
    `ISO 9001` and `ISO9001` are one certification spelled by two marketing
    departments.

    **The unmatched tail is not a residue to be tidied away.** The pack names
    11 traps; 46 domains produced 57 distinct certifications and 500 will
    produce hundreds. The pack list is the set the prompt calls out by name,
    and the prompt's shape rule covers everything else. Both halves go to the
    model, and both are neutral.
    """
    index = _surface_index(pack.trap_terms, loose=True)
    matched: dict[str, list[str]] = {}
    unmatched: list[str] = []
    for value in values:
        key = _loose(canonical(value))
        if not key:
            continue
        hit = next((term for form, term in index if form in key or key in form), None)
        if hit is None:
            unmatched.append(value)
        else:
            matched.setdefault(hit, []).append(value)
    return matched, unmatched


#: Trailing plural `s` removed from words of four characters or more. Four so
#: that `is`, `as` and `gas` survive; a product category short enough to be
#: mangled by this does not exist in the pack.
_PLURAL = re.compile(r"(\w{3,})s\b")


def _singular(text: str) -> str:
    """Fold plural and singular spellings of the same product category together.

    Scoped to the product taxonomy, exactly as `_loose` is scoped to
    certification names. A quote is evidence and stays strict; a product
    category is a token, and `catheter` and `catheters` are one category spelled
    by a marketing page and a pack author.
    """
    return _PLURAL.sub(r"\1", text)


def _surface_index(
    terms: Sequence[Term], *, loose: bool = False, singular: bool = False
) -> list[tuple[str, str]]:
    """(canonical surface form, canonical term) pairs, longest form first.

    Longest first so `hospital beds` is tried before `beds` would be, if the
    pack ever carries both.
    """
    pairs: list[tuple[str, str]] = []
    for term in terms:
        for form in term.surface_forms:
            key = canonical(form)
            if loose:
                key = _loose(key)
            if singular:
                key = _singular(key)
            if key:
                pairs.append((key, term.term))
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# The system prompt
# ---------------------------------------------------------------------------
#
# The label definitions and edge cases below are copied verbatim from BUILD.md,
# on purpose. They are the specification of the output, and paraphrasing a
# specification into a prompt is how a pipeline quietly starts answering a
# different question than the one it was built to answer.

_ROLE = """\
You are classifying ONE medical device company. You are given every verified
quote its website produced, the products it names, and the certifications it
carries. You return one verdict about whether that company operates a cleanroom.

You are not reading the website. You are reading evidence collected from it by
a separate system, which recorded what the pages said and decided nothing. The
deciding is yours, and it is the only judgment in this pipeline.

The input list has already been filtered to medical device manufacturers. That
label is too broad to be useful: a wheelchair manufacturer and a catheter
manufacturer both carry it, and only one of them needs a cleanroom. Separating
those two is the entire job."""

_QUESTION = """\
## The question

DOES A CLEANROOM EXIST AT ALL?

Not "is this company's product made in a cleanroom", and not "is a cleanroom
central to what they do". A company running an ISO 7 room to package springs
operates a cleanroom, and the answer is yes.

This is the broad reading and it is deliberate. Read it that way even when the
room is used for something peripheral, for a different industry, or for a
product line that is not the company's main business.

## SIZE AND PURPOSE DO NOT MATTER

There is no minimum size and no required use. A single room the size of a lab
or a corner of a warehouse counts exactly as much as a 50,000 square foot
production hall.

The room does not have to be for manufacturing. Packaging, kitting, labelling,
final assembly, sterile barrier sealing, inspection, compounding, testing and
laboratory work are all real uses of a controlled environment, and a company
doing any of them in one is a yes.

The room does not have to be for medical devices, or even for anything medical.
A semiconductor, optics, aerospace or pharmaceutical cleanroom is still a
cleanroom.

Nothing in the evidence has to describe a factory. "Our ISO 8 packaging suite",
"we hand-assemble in a controlled environment", "our clean room in the back of
the Fremont facility" are all sufficient. Do not hold out for square footage, a
plant, or a production line: most companies of this size have none of those and
still run a room."""

_LABELS = """\
## The five labels

- `in_house` — operates its own cleanroom. ISO class stated, cleanroom square
  footage, gowning procedures, cleanroom job postings, facility described as a
  controlled production environment. Do NOT use this when the evidence only
  shows the product is sterile.
- `likely_in_house` — the products strongly imply a cleanroom (catheters,
  stents, implantables, other work that is made in a controlled environment as
  a matter of industry practice) but nothing on the site explicitly describes
  one: no class, no cleanroom word, no gowning, no controlled-environment
  language anywhere. An inference, not a statement. ALWAYS set
  `needs_manual_review: true` when you use this label — see the worked example
  below (D2).
- `outsourced` — sells sterile product, manufactures none of it. Contract
  manufacturer named, manufacturing-partner language, design-and-marketing only
  with no facility mentioned anywhere.
- `no_cleanroom_needed` — the product line does not require a cleanroom.
  Catalogue is entirely durable equipment, furniture, reusable instruments, or
  software. This is a POSITIVE DETERMINATION FROM THE PRODUCT TAXONOMY, not an
  absence of evidence.
- `unknown` — the site is too thin to say what the company even does. Fewer
  than three pages retrieved, no products listed, no facility or manufacturing
  language at all. Reserve this one for genuine absence of evidence, not for
  "the evidence is there but doesn't reach a class statement" — that shape is
  `likely_in_house`, not `unknown`.

## Edge cases

- HYBRID (some in-house, some outsourced) resolves to `in_house`. The question
  is whether a cleanroom exists at all.
- CONTROLLED BUT UNCLASSIFIED (a controlled environment with no ISO class
  stated) resolves to `in_house` with `iso_class: null`.
- DISTRIBUTOR ONLY resolves to `no_cleanroom_needed`. A company that sells
  cleanroom supplies, cleanroom furniture or cleanroom consumables without
  operating a room is a distributor. Its catalogue is full of the word and it
  is still `no_cleanroom_needed`.
- PARENT/SUBSIDIARY, RECENTLY ACQUIRED, NON-ENGLISH — classify normally but set
  `needs_manual_review: true`.

## `in_house` REQUIRES FACILITY EVIDENCE — `likely_in_house` IS THE INFERENCE BAND

A positive `in_house` verdict needs at least one quote describing a controlled
environment: a stated cleanliness class, a cleanroom, a controlled or critical
environment, gowning, an airlock, HEPA or laminar flow, particulate control, or
a job posting for someone who works in one.

**A product catalogue is not facility evidence for `in_house`.** Catheters,
implantables and stents are made in cleanrooms as a matter of industry
practice, and that fact tells you about the industry, not about this company.
Neither is a manufacturing statement on its own: "all assembly is performed in
house" says there is a factory, not that any part of it is controlled.

**But it IS `likely_in_house`.** If the products imply a cleanroom and nothing
on the site describes one, the answer is `likely_in_house` with
`needs_manual_review: true` — not `in_house` (no facility statement exists to
support it) and not `unknown` (you can name what they make, and what they make
is genuinely informative). Say in your reasoning that the catalogue implies a
controlled environment and the site never evidences one. That is a useful,
honest answer and it is not a failure.

The population this runs against is small and mid-sized manufacturers. A
company of that size that operates a cleanroom says so, usually on more than
one page, because it is the thing they are selling. Silence is genuinely
informative, which is exactly why the inference is worth flagging rather than
discarding into `unknown`. The exception is a large multi-division enterprise,
where ten crawled pages cannot cover the estate; that is still `unknown`,
still flagged, and the review queue is where it gets resolved.

## The distinctions to enforce hardest

Three different kinds of "not certain", and collapsing any pair of them
destroys the population that gets re-run later:

- `no_cleanroom_needed` is a DETERMINATION. You can name what they sell and it
  plainly needs no controlled environment.
- `likely_in_house` is an INFERENCE. You can name what they sell, it plainly
  DOES suggest a controlled environment, but nothing states one exists.
- `unknown` is ABSENCE OF EVIDENCE. You cannot even say what the company does.

Ask it in this order: could you name what this company sells? If no, that is
`unknown`. If yes — does what they sell need a controlled environment? If no,
that is `no_cleanroom_needed`. If yes, and nothing on the site states a
facility, that is `likely_in_house`. Only when a page actually describes the
facility does it become `in_house`.

**`no_cleanroom_needed` is a judgment about the product line, not a lookup.**
The four categories named in its definition are examples, not the permitted
set. A specialty chemical blender, an electronic component distributor, a
machine shop, a software vendor, a design consultancy, a crane manufacturer:
none of them appears in that list and every one of them is
`no_cleanroom_needed`, because you can see what they sell and it plainly needs
no controlled environment.

**Products matching no taxonomy term is the normal case, not a signal.** The
pack's product lists are short on purpose and most catalogues match nothing in
them. A domain reporting 200 product names and 3 taxonomy hits has told you
what it sells; it has not told you that you are ignorant. Never reach for
`unknown` because the taxonomy came back empty. Reach for it when the *page
text* is too thin to say what the business does.

Note also that the facility-evidence requirement above applies to `in_house`
ONLY. `no_cleanroom_needed` and `outsourced` are determinations from the
catalogue and the manufacturing language, and neither needs a cleanroom quote
to be made.

**`outsourced` needs a product.** It means the company SELLS something physical
that it does not make. A consultancy, design house or engineering service that
sells hours rather than goods is `no_cleanroom_needed`, not `outsourced`,
because there is no product being manufactured elsewhere. Ask what is in the
box the customer receives: if the answer is "nothing, they receive a service",
the label is `no_cleanroom_needed`.

**Doubt belongs in `confidence`, not in the label.** `unknown` does not mean
"not sure", it means "cannot tell what this company does". If you can name the
products and your best reading is that they need no controlled environment,
answer `no_cleanroom_needed` and carry the doubt in a lower confidence number.
If you can name the products and they imply a cleanroom with nothing stating
one, that doubt is exactly what `likely_in_house` exists to carry — do not
round it down to `unknown` just because you are not certain. A hedged
`unknown` sitting on a catalogue you have just described in your own reasoning
reads as caution and is really a refusal to answer. Measured on batch 1: three
verdicts named the product line, said in the reasoning that it plainly needs no
cleanroom, and still declined to say `no_cleanroom_needed`.

Roughly 15 to 20 percent of a normal population lands on `unknown`. If you find
yourself reaching for it because the evidence is mixed or inferred rather than
missing, the answer is one of the other four."""

_ATTRIBUTION = """\
## Attribution — whose facility, whose product, whose statement

Three ways evidence gets attached to the wrong entity. All three are
grammar-level judgment a keyword list cannot make, which is why they live
here rather than in the vocabulary below.

**CONTRACT-MANUFACTURER POLARITY.** "Contract manufacturer" cuts both ways
depending on who is the subject. When the company describes ITSELF that way —
"we are a contract manufacturer", "our contract manufacturing services", on
its own capabilities page — that establishes the company as a manufacturer
running its own facility, not evidence that it outsources. It is only
outsourcing evidence when the company names a contract manufacturer IT uses:
the company is the buyer of the service, not the seller of it. Read the
sentence's subject before deciding which way it points.

**CUSTOMER-ENVIRONMENT SCOPING.** A sentence describing where the CUSTOMER's
product ends up — the operating room, "your GMP facility", "your clinical
trials", a single OR, a hospital's own cleanroom — is never facility evidence
about the company on this page. It describes the customer's environment, not
the seller's. This is the same test as R5's possessive check, applied to the
object of the sentence instead of the subject: if the controlled environment
belongs to the reader ("your"), the vendor ("their"), or an unnamed third
party, it does not count toward this company's `in_house` determination.

**ENTITY-ESTABLISHING CUES ARE NOT CLEANROOM EVIDENCE.** Phrases like "hand
crafted", "designed and manufactured in [place]", "full service manufacturer"
tell you the company makes its own product — useful for confirming it is a
manufacturer rather than a distributor or design house — but they say nothing
about whether any part of that manufacturing happens in a controlled
environment. Never let language like this alone move `cleanroom_status` above
`unknown`. It can corroborate a real facility statement; it cannot substitute
for one.

**PRIOR-EMPLOYER BIOGRAPHIES.** An outsourcing cue ("contract manufacturer",
"outsources production", etc.) only counts when its subject is the company
itself, in the present tense. An executive biography describing where they
worked BEFORE joining the company — "prior to joining Acme, Jane was CEO of a
contract manufacturer", "before founding the company, he ran a CDMO" — is a
fact about a person's career history, not a statement about this company's
supply chain. Read who the sentence is actually about and when, before
treating an outsourcing word as evidence.

**MULTINATIONAL LOCAL-FACILITY TEST.** When the domain is a country or
regional site of a multinational, ask two questions in order: is this a US
location at all, and does THIS specific location — not the global brand —
operate the cleanroom. A brand-level claim ("PARI has been manufacturing
inhalation devices for 100 years", "the trusted brand for decades") describes
the company's history everywhere it operates, not necessarily what happens at
the address on this page. If the site itself calls this location an "office"
or "headquarters" and names no plant, facility, capabilities, quality, or
careers content of its own, the manufacturing claim belongs to the brand or
parent, not this entity — lean toward `distributor` unless the page
separately states that this location manufactures."""

_ENTITY_TYPE = """\
## entity_type

What kind of company this is, independent of the cleanroom question. Pick
exactly one, from what the evidence actually shows about how the company
makes money — not from a guess about its size or its certifications.

- `manufacturer` — makes its own product, for itself, under its own brand.
  The default when nothing below applies more specifically.
- `contract_manufacturer` — makes product FOR OTHER COMPANIES under their
  brands (CM, CDMO). The R20 test applies: this is the company's own
  description of itself ("we are a contract manufacturer", its own
  capabilities page), not a contract manufacturer it merely mentions using.
- `distributor` — resells other companies' products. Does not manufacture
  what it sells.
- `design_house` — designs and develops product but does not run its own
  manufacturing; production is contracted out or licensed elsewhere.
- `service_lab` — provides testing, diagnostic, or clinical laboratory
  services (CLIA/CAP-accredited labs and similar). Sells expertise applied to
  samples or specimens, not a manufactured physical product.
- `consultancy` — sells expertise or engineering hours, not a product. If the
  answer to "what does the customer receive" is "a service, not a thing",
  this is it.
- `cleanroom_vendor` — sells cleanroom equipment, furniture, consumables, or
  build-out services TO other companies. A supplier into this market, not a
  company operating a cleanroom for its own product.
- `acquired_stub` — a legacy brand or site for a company now operating under
  a parent, with no independent facility or operations of its own described.

This does not have to agree with `cleanroom_status` in a fixed way — a
`contract_manufacturer` can be `in_house`, a `distributor` is usually
`no_cleanroom_needed`, and so on — but describe what the evidence actually
shows for each independently rather than forcing them to match a pattern.

## referred_domain

Only when `cleanroom_status` is `outsourced` and the evidence names a
specific contract manufacturer or manufacturing partner (a company name, and
ideally a domain if one is stated or obviously inferable). Otherwise null.
Do not guess a domain that is not supported by the evidence."""

_TRAPS = """\
## Certifications are neutral

The certifications listed for each company are TRAP TERMS. They are carried by
nearly every medical device firm, including ones that will never see a
cleanroom, and reporting one is not evidence in either direction.

ISO 13485 is a quality management standard. So is ISO 9001. A wheelchair
manufacturer holds both. Seeing them tells you the company is a medical device
firm, which you already knew, and nothing whatsoever about its facilities.

Never subtract points for a trap term. Never add them. See it, discount it,
move on. Treating a long certification list as evidence of sophistication, or
its absence as evidence of nothing, is the single most common way to get this
classification wrong.

**The rule that makes this finite.** The only standard numbers that indicate a
controlled environment are ISO 14644 (cleanroom classification) and EU GMP
Grades A to D. Every other ISO, IEC, AS, IATF, EN or CFR number is a
management-system, product-safety or testing standard and is neutral, whether
or not it appears in the vocabulary below.

That covers ISO 14971, ISO 10993, ISO 15189, ISO 27001, ISO 22301, IATF 16949
and IEC 60601 without any of them needing to be listed. Apply it to numbers you
have never seen before. A standard you cannot identify is neutral, not
suspicious.

Two exceptions where the vocabulary carries a policy rather than a fact:
`GMP compliant` and `cGMP`. Pharmaceutical GMP does imply controlled areas; a
device firm's use of the word usually does not. Weigh them as weak support,
never as proof."""

_EVIDENCE_RULES = """\
## Evidence

Every quote you are shown has already been verified against the page it came
from, character for character. Treat them as trustworthy source text.

Return two to four of them in `evidence`, the ones a human would need to check
your verdict. COPY THEM EXACTLY AS GIVEN, along with the `source_url` printed
beside them. A quote you paraphrase, trim or merge cannot be traced back and
will be dropped from the record.

Choose evidence that carries the verdict, not evidence that is merely present:

- For `in_house`, the quote naming the class, the room, the gowning or the job.
- For `likely_in_house`, the product statement the inference rests on.
- For `outsourced`, the sentence saying who makes the product.
- For `no_cleanroom_needed`, the product or catalogue statement that shows
  what they sell.
- For `unknown`, whatever little there is. An empty list is acceptable here and
  nowhere else.

## Reasoning

Two to four sentences. Say which evidence decided it and which evidence you
discounted. If you are choosing between two labels, name the other one and say
what would have made you pick it. Write for someone auditing this verdict six
months from now with only your reasoning and your quotes.

## Confidence

How clearly the evidence supports the verdict, not how common the verdict is.

- 0.9 to 1.0 — stated plainly and directly about this company.
- 0.6 to 0.8 — a sound inference from products or capability language.
- 0.3 to 0.5 — thin, mixed, or resting on a single fragment.
- Below 0.3 — you are guessing. Consider `unknown` instead, and only if the
  evidence is missing rather than merely mixed.

## needs_manual_review

Set it to true for: an enterprise multi-division company where one division's
facilities say nothing about another's, a recent acquisition or parent/
subsidiary structure, a site that is largely not in English, or a verdict you
would want a person to look at before it reaches a sales list. It is a flag,
not a hedge; the verdict still has to be your best answer."""

_OUTPUT = """\
## Output format

Return exactly one JSON object, with exactly these eight keys, and nothing
else. No preamble, no explanation, no markdown fence.

{
  "cleanroom_status": "in_house" | "likely_in_house" | "outsourced" |
                       "no_cleanroom_needed" | "unknown",
  "iso_class": "ISO 7" | null,
  "confidence": 0.0,
  "reasoning": "...",
  "evidence": [{"quote": "...", "source_url": "https://..."}],
  "needs_manual_review": false,
  "entity_type": "manufacturer" | "contract_manufacturer" | "distributor" |
                  "design_house" | "service_lab" | "consultancy" |
                  "cleanroom_vendor" | "acquired_stub",
  "referred_domain": "example.com" | null
}

`cleanroom_status` is exactly one of the five strings, lower case with the
underscore. `iso_class` is the cleanest single statement of class if one is
stated ("ISO 7", "ISO Class 8", "Class 100,000"), otherwise null. A controlled
environment with no class stated is `in_house` with `iso_class: null`, never
`unknown` or `likely_in_house`. `entity_type` is always one of the eight
strings above — see below. `referred_domain` is null unless `cleanroom_status`
is `outsourced` and a specific manufacturing partner is named."""

_WORKED = """\
## Worked examples

### A — states a class

Quotes include "12,000 square feet of ISO Class 7 cleanroom space" and
"gowning rooms and HEPA-filtered laminar flow at every moulding cell".
Certifications: ISO 13485. Products match the positive taxonomy on catheters.

{"cleanroom_status": "in_house", "iso_class": "ISO Class 7", "confidence": 0.97,
 "reasoning": "The capabilities page states 12,000 sq ft of ISO Class 7 cleanroom
 space and describes gowning and laminar flow at the moulding cells, which is a
 facility rather than a product claim. ISO 13485 is carried but discounted as a
 quality management standard. Catheters in the catalogue are consistent with the
 facility evidence but are not what decided it.", ...}

### B — sterile product, no facility

Quotes include "Our devices are manufactured for us by trusted partners in
Ireland and Costa Rica" and "allowing our team to focus on design and clinical
education". No cleanroom or ISO class quotes at all. Certifications: FDA
registered.

{"cleanroom_status": "outsourced", "iso_class": null, "confidence": 0.93,
 "reasoning": "The company states plainly that its devices are made by partners
 abroad and describes its own role as design and clinical education, with no
 facility named anywhere in the evidence. Products ship sterile, which is a
 product property and not a facility, so in_house is not available on this
 evidence.", ...}

### C — durable equipment

Products match the negative taxonomy on wheelchairs and hospital beds, nothing
on the positive side. Manufacturing quote: "All frames are welded and
powder-coated at our plant in Grand Rapids". Certifications: ISO 9001.

{"cleanroom_status": "no_cleanroom_needed", "iso_class": null, "confidence": 0.9,
 "reasoning": "The catalogue is entirely durable patient-handling equipment,
 which is a positive determination from the taxonomy rather than an absence of
 evidence. The company does manufacture, at its own plant, but welding and
 powder-coating frames needs no controlled environment. Not unknown: the
 evidence says clearly what they sell.", ...}

### D2 — strong products, no facility evidence anywhere

Products match the positive taxonomy on catheters, stents and pacemakers. No
quote in any field mentions a cleanroom, a class, gowning or a controlled
environment. Manufacturing language is limited to "our global manufacturing
network".

{"cleanroom_status": "likely_in_house", "iso_class": null, "confidence": 0.55,
 "reasoning": "The catalogue is implantable and interventional product lines
 that are made in cleanrooms as a matter of industry practice, but nothing on
 the site describes a controlled environment, a class, gowning or a facility of
 any kind. That is an inference from the products, not a stated fact, so
 likely_in_house rather than in_house: the products imply a cleanroom and the
 site never evidences one. Not unknown, either: the catalogue plainly says what
 they make, and not no_cleanroom_needed, because implantables are not durable
 equipment. Flagged for review.",
 "needs_manual_review": true, ...}

### D — a cleanroom for something else

Quotes include "class 1,000 cleanroom for semiconductor wafer fabrication".
Products are optical encoders sold into medical and laboratory equipment.

{"cleanroom_status": "in_house", "iso_class": "Class 1,000", "confidence": 0.85,
 "reasoning": "A class 1,000 cleanroom is operated for semiconductor
 fabrication rather than for the medical product line. Under the broad question,
 does a cleanroom exist at all, that is in_house: the room exists and is theirs.
 Flagged for review because the room's purpose is not the medical business.",
 "needs_manual_review": true, ...}

### D3 — one small packaging room

A 40-person company that makes single-use surgical kits. One quote:
"Kitting and final packaging are performed in our ISO Class 8 clean room."
No square footage, no plant, no production line described anywhere.

{"cleanroom_status": "in_house", "iso_class": "ISO Class 8", "confidence": 0.95,
 "reasoning": "The company states that kitting and final packaging happen in its
 own ISO Class 8 clean room. That is a controlled environment it operates, and
 nothing more is required: the room's size is never stated and does not matter,
 and packaging is a real use of a cleanroom rather than a lesser one. No
 outsourcing language appears anywhere.", ...}

A single room used only for packaging is a full yes. Companies of this size
rarely describe square footage or a production line, and waiting for either is
how a real cleanroom gets missed.

### E — a distributor of cleanroom supplies

Products include "Controlled Environments and Cleanroom Solutions" as catalogue
categories. No quote describes a room the company operates.

{"cleanroom_status": "no_cleanroom_needed", "iso_class": null, "confidence": 0.88,
 "reasoning": "Every cleanroom mention is a product category in a distribution
 catalogue, not a facility this company runs. Distributor only resolves to
 no_cleanroom_needed. The word cleanroom appearing throughout the evidence is
 exactly the trap this label exists to catch.", ...}

### F — too thin to judge

Five pages retrieved, 688 words in total, no products named, no manufacturing
or facility language of any kind.

{"cleanroom_status": "unknown", "iso_class": null, "confidence": 0.4,
 "reasoning": "Five pages of brochureware with no product catalogue and no
 statement about manufacturing or facilities. There is not enough here to say
 what the company sells, let alone where it is made, so this is absence of
 evidence rather than a determination that no cleanroom is needed.", ...}"""

_CLOSE = """\
Read the evidence. Weigh it. Return the JSON object."""


def _render_terms(terms: Sequence[Term]) -> str:
    lines = []
    for term in terms:
        line = f"- {term.term}"
        if term.aliases:
            line += f" (also: {', '.join(term.aliases)})"
        if term.note:
            line += f"  — {term.note}"
        lines.append(line)
    return "\n".join(lines)


def _vocabulary(pack: SectorPack) -> str:
    """The pack's vocabulary, rendered into the prompt.

    All five lists appear here, including the two product taxonomies that are
    deliberately withheld from the extraction prompt. Withholding them there
    stops the extractor filtering its catalogue to the list; showing them here
    is the whole point of matching at classify time, and the model needs them to
    read the match summary it is given and to judge the names that matched
    nothing.
    """
    return "\n".join(
        [
            f"## Vocabulary (sector pack {pack.version}, sector {pack.sector})",
            "",
            "Weight is signal RELIABILITY, not direction. Direction comes from",
            "which list a term sits in: a `strong` negative product pushes",
            "toward `no_cleanroom_needed` exactly as firmly as a `strong` positive",
            "product pushes toward `likely_in_house`.",
            "",
            "None of these lists is exhaustive. Equivalent language that is not",
            "listed still counts, and a listed term is not a verdict on its own.",
            "",
            "### Positive products — consistent with needing a cleanroom",
            "",
            _render_terms(pack.positive_products),
            "",
            "### Negative products — consistent with needing none",
            "",
            _render_terms(pack.negative_products),
            "",
            "### Evidence vocabulary — controlled environments and processes",
            "",
            _render_terms(pack.evidence_terms),
            "",
            "### Outsourcing cues",
            "",
            _render_terms(pack.outsourcing_cues),
            "",
            "### Trap vocabulary — neutral, never subtract points",
            "",
            _render_terms(pack.trap_terms),
        ]
    )


def build_system_prompt(pack: SectorPack) -> str:
    """The classification system prompt for a given pack.

    Assembled from constant blocks plus the pack's rendered vocabulary, so a
    pack edit changes the prompt with no code change. The prefix is identical
    across every domain in a run, which is what lets it cache.
    """
    return "\n\n".join(
        [
            _ROLE,
            _QUESTION,
            _LABELS,
            _ATTRIBUTION,
            _ENTITY_TYPE,
            _TRAPS,
            _EVIDENCE_RULES,
            _OUTPUT,
            _vocabulary(pack),
            _WORKED,
            _CLOSE,
        ]
    )


def build_user_prompt(bundle: DomainEvidence) -> str:
    """One domain's evidence, in the order a reader would want it."""
    lines: list[str] = [
        f"DOMAIN: {bundle.domain}",
        f"PAGES ANALYSED: {bundle.pages}",
        f"PAGE TYPES: {', '.join(bundle.page_types) or 'none'}",
        f"TOTAL WORDS: {bundle.total_words}",
        "",
    ]

    if bundle.pages < THIN_SITE_PAGES:
        lines += [
            f"NOTE: fewer than {THIN_SITE_PAGES} pages were retrieved for this "
            "domain, which is the shape BUILD.md describes for `unknown`. Weigh "
            "that, but do not treat it as automatic: judge the evidence below.",
            "",
        ]

    lines.append("## Verified quotes")
    lines.append("")
    if bundle.quotes:
        lines.append(
            "Each quote was checked against its page character for character. "
            "Copy any you use exactly, with its URL."
        )
        lines.append("")
        shown = bundle.quotes[:MAX_QUOTES_PER_DOMAIN]
        by_field: dict[str, list[Quote]] = defaultdict(list)
        for quote in shown:
            by_field[quote.field].append(quote)
        for field_name in FINDING_FIELDS:
            group = by_field.get(field_name)
            if not group:
                continue
            lines.append(f"### {field_name} ({len(group)})")
            for quote in group:
                lines.append(f'- "{quote.quote}"')
                lines.append(
                    f"    claim: {quote.claim}  |  confidence: {quote.confidence:.2f}"
                )
                lines.append(f"    source_url: {quote.source_url}  ({quote.page_type})")
            lines.append("")
        if len(bundle.quotes) > MAX_QUOTES_PER_DOMAIN:
            lines.append(
                f"({len(bundle.quotes) - MAX_QUOTES_PER_DOMAIN} further quotes of "
                "lower priority were not shown.)"
            )
            lines.append("")
    else:
        lines += [
            "NONE. No page produced a verifiable statement about a controlled "
            "environment, a sterilisation method or how the product is made.",
            "",
        ]

    lines.append("## Products named on the site")
    lines.append("")
    taxonomy = bundle.taxonomy
    lines.append(f"{len(bundle.products)} distinct names.")
    lines.append("")
    lines.append(
        "Matched against the pack taxonomy in code, so this match is exact and "
        "you do not need to redo it. Judge whether it is representative."
    )
    lines.append("")
    lines.append(f"- POSITIVE taxonomy hits: {len(taxonomy.positive)} term(s)")
    for term in taxonomy.positive_terms:
        names = taxonomy.positive[term]
        lines.append(f"    {term}: {_join_sample(names)}")
    lines.append(f"- NEGATIVE taxonomy hits: {len(taxonomy.negative)} term(s)")
    for term in taxonomy.negative_terms:
        names = taxonomy.negative[term]
        lines.append(f"    {term}: {_join_sample(names)}")
    lines.append(f"- Matched no taxonomy term: {taxonomy.unmatched} name(s)")
    lines.append("")

    if bundle.products:
        sample = bundle.products[:MAX_PRODUCTS_PER_DOMAIN]
        lines.append(f"Raw names ({len(sample)} of {len(bundle.products)} shown):")
        lines.append(", ".join(sample))
        lines.append("")

    lines.append("## Certifications reported (NEUTRAL)")
    lines.append("")
    if bundle.trap_matched or bundle.trap_unmatched:
        if bundle.trap_matched:
            named = ", ".join(
                f"{term} ({', '.join(sorted(set(values)))})"
                for term, values in sorted(bundle.trap_matched.items())
            )
            lines.append(f"- In the pack's trap vocabulary: {named}")
        if bundle.trap_unmatched:
            lines.append(
                "- Not in the vocabulary: "
                + ", ".join(sorted(set(bundle.trap_unmatched)))
            )
            lines.append(
                "  Apply the standard-number rule to these. Being unlisted "
                "makes a certification neither suspicious nor significant."
            )
    else:
        lines.append("NONE reported.")
    lines.append("")
    lines.append("Return the JSON object now.")
    return "\n".join(lines)


def _join_sample(names: Sequence[str], limit: int = 6) -> str:
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown}, ... (+{len(names) - limit})"


# ---------------------------------------------------------------------------
# Verifying what came back
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerdictDrops:
    """What verification threw away on one domain, and what it was."""

    dropped_evidence: tuple[str, ...] = ()

    @property
    def evidence(self) -> int:
        return len(self.dropped_evidence)


def verify_verdict(
    verdict: DomainVerdict, bundle: DomainEvidence
) -> tuple[DomainVerdict, VerdictDrops]:
    """Drop evidence the extractions do not actually contain.

    The same discipline as Phase 5's quote check, one stage further down. There
    the question was "did the page say this"; here it is "did the pipeline
    collect this", and the answer has to be yes or the verdict cannot be
    audited without a re-crawl.

    Two checks per item, because an evidence entry is a pair:

    - the quote has to appear among the verified quotes sent for this domain;
    - the `source_url` has to be a page of this domain that was actually read.

    A URL that fails while the quote passes is repaired rather than dropped:
    the quote identifies its own page unambiguously, and losing real evidence
    to a mis-copied URL would be the same false-drop mistake that cost Phase 5
    a certification profile.
    """
    pool = bundle.quote_pool()
    by_quote = {canonical(q.quote): q for q in bundle.quotes}
    urls = set(bundle.urls)

    kept: list[Evidence] = []
    dropped: list[str] = []
    seen: set[str] = set()

    for item in verdict.evidence:
        key = canonical(item.quote)
        if not key or key not in pool:
            dropped.append(_sample(item.quote))
            log.debug("%s: dropped unsourced evidence: %r", bundle.domain, item.quote[:120])
            continue
        if key in seen:
            continue
        seen.add(key)

        source = item.source_url.strip()
        if source not in urls:
            original = by_quote.get(key)
            repaired = original.source_url if original else None
            if repaired is None:
                # An exact-match quote whose page cannot be identified means the
                # model stitched two quotes together. That is not evidence.
                dropped.append(_sample(item.quote))
                continue
            log.debug(
                "%s: repaired evidence url %r -> %r", bundle.domain, source, repaired
            )
            source = repaired
        kept.append(Evidence(quote=item.quote, source_url=source))

    return verdict.model_copy(update={"evidence": kept}), VerdictDrops(tuple(dropped))


DROP_SAMPLE_CHARS = 160


def _sample(value: str) -> str:
    text = " ".join(str(value).split())
    return text[:DROP_SAMPLE_CHARS] + ("..." if len(text) > DROP_SAMPLE_CHARS else "")


# ---------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------


class DomainClassification(BaseModel):
    """The outcome of one domain's classification, for the run report.

    Separate from `ClassificationRecord` because it describes the *call*, not
    the company. Failures live here and only here: a record asserting `unknown`
    because a call failed is indistinguishable from a genuinely thin site.
    """

    model_config = ConfigDict(extra="forbid")

    domain: str
    ok: bool = False

    cleanroom_status: str | None = None
    iso_class: str | None = None
    confidence: float = 0.0
    needs_manual_review: bool = False
    entity_type: str | None = None
    referred_domain: str | None = None

    pages_analyzed: int = 0
    quotes_available: int = 0
    evidence_quotes: int = 0
    evidence_kept: int = 0
    evidence_dropped: int = 0
    dropped_evidence: list[str] = Field(default_factory=list)

    positive_terms: list[str] = Field(default_factory=list)
    negative_terms: list[str] = Field(default_factory=list)
    trap_terms: list[str] = Field(default_factory=list)

    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None


class ClassificationRun(BaseModel):
    """Everything one classification run did.

    `label_counts` is the instrument this stage is read by. BUILD.md expects 15
    to 20 percent `unknown`; a run that collapses `unknown` into
    `no_cleanroom_needed`, or one that hides behind `unknown` when the
    evidence is merely mixed or inferred (that shape is `likely_in_house`), shows up
    here before it shows up in Phase 7b's precision numbers.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    pack_version: str
    crawl_date: date
    extraction_run_id: str
    provider: str
    model: str
    system_prompt_chars: int = 0

    started_at: datetime
    duration_s: float = 0.0

    domains_seen: int = 0
    domains_classified: int = 0
    domains_failed: int = 0
    domains_skipped: int = 0

    evidence_kept: int = 0
    evidence_dropped: int = 0
    needs_manual_review: int = 0

    records_written: int = 0
    parts: list[str] = Field(default_factory=list)

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    stopped_reason: str | None = None

    domains: list[DomainClassification] = Field(default_factory=list)

    @property
    def label_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.domains:
            if row.ok and row.cleanroom_status:
                counts[row.cleanroom_status] = counts.get(row.cleanroom_status, 0) + 1
        return counts

    @property
    def unknown_share(self) -> float:
        counts = self.label_counts
        total = sum(counts.values())
        return counts.get("unknown", 0) / total if total else 0.0

    @property
    def evidence_drop_rate(self) -> float:
        offered = self.evidence_kept + self.evidence_dropped
        return self.evidence_dropped / offered if offered else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "records": self.records_written,
            "failed": self.domains_failed,
            "labels": self.label_counts,
            "unknown_share": round(self.unknown_share, 4),
            "evidence_dropped": self.evidence_dropped,
            "cost_usd": round(self.cost_usd, 4),
        }


def extraction_runs(store: Store, pack_version: str) -> list[str]:
    """Every extraction run id present for a pack, ascending.

    Run ids are `YYYYmmdd-HHMMSS`, so lexical order is chronological order and
    the last one is the newest. Runs with no part file are ignored: a directory
    holding only a report is a run that wrote nothing.
    """
    prefix = f"extractions/pack={pack_version}/"
    runs: set[str] = set()
    for path in store.list_paths(prefix):
        if not path.endswith(".jsonl"):
            continue
        segments = path.split("/")
        if len(segments) >= 4 and segments[2].startswith("run="):
            runs.add(segments[2][len("run=") :])
    return sorted(runs)


#: A run id produced by `default_run_id`. Hand-named runs like `smoke-01` sort
#: after every digit and would otherwise win "latest" forever.
_TIMESTAMP_RUN = re.compile(r"^\d{8}-\d{6}$")


def latest_extraction_run(store: Store, pack_version: str) -> str | None:
    """The newest timestamped extraction run, or None.

    Hand-named runs are ignored rather than sorted: `smoke-01` is a 4-domain
    smoke test and picking it up by default would classify a tenth of the
    goldset and look like it worked.
    """
    runs = [run for run in extraction_runs(store, pack_version) if _TIMESTAMP_RUN.match(run)]
    return runs[-1] if runs else None


def report_path(pack_version: str, run_id: str) -> str:
    return f"classifications/pack={pack_version}/run={run_id}/report.json"


def csv_path(pack_version: str, run_id: str) -> str:
    return f"classifications/pack={pack_version}/run={run_id}/verdicts.csv"


def read_classifications(
    store: Store, pack_version: str, run_id: str
) -> list[ClassificationRecord]:
    """Every classification record in a run, in part order. Phase 7 reads this."""
    prefix = f"classifications/pack={pack_version}/run={run_id}"
    records: list[ClassificationRecord] = []
    for path in sorted(store.list_paths(prefix)):
        if path.endswith(".jsonl"):
            records.extend(store.read_jsonl(path, ClassificationRecord))
    return records


def existing_domains(store: Store, pack_version: str, run_id: str) -> set[str]:
    """Domains already classified under this run id, for resume."""
    prefix = f"classifications/pack={pack_version}/run={run_id}"
    done: set[str] = set()
    for path in store.list_paths(prefix):
        if not path.endswith(".jsonl"):
            continue
        for row in store.read_jsonl(path):
            domain = row.get("domain")
            if domain:
                done.add(domain)
    return done


class PartWriter:
    """Buffers records and flushes them to numbered part files.

    Identical in shape to Phase 5's writer, and identical in intent: part
    numbering continues past whatever the run directory already holds, so
    re-running a run id appends rather than overwrites.
    """

    def __init__(
        self,
        store: Store,
        pack_version: str,
        run_id: str,
        *,
        records_per_part: int = DEFAULT_RECORDS_PER_PART,
    ):
        self.store = store
        self.pack_version = pack_version
        self.run_id = run_id
        self.records_per_part = max(records_per_part, 1)
        self.index = self._next_index()
        self.written: list[str] = []
        self.count = 0
        self._buffer: list[ClassificationRecord] = []

    def _next_index(self) -> int:
        prefix = f"classifications/pack={self.pack_version}/run={self.run_id}"
        existing = [p for p in self.store.list_paths(prefix) if p.endswith(".jsonl")]
        return len(existing)

    def add(self, record: ClassificationRecord) -> None:
        self._buffer.append(record)
        if len(self._buffer) >= self.records_per_part:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        path = classifications_path(self.pack_version, self.run_id, self.index)
        self.store.write_jsonl(path, self._buffer)
        log.info("wrote %d records to %s", len(self._buffer), self.store.uri(path))
        self.written.append(path)
        self.count += len(self._buffer)
        self.index += 1
        self._buffer.clear()


# ---------------------------------------------------------------------------
# Classifying one domain
# ---------------------------------------------------------------------------


def _matcher_pass(
    store: Store, bundle: DomainEvidence, entity_type: str = "manufacturer"
) -> MatcherOutcome | None:
    """The medical_v2.2 deterministic QC layer, best-effort.

    Independent of the LLM call above and never allowed to break it: this is a
    new, additive read of the same raw pages (see match.py), and one domain's
    matcher hiccup — e.g. a raw page pruned from the store after extraction —
    must not lose a classification the LLM already produced successfully.

    `entity_type` should be the LLM's own answer for this domain (available
    since the LLM call above already ran) rather than the "manufacturer"
    default — decide()'s entity gate is only as good as what it's given.
    """
    try:
        return run_matcher(store, bundle.domain, bundle.crawl_date, entity_type=entity_type)
    except Exception as exc:  # noqa: BLE001 - best-effort QC layer, see docstring
        log.warning("%s: matcher pass failed, continuing without it: %s", bundle.domain, exc)
        return None


def _acquired_stub(
    bundle: DomainEvidence, *, run_id: str
) -> tuple[ClassificationRecord, DomainClassification, None]:
    """A domain redirected wholesale to a different registrable domain.

    HANDOFF.md's `site_moved` case: the company crawled is not the company
    that answered. Discovery already knows this (`DiscoveryResult`, threaded
    through as `resolved_domain`/`redirected_off_domain`); asking the LLM to
    render a cleanroom verdict from whatever the acquirer's site happens to
    say would be answering a question about the wrong company, and paying for
    the privilege. No call, no spend — the verdict a human needs here is
    "go look at who owns this now," not a five-way label.
    """
    reasoning = (
        f"{bundle.domain} redirects to a different registrable domain "
        f"({bundle.resolved_domain}); not classified as itself."
    )
    record = ClassificationRecord(
        domain=bundle.domain,
        cleanroom_status="unknown",
        confidence=0.0,
        reasoning=reasoning,
        pages_analyzed=bundle.pages,
        page_types_found=bundle.page_types,
        needs_manual_review=True,
        run_id=run_id,
        pack_version=bundle.pack_version,
        source_crawl_date=bundle.crawl_date,
        # Pipeline health, not "ok" — decide()'s own convention (matcher.py)
        # already treats run_status != "ok" as never a business verdict; this
        # is the one producer in classify.py that has a real, non-"ok" fact to
        # report. Distinguishes this from a genuinely thin-evidence `unknown`,
        # which a reader could otherwise not tell apart from "wrong company."
        run_status="site_moved",
        entity_type="acquired_stub",
        referred_domain=bundle.resolved_domain,
    )
    outcome = DomainClassification(
        domain=bundle.domain,
        ok=True,
        cleanroom_status=record.cleanroom_status,
        confidence=record.confidence,
        needs_manual_review=True,
        entity_type=record.entity_type,
        referred_domain=record.referred_domain,
        pages_analyzed=bundle.pages,
        quotes_available=len(bundle.quotes),
        evidence_quotes=len(bundle.evidence_quotes),
    )
    return record, outcome, None


def _thin_render_stub(
    bundle: DomainEvidence, *, run_id: str
) -> tuple[ClassificationRecord, DomainClassification, None]:
    """A domain whose median page is functionally empty.

    A bot wall, a consent/geo interstitial, or a JS shell the crawler never
    rendered through all answer 200 and bill a credit, but leave nothing for
    either consumer to read. No LLM call, no matcher pass, for the same
    reason as `_acquired_stub`: there is no verdict to render, only a
    pipeline-health fact to report. Confirmed on frantellgroup.com that
    skipping the matcher pass loses nothing real — its independent scan of
    the same raw pages also matched zero terms.
    """
    reasoning = (
        f"{bundle.domain}'s median crawled page is {bundle.median_page_word_count:g} "
        f"words, under the {THIN_RENDER_WORD_THRESHOLD}-word thin-render threshold; "
        "not enough content to classify."
    )
    record = ClassificationRecord(
        domain=bundle.domain,
        cleanroom_status="unknown",
        confidence=0.0,
        reasoning=reasoning,
        pages_analyzed=bundle.pages,
        page_types_found=bundle.page_types,
        needs_manual_review=True,
        run_id=run_id,
        pack_version=bundle.pack_version,
        source_crawl_date=bundle.crawl_date,
        # Pipeline health, not "ok" — same convention as site_moved. Keeps a
        # near-empty crawl distinguishable from a genuinely thin-evidence
        # `unknown`, which needs a different fix (better evidence, not a
        # re-crawl).
        run_status="thin_render",
        entity_type=None,
    )
    outcome = DomainClassification(
        domain=bundle.domain,
        ok=True,
        cleanroom_status=record.cleanroom_status,
        confidence=record.confidence,
        needs_manual_review=True,
        entity_type=record.entity_type,
        referred_domain=record.referred_domain,
        pages_analyzed=bundle.pages,
        quotes_available=len(bundle.quotes),
        evidence_quotes=len(bundle.evidence_quotes),
    )
    return record, outcome, None


def classify_domain(
    client: LLMClient,
    pack: SectorPack,
    bundle: DomainEvidence,
    store: Store,
    *,
    run_id: str,
    system: str | None = None,
) -> tuple[ClassificationRecord, DomainClassification, Completion[DomainVerdict] | None]:
    """One domain's evidence in, one verified verdict out.

    Raises whatever `llm.py` raises. The caller decides whether one domain's
    failure is worth stopping a batch for.
    """
    if bundle.redirected_off_domain:
        return _acquired_stub(bundle, run_id=run_id)

    if bundle.median_page_word_count < THIN_RENDER_WORD_THRESHOLD:
        return _thin_render_stub(bundle, run_id=run_id)

    completion = client.complete(
        stage=STAGE,
        system=system if system is not None else build_system_prompt(pack),
        user=build_user_prompt(bundle),
        schema=DomainVerdict,
    )

    verdict, drops = verify_verdict(completion.parsed, bundle)
    matcher_outcome = _matcher_pass(store, bundle, entity_type=verdict.entity_type)

    # A disagreement between the LLM and matcher.py's independent read is
    # exactly the signal the second opinion exists to surface. Silently
    # trusting the LLM's own needs_manual_review here would mean the QC layer
    # can be outvoted by the very verdict it is meant to check.
    #
    # A `no_cleanroom_needed` with no literal negative-taxonomy grounding is a
    # separate, more dangerous gap: it can be a correct product-line judgment
    # (the prompt deliberately allows this — see "Products matching no
    # taxonomy term is the normal case" above), but when it is wrong, it is
    # wrong silently and permanently — a real lead deleted from the pipeline
    # with nobody ever looking again. This does not touch the label; it only
    # guarantees the row enters the review queue instead of skipping it.
    ungrounded_negative = (
        verdict.cleanroom_status == "no_cleanroom_needed"
        and not bundle.taxonomy.negative
    )
    matcher_disagrees = (
        matcher_outcome is not None
        and matcher_outcome.verdict.cleanroom_status != verdict.cleanroom_status
    )

    # Named individually, not just OR'd into one bit, so a reader (or
    # eval/review_priority.py) can see WHY a row was flagged without
    # re-deriving it from taxonomy/matcher state that isn't even persisted.
    #
    # Every signal is RECORDED; only the ones in REVIEW_GATES decide whether the
    # row is queued. Before 2026-08-26 this was `bool(flag_reasons)`, i.e. every
    # signal was a gate, which put 84% of rows in the queue to catch 5 errors
    # that confidence alone catches at 61%. See REVIEW_GATES for the numbers.
    flag_reasons: list[str] = []
    if verdict.needs_manual_review:
        flag_reasons.append("llm_flagged")
    if verdict.confidence < AUTO_ACCEPT_CONFIDENCE:
        flag_reasons.append("low_confidence")
    if ungrounded_negative:
        flag_reasons.append("ungrounded_negative")
    if matcher_disagrees:
        flag_reasons.append("matcher_disagreement")
    needs_manual_review = any(reason in REVIEW_GATES for reason in flag_reasons)

    record = ClassificationRecord(
        domain=bundle.domain,
        cleanroom_status=verdict.cleanroom_status,
        iso_class=verdict.iso_class,
        confidence=verdict.confidence,
        reasoning=verdict.reasoning,
        evidence=verdict.evidence,
        pages_analyzed=bundle.pages,
        page_types_found=bundle.page_types,
        needs_manual_review=needs_manual_review,
        flag_reasons=flag_reasons,
        run_id=run_id,
        pack_version=bundle.pack_version,
        source_crawl_date=bundle.crawl_date,
        matched_terms=matcher_outcome.matched_terms if matcher_outcome else [],
        matched_terms_pack_version=matcher_outcome.pack_version if matcher_outcome else None,
        matcher_verdict=(asdict(matcher_outcome.verdict) if matcher_outcome else None),
        run_status="ok",
        entity_type=verdict.entity_type,
        referred_domain=verdict.referred_domain,
    )

    usage = completion.usage
    outcome = DomainClassification(
        domain=bundle.domain,
        ok=True,
        cleanroom_status=record.cleanroom_status,
        iso_class=record.iso_class,
        confidence=record.confidence,
        needs_manual_review=record.needs_manual_review,
        entity_type=record.entity_type,
        referred_domain=record.referred_domain,
        pages_analyzed=bundle.pages,
        quotes_available=len(bundle.quotes),
        evidence_quotes=len(bundle.evidence_quotes),
        evidence_kept=len(record.evidence),
        evidence_dropped=drops.evidence,
        dropped_evidence=list(drops.dropped_evidence),
        positive_terms=bundle.taxonomy.positive_terms,
        negative_terms=bundle.taxonomy.negative_terms,
        trap_terms=sorted(bundle.trap_matched),
        attempts=completion.attempts,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        cost_usd=usage.cost_usd,
    )
    return record, outcome, completion


# ---------------------------------------------------------------------------
# Classifying a batch
# ---------------------------------------------------------------------------


def _is_fatal(exc: BaseException) -> bool:
    """A failure that will repeat identically on every remaining domain."""
    return isinstance(exc, LLMError) and not isinstance(
        exc, (LLMResponseError, LLMTransportError)
    )


def group_records(
    records: Sequence[ExtractionRecord],
) -> dict[str, list[ExtractionRecord]]:
    """Extraction records grouped by domain, in first-seen order."""
    grouped: dict[str, list[ExtractionRecord]] = {}
    for record in records:
        grouped.setdefault(record.domain, []).append(record)
    return grouped


def classify_all(
    domains: Sequence[str],
    pack: SectorPack,
    store: Store,
    client: LLMClient,
    *,
    extraction_run_id: str,
    run_id: str | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    budget_usd: float | None = None,
    records_per_part: int = DEFAULT_RECORDS_PER_PART,
    resume: bool = True,
    on_domain: Callable[[DomainClassification], None] | None = None,
    write_report: bool = True,
) -> ClassificationRun:
    """Classify every domain that has extractions, in input order.

    Reads the extraction run once and holds it in memory. 448 records is a few
    megabytes; 5,000 will still be, and re-reading the store per domain would
    cost more than it saves.

    `budget_usd` is a stop line rather than a ceiling: calls already in flight
    when it is crossed finish and can overshoot.
    """
    run_id = run_id or default_run_id()
    started = datetime.now(timezone.utc)
    clock = time.monotonic()
    system = build_system_prompt(pack)

    records = read_records(store, pack.version, extraction_run_id)
    grouped = group_records(records)

    bundles: list[DomainEvidence] = []
    missing: dict[str, str] = {}
    for domain in domains:
        rows = grouped.get(domain)
        if not rows:
            # Same rule as Phase 4's withheld manifest: absence means "extract
            # this domain", never "this company had nothing to say".
            missing[domain] = f"no records in extraction run {extraction_run_id}"
            continue
        bundles.append(aggregate(rows, pack))

    crawl_date = bundles[0].crawl_date if bundles else date.today()

    run = ClassificationRun(
        run_id=run_id,
        pack_version=pack.version,
        crawl_date=crawl_date,
        extraction_run_id=extraction_run_id,
        provider=client.settings.provider,
        model=client.settings.model_id(STAGE),
        system_prompt_chars=len(system),
        started_at=started,
        domains_seen=len(domains),
    )

    done = existing_domains(store, pack.version, run_id) if resume else set()
    if done:
        log.info("resuming run %s: %d domains already classified", run_id, len(done))

    writer = PartWriter(store, pack.version, run_id, records_per_part=records_per_part)
    lock = threading.Lock()
    stop = threading.Event()
    stop_reason: list[str] = []
    spent = 0.0

    def halt(reason: str) -> None:
        if not stop.is_set():
            stop.set()
            stop_reason.append(reason)
            log.error("stopping the run: %s", reason)

    def run_one(
        bundle: DomainEvidence,
    ) -> tuple[ClassificationRecord | None, DomainClassification]:
        nonlocal spent
        if bundle.domain in done:
            return None, _skipped(bundle, "already classified in this run")
        if stop.is_set():
            return None, _skipped(bundle, stop_reason[0] if stop_reason else "run stopped")
        with lock:
            if budget_usd is not None and spent >= budget_usd:
                halt(f"budget of ${budget_usd:.2f} reached after ${spent:.2f}")
                return None, _skipped(bundle, stop_reason[0])

        try:
            record, outcome, _ = classify_domain(
                client, pack, bundle, store, run_id=run_id, system=system
            )
        except Exception as exc:  # noqa: BLE001 - one domain must not lose the run
            if _is_fatal(exc):
                halt(f"{type(exc).__name__}: {exc}")
            else:
                log.warning("%s: classification failed: %s", bundle.domain, exc)
            return None, _failed(bundle, f"{type(exc).__name__}: {exc}")

        # Tallied in the worker, not the consumer, for the same reason Phase 5
        # does it: `pool.map` submits everything up front, so a budget checked
        # against the consumer's tally is checked against a stale number.
        with lock:
            spent += outcome.cost_usd
        return record, outcome

    def consume(
        results: Iterable[tuple[ClassificationRecord | None, DomainClassification]],
    ) -> None:
        for record, outcome in results:
            run.domains.append(outcome)
            if on_domain is not None:
                on_domain(outcome)

            if outcome.skipped:
                run.domains_skipped += 1
                continue
            if not outcome.ok or record is None:
                run.domains_failed += 1
                continue

            run.domains_classified += 1
            run.evidence_kept += outcome.evidence_kept
            run.evidence_dropped += outcome.evidence_dropped
            run.needs_manual_review += int(outcome.needs_manual_review)
            run.input_tokens += outcome.input_tokens
            run.output_tokens += outcome.output_tokens
            run.cache_read_tokens += outcome.cache_read_tokens
            run.cache_write_tokens += outcome.cache_write_tokens
            run.cost_usd += outcome.cost_usd

            writer.add(record)

    if bundles:
        with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as pool:
            consume(pool.map(run_one, bundles))

    writer.flush()

    for domain, reason in missing.items():
        row = DomainClassification(domain=domain, error=reason)
        run.domains.append(row)
        run.domains_failed += 1
        log.warning("%s: %s", domain, reason)

    run.records_written = writer.count
    run.parts = writer.written
    run.duration_s = round(time.monotonic() - clock, 2)
    run.stopped_reason = stop_reason[0] if stop_reason else None

    if write_report:
        store.write_json(
            report_path(pack.version, run_id), run.model_dump(mode="json"), overwrite=True
        )
        store.write_text(
            csv_path(pack.version, run_id),
            build_csv(read_classifications(store, pack.version, run_id)),
            overwrite=True,
        )

    log.info("classification run %s", run.summary())
    return run


def _skipped(bundle: DomainEvidence, reason: str) -> DomainClassification:
    return DomainClassification(
        domain=bundle.domain,
        pages_analyzed=bundle.pages,
        skipped=True,
        skip_reason=reason,
    )


def _failed(bundle: DomainEvidence, error: str) -> DomainClassification:
    return DomainClassification(
        domain=bundle.domain,
        ok=False,
        pages_analyzed=bundle.pages,
        quotes_available=len(bundle.quotes),
        error=error,
    )


# ---------------------------------------------------------------------------
# The CSV report
# ---------------------------------------------------------------------------

CSV_COLUMNS: tuple[str, ...] = (
    "domain",
    "cleanroom_status",
    "iso_class",
    "confidence",
    "needs_manual_review",
    "pages_analyzed",
    "page_types_found",
    "evidence_count",
    "reasoning",
    "quote_1",
    "source_1",
    "quote_2",
    "source_2",
    "quote_3",
    "source_3",
    "entity_type",
    "referred_domain",
    "run_status",
    "matcher_cleanroom_status",
    "matcher_confidence",
    "matcher_agrees",
    "flag_reasons",
)

#: Top evidence quotes given their own columns.
CSV_EVIDENCE_SLOTS = 3


def build_csv(records: Sequence[ClassificationRecord]) -> str:
    """One row per domain, flat, with the top three quotes as columns.

    A CSV rather than the JSONL because this is the artifact a person opens.
    Flattening the evidence into columns is what lets it sort and filter in
    Excel with no join, at the cost of the fourth quote onward, which stays in
    the JSONL for anyone who needs it.
    """
    buffer = io.StringIO()
    # Excel on Windows reads a bare \n as one cell; newline="" plus the writer's
    # own \r\n is what keeps a multi-sentence `reasoning` in one row.
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(CSV_COLUMNS)
    for record in sorted(records, key=lambda r: r.domain):
        row: list[Any] = [
            record.domain,
            record.cleanroom_status,
            record.iso_class or "",
            f"{record.confidence:.2f}",
            "yes" if record.needs_manual_review else "",
            record.pages_analyzed,
            " ".join(record.page_types_found),
            len(record.evidence),
            " ".join(record.reasoning.split()),
        ]
        for slot in range(CSV_EVIDENCE_SLOTS):
            if slot < len(record.evidence):
                item = record.evidence[slot]
                row += [" ".join(item.quote.split()), item.source_url]
            else:
                row += ["", ""]
        row += [record.entity_type or "", record.referred_domain or ""]
        mv = record.matcher_verdict
        if mv is None:
            row += [record.run_status or "", "", "", ""]
        else:
            agrees = "yes" if mv["cleanroom_status"] == record.cleanroom_status else "no"
            row += [
                record.run_status or "",
                mv["cleanroom_status"],
                f"{mv['confidence']:.2f}",
                agrees,
            ]
        row.append(" ".join(record.flag_reasons))
        writer.writerow(row)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Pricing a run before it runs
# ---------------------------------------------------------------------------


def cost_estimate(
    store: Store,
    domains: Sequence[str],
    pack: SectorPack,
    client: LLMClient,
    *,
    extraction_run_id: str,
) -> dict[str, Any]:
    """What a classification pass would cost, from the records already on disk.

    Assumes the system prefix caches after the first call. Rough by design: it
    measures prompt characters, not tokens.
    """
    records = read_records(store, pack.version, extraction_run_id)
    grouped = group_records(records)
    spec = client.settings.spec(STAGE)
    system = build_system_prompt(pack)
    system_tokens = len(system) // 4

    missing: dict[str, str] = {}
    user_tokens = 0
    counted = 0
    for domain in domains:
        rows = grouped.get(domain)
        if not rows:
            missing[domain] = f"no records in extraction run {extraction_run_id}"
            continue
        user_tokens += len(build_user_prompt(aggregate(rows, pack))) // 4
        counted += 1

    output_tokens = counted * ESTIMATED_OUTPUT_TOKENS
    cost = spec.cost_usd(
        user_tokens,
        output_tokens,
        cache_read_tokens=system_tokens * max(counted - 1, 0),
        cache_write_tokens=system_tokens if counted else 0,
        batch=client.settings.use_batch,
    )
    return {
        "domains": counted,
        "pages": len(records),
        "missing": missing,
        "system_prompt_tokens": system_tokens,
        "caching": "on" if system_tokens >= spec.cache_min_tokens else "OFF: prompt too short",
        "input_tokens": user_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 4),
    }
