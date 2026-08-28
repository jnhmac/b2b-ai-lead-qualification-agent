"""Phase 5: one LLM call per page, and nothing leaves it without a quote.

This is the first stage below the raw store, which means it runs fifty times
against pages that were scraped once. Everything here is designed to be cheap
to re-run and safe to throw away.

**The extractor decides nothing.** It reports what a page says. Whether a
company operates a cleanroom is Phase 6's call, made from the quotes collected
here, so a paraphrase at this layer is a judgment smuggled in one stage early.
`products_named` in particular stays uninterpreted: the pack's product taxonomy
is deliberately kept out of the prompt, because a model shown a list of
"products that indicate a cleanroom" quietly starts filtering to it, and then
changing the taxonomy means re-extracting instead of re-classifying.

**A schema guarantees a quote exists; only this module checks it is real.**
`Finding` cannot be constructed without a quote, but nothing in pydantic knows
whether that quote appears on the page. That gap is the entire gate, so after
parsing, both sides are canonicalised (markdown syntax and list markers
removed, trademark marks and typography folded, whitespace collapsed, case
folded) and every quote is checked for containment against the text the model
was shown. Findings that fail are dropped, per BUILD.md's "no quote, the field
stays empty", and both the count and the dropped text are reported per run: the
count says the rate moved, and only the text says whether the verifier is too
strict or the prompt too loose.

**Six fields are asked for, twelve are written.** `domain`, `url`, `page_type`,
`crawl_date`, `pack_version` and `page_word_count` all come from the crawl
manifest, so they are assembled here rather than generated. Cheaper in output
tokens, and a URL that is never generated is a URL that cannot be hallucinated.

**A failed page writes no record.** An `ExtractionRecord` with six empty lists
is indistinguishable from a page that had nothing to say, and Phase 6 counts
pages. Failures land in the run report instead, and re-running the same
`--run-id` retries exactly those pages, because the resume scan skips whatever
already made it into a part file.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .config import LLMStage, SectorPack, Term
from .llm import Completion, LLMClient, LLMError, LLMResponseError, LLMTransportError
from .models import ExtractionRecord, Finding
from .store import ObjectNotFoundError, Store, extractions_path, raw_manifest_path

log = logging.getLogger(__name__)

STAGE: LLMStage = "extract"

#: Pages run against a provider, not a local file, so this is bounded by the
#: provider's rate limits rather than by CPU. Four is comfortable on
#: OpenRouter and leaves headroom for the classify stage to share the account.
DEFAULT_CONCURRENCY = 4

#: Records per part file. Small enough that a crash costs minutes rather than a
#: whole run, large enough that a 500-domain run does not write 5,000 objects.
DEFAULT_RECORDS_PER_PART = 200

#: Page markdown is truncated at this many characters before it is sent. The
#: median stored page is ~610 words, so this fires on catalogue dumps only, and
#: an untruncated one of those costs more in input tokens than everything it
#: adds is worth. Quotes are verified against the truncated text, since that is
#: what the model actually saw.
MAX_PAGE_CHARS = 60_000

#: Below this a "quote" carries no evidence and would match almost any page.
MIN_QUOTE_CHARS = 3

#: Half-width, in characters, of the excerpt kept around each pack-keyword hit
#: on an oversized `investors` page. Head-truncation (MAX_PAGE_CHARS above) is
#: the wrong strategy for a filing: nvea.com's 10-K states the decisive fact
#: nowhere near the front matter, so the excerpt has to follow the evidence
#: instead of the document's start.
INVESTOR_WINDOW_CHARS = 500

#: Rough output size per page, used only by the pre-run estimate.
ESTIMATED_OUTPUT_TOKENS = 500

#: The finding fields, in schema order. `products_named` and
#: `trap_terms_present` are absent on purpose: they are lists of strings, not
#: claims, and are verified differently.
FINDING_FIELDS: tuple[str, ...] = (
    "iso_class_mentions",
    "cleanroom_terms",
    "sterilization_methods",
    "manufacturing_language",
)


# ---------------------------------------------------------------------------
# What the model is asked for
# ---------------------------------------------------------------------------


class PageFindings(BaseModel):
    """The six fields the model produces. The other six are assembled locally.

    Deliberately not `ExtractionRecord`: asking for `url` or `crawl_date` spends
    output tokens re-stating something the manifest already knows, and invites a
    hallucinated URL into a record whose whole job is to be auditable.
    """

    model_config = ConfigDict(extra="forbid")

    iso_class_mentions: list[Finding] = Field(default_factory=list)
    cleanroom_terms: list[Finding] = Field(default_factory=list)
    sterilization_methods: list[Finding] = Field(default_factory=list)
    products_named: list[str] = Field(default_factory=list)
    manufacturing_language: list[Finding] = Field(default_factory=list)
    trap_terms_present: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Verbatim checking
# ---------------------------------------------------------------------------
#
# The canonical form removes exactly two classes of difference: markdown syntax
# and typographic variation. It does not remove punctuation, reorder words or
# tolerate near-misses, because the point of the check is that the quote is the
# page's words and not the model's.

#: Images keep their alt text and lose their URL, exactly as links do. Dropping
#: the alt text was a real bug and an expensive one: a certifications page that
#: renders its certificates as logos has the certificate names *only* in the
#: alt text, and arterismedical.com lost its entire certification profile plus
#: a genuine "9,000 Sq. Ft. ISO Class 8 Clean Room Facility" headline that way.
#: Alt text is authored page content that a screen reader reads aloud; it is
#: not the model's invention, and it is the one place some pages state a fact.
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")

#: List markers at the start of a line. Markdown syntax exactly as `*` and `#`
#: are, and removing them is what lets a bulleted capability list match when it
#: is quoted as the run of text a reader sees. Measured on the first goldset
#: pass: a third of all dropped quotes were flattened bullet lists, every one
#: of them contiguous page content that a reader would call a single quote.
#: Contiguity is still enforced — the items have to be adjacent and in order —
#: so this loosens the syntax tolerated, not the "is it really on the page"
#: question the check exists to answer.
_LIST_MARKER_RE = re.compile(r"(?m)^[ \t]*(?:[-*+•·]|\d+[.)])[ \t]+")

#: Trademark and copyright marks. Product names in this sector are full of
#: them and models omit them roughly half the time; the page and the quote are
#: then the same words with different decoration. Measured: 16 real product
#: names lost to a stray registered-trademark sign on the first pass.
_MARKS = "®™℠©"

#: Emphasis, headings, code, quoting, table pipes and leftover link brackets.
#: A model reading rendered text quotes the words without the syntax around
#: them, and dropping it from both sides is what makes that a match.
_MARKDOWN_CHARS = "*_`~#>|[]\\"

#: Smart quotes, dashes and non-breaking spaces survive a scrape and rarely
#: survive a model repeating them.
_TYPOGRAPHY = {
    " ": " ",   # non-breaking space
    " ": " ",   # figure space
    " ": " ",   # narrow no-break space
    "​": "",    # zero-width space
    "­": "",    # soft hyphen
    "‘": "'",   # left single quote
    "’": "'",   # right single quote / apostrophe
    "‚": "'",
    "“": '"',
    "”": '"',
    "–": "-",   # en dash
    "—": "-",   # em dash
    "−": "-",   # minus sign
    "…": "...", # ellipsis
}

_CANONICAL_TABLE = str.maketrans(
    {
        **{char: None for char in _MARKDOWN_CHARS},
        **{char: None for char in _MARKS},
        **_TYPOGRAPHY,
    }
)


def canonical(text: str) -> str:
    """Markdown and typography stripped, whitespace collapsed, case folded.

    Order matters: list markers are removed line by line, before whitespace is
    collapsed, because once the newlines are gone there is no line start left
    to anchor them to.
    """
    # Images first: a thumbnail wrapped in a link, `[![alt](img)](href)`, has to
    # collapse from the inside out to leave `alt` rather than URL debris.
    without_images = _IMAGE_RE.sub(r"\1", text)
    linked = _LINK_RE.sub(r"\1", without_images)
    unlisted = _LIST_MARKER_RE.sub("", linked)
    return " ".join(unlisted.translate(_CANONICAL_TABLE).split()).casefold()


def is_verbatim(quote: str, canonical_page: str) -> bool:
    """True when `quote` appears in the page, ignoring markdown and case."""
    needle = canonical(quote)
    return len(needle) >= MIN_QUOTE_CHARS and needle in canonical_page


#: Dropped values are kept, not just counted. A count tells you the rate; only
#: the text tells you whether the verifier is too strict or the prompt too
#: loose, and those two have opposite fixes. Truncated because a dropped quote
#: can be a whole paragraph and the run report is read by a human.
DROP_SAMPLE_CHARS = 160


@dataclass(frozen=True)
class Drops:
    """What verification threw away on one page, and what it was."""

    dropped_quotes: tuple[str, ...] = ()
    dropped_products: tuple[str, ...] = ()
    dropped_trap_terms: tuple[str, ...] = ()

    @property
    def quotes(self) -> int:
        return len(self.dropped_quotes)

    @property
    def products(self) -> int:
        return len(self.dropped_products)

    @property
    def trap_terms(self) -> int:
        return len(self.dropped_trap_terms)

    @property
    def total(self) -> int:
        return self.quotes + self.products + self.trap_terms


def _sample(value: str) -> str:
    text = " ".join(str(value).split())
    return text[:DROP_SAMPLE_CHARS] + ("..." if len(text) > DROP_SAMPLE_CHARS else "")


def verify(findings: PageFindings, page_text: str) -> tuple[PageFindings, Drops]:
    """Drop everything the page does not actually say.

    Note what this does *not* take: the pack. Verification asks one question
    only, "is this on the page", and every pack-dependent judgment now happens
    at classify time. That is the same separation that keeps `products_named`
    uninterpreted, applied to the whole stage.

    Three checks, because the three kinds of field mean different things:

    - A `Finding` needs its quote to appear on the page. That is the gate.
    - A product name has no quote to check, so the name itself has to appear.
      Plural and singular are different strings and the prompt asks for the
      page's spelling, so a normalised name is a dropped name.
    - A trap term is a certification name, compared ignoring internal spacing
      and hyphens. See `_verify_trap_terms` for why that is the right equality
      for this one field and the wrong one everywhere else.
    """
    page = canonical(page_text)
    kept: dict[str, Any] = {}
    dropped_quotes: list[str] = []

    for field_name in FINDING_FIELDS:
        survivors = []
        for finding in getattr(findings, field_name):
            if is_verbatim(finding.quote, page):
                survivors.append(finding)
            else:
                dropped_quotes.append(f"{field_name}: {_sample(finding.quote)}")
                log.debug("dropped unverifiable quote in %s: %r", field_name, finding.quote[:120])
        kept[field_name] = survivors

    products, dropped_products = _verify_products(findings.products_named, page)
    traps, dropped_traps = _verify_trap_terms(findings.trap_terms_present, page)

    kept["products_named"] = products
    kept["trap_terms_present"] = traps

    return PageFindings(**kept), Drops(
        tuple(dropped_quotes), tuple(dropped_products), tuple(dropped_traps)
    )


def _verify_products(names: Sequence[str], canonical_page: str) -> tuple[list[str], list[str]]:
    seen: set[str] = set()
    kept: list[str] = []
    dropped: list[str] = []
    for name in names:
        key = canonical(name)
        if not key or key not in canonical_page:
            dropped.append(_sample(name))
            log.debug("dropped product name absent from the page: %r", name)
            continue
        if key in seen:
            continue
        seen.add(key)
        kept.append(name.strip())
    return kept, dropped


def _verify_trap_terms(values: Sequence[str], canonical_page: str) -> tuple[list[str], list[str]]:
    """Keep the certifications the page carries, uninterpreted.

    **Not matched against the pack.** Certifications are an unbounded long
    tail: 46 goldset domains carried 57 distinct ones, and a closed-vocabulary
    check threw away 88 real certifications that were plainly on their pages,
    for no safety benefit. The guard that catches a model inventing "FDA
    registered" from sector knowledge is page containment, not pack membership,
    and page containment is kept. So this field works exactly like
    `products_named`: report what is there, match it against the pack at
    classify time, where changing the list costs a re-classify instead of a
    re-extract.

    Compared ignoring internal spaces and hyphens, which is the right equality
    for certification names specifically: `ISO 9001` / `ISO9001`,
    `FDA registered` / `FDA-registered`, `AS9100D` / `AS 9100 D` are the same
    certification spelled by different marketing departments. That tolerance is
    deliberately confined to this field. A quote is evidence and must stay
    strict; a certification name is a token.
    """
    loose_page = _loose(canonical_page)
    seen: set[str] = set()
    kept: list[str] = []
    dropped: list[str] = []
    for value in values:
        key = _loose(canonical(value))
        if not key or key not in loose_page:
            dropped.append(_sample(value))
            log.debug("dropped trap term absent from the page: %r", value)
            continue
        if key in seen:
            continue
        seen.add(key)
        kept.append(value.strip())
    return kept, dropped


_PUNCTUATION_RE = re.compile(r"[\s\-]+")


def _loose(canonical_text: str) -> str:
    """Canonical text with spacing and hyphens removed. Certification names only."""
    return _PUNCTUATION_RE.sub("", canonical_text)


# ---------------------------------------------------------------------------
# The system prompt
# ---------------------------------------------------------------------------
#
# Long on purpose, and the length is an economy rather than a cost. Haiku 4.5
# caches nothing below 4,096 tokens: no error, no warning from the provider,
# just `cache_read_tokens` stuck at zero on the stage that is 90% of all calls.
# Above the threshold the whole prefix bills at a tenth of the input rate, so a
# detailed prompt with worked examples is *cheaper per page* than a terse one
# that never caches, as well as being a better prompt. `llm.py` warns once per
# client if this ever drops back under the line.

_ROLE = """\
You are an extraction engine. You are reading ONE page from ONE medical device
company's website, and you report what that page says about manufacturing
environments, sterilisation and products.

A separate system decides whether this company operates a cleanroom. That
decision is not yours. It is made months from now, by someone reading the quotes
you return and nothing else. A quote you tidy up, translate, join together or
reconstruct from memory is a decision you made on that person's behalf, and they
have no way to know you made it."""

_RULES = """\
## The two rules

RULE 1 — EVERY CLAIM CARRIES A VERBATIM QUOTE.
Copy the supporting words off the page character for character. Do not correct
spelling, expand abbreviations, fix capitalisation, translate from another
language, or stitch two sentences from different parts of the page into one
quote. If you cannot find words on the page that support a claim, do not make
the claim.

Every quote you return is checked against the page automatically. A quote that
cannot be found is deleted along with its claim, so an invented quote does not
get your claim through, it only loses it.

RULE 2 — YOU DECIDE NOTHING.
You are not judging whether this company has a cleanroom, whether its products
need one, or whether its certifications are impressive. You are not weighing
evidence, resolving contradictions, or filling gaps with what companies like
this usually do. If the page contradicts itself, report both statements. If the
page is a stub, return empty lists.

EMPTY LISTS ARE THE NORMAL ANSWER. Most pages of most websites say nothing
about sterilisation or manufacturing environments. A contact page, a leadership
page, a privacy policy: all six lists empty is the correct extraction, and it
costs nothing. Reaching for something to report is the expensive mistake."""

_OUTPUT = """\
## Output format

Return exactly one JSON object, with exactly these six keys, and nothing else.
No preamble, no explanation, no markdown fence.

{
  "iso_class_mentions": [{"claim": "...", "quote": "...", "confidence": 0.0}],
  "cleanroom_terms": [{"claim": "...", "quote": "...", "confidence": 0.0}],
  "sterilization_methods": [{"claim": "...", "quote": "...", "confidence": 0.0}],
  "products_named": ["...", "..."],
  "manufacturing_language": [{"claim": "...", "quote": "...", "confidence": 0.0}],
  "trap_terms_present": ["...", "..."]
}

All six keys must be present on every response. Use an empty list, never null,
never a string, never an object. `claim` is one short sentence in your own
words saying what the page asserts. `quote` is the page's words. `confidence`
is a number between 0 and 1."""

_FIELDS = """\
## The fields

### iso_class_mentions
A stated cleanliness classification for a room, a facility or a process.

Record: "ISO Class 7", "ISO 8 cleanroom", "Class 100,000", "Class 1,000",
"ISO 14644-1", "Grade C", "Class 10K", and any statement of how many rooms of
which class exist or how large they are.

Do NOT record: ISO 13485, ISO 9001, ISO 45001, ISO 27001. Those are management
system standards and say nothing about a room. They belong in
trap_terms_present. This is the single most common mistake on this field: the
string "ISO" is not the signal, a cleanliness class is.

If the page states a class without saying it is theirs — "products are made to
ISO Class 7 standards" — still record it, and let the claim say exactly what
the page said. Do not upgrade it to a facility they own.

### cleanroom_terms
Language describing a controlled production environment, whether or not a class
is attached.

Record: "cleanroom", "clean room", "controlled environment", "critical
environment", "white room", "gowning", "gown room", "airlock", "HEPA
filtration", "laminar flow", "positive pressure", "air changes per hour",
"particulate control", "environmental monitoring", "cleanroom square footage",
"cleanroom technician" in a job posting, "our clean rooms in X".

Do NOT record: "clean", "sterile", "hygienic", "state of the art facility",
"controlled process" on its own. A sterile product is not a cleanroom, and
"sterile" is the word most often mistaken for this field. A clean facility is
not a controlled environment. If the page only says the product is sterile,
this list stays empty.

Record cleanroom language even when the page says it belongs to someone else —
"our partner's ISO 7 cleanroom" — and make the claim say whose it is.

### sterilization_methods
A NAMED process for sterilising or controlling bioburden.

Record: ethylene oxide, EtO, EO, gamma irradiation, e-beam, electron beam,
X-ray sterilisation, steam, autoclave, dry heat, vaporised hydrogen peroxide,
VHP, chlorine dioxide, aseptic processing, terminal sterilisation, sterile
barrier packaging, bioburden testing, endotoxin testing, validation to
ISO 11135 or ISO 11137.

Do NOT record: "sterile", "sterilised", "sterility assured" with no method
named. "Our products are supplied sterile" names no process and belongs
nowhere. If the page says who performs the sterilisation, or that it is sent
out, put that statement in manufacturing_language as well, quoted separately.

### products_named
Every product, product family or device type this page names. A plain list of
strings, no quotes, no judgment.

Copy the page's own spelling and its own grammatical number. If the page says
"Balloon Catheters", write "Balloon Catheters", not "catheter". A name you
normalise is a name that fails verification and is deleted.

List them all, including products that have nothing to do with sterility:
wheelchairs, cabinets, software, cranes, encoders, fasteners. You are building
a catalogue, not evidence. Do not decide which products matter — something
downstream matches this list against a taxonomy you cannot see, and a filtered
list corrupts that match silently.

Include: device names, product families, categories the company says it makes
or sells, and named services when the service IS the product ("contract
sterilisation", "injection moulding"). Exclude: the names of standards,
certifications, customers, materials in the abstract, and navigation labels
like "Products" or "Shop All".

### manufacturing_language
Anything the page says about WHERE and BY WHOM its products are made. This is
the field that separates a company that makes sterile product from one that
sells it, so it holds both directions.

Record statements of in-house manufacture: "manufactured at our facility in
Ohio", "we operate three production sites", "vertically integrated", "our
50,000 sq ft plant", "all assembly is performed in house".

Record statements of outsourcing just as carefully: "manufactured for us by",
"our manufacturing partners", "we work with contract manufacturers", "designed
in California, made in Malaysia", "we are a design and marketing company",
"our supplier network". A page saying the company does NOT manufacture is one
of the most valuable things this pipeline can find, and there is no other field
for it.

Record capability language that implies a facility: "injection moulding",
"extrusion", "machining", "assembly", "packaging", "converting", when the page
presents them as things this company does.

Do NOT record: "quality manufacturing", "manufacturing excellence", "industry
leading manufacturer" and similar slogans with no facility, no location, no
process and no partner attached.

### trap_terms_present
Certifications and registrations that nearly every medical device company
carries, listed here so the downstream classifier can see them and DISCOUNT
them.

These are NEUTRAL, not negative. A wheelchair manufacturer and a catheter
manufacturer both hold ISO 13485; it says nothing whatsoever about facilities.
Reporting one is not evidence against a cleanroom any more than it is evidence
for one.

Report only terms drawn from the trap vocabulary listed below, spelled as the
vocabulary spells them. A plain list of strings. If the page carries none, the
list is empty."""

_PAGE_TYPES = """\
## What each page type usually holds

The PAGE TYPE line tells you which slot discovery selected this page for. It is
context, not an instruction: a careers page that describes a facility is still
describing a facility. Never let the type stop you recording something that is
plainly on the page, and never let it make you expect something that is not.

- homepage — positioning and a product summary. Usually thin on facilities.
  Watch for the one-line self-description: "a contract manufacturer of", "a
  design house for", "the leading distributor of". That sentence often decides
  the whole verdict, so quote it.
- capabilities / manufacturing — the richest pages in this pipeline. Process
  lists, equipment, floor space, cleanroom classes, and the sentence that says
  whether the work happens here or elsewhere.
- products — a catalogue. Get every product name. Product pages also carry
  sterility claims ("supplied sterile, single use"), which name no method and
  belong nowhere unless a process is named.
- facilities — floor plans, square footage, plant locations, room classes.
  Distinguish a manufacturing plant from a warehouse, an office or a
  distribution centre, and quote what the page actually calls it.
- quality — certifications and registrations. Mostly trap terms. A cleanroom
  class occasionally appears here as part of a validation statement.
- careers — job postings. A "Cleanroom Technician" or "Sterile Processing
  Operator" posting is real evidence a controlled environment exists, and it
  is evidence that survives on sites whose marketing copy never mentions one.
- about — history, leadership, locations. Watch for acquisition language and
  for parent-company structure; quote it plainly and let the classifier decide
  what it means.
- other — anything unpatterned: industries served, clinical pages, equipment
  lists, press releases. A press release is where "we have opened a new Class 7
  cleanroom" appears, so read these as carefully as the rest."""

_QUOTES = """\
## Quote discipline

- One sentence is usually the right length. Include enough that the claim
  stands up without the surrounding page, and stop.
- A quote may be a fragment, a heading, a table cell, a bullet, a job title, or
  the alt text of an image. Any of those is fine as long as it is contiguous
  text on the page.
- Alt text is text. DESCRIBING WHAT A PICTURE SHOWS IS NOT. If an image's alt
  text reads "ISO 13485 certification logo", that is quotable. If you are
  looking at a photograph and writing your own caption for what is happening in
  it, you are inventing a quote, and it will not verify.
- Never join text across a line break with different content on either side.
  Two facts are two findings.
- Never use an ellipsis to skip over words. Quote the shorter span instead.
- If the page is in a language other than English, quote it in that language
  and write the claim in English. Do not translate inside the quote.
- If the same fact appears twice on the page, report it once.
- Numbers, units and capitalisation come from the page exactly: "220,000 sq ft"
  not "220000 square feet".

## Confidence

Confidence is about how clearly the PAGE STATES the thing, not how important it
is and not how sure you are that a cleanroom exists.

- 0.9 to 1.0 — the page states it plainly and unambiguously about itself.
- 0.6 to 0.8 — stated, but with hedging, or about a partner, a customer or an
  unnamed party.
- 0.3 to 0.5 — implied by a heading, a job title, a caption or a fragment.
- Below 0.3 — do not report it. If it is that faint, the quote will not carry
  it either."""

_MISTAKES = """\
## The five mistakes that matter

1. Treating "sterile" as a cleanroom. A sterile product can be made anywhere
   and sterilised by a third party. cleanroom_terms stays empty unless the page
   describes an environment.
2. Filing ISO 13485 under iso_class_mentions. It is a quality management
   standard. It goes in trap_terms_present, and nowhere else.
3. Filtering products_named down to the medically interesting ones. List the
   cabinets and the cranes too.
4. Inferring a facility from an address. A headquarters, an office or a
   distribution centre is not a manufacturing site unless the page says so.
5. Reconstructing a quote from memory of the page instead of copying it. It
   will not verify, and the claim is lost with it.

Read the page. Report what it says. Return the JSON object."""


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

    Three lists only. `positive_products` and `negative_products` are held back
    deliberately: they are the taxonomy, and a model shown "products that
    indicate a cleanroom" starts reporting the ones on the list and dropping
    the ones that are not. Matching happens at classify time, which is what
    lets the taxonomy change without re-extracting a single page.
    """
    return "\n".join(
        [
            f"## Vocabulary (sector pack {pack.version}, sector {pack.sector})",
            "",
            "This vocabulary is a prompt for your attention, NOT a whitelist.",
            "Equivalent language that is not listed here still counts, and a",
            "listed term still needs a verbatim quote like anything else.",
            "",
            "### Evidence vocabulary — controlled environments and processes",
            "Language of this kind belongs in cleanroom_terms,",
            "iso_class_mentions or sterilization_methods, whichever fits.",
            "",
            _render_terms(pack.evidence_terms),
            "",
            "### Outsourcing cues — put these in manufacturing_language",
            "A page carrying these is telling you it may not make its own",
            "product. Quote it exactly; this is often the highest-value",
            "sentence on the whole site.",
            "",
            _render_terms(pack.outsourcing_cues),
            "",
            "### Trap vocabulary — report in trap_terms_present, spelled as below",
            "Neutral. Carried by nearly every company in this sector,",
            "including ones that will never see a cleanroom.",
            "",
            _render_terms(pack.trap_terms),
        ]
    )


_EXAMPLES = """\
## Worked examples

### Example A — a contract manufacturer's capabilities page

PAGE:
    # Precision Molding Capabilities
    Acme Medical is an ISO 13485:2016 certified contract manufacturer serving
    the cardiovascular market. Our Akron facility includes 12,000 square feet
    of ISO Class 7 cleanroom space, with gowning rooms and HEPA-filtered
    laminar flow at every moulding cell. We produce balloon catheters,
    guidewires and introducer sheaths, and offer validated ethylene oxide
    sterilisation through our sister site.

CORRECT:
{
  "iso_class_mentions": [
    {"claim": "The Akron site has 12,000 sq ft of ISO Class 7 cleanroom.",
     "quote": "12,000 square feet of ISO Class 7 cleanroom space",
     "confidence": 0.97}
  ],
  "cleanroom_terms": [
    {"claim": "Gowning rooms and HEPA laminar flow at the moulding cells.",
     "quote": "with gowning rooms and HEPA-filtered laminar flow at every moulding cell",
     "confidence": 0.95}
  ],
  "sterilization_methods": [
    {"claim": "Validated EtO sterilisation, performed at a sister site.",
     "quote": "offer validated ethylene oxide sterilisation through our sister site",
     "confidence": 0.9}
  ],
  "products_named": ["balloon catheters", "guidewires", "introducer sheaths"],
  "manufacturing_language": [
    {"claim": "Describes itself as a contract manufacturer with its own plant.",
     "quote": "Acme Medical is an ISO 13485:2016 certified contract manufacturer",
     "confidence": 0.9}
  ],
  "trap_terms_present": ["ISO 13485"]
}

Note three things. ISO 13485 appears in trap_terms_present and NOT in
iso_class_mentions, even though it starts with "ISO". The product names keep
the page's plural spelling. The sterilisation claim records that it happens at
a sister site, because that is what the page said.

### Example B — a brand page that outsources everything

PAGE:
    ## About Northlake Surgical
    Northlake designs single-use surgical instruments for ambulatory centres.
    Every product ships sterile and is FDA registered. Our devices are
    manufactured for us by trusted partners in Ireland and Costa Rica, allowing
    our team to focus on design and clinical education.

CORRECT:
{
  "iso_class_mentions": [],
  "cleanroom_terms": [],
  "sterilization_methods": [],
  "products_named": ["single-use surgical instruments"],
  "manufacturing_language": [
    {"claim": "States that its devices are made by third-party partners abroad.",
     "quote": "Our devices are manufactured for us by trusted partners in Ireland and Costa Rica",
     "confidence": 0.96},
    {"claim": "Describes its own role as design and clinical education.",
     "quote": "allowing our team to focus on design and clinical education",
     "confidence": 0.85}
  ],
  "trap_terms_present": ["FDA registered"]
}

"Every product ships sterile" produced nothing at all. It names no method, so
it is not a sterilization_method, and a sterile product is not a cleanroom, so
it is not a cleanroom_term. The outsourcing sentence is the whole value of this
page, and it is quoted exactly.

### Example C — a durable equipment catalogue

PAGE:
    # Patient Handling
    Meridian builds powered wheelchairs, transfer benches, bariatric hospital
    beds and overbed tables. All frames are welded and powder-coated at our
    plant in Grand Rapids. ISO 9001 certified since 1998.

CORRECT:
{
  "iso_class_mentions": [],
  "cleanroom_terms": [],
  "sterilization_methods": [],
  "products_named": ["powered wheelchairs", "transfer benches",
                     "bariatric hospital beds", "overbed tables"],
  "manufacturing_language": [
    {"claim": "Frames are welded and powder-coated at the company's own plant.",
     "quote": "All frames are welded and powder-coated at our plant in Grand Rapids",
     "confidence": 0.95}
  ],
  "trap_terms_present": ["ISO 9001"]
}

The manufacturing statement is recorded even though the products obviously need
no cleanroom. That is not your call. Report the facility, list the products,
and let the classifier put the two together.

### Example D — a careers page

PAGE:
    ### Open roles — Tijuana
    Cleanroom Assembly Technician (2nd shift)
    Quality Engineer, Sterile Packaging
    Maintenance Technician

CORRECT:
{
  "iso_class_mentions": [],
  "cleanroom_terms": [
    {"claim": "Hiring a cleanroom assembly technician at the Tijuana site.",
     "quote": "Cleanroom Assembly Technician (2nd shift)",
     "confidence": 0.8}
  ],
  "sterilization_methods": [
    {"claim": "Hiring a quality engineer for sterile packaging.",
     "quote": "Quality Engineer, Sterile Packaging",
     "confidence": 0.5}
  ],
  "products_named": [],
  "manufacturing_language": [],
  "trap_terms_present": []
}

A job title is a legitimate quote. Confidence is middling because a title is a
fragment, not a statement about the facility, and the sterile packaging role is
fainter still.

### Example E — a page with nothing to say

PAGE:
    # Contact
    Head office: 4400 Pine Street, Suite 210, Chicago IL.
    Phone: (312) 555-0134. For distributor enquiries, email sales@example.com.
    We are a leading manufacturer of quality medical products.

CORRECT:
{
  "iso_class_mentions": [],
  "cleanroom_terms": [],
  "sterilization_methods": [],
  "products_named": [],
  "manufacturing_language": [],
  "trap_terms_present": []
}

An address is not a facility. "A leading manufacturer of quality medical
products" is a slogan with no location, process or partner in it. Six empty
lists is the right answer and it is a common one."""


def build_system_prompt(pack: SectorPack) -> str:
    """The extraction system prompt for a given pack.

    Assembled from constant blocks plus the pack's rendered vocabulary, so a
    pack edit changes the prompt with no code change — and, because the prefix
    is stable across every page in a run, it caches.
    """
    return "\n\n".join(
        [
            _ROLE,
            _RULES,
            _OUTPUT,
            _FIELDS,
            _PAGE_TYPES,
            _QUOTES,
            _vocabulary(pack),
            _EXAMPLES,
            _MISTAKES,
        ]
    )


def build_user_prompt(page: "ManifestPage", markdown: str) -> str:
    """One page, with the two facts that help read it and nothing else."""
    return (
        f"URL: {page.url}\n"
        f"PAGE TYPE: {page.page_type}\n\n"
        "Everything between the markers is the page. Quote only from inside "
        "them; the URL line above is not part of the page.\n\n"
        "--- PAGE MARKDOWN BEGINS ---\n"
        f"{markdown}\n"
        "--- PAGE MARKDOWN ENDS ---\n\n"
        "Return the JSON object now."
    )


# ---------------------------------------------------------------------------
# Reading the crawl manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestPage:
    """One stored page, as the crawl manifest records it.

    The manifest's `fetch` block is authoritative. Listing the raw directory
    would find the same markdown files, but a `{url_hash}.md` filename does not
    say which URL it came from or which page type it was selected as.
    """

    domain: str
    url: str
    page_type: str
    stored_path: str
    word_count: int
    #: Phase 4's fact (see fetch.DomainCrawl), carried through so classify
    #: time knows this domain redirected wholesale to a different registrable
    #: domain, without re-deriving it from whatever evidence turned up.
    resolved_domain: str | None = None
    redirected_off_domain: bool = False


def manifest_pages(store: Store, domain: str, crawl_date: date | str) -> list[ManifestPage]:
    """Every successfully stored page for one domain, deduped by URL.

    Raises `ObjectNotFoundError` when the domain has no manifest. That is not
    the same as a domain with nothing to say: Phase 4 withholds the manifest
    when a crawl is incomplete, so its absence means "retry this domain".
    """
    manifest = store.read_json(raw_manifest_path(domain, crawl_date))
    fetch = manifest.get("fetch") or {}
    resolved_domain = fetch.get("resolved_domain")
    redirected_off_domain = bool(fetch.get("redirected_off_domain"))

    pages: dict[str, ManifestPage] = {}
    for entry in fetch.get("pages") or []:
        if not entry.get("ok") or not entry.get("stored_path"):
            continue
        url = str(entry.get("url") or "").strip()
        if not url or url in pages:
            continue
        pages[url] = ManifestPage(
            domain=domain,
            url=url,
            page_type=str(entry.get("page_type") or "other"),
            stored_path=str(entry["stored_path"]),
            word_count=int(entry.get("word_count") or 0),
            resolved_domain=resolved_domain,
            redirected_off_domain=redirected_off_domain,
        )
    return list(pages.values())


def crawl_dates(store: Store) -> list[str]:
    """Every crawl date present in the raw store, ascending."""
    return sorted(
        {
            path.split("/")[2]
            for path in store.list_paths("raw")
            if path.endswith("/manifest.json") and len(path.split("/")) >= 4
        }
    )


def latest_crawl_date(store: Store) -> str | None:
    dates = crawl_dates(store)
    return dates[-1] if dates else None


def crawled_domains(store: Store, crawl_date: date | str) -> list[str]:
    """Domains with a manifest for `crawl_date`, sorted."""
    wanted = crawl_date.isoformat() if isinstance(crawl_date, date) else str(crawl_date)
    return sorted(
        path.split("/")[1]
        for path in store.list_paths("raw")
        if path.endswith("/manifest.json") and path.split("/")[2:3] == [wanted]
    )


# ---------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------


class PageExtraction(BaseModel):
    """The outcome of one page's extraction, for the run report.

    Separate from `ExtractionRecord` because it describes the *call*, not the
    page. Failures live here and only here: an empty record in the JSONL would
    be indistinguishable from a page that genuinely said nothing.
    """

    model_config = ConfigDict(extra="forbid")

    domain: str
    url: str
    page_type: str

    ok: bool = False
    page_word_count: int = 0
    chars_sent: int = 0
    truncated: bool = False
    #: A local keyword scan found nothing in an oversized `investors` page,
    #: so the LLM was never called. Not the same as `skipped`: this is a
    #: real, completed determination with a real (empty) record, not an
    #: attempt that never ran.
    local_scan_only: bool = False

    findings_kept: int = 0
    quotes_dropped: int = 0
    products_dropped: int = 0
    trap_terms_dropped: int = 0

    #: What was dropped, not just how much. Reading these is how you tell a
    #: verifier that is too strict from a prompt that is too loose.
    dropped_quotes: list[str] = Field(default_factory=list)
    dropped_products: list[str] = Field(default_factory=list)
    dropped_trap_terms: list[str] = Field(default_factory=list)

    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None


class DomainExtraction(BaseModel):
    """Per-domain totals, printed live and kept in the report."""

    model_config = ConfigDict(extra="forbid")

    domain: str
    pages_seen: int = 0
    pages_extracted: int = 0
    pages_failed: int = 0
    pages_skipped: int = 0
    findings_kept: int = 0
    quotes_dropped: int = 0
    products_named: int = 0
    evidence_pages: int = 0
    cost_usd: float = 0.0
    error: str | None = None


class ExtractionRun(BaseModel):
    """Everything one extraction run did, including what it threw away.

    The drop counts are the gate's instrument. `quote_drop_rate` rising between
    two runs of the same pages is a prompt regression, and it is a number in a
    stored report rather than a log line precisely so two runs can be compared.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    pack_version: str
    crawl_date: date
    provider: str
    model: str
    system_prompt_chars: int = 0

    started_at: datetime
    duration_s: float = 0.0

    domains_seen: int = 0
    pages_seen: int = 0
    pages_extracted: int = 0
    pages_failed: int = 0
    pages_skipped: int = 0

    findings_kept: int = 0
    quotes_dropped: int = 0
    products_dropped: int = 0
    trap_terms_dropped: int = 0

    records_written: int = 0
    parts: list[str] = Field(default_factory=list)

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    stopped_reason: str | None = None

    domains: list[DomainExtraction] = Field(default_factory=list)
    pages: list[PageExtraction] = Field(default_factory=list)

    @property
    def quote_drop_rate(self) -> float:
        claimed = self.findings_kept + self.quotes_dropped
        return self.quotes_dropped / claimed if claimed else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "records": self.records_written,
            "pages_failed": self.pages_failed,
            "findings": self.findings_kept,
            "quotes_dropped": self.quotes_dropped,
            "quote_drop_rate": round(self.quote_drop_rate, 4),
            "cost_usd": round(self.cost_usd, 4),
        }


def default_run_id(now: datetime | None = None) -> str:
    """A sortable, path-safe run id. Colons are not allowed in a store segment."""
    moment = now or datetime.now(timezone.utc)
    return moment.strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------------
# Writing parts
# ---------------------------------------------------------------------------


class PartWriter:
    """Buffers records and flushes them to numbered part files.

    Part numbering continues past whatever is already in the run directory, so
    re-running an existing `run_id` appends rather than overwrites. The store
    refuses to clobber anyway; this makes the resume path the normal one
    instead of an error to work around.
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
        self._buffer: list[ExtractionRecord] = []

    def _next_index(self) -> int:
        prefix = f"extractions/pack={self.pack_version}/run={self.run_id}"
        existing = [
            path for path in self.store.list_paths(prefix) if path.endswith(".jsonl")
        ]
        return len(existing)

    def add(self, record: ExtractionRecord) -> None:
        self._buffer.append(record)
        if len(self._buffer) >= self.records_per_part:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        path = extractions_path(self.pack_version, self.run_id, self.index)
        self.store.write_jsonl(path, self._buffer)
        log.info("wrote %d records to %s", len(self._buffer), self.store.uri(path))
        self.written.append(path)
        self.count += len(self._buffer)
        self.index += 1
        self._buffer.clear()


def existing_pages(store: Store, pack_version: str, run_id: str) -> set[tuple[str, str]]:
    """The (domain, url) pairs already extracted under this run id.

    Re-running the same run id then retries exactly the pages that failed, at
    the cost of one listing and a read of what is already there.
    """
    prefix = f"extractions/pack={pack_version}/run={run_id}"
    done: set[tuple[str, str]] = set()
    for path in store.list_paths(prefix):
        if not path.endswith(".jsonl"):
            continue
        for row in store.read_jsonl(path):
            domain, url = row.get("domain"), row.get("url")
            if domain and url:
                done.add((domain, url))
    return done


def read_records(store: Store, pack_version: str, run_id: str) -> list[ExtractionRecord]:
    """Every extraction record in a run, in part order. Phase 6 reads this."""
    prefix = f"extractions/pack={pack_version}/run={run_id}"
    records: list[ExtractionRecord] = []
    for path in sorted(store.list_paths(prefix)):
        if path.endswith(".jsonl"):
            records.extend(store.read_jsonl(path, ExtractionRecord))
    return records


def report_path(pack_version: str, run_id: str) -> str:
    return f"extractions/pack={pack_version}/run={run_id}/report.json"


# ---------------------------------------------------------------------------
# Windowing an oversized `investors` page (HANDOFF.md item 7)
# ---------------------------------------------------------------------------


def _pack_keywords(pack: SectorPack) -> list[str]:
    """Every evidence term and alias, lowercased, for a local text scan.

    Not a substitute for the model's judgment — a cheap, deterministic
    pre-filter so a 100+ page filing is only ever paid for when something in
    it is actually worth reading.
    """
    keywords: list[str] = []
    for term in pack.evidence_terms:
        keywords.append(term.term.lower())
        keywords.extend(alias.lower() for alias in term.aliases)
    return keywords


def _windowed_excerpt(
    text: str, keywords: Sequence[str], *, window: int = INVESTOR_WINDOW_CHARS, budget: int = MAX_PAGE_CHARS
) -> str | None:
    """Excerpt `text` around every keyword hit, merging overlaps, or `None`.

    `None` means no keyword hit anywhere in the document — the caller's
    signal to skip the LLM call rather than pay it to confirm silence.
    Windows are kept in document order up to `budget` characters, not
    truncated from the front: the hit that matters in a large filing is as
    likely to be on page 80 as page 1.
    """
    lower = text.lower()
    hits = sorted({match.start() for kw in keywords for match in re.finditer(re.escape(kw), lower)})
    if not hits:
        return None

    spans: list[list[int]] = []
    for pos in hits:
        start, end = max(0, pos - window), min(len(text), pos + window)
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])

    pieces: list[str] = []
    kept = 0
    for start, end in spans:
        if kept >= budget:
            break
        piece = text[start : start + min(end - start, budget - kept)]
        pieces.append(piece)
        kept += len(piece)

    return "\n\n[...]\n\n".join(pieces)


# ---------------------------------------------------------------------------
# Extracting one page
# ---------------------------------------------------------------------------


def extract_page(
    client: LLMClient,
    pack: SectorPack,
    page: ManifestPage,
    markdown: str,
    *,
    crawl_date: date,
    system: str | None = None,
) -> tuple[ExtractionRecord, PageExtraction, Completion[PageFindings] | None]:
    """One page in, one verified record out.

    Raises whatever `llm.py` raises. The caller decides whether one page's
    failure is worth stopping a batch for.
    """
    text = markdown
    truncated = len(text) > MAX_PAGE_CHARS
    if truncated and page.page_type == "investors":
        excerpt = _windowed_excerpt(text, _pack_keywords(pack))
        if excerpt is None:
            log.info("%s: no pack keyword hit in %d chars; skipping the call", page.url, len(text))
            record = ExtractionRecord(
                domain=page.domain,
                url=page.url,
                page_type=page.page_type,
                crawl_date=crawl_date,
                pack_version=pack.version,
                page_word_count=page.word_count,
                resolved_domain=page.resolved_domain,
                redirected_off_domain=page.redirected_off_domain,
            )
            outcome = PageExtraction(
                domain=page.domain,
                url=page.url,
                page_type=page.page_type,
                ok=True,
                page_word_count=page.word_count,
                chars_sent=0,
                truncated=True,
                local_scan_only=True,
            )
            return record, outcome, None
        log.info("%s: windowing %d chars to %d around keyword hits", page.url, len(text), len(excerpt))
        text = excerpt
    elif truncated:
        log.info("%s: truncating %d chars to %d", page.url, len(text), MAX_PAGE_CHARS)
        text = text[:MAX_PAGE_CHARS] + "\n\n[page truncated]"

    completion = client.complete(
        stage=STAGE,
        system=system if system is not None else build_system_prompt(pack),
        user=build_user_prompt(page, text),
        schema=PageFindings,
    )

    # Verified against the text that was sent, not the file on disk: a quote
    # cannot come from a part of the page the model never saw.
    findings, drops = verify(completion.parsed, text)

    record = ExtractionRecord(
        domain=page.domain,
        url=page.url,
        page_type=page.page_type,
        crawl_date=crawl_date,
        pack_version=pack.version,
        page_word_count=page.word_count,
        resolved_domain=page.resolved_domain,
        redirected_off_domain=page.redirected_off_domain,
        **findings.model_dump(),
    )

    usage = completion.usage
    outcome = PageExtraction(
        domain=page.domain,
        url=page.url,
        page_type=page.page_type,
        ok=True,
        page_word_count=page.word_count,
        chars_sent=len(text),
        truncated=truncated,
        findings_kept=count_findings(record),
        quotes_dropped=drops.quotes,
        products_dropped=drops.products,
        trap_terms_dropped=drops.trap_terms,
        dropped_quotes=list(drops.dropped_quotes),
        dropped_products=list(drops.dropped_products),
        dropped_trap_terms=list(drops.dropped_trap_terms),
        attempts=completion.attempts,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        cost_usd=usage.cost_usd,
    )
    return record, outcome, completion


def count_findings(record: ExtractionRecord) -> int:
    return sum(len(getattr(record, field_name)) for field_name in FINDING_FIELDS)


def has_evidence(record: ExtractionRecord) -> bool:
    """Did the page say anything about a controlled environment at all?"""
    return bool(record.iso_class_mentions or record.cleanroom_terms)


# ---------------------------------------------------------------------------
# Extracting a batch
# ---------------------------------------------------------------------------


@dataclass
class _Task:
    domain: str
    page: ManifestPage


def _is_fatal(exc: BaseException) -> bool:
    """A failure that will repeat identically on every remaining page.

    `llm.py` raises a bare `LLMError` for a non-retryable HTTP status — a bad
    key, an exhausted balance, a malformed request. Retrying 450 more pages
    against it wastes wall clock and, on a 402, credit.
    """
    return isinstance(exc, LLMError) and not isinstance(
        exc, (LLMResponseError, LLMTransportError)
    )


def extract_all(
    domains: Sequence[str],
    pack: SectorPack,
    store: Store,
    client: LLMClient,
    *,
    crawl_date: date,
    run_id: str | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    budget_usd: float | None = None,
    records_per_part: int = DEFAULT_RECORDS_PER_PART,
    resume: bool = True,
    on_domain: Callable[[DomainExtraction], None] | None = None,
    write_report: bool = True,
) -> ExtractionRun:
    """Extract every stored page of every domain, in input order.

    `budget_usd` is a stop line rather than a ceiling: pages already in flight
    when it is crossed finish and can overshoot. It exists so a mis-typed
    `--limit` cannot turn a $3 pass into a $300 one.

    Results are consumed in submission order, which is what lets records be
    flushed to part files as the run proceeds and lets a per-domain row be
    printed the moment that domain's last page lands.
    """
    run_id = run_id or default_run_id()
    started = datetime.now(timezone.utc)
    clock = time.monotonic()
    system = build_system_prompt(pack)

    run = ExtractionRun(
        run_id=run_id,
        pack_version=pack.version,
        crawl_date=crawl_date,
        provider=client.settings.provider,
        model=client.settings.model_id(STAGE),
        system_prompt_chars=len(system),
        started_at=started,
        domains_seen=len(domains),
    )

    tasks, missing = _plan(store, domains, crawl_date)
    done = existing_pages(store, pack.version, run_id) if resume else set()
    if done:
        log.info("resuming run %s: %d pages already extracted", run_id, len(done))

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

    def run_one(task: _Task) -> tuple[ExtractionRecord | None, PageExtraction]:
        nonlocal spent
        page = task.page
        if (page.domain, page.url) in done:
            return None, _skipped(page, "already extracted in this run")
        if stop.is_set():
            return None, _skipped(page, stop_reason[0] if stop_reason else "run stopped")
        with lock:
            if budget_usd is not None and spent >= budget_usd:
                halt(f"budget of ${budget_usd:.2f} reached after ${spent:.2f}")
                return None, _skipped(page, stop_reason[0])

        try:
            markdown = store.read_text(page.stored_path)
        except Exception as exc:  # noqa: BLE001 - one page must not lose the run
            return None, _failed(page, f"{type(exc).__name__}: {exc}")

        try:
            record, outcome, _ = extract_page(
                client, pack, page, markdown, crawl_date=crawl_date, system=system
            )
        except Exception as exc:  # noqa: BLE001 - same
            if _is_fatal(exc):
                halt(f"{type(exc).__name__}: {exc}")
            else:
                log.warning("%s: extraction failed: %s", page.url, exc)
            return None, _failed(page, f"{type(exc).__name__}: {exc}")

        # Tallied here rather than in the consumer below. `pool.map` submits
        # every task up front, so the workers run arbitrarily far ahead of the
        # thread reading their results: a budget checked against a tally the
        # consumer maintains would be checked against a number minutes out of
        # date, and would not stop anything. The money is spent when the call
        # returns, so that is where it is counted.
        with lock:
            spent += outcome.cost_usd
        return record, outcome

    def consume(results: Iterable[tuple[ExtractionRecord | None, PageExtraction]]) -> None:
        current: DomainExtraction | None = None
        for record, outcome in results:
            if current is None or current.domain != outcome.domain:
                if current is not None:
                    _finish_domain(run, current, on_domain)
                current = DomainExtraction(domain=outcome.domain)

            current.pages_seen += 1
            run.pages_seen += 1
            run.pages.append(outcome)

            if outcome.skipped:
                current.pages_skipped += 1
                run.pages_skipped += 1
                continue
            if not outcome.ok or record is None:
                current.pages_failed += 1
                run.pages_failed += 1
                continue

            current.pages_extracted += 1
            current.findings_kept += outcome.findings_kept
            current.quotes_dropped += outcome.quotes_dropped
            current.products_named += len(record.products_named)
            current.evidence_pages += int(has_evidence(record))
            current.cost_usd += outcome.cost_usd

            run.pages_extracted += 1
            run.findings_kept += outcome.findings_kept
            run.quotes_dropped += outcome.quotes_dropped
            run.products_dropped += outcome.products_dropped
            run.trap_terms_dropped += outcome.trap_terms_dropped
            run.input_tokens += outcome.input_tokens
            run.output_tokens += outcome.output_tokens
            run.cache_read_tokens += outcome.cache_read_tokens
            run.cache_write_tokens += outcome.cache_write_tokens
            run.cost_usd += outcome.cost_usd

            writer.add(record)

        if current is not None:
            _finish_domain(run, current, on_domain)

    if tasks:
        with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as pool:
            consume(pool.map(run_one, tasks))

    writer.flush()

    for domain, reason in missing.items():
        run.domains.append(DomainExtraction(domain=domain, error=reason))
        log.warning("%s: %s", domain, reason)

    run.records_written = writer.count
    run.parts = writer.written
    run.duration_s = round(time.monotonic() - clock, 2)
    run.stopped_reason = stop_reason[0] if stop_reason else None

    if write_report:
        store.write_json(
            report_path(pack.version, run_id), run.model_dump(mode="json"), overwrite=True
        )

    log.info("extraction run %s", run.summary())
    return run


def _plan(
    store: Store, domains: Sequence[str], crawl_date: date
) -> tuple[list[_Task], dict[str, str]]:
    """Turn a domain list into a page list, recording domains with no manifest."""
    tasks: list[_Task] = []
    missing: dict[str, str] = {}
    for domain in domains:
        try:
            pages = manifest_pages(store, domain, crawl_date)
        except ObjectNotFoundError:
            # Phase 4 withholds the manifest on an incomplete crawl, so this
            # means "never crawled, retry it", not "this site said nothing".
            missing[domain] = f"no manifest for {crawl_date.isoformat()}; crawl it first"
            continue
        except Exception as exc:  # noqa: BLE001 - a bad manifest is one domain's problem
            missing[domain] = f"unreadable manifest: {type(exc).__name__}: {exc}"
            continue
        if not pages:
            missing[domain] = "manifest holds no successfully stored pages"
            continue
        tasks.extend(_Task(domain, page) for page in pages)
    return tasks, missing


def _finish_domain(
    run: ExtractionRun,
    summary: DomainExtraction,
    on_domain: Callable[[DomainExtraction], None] | None,
) -> None:
    run.domains.append(summary)
    if on_domain is not None:
        on_domain(summary)


def _skipped(page: ManifestPage, reason: str) -> PageExtraction:
    return PageExtraction(
        domain=page.domain,
        url=page.url,
        page_type=page.page_type,
        page_word_count=page.word_count,
        skipped=True,
        skip_reason=reason,
    )


def _failed(page: ManifestPage, error: str) -> PageExtraction:
    return PageExtraction(
        domain=page.domain,
        url=page.url,
        page_type=page.page_type,
        page_word_count=page.word_count,
        ok=False,
        error=error,
    )


# ---------------------------------------------------------------------------
# Pricing a run before it runs
# ---------------------------------------------------------------------------


def cost_estimate(
    store: Store,
    domains: Sequence[str],
    pack: SectorPack,
    client: LLMClient,
    *,
    crawl_date: date,
) -> dict[str, Any]:
    """What an extraction pass would cost, from the pages already on disk.

    Assumes the system prefix caches after the first call, which is the whole
    reason it is sized past 4,096 tokens. Rough by design: it reads file sizes,
    not tokenisers.
    """
    tasks, missing = _plan(store, domains, crawl_date)
    spec = client.settings.spec(STAGE)
    system_tokens = len(build_system_prompt(pack)) // 4

    page_tokens = 0
    for task in tasks:
        try:
            size = len(store.read_text(task.page.stored_path))
        except Exception:  # noqa: BLE001 - an estimate must not fail on one page
            size = task.page.word_count * 6
        page_tokens += min(size, MAX_PAGE_CHARS) // 4

    pages = len(tasks)
    output_tokens = pages * ESTIMATED_OUTPUT_TOKENS
    cost = spec.cost_usd(
        page_tokens,
        output_tokens,
        cache_read_tokens=system_tokens * max(pages - 1, 0),
        cache_write_tokens=system_tokens if pages else 0,
        batch=client.settings.use_batch,
    )
    return {
        "domains": len({task.domain for task in tasks}),
        "pages": pages,
        "missing": missing,
        "system_prompt_tokens": system_tokens,
        "caching": "on" if system_tokens >= spec.cache_min_tokens else "OFF: prompt too short",
        "input_tokens": page_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 4),
    }
