"""Record schemas for the extraction and classification stages.

Two rules from the spec are enforced here rather than in a prompt, because a
prompt is a request and a schema is a guarantee:

1. Every claim carries a verbatim quote. `Finding` cannot be built without one,
   so a model that wants to assert something without support has no way to
   express it and the field stays empty.
2. The required identity fields on a record are always populated. An empty
   `domain` or `url` is a parsing failure, not a finding, and must fail at the
   boundary instead of travelling downstream as a valid-looking record.

The list fields are allowed to be empty. Most pages have nothing to say about
sterilisation.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Storage-side type: permissive, spans both label generations so old JSONL
# (written "none") stays readable. Never widen the pipeline's OUTPUT to include
# "none" again — that's NewCleanroomStatus's job, see below.
CleanroomStatus = Literal[
    "in_house", "likely_in_house", "outsourced", "none", "no_cleanroom_needed", "unknown"
]

CLEANROOM_STATUSES: tuple[str, ...] = (
    "in_house", "likely_in_house", "outsourced", "none", "no_cleanroom_needed", "unknown"
)

# What a fresh classification is allowed to produce. "none" is retired here
# (-> "no_cleanroom_needed") and cannot be emitted by a new call, only read
# from old records via the wider CleanroomStatus above.
NewCleanroomStatus = Literal["in_house", "likely_in_house", "outsourced", "no_cleanroom_needed", "unknown"]

# medical_v2.2.yaml's labels.entity_type. Gates cleanroom_status in matcher.py's
# decide(); reported by the LLM here so decide()'s independent QC read can use
# a real value instead of the "manufacturer" default.
EntityType = Literal[
    "manufacturer", "contract_manufacturer", "distributor", "design_house",
    "service_lab", "consultancy", "cleanroom_vendor", "acquired_stub",
]

ENTITY_TYPES: tuple[str, ...] = (
    "manufacturer", "contract_manufacturer", "distributor", "design_house",
    "service_lab", "consultancy", "cleanroom_vendor", "acquired_stub",
)


def _require_text(value: str, field: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


# ---------------------------------------------------------------------------
# Phase 5: extraction
# ---------------------------------------------------------------------------


class Finding(BaseModel):
    """One thing a page says, with the words it said it in.

    `quote` is verbatim source text. It is what makes a classification
    auditable months later without a re-crawl, so it is mandatory and cannot be
    blank.
    """

    model_config = ConfigDict(extra="forbid")

    claim: str
    quote: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("claim")
    @classmethod
    def _claim_present(cls, value: str) -> str:
        return _require_text(value, "claim")

    @field_validator("quote")
    @classmethod
    def _quote_present(cls, value: str) -> str:
        # Not stripped to a canonical form: a quote is verbatim, and collapsing
        # its internal whitespace would break substring checks against the
        # stored markdown. Only the outer padding goes.
        return _require_text(value, "quote")


class ExtractionRecord(BaseModel):
    """What one page said. One record per page, one LLM call per record.

    The extractor decides nothing. `products_named` in particular stays
    uninterpreted: taxonomy matching happens at classify time against the pack,
    so changing the taxonomy never requires re-scraping.
    """

    model_config = ConfigDict(extra="forbid")

    domain: str
    url: str
    page_type: str
    crawl_date: date
    pack_version: str
    page_word_count: int = Field(ge=0)

    #: The registrable domain the crawl actually landed on, and whether that
    #: is off `domain`. Discovery's own fact (see discover.DiscoveryResult),
    #: carried per-record the same way crawl_date/pack_version are so classify
    #: time never has to re-derive an acquisition from whatever evidence the
    #: redirect happened to serve up.
    resolved_domain: str | None = None
    redirected_off_domain: bool = False

    iso_class_mentions: list[Finding] = Field(default_factory=list)
    cleanroom_terms: list[Finding] = Field(default_factory=list)
    sterilization_methods: list[Finding] = Field(default_factory=list)
    products_named: list[str] = Field(default_factory=list)
    manufacturing_language: list[Finding] = Field(default_factory=list)
    trap_terms_present: list[str] = Field(default_factory=list)

    @field_validator("domain", "url", "page_type", "pack_version")
    @classmethod
    def _required_text(cls, value: str, info) -> str:
        return _require_text(value, info.field_name)


# ---------------------------------------------------------------------------
# Phase 6: classification
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """A quote plus the page it came from, so a verdict can be spot-checked."""

    model_config = ConfigDict(extra="forbid")

    quote: str
    source_url: str

    @field_validator("quote", "source_url")
    @classmethod
    def _required_text(cls, value: str, info) -> str:
        return _require_text(value, info.field_name)


class ClassificationRecord(BaseModel):
    """One verdict per domain, over that domain's aggregated extractions.

    `none` is a positive determination from the product taxonomy. `unknown` is
    absence of evidence. They are different labels because collapsing them
    destroys the re-run population, so nothing in this schema lets one stand in
    for the other.

    `iso_class` is nullable on purpose: a controlled environment with no ISO
    class stated is still `in_house`.
    """

    model_config = ConfigDict(extra="forbid")

    domain: str
    cleanroom_status: CleanroomStatus
    iso_class: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    evidence: list[Evidence] = Field(default_factory=list)
    pages_analyzed: int = Field(ge=0)
    page_types_found: list[str] = Field(default_factory=list)
    needs_manual_review: bool = False
    #: Which of the checks OR'd into needs_manual_review actually fired:
    #: "llm_flagged", "ungrounded_negative", "matcher_disagreement". Lets a
    #: reader (or eval/review_priority.py) see WHY a row was flagged without
    #: re-deriving it — needs_manual_review alone collapses three different
    #: situations into one bit. Empty by default, including on every record
    #: written before this field existed.
    flag_reasons: list[str] = Field(default_factory=list)
    run_id: str
    pack_version: str
    source_crawl_date: date

    # -- matcher.py (medical_v2.2) deterministic QC layer, additive ---------
    # An independent, code-only read of the same raw pages, computed and
    # stored alongside the LLM verdict above rather than in place of it. See
    # match.py. `matcher_verdict` is `matcher.decide()`'s own Verdict
    # (cleanroom_status/confidence/entity_type/run_status/reasons), dumped
    # as-is: a second opinion to compare against `cleanroom_status`, not a
    # value that overwrites it.
    matched_terms: list[dict] = Field(default_factory=list)
    matched_terms_pack_version: str | None = None
    matcher_verdict: dict | None = None

    # -- verdict schema fields from the medical_v2.2 handoff (HANDOFF.md §4) -
    # Nullable: populated as each source comes online (run_status first, from
    # pipeline code; entity_type/referred_domain from the LLM, see classify.py).
    entity_type: EntityType | None = None
    run_status: str | None = None
    referred_domain: str | None = None

    @field_validator("domain", "reasoning", "run_id", "pack_version")
    @classmethod
    def _required_text(cls, value: str, info) -> str:
        return _require_text(value, info.field_name)

    @field_validator("iso_class")
    @classmethod
    def _blank_iso_class_is_null(cls, value: str | None) -> str | None:
        # "" and "none" are the two ways a model tries to say null in a string
        # field. Normalise both, otherwise downstream sees three empty values.
        if value is None:
            return None
        text = value.strip()
        if not text or text.lower() in {"null", "none", "n/a"}:
            return None
        return text
