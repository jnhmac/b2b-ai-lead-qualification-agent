"""The conditional crawl ladder: steps 0 to 3 of `docs/handoff-v2.2/crawl_policy.md`.

The budget is conditional, not fixed. Two facts from the 2026-08-27 batch set
the shape: nine of thirty rows spent nine or ten pages and captured zero
evidence, and three of six group-1-eligible domains were misfiled into group 2
because a page was never fetched. So the only rows worth a deep crawl are the
ones where the product already qualifies as a lead but the cleanroom statement
has not been found. Everything else should resolve in two or three pages.

**Step 1 is model-driven, and this is the load-bearing decision in the module.**
Exit A asks whether a first-party cleanroom statement is present. The matcher
cannot answer that question: it fails to detect `avantalign.com`'s "Class 10K",
`katechon.com`'s "controlled-environment room" and, at domain level,
`nereidmedical.com`'s "clean room facilities" — three stated cleanrooms, two
missed on vocabulary and one outvoted by negative products. Gating the exit on a
matcher hit would rebuild the exact defect that misfiled three group-1 leads
this batch, because any paraphrase the pack does not carry would silently demote
a confirmed cleanroom to group 2.

The matcher still runs, and its read is recorded. Under Appendix B it is the
B-iv escalation trigger: a disagreement between "the matcher found a first-party
statement" and "the model cited none" is worth a second look. That is
corroboration, not a gate.

**The contract across the model boundary: the model returns quotes and source
URLs, never a verdict.** `Step1Read` carries no label. The branch is decided
here, deterministically, and every quote is verified verbatim against the page
before it can move a row. Retrieval is checkable because a quote either appears
on the page or it does not; a verdict from a model is not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .config import SectorPack
from .extract import canonical, is_verbatim
from .llm import LLMClient
from .matcher import POSSESSIVE
from .matcher import Pack as MatcherPack

log = logging.getLogger(__name__)

#: Step 1 fetches the homepage plus the single highest-scoring product page.
#: Two pages, because that is what resolved Neptune, Avalign, Clinton, iCAD and
#: Ingmar in the batch while the crawler went on to spend ten.
STEP_1_PAGE_BUDGET = 2

Exit = Literal["A", "B", "C"]


# ---------------------------------------------------------------------------
# What the model returns. Quotes and source URLs, never a verdict.
# ---------------------------------------------------------------------------


class Step1Read(BaseModel):
    """One reading of the step-1 pages. Every field is an observation.

    Deliberately carries no `cleanroom_status`, no confidence and no label. The
    moment this schema gains one, the branch below stops being auditable and
    the model is deciding rather than reporting.
    """

    model_config = ConfigDict(extra="ignore")

    #: Verbatim sentence stating a cleanroom, or null. Verified against the
    #: page before it is allowed to move anything.
    cleanroom_quote: str | None = None
    cleanroom_quote_url: str | None = None
    #: R5: is the company the subject of that sentence, or is it selling into
    #: someone else's cleanroom?
    company_is_subject: bool = False

    #: What the catalogue plainly is, in the model's words.
    catalogue_summary: str = ""
    #: Does that catalogue require a cleanroom to produce or package?
    catalogue_requires_cleanroom: Literal["yes", "no", "unclear"] = "unclear"
    #: The specific durable-equipment / furniture / software item that makes
    #: this a negative catalogue. Required for exit B, per tab 2: a positive
    #: determination names the item, absence of evidence does not.
    negative_catalogue_item: str | None = None


@dataclass
class Step1Branch:
    """The exit taken, why, and everything needed to audit it later."""

    exit: Exit
    reason: str
    read: Step1Read
    model_used: str
    ladder_step: int = 1
    quote: str = ""
    quote_url: str = ""

    #: The verdict this branch emits, or None when it emits nothing.
    verdict: str | None = None
    #: False whenever `verdict` is withheld. Exit B sets this False in phase 1
    #: by design; see `needs_strong_model_confirmation`.
    emitted: bool = False
    #: Appendix B.2. Exit B is the only irreversible call in the ladder: nobody
    #: revisits a `no_cleanroom_needed`, and a wrong one deletes a real lead
    #: silently and permanently. It must be confirmed on the strong model
    #: before it is emitted, and that routing lands in phase 3.
    needs_strong_model_confirmation: bool = False

    #: The matcher's independent answer to the same question, recorded for the
    #: B-iv escalation trigger. Never gates the exit.
    matcher_found_statement: bool = False

    #: The model returned a cleanroom quote that is not on any page it was
    #: given. Recorded rather than merely discarded: Appendix B.6 asks whether
    #: the small model is good enough for step 1, and this answers it
    #: continuously and for free on every row instead of by a ten-domain
    #: sample. It is the single most diagnostic signal available about a model
    #: at this step, because a fabricated quote is unambiguous in a way a
    #: wrong judgement is not.
    quote_verification_failed: bool = False

    pages_read: list[str] = field(default_factory=list)

    @property
    def should_continue(self) -> bool:
        return self.exit == "C"


# ---------------------------------------------------------------------------
# Page selection
# ---------------------------------------------------------------------------


def select_step_1_pages(pages: Sequence) -> list:
    """The homepage plus the highest-scoring product page, in that order.

    Takes `DiscoveredPage`-shaped objects. The homepage is always first because
    the pack marks exactly one type `always_fetch`, and the second slot goes to
    the best product candidate because a catalogue is what decides exits A and B
    on the sites that resolve in two pages.
    """
    homepage = [p for p in pages if p.page_type == "homepage"][:1]
    products = sorted(
        (p for p in pages if p.page_type == "products"),
        key=lambda p: (-p.score, p.url),
    )[:1]
    selected = homepage + products
    if len(selected) < STEP_1_PAGE_BUDGET:
        # No product page was discovered. Fill from the next-best typed page
        # rather than spending the slot on nothing: capabilities and
        # manufacturing describe a catalogue on sites that have no /products.
        rest = [
            p for p in pages
            if p not in selected and p.page_type not in ("homepage", "other")
        ]
        rest.sort(key=lambda p: (-p.score, p.url))
        selected += rest[: STEP_1_PAGE_BUDGET - len(selected)]
    return selected


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

SYSTEM = """\
You read one or two pages from a medical device company's website and report \
what they say. You do not classify the company.

Report exactly these observations:

1. cleanroom_quote: if any sentence states that a cleanroom, controlled \
environment, ISO class, gowning procedure or clean manufacturing area exists, \
copy that sentence VERBATIM. Copy the characters exactly as they appear. If no \
such sentence appears, use null. Do not paraphrase and do not infer.
2. cleanroom_quote_url: the URL of the page that sentence came from.
3. company_is_subject: true only if THIS company operates the environment \
described. False if the sentence is about selling into, supplying, building, \
certifying or cleaning someone else's cleanroom, or about a customer's site.
4. catalogue_summary: what this company makes or sells, in a few words.
5. catalogue_requires_cleanroom: "yes" if producing or packaging that catalogue \
needs a controlled environment; "no" only if the catalogue is entirely durable \
equipment, furniture, reusable instruments, software or services; "unclear" \
otherwise. Absence of evidence is "unclear", never "no".
6. negative_catalogue_item: when and only when you answered "no", name the \
specific item that makes it so. Otherwise null.

Never state a conclusion about whether the company operates a cleanroom. \
Report the sentences; something else decides."""


def build_user_prompt(pages: Iterable[tuple[str, str, str]]) -> str:
    blocks = []
    for url, page_type, text in pages:
        blocks.append(f"URL: {url}\nPAGE TYPE: {page_type}\n\n{text}")
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Step 1
# ---------------------------------------------------------------------------


def read_step_1(
    client: LLMClient,
    pack: SectorPack,
    pages: Sequence[tuple[str, str, str]],
    *,
    domain: str = "",
    matcher: MatcherPack | None = None,
    stage: str = "extract",
) -> Step1Branch:
    """Read the step-1 pages and take exit A, B or C.

    `stage` names the LLM stage, which is what carries the model tier. Phase 3
    turns Appendix B's routing table on by giving steps their own stages in
    `STAGE_MODELS`; nothing here branches on the model, because the ladder
    already knows which step it is on.
    """
    completion = client.complete(
        stage=stage,  # type: ignore[arg-type]
        system=SYSTEM,
        user=build_user_prompt(pages),
        schema=Step1Read,
    )
    read = completion.parsed
    urls = [url for url, _, _ in pages]

    branch = Step1Branch(
        exit="C",
        reason="",
        read=read,
        model_used=completion.model,
        pages_read=urls,
        matcher_found_statement=_matcher_sees_statement(matcher, pages),
    )

    # ---- Exit A: a first-party cleanroom statement, verified.
    if read.cleanroom_quote:
        branch.quote = read.cleanroom_quote
        branch.quote_url = read.cleanroom_quote_url or ""
        corpus = canonical("\n".join(text for _, _, text in pages))
        if not is_verbatim(read.cleanroom_quote, corpus):
            # A quote that is not on the page is a fabrication, and an exit A
            # on one emits in_house on no evidence at all. Falling through to
            # C is the safe direction: the row costs more to crawl, and the
            # deeper search may find the real sentence.
            branch.exit = "C"
            branch.quote_verification_failed = True
            branch.reason = (
                f"cleanroom quote is not verbatim on any step-1 page: "
                f"{read.cleanroom_quote[:80]!r}"
            )
            # Structured on purpose: event, domain, model. Grepping
            # `quote_not_verbatim` across a batch's logs gives the failure rate
            # per model, which is what B.6 is trying to buy with a ten-domain
            # comparison.
            log.warning(
                "quote_not_verbatim domain=%s model=%s step=1 quote=%r",
                domain or "?", completion.model, read.cleanroom_quote[:160],
            )
            return branch

        if not read.company_is_subject:
            branch.exit = "C"
            branch.reason = (
                "cleanroom statement found but not first-party (R5): "
                f"{read.cleanroom_quote[:80]!r}"
            )
            return branch

        branch.exit = "A"
        branch.verdict = "in_house"
        branch.emitted = True
        branch.reason = f"first-party cleanroom statement on {branch.quote_url or 'a step-1 page'}"
        return branch

    # ---- Exit B: a positively negative catalogue.
    if read.catalogue_requires_cleanroom == "no" and read.negative_catalogue_item:
        branch.exit = "B"
        # Recorded, NOT emitted. Appendix B.2 requires strong-model
        # confirmation before `no_cleanroom_needed` is written, and that
        # routing is phase 3. A wrong exit B deletes a lead permanently and
        # nobody revisits it, so the withheld verdict is the safe default until
        # the confirmation exists.
        branch.verdict = None
        branch.emitted = False
        branch.needs_strong_model_confirmation = True
        branch.reason = (
            f"catalogue is plainly negative ({read.negative_catalogue_item}); "
            "withheld pending strong-model confirmation (Appendix B.2)"
        )
        return branch

    # ---- Exit C: continue to step 2.
    branch.exit = "C"
    branch.reason = (
        f"catalogue {read.catalogue_summary!r} requires a cleanroom="
        f"{read.catalogue_requires_cleanroom}; no first-party statement found in "
        f"{len(pages)} page(s)"
    )
    return branch


def _matcher_sees_statement(
    matcher: MatcherPack | None, pages: Sequence[tuple[str, str, str]]
) -> bool:
    """The matcher's narrowed question: is there a first-party statement here?

    Recorded so phase 3 can fire the B-iv trigger on a disagreement. It is not
    consulted by the branch above and must not become so.
    """
    if matcher is None:
        return False
    try:
        text = "\n".join(t for _, _, t in pages)
        result = matcher.score_domain(list(pages))
        return bool(result.by_tier("DIRECT")) and bool(POSSESSIVE.search(text))
    except Exception as exc:  # noqa: BLE001 - a corroboration read must never break the crawl
        log.warning("step 1: matcher corroboration failed, continuing without it: %s", exc)
        return False
