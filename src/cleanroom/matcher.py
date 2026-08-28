"""
Cleanroom classifier — reference matcher.

THIS FILE IS THE SPEC. If prose in the workbook or the YAML disagrees with this
file, this file is right. Every future change to matching lands as a failing test
in test_matcher.py first.

Implements matching rules M1-M11 from medical_v2.2.yaml.

    from matcher import Pack
    pack = Pack.load("medical_v2.2.yaml")
    result = pack.score(page_text, url="https://example.com/capabilities")
    print(result.total, result.verdict_inputs())
    for m in result.matches:
        print(m.weight, m.tier, m.term, repr(m.quote))
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import yaml

# --------------------------------------------------------------------------
# M-normalize: canonicalization map. Applied to page text AND to pack terms.
# Order matters. Keep this list in sync with the YAML `matching.canonicalize`.
# --------------------------------------------------------------------------
_CANON: list[tuple[re.Pattern, object]] = [
    (re.compile(r"clean[\s\-]?rooms?", re.I), "cleanroom"),
    (re.compile(r"white[\s\-]?rooms?", re.I), "whiteroom"),
    (re.compile(r"grey[\s\-]?rooms?|gray[\s\-]?rooms?", re.I), "greyroom"),
    (re.compile(r"hard[\s\-]?wall", re.I), "hardwall"),
    (re.compile(r"soft[\s\-]?wall", re.I), "softwall"),
    (re.compile(r"\biso[\s\-]?(?:class[\s\-]?)?([1-9])\b", re.I), r"iso \1"),
    (re.compile(r"\bclass[\s\-]?(\d{1,3}(?:[,.]\d{3})*)\b", re.I),
     lambda m: "class " + m.group(1).replace(",", "").replace(".", "")),
    (re.compile(r"\bgrade[\s\-]?([a-d])\b", re.I), r"grade \1"),
    (re.compile(r"\biso[\s\-]?(\d{4,5})\b", re.I), r"iso \1"),
    (re.compile(r"\b21[\s\-]?cfr[\s\-]?(?:part[\s\-]?)?", re.I), "21 cfr "),
    (re.compile(r"\busp[\s\-]?<?\s*(\d+)\s*>?", re.I), r"usp \1"),
    (re.compile(r"in[\s\-]house", re.I), "in-house"),
    (re.compile(r"e[\s\-]?beam|electron[\s\-]beam", re.I), "ebeam"),
    (re.compile(r"\beto\b|\bethylene[\s\-]?oxide\b", re.I), "eo"),
    (re.compile(r"single[\s\-]?use", re.I), "single-use"),
    (re.compile(r"form[\s\-]?fill[\s\-]?seal", re.I), "form-fill-seal"),
    (re.compile(r"\bsq\.?\s?ft\.?|square\s?f(?:ee|oo)t", re.I), "sqft"),
    (re.compile(r"\bhepa[\s\-]?filters?\b", re.I), "hepa filter"),
]


def normalize(text: str) -> tuple[str, list[int]]:
    """Lowercase + canonicalize + collapse whitespace.

    Returns (normalized_text, offset_map) where offset_map[i] is the index in
    the ORIGINAL string that normalized character i came from. M10 requires the
    evidence quote to be verbatim from the original, so we must be able to map
    a normalized span back.
    """
    # Step 1: character-preserving transforms, tracking origin per character.
    src = text.replace("’", "'").replace("–", "-").replace("—", "-")
    lowered = src.lower()

    # Canonicalization can change length, so rebuild with an offset map.
    out_chars: list[str] = []
    out_map: list[int] = []
    i = 0
    while i < len(lowered):
        for rx, rep in _CANON:
            m = rx.match(lowered, i)
            if m:
                sub = rep(m) if callable(rep) else rx.sub(rep, m.group(0), count=1)
                out_chars.extend(sub)
                out_map.extend([m.start()] * len(sub))
                i = m.end()
                break
        else:
            out_chars.append(lowered[i])
            out_map.append(i)
            i += 1

    # Step 2: collapse whitespace runs, still tracking origin.
    final_chars: list[str] = []
    final_map: list[int] = []
    prev_ws = False
    for ch, src_i in zip(out_chars, out_map):
        if ch.isspace():
            if prev_ws:
                continue
            final_chars.append(" ")
            final_map.append(src_i)
            prev_ws = True
        else:
            final_chars.append(ch)
            final_map.append(src_i)
            prev_ws = False

    # trim
    start, end = 0, len(final_chars)
    while start < end and final_chars[start] == " ":
        start += 1
    while end > start and final_chars[end - 1] == " ":
        end -= 1
    return "".join(final_chars[start:end]), final_map[start:end]


def normalize_term(term: str) -> str:
    return normalize(term)[0]


# --------------------------------------------------------------------------
@dataclass
class Match:
    term: str
    tier: str
    weight: int
    category: str
    start: int          # span in NORMALIZED text
    end: int
    via: str            # which term/variant fired
    quote: str = ""     # verbatim from ORIGINAL text (M10)
    url: str = ""
    # Domain-level aggregation only (Pack.score_domain). A single-page score()
    # leaves these at their defaults: one occurrence, on the page it fired on.
    page_count: int = 1
    page_types: list[str] = field(default_factory=list)
    all_urls: list[str] = field(default_factory=list)

    @property
    def span_len(self) -> int:
        return self.end - self.start


@dataclass
class Result:
    matches: list[Match] = field(default_factory=list)
    suppressed: list[Match] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(m.weight for m in self.matches)

    def by_tier(self, tier: str) -> list[Match]:
        return [m for m in self.matches if m.tier == tier]

    def verdict_inputs(self) -> dict:
        neg = sum(m.weight for m in self.matches if m.weight < 0)
        return {
            "score": self.total,
            "negative_subtotal": neg,
            "has_direct": bool(self.by_tier("DIRECT")),
            "n_direct": len(self.by_tier("DIRECT")),
            "seller_cue": any(m.category == "seller_cues" for m in self.matches),
            "negative_product_hit": any(m.category == "negative_products" for m in self.matches),
            "strong_positive_product": any(
                m.category == "positive_products" and m.tier == "STRONG_INFERRED"
                for m in self.matches
            ),
            # R10's "no company-subject outsourcing cue" qualifier, simplified to
            # presence rather than subject-scoped: subject scoping is grammar-level
            # attribution, which is the LLM's job (see the note above `decide`).
            "outsourcing_cue": any(m.category == "outsourcing_cues" for m in self.matches),
        }

    # M11 — this is what makes every future weight a measurement instead of a guess.
    def matched_terms(self) -> list[dict]:
        return [
            {"term": m.term, "tier": m.tier, "weight": m.weight, "category": m.category,
             "url": m.url, "span": [m.start, m.end], "via": m.via, "quote": m.quote,
             "page_count": m.page_count, "page_types": m.page_types, "urls": m.all_urls or [m.url]}
            for m in self.matches
        ]


# --------------------------------------------------------------------------
class Pack:
    TERM_LISTS_SKIP = {"scoring", "matching", "labels", "attribution_rules",
                       "label_rules", "open_from_pass_1", "review_instrument"}

    def __init__(self, data: dict):
        self.data = data
        self.rows: list[dict] = []
        for key, val in data.items():
            if key in self.TERM_LISTS_SKIP or not isinstance(val, list):
                continue
            if not val or not isinstance(val[0], dict) or "term" not in val[0]:
                continue
            for r in val:
                r = dict(r)
                r["_category"] = key
                self.rows.append(r)
        fams = (data.get("matching", {}) or {}).get("dedup_families", {}) or {}
        self._fam = {cat: fam for fam, cats in fams.items() for cat in cats}
        self._compile()

    def family_of(self, category: str) -> str:
        return self._fam.get(category, category)

    @classmethod
    def load(cls, path: str) -> "Pack":
        with open(path) as fh:
            return cls(yaml.safe_load(fh))

    # ---- M1 / M2 / M3 / M4 -------------------------------------------------
    def _compile(self) -> None:
        for r in self.rows:
            pats: list[tuple[re.Pattern, str, bool]] = []   # (pattern, via, needs_context)
            ctx = {str(c).lower() for c in (r.get("needs_context") or [])}

            if r.get("match") == "regex":
                # M2: use term_pattern, never the bare term field.
                tp = r.get("term_pattern") or (r"\b" + re.escape(normalize_term(r["term"])) + r"\b")
                pats.append((re.compile(tp, re.I), r["term"], False))
                for v in (r.get("variants") or []):
                    try:
                        pats.append((re.compile(str(v), re.I), str(v), str(v).lower() in ctx))
                    except re.error:
                        pass
            else:
                # M1: whole-word literal, EVERY length.
                for cand in [r["term"], *(r.get("variants") or [])]:
                    n = normalize_term(str(cand))
                    if not n:
                        continue
                    pats.append((re.compile(r"\b" + re.escape(n) + r"\b", re.I),
                                 str(cand), str(cand).lower() in ctx))
            r["_pats"] = pats
            r["_suppress"] = (re.compile(r["suppress_if_preceded_by"], re.I)
                              if r.get("suppress_if_preceded_by") else None)
            # M7's original 30-char default suits a same-clause qualifier
            # ("temperature controlled"). A biography aside ("Prior to X,
            # the executive was Founder and CEO of Y, a ... contract manufacturer")
            # needs a wider look-back — override per row, default unchanged.
            r["_suppress_window"] = int(r.get("suppress_window", 30))
            r["_ctx_anchor"] = [p for p, via, nc in pats if not nc]

    # ---- scoring -----------------------------------------------------------
    def score(self, text: str, url: str = "") -> Result:
        norm, omap = normalize(text)
        cands: list[Match] = []
        suppressed: list[Match] = []

        for r in self.rows:
            w = r.get("weight", 0)
            if not isinstance(w, int):
                w = 0
            best: Match | None = None
            for pat, via, needs_ctx in r["_pats"]:
                for m in pat.finditer(norm):
                    # M7 — negative left-context suppression
                    if r["_suppress"] is not None:
                        left = norm[max(0, m.start() - r["_suppress_window"]):m.start()]
                        if r["_suppress"].search(left):
                            suppressed.append(Match(r["term"], r.get("tier", ""), w,
                                                    r["_category"], m.start(), m.end(), via, url=url))
                            continue
                    # M4 — acronym needs its expansion within 200 chars
                    if needs_ctx:
                        window = norm[max(0, m.start() - 200):m.end() + 200]
                        if not any(a.search(window) for a in r["_ctx_anchor"]):
                            suppressed.append(Match(r["term"], r.get("tier", ""), w,
                                                    r["_category"], m.start(), m.end(), via, url=url))
                            continue
                    cand = Match(r["term"], r.get("tier", ""), w, r["_category"],
                                 m.start(), m.end(), via, url=url)
                    # M6 — a row scores at most once; keep its longest span
                    if best is None or cand.span_len > best.span_len:
                        best = cand
            if best is not None:
                cands.append(best)

        # ---- M5: SPAN DEDUP, scoped per family. Longest span wins inside a
        # family; families are independent so an attribution cue does not
        # swallow the evidence term underneath it.
        cands.sort(key=lambda m: (-m.span_len, -abs(m.weight), m.start))
        taken: dict[str, list[tuple[int, int]]] = {}
        kept: list[Match] = []
        for m in cands:
            fam = self.family_of(m.category)
            spans = taken.setdefault(fam, [])
            if any(m.start < e and s < m.end for s, e in spans):
                suppressed.append(m)
                continue
            spans.append((m.start, m.end))
            # M10 — verbatim quote from the ORIGINAL text, widened to sentence-ish bounds
            if omap:
                a = omap[min(m.start, len(omap) - 1)]
                b = omap[min(m.end - 1, len(omap) - 1)] + 1
                lo = max(0, text.rfind(".", 0, a) + 1)
                hi = text.find(".", b)
                hi = len(text) if hi == -1 else hi + 1
                m.quote = text[lo:hi].strip()[:400]
            kept.append(m)

        kept.sort(key=lambda m: m.start)
        return Result(matches=kept, suppressed=suppressed)

    # ---- domain-level aggregation ------------------------------------------
    def score_domain(self, pages: Iterable[tuple[str, str, str]]) -> Result:
        """Collapse a whole domain's pages into one Result. `pages` is
        (url, page_type, text) tuples.

        M5/M6 dedup is per page and stays that way here: each page is scored
        exactly as `score()` scores it today. What changes is what happens
        after — a pack row that fires on five of a domain's pages (a nav
        banner, a footer line, a repeated headline) is one fact restated by a
        template, not five independent pieces of evidence, so it contributes
        its weight to the domain total exactly once. The repetition itself
        isn't discarded, it moves from the score to `page_count` /
        `page_types` on the kept Match: confidence inputs, not score inputs.
        """
        seen: dict[tuple[str, str], Match] = {}
        order: list[tuple[str, str]] = []

        for url, page_type, text in pages:
            page_result = self.score(text, url=url)
            for m in page_result.matches:
                key = (m.category, m.term)
                rep = seen.get(key)
                if rep is None:
                    rep = Match(m.term, m.tier, m.weight, m.category, m.start, m.end,
                                m.via, quote=m.quote, url=m.url, page_count=0)
                    seen[key] = rep
                    order.append(key)
                rep.page_count += 1
                if page_type not in rep.page_types:
                    rep.page_types.append(page_type)
                if url not in rep.all_urls:
                    rep.all_urls.append(url)

        return Result(matches=[seen[key] for key in order])

    # ---- iso_class extractor (M8) -----------------------------------------
    LEGACY = {"100": "ISO 5", "1000": "ISO 6", "10000": "ISO 7", "100000": "ISO 8"}
    GRADE = {"a": "ISO 5", "b": "ISO 5", "c": "ISO 7", "d": "ISO 8"}

    def iso_class(self, text: str) -> str | None:
        norm, _ = normalize(text)
        found: list[str] = []
        for m in re.finditer(r"\biso ([1-9])\b(?!\d)", norm):
            found.append(f"ISO {m.group(1)}")
        for m in re.finditer(r"\bclass (\d+)\b", norm):
            if m.group(1) in self.LEGACY:
                found.append(self.LEGACY[m.group(1)])
        for m in re.finditer(r"\bgrade ([a-d])\b", norm):
            found.append(self.GRADE[m.group(1)])
        if not found:
            return None
        # report the cleanest (lowest ISO number) stated
        return sorted(set(found), key=lambda s: int(s.split()[1]))[0]


def score_pages(pack: Pack, pages: Iterable[tuple[str, str]]) -> Result:
    """pages = iterable of (url, text). Rows score once per page (M6) and are
    NOT deduped across pages: a row that fires on every page of a domain is
    counted once per page here. For a domain-level verdict, use
    `Pack.score_domain()` instead, which collapses repeats across pages."""
    all_m: list[Match] = []
    all_s: list[Match] = []
    for url, txt in pages:
        r = pack.score(txt, url=url)
        all_m.extend(r.matches)
        all_s.extend(r.suppressed)
    return Result(matches=all_m, suppressed=all_s)


# --------------------------------------------------------------------------
# Verdict layer. Implements the bands in medical_v2.2.yaml scoring.bands plus
# the attribution gates R5/R8/R10 and the entity gate. Kept separate from
# matching on purpose: matching reports what the page says, this decides.
# --------------------------------------------------------------------------
POSSESSIVE = re.compile(
    r"\b(our|we operate|we maintain|we run|we manufacture|at our|in our|"
    r"located at our|on[- ]site|in-house|housed in our|within our)\b", re.I)

ELIGIBLE_ENTITIES = {"manufacturer", "contract_manufacturer"}


@dataclass
class Verdict:
    cleanroom_status: str
    confidence: float
    entity_type: str
    run_status: str
    iso_class: str | None
    needs_manual_review: bool
    reasons: list[str] = field(default_factory=list)


def decide(result: Result, text: str, *, entity_type: str = "manufacturer",
           run_status: str = "ok", iso_class: str | None = None,
           in_house_cue_cap: int = 6) -> Verdict:
    """Order matters and is enforced here, not left to condition ordering:

    1. run_status — pipeline health, never a business verdict.
    2. entity_type gate — computed once, up front. `cleanroom_vendor` returns
       immediately (a distinct, high-confidence determination: it sells into
       cleanrooms rather than operating one). Every other ineligible type
       (service_lab, distributor, design_house, consultancy, acquired_stub)
       is barred from BOTH `in_house` and `likely_in_house` for the rest of
       this call — not just the first band that happens to check `eligible`.
    3. R8 seller-cue suppression.
    4. Score bands — `in_house` first, then R10, applied LAST and only as a
       floor: a STRONG positive product with no outsourcing cue forbids
       `no_cleanroom_needed`, it does not by itself mint `likely_in_house`.
       Once that veto is resolved, the label is still score-determined.

    R20/R21/R22 (medical_v2.2.yaml's attribution_rules) are not implemented
    here on purpose: they all turn on the grammatical subject of a sentence
    (who is outsourcing to whom, whose environment is being described, which
    cues merely establish entity_type). Matching reports what a page says;
    subject-level attribution is judgment the LLM classifier makes over the
    same evidence (see classify.py's system prompt) — building it twice here
    would be exactly the two-independent-implementations failure this file
    exists to prevent.
    """
    vi = result.verdict_inputs()
    reasons: list[str] = []

    # 1. run_status is pipeline health and is never a business verdict.
    if run_status != "ok":
        return Verdict("unknown", 0.0, entity_type, run_status, None, True,
                       [f"run_status={run_status}; no business verdict emitted"])

    # 2. entity_type gate, computed once and applied to every band below.
    eligible = entity_type in ELIGIBLE_ENTITIES
    if entity_type == "cleanroom_vendor":
        return Verdict("no_cleanroom_needed", 0.85, entity_type, run_status, None, False,
                       ["entity_type=cleanroom_vendor: sells into cleanrooms rather than operating one"])
    if not eligible:
        reasons.append(f"entity_type={entity_type} is not eligible for in_house or likely_in_house")

    possessive = bool(POSSESSIVE.search(text))

    # 3. R8 — a seller cue suppresses co-occurring cleanroom evidence unless
    # the company is the grammatical subject of the cleanroom claim.
    if vi["seller_cue"] and not possessive:
        return Verdict("no_cleanroom_needed", 0.75, entity_type, run_status, None, True,
                       ["seller cue present with no possessive/self-subject cleanroom claim (R8)"])

    # cap attribution-cue stacking so cues alone cannot reach the in_house floor
    cue_total = sum(m.weight for m in result.matches if m.tier == "IN_HOUSE_CUE")
    score = vi["score"] - max(0, cue_total - in_house_cue_cap)
    has_direct = vi["has_direct"]

    # 4. Score bands.
    if eligible and score >= 10 and has_direct:
        return Verdict("in_house", 0.90, entity_type, run_status, iso_class, False,
                       reasons + [f"score {score} with {vi['n_direct']} DIRECT term(s)"])

    if eligible and 6 <= score < 10 and has_direct and possessive:
        return Verdict("in_house", 0.60, entity_type, run_status, iso_class, True,
                       reasons + [f"score {score} with a DIRECT term and a self-subject cue (R5)"])

    # R10, applied last: a STRONG positive product with no outsourcing cue
    # forbids no_cleanroom_needed. It is a veto on that one band, not an
    # alternate path to likely_in_house — the label past this point is still
    # decided by score alone.
    r10_blocks_negative = vi["strong_positive_product"] and not vi["outsourcing_cue"]

    if (vi["negative_subtotal"] <= -8 and vi["negative_product_hit"]
            and not r10_blocks_negative):
        return Verdict("no_cleanroom_needed", 0.80, entity_type, run_status, None, False,
                       reasons + [f"negative subtotal {vi['negative_subtotal']} with a negative-list match"])

    if eligible and score >= 6:
        return Verdict("likely_in_house", 0.55, entity_type, run_status, None, True,
                       reasons + [f"score {score}, no in_house band met; R10 veto={r10_blocks_negative}"])

    if not eligible:
        return Verdict("no_cleanroom_needed", 0.70, entity_type, run_status, None, True, reasons)

    return Verdict("unknown", 0.40, entity_type, run_status, None, True,
                   reasons + [f"score {score}, direct={has_direct}: matches no band"])
