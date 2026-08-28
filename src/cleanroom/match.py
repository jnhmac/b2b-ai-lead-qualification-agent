"""Glue between the store and `matcher.py`.

`matcher.py` is the spec for deterministic term matching (see its own
docstring). Nothing in this module ports or reimplements any of its scoring,
dedup or verdict logic — it only gets raw page text out of the store and into
`matcher.Pack.score()` / `matcher.decide()`, unmodified.

This runs as a new, independent QC/evidence layer alongside the existing
Haiku-extract + Sonnet-classify LLM pipeline (see classify.py), not a
replacement for it. It reads only what Phase 4 already crawled and stored, so
it costs nothing to run or re-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

from . import matcher
from .config import REPO_ROOT
from .extract import manifest_pages
from .store import Store

DEFAULT_PACK_NAME = "medical_v2.2"
PACK_DIR = REPO_ROOT / "packs"


def matcher_pack_path(name: str = DEFAULT_PACK_NAME, pack_dir: Path | None = None) -> Path:
    directory = pack_dir or PACK_DIR
    return directory / f"{name}.yaml"


@lru_cache(maxsize=4)
def get_matcher_pack(name: str = DEFAULT_PACK_NAME, pack_dir: Path | None = None) -> matcher.Pack:
    """Cached load of a matcher.py pack, by name. Separate cache from config.get_pack:

    this is a different pack schema (weighted terms + dedup families + score
    bands) loaded by matcher.Pack.load(), not config.SectorPack.
    """
    return matcher.Pack.load(str(matcher_pack_path(name, pack_dir)))


@dataclass
class MatcherOutcome:
    """One domain's independent, deterministic read: matched terms plus decide()'s verdict."""

    domain: str
    crawl_date: date
    pack_version: str
    pages_scored: int
    result: matcher.Result
    verdict: matcher.Verdict
    matched_terms: list[dict] = field(default_factory=list)


def run_matcher(
    store: Store,
    domain: str,
    crawl_date: date,
    *,
    pack: matcher.Pack | None = None,
    entity_type: str = "manufacturer",
    run_status: str = "ok",
) -> MatcherOutcome:
    """Score every stored raw page for one domain/crawl_date and decide a verdict.

    Raises `ObjectNotFoundError` (from `manifest_pages`) exactly as the rest of
    the pipeline does when a domain has no manifest for this crawl date: absence
    means "crawl it first", not "nothing to score".

    `entity_type` and `run_status` default to the values the rest of the
    pipeline implicitly assumes today (every domain is a manufacturer, every
    crawl succeeded); callers with real values should pass them once Step 4's
    schema fields exist end to end.
    """
    active_pack = pack or get_matcher_pack()

    pages = manifest_pages(store, domain, crawl_date)
    texts = [store.read_text(page.stored_path) for page in pages]

    # score_domain(), not score_pages(): the latter has no cross-page dedup,
    # so a phrase repeated in a site-wide template (nav, footer) is counted
    # once per page it appears on. See matcher.py's score_domain docstring.
    result = active_pack.score_domain(
        (page.url, page.page_type, text) for page, text in zip(pages, texts)
    )
    combined_text = "\n\n".join(texts)
    verdict = matcher.decide(
        result,
        combined_text,
        entity_type=entity_type,
        run_status=run_status,
        iso_class=active_pack.iso_class(combined_text),
    )

    return MatcherOutcome(
        domain=domain,
        crawl_date=crawl_date,
        pack_version=active_pack.data.get("version", DEFAULT_PACK_NAME),
        pages_scored=len(pages),
        result=result,
        verdict=verdict,
        matched_terms=result.matched_terms(),
    )
