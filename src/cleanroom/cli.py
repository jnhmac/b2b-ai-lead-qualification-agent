"""Command line entry points.

Each stage reads from the store and writes back to it. No stage calls the next
one implicitly: a crawl that finishes does not start an extraction, because the
whole point of the raw store is that the two version independently.

Built so far:

    crawl     --domains file.csv [--force] [--dry-run] [--budget N]
    extract   --crawl-date YYYY-MM-DD [--dry-run] [--budget-usd N]
    classify  --pack medical_v1 [--extraction-run ID] [--dry-run]
    credits   balance and concurrency, free

`evaluate` and `compare` land with their phases.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from .classify import (
    ClassificationRun,
    DomainClassification,
    classify_all,
    csv_path,
    latest_extraction_run,
)
from .classify import DEFAULT_CONCURRENCY as DEFAULT_CLASSIFY_CONCURRENCY
from .classify import cost_estimate as classify_cost_estimate
from .config import ConfigError, PackValidationError, Settings, load_pack, load_settings
from .discover import DiscoveryResult, discover_all
from .extract import (
    DomainExtraction,
    ExtractionRun,
    cost_estimate,
    crawled_domains,
    default_run_id,
    extract_all,
    latest_crawl_date,
)
from .extract import DEFAULT_CONCURRENCY as DEFAULT_EXTRACT_CONCURRENCY
from .fetch import (
    DEFAULT_CONCURRENCY,
    DEFAULT_RATE_LIMIT_PER_MINUTE,
    DomainCrawl,
    Fetcher,
    crawl_all,
    credit_estimate,
)
from .llm import get_llm_client
from .normalize import NormalizeResult, normalize_all
from .store import get_store

log = logging.getLogger("cleanroom")

DEFAULT_PACK = "medical_v1"


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def read_domains(path: Path) -> list[str]:
    """Read a domain list from CSV, JSONL or a plain text file.

    Three formats because the list arrives from three places: a spreadsheet
    export, the goldset, and a hand-written scratch file. A CSV with a `domain`
    column uses it; otherwise the first column is taken. Values are raw here —
    normalization happens once, afterwards, so the same cleanup runs regardless
    of which format the list came in.
    """
    if not path.is_file():
        raise ConfigError(f"domain list not found: {path}")

    text = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return [str(row["domain"]) for row in rows if row.get("domain")]

    if suffix == ".csv":
        reader = csv.reader(text.splitlines())
        rows = [row for row in reader if row and any(cell.strip() for cell in row)]
        if not rows:
            return []
        header = [cell.strip().lower() for cell in rows[0]]
        column = header.index("domain") if "domain" in header else 0
        body = rows[1:] if ("domain" in header or header[column] in ("url", "website")) else rows
        return [row[column].strip() for row in body if len(row) > column and row[column].strip()]

    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


#: Optional columns a domain list may carry, and what each one warns about.
#: Both are advisory. BUILD.md locks "no exclusion filters" and these do not
#: break it: nothing is skipped, the operator is told before the credits go.
_WARNING_COLUMNS = {
    "identity_risk": (
        "the join to this domain has no recorded basis, or is name-only against "
        "a different website -- it may not be the intended company at all"
    ),
    "junk_sector": "the source list's own sector for this domain is not plausible",
}

_TRUTHY = {"yes", "y", "true", "1"}


@dataclass(frozen=True)
class DomainWarnings:
    """Advisory flags read off a domain list's own optional columns."""

    identity_risk: list[str] = field(default_factory=list)
    junk_sector: list[str] = field(default_factory=list)
    #: Whether the list carried any warning column at all. A list without them
    #: warns about nothing, which is not the same as a list that warns clean --
    #: the goldset and the 2026-08-11 batch files have no such columns.
    columns_present: bool = False

    @property
    def total(self) -> int:
        """Distinct domains carrying at least one warning."""
        return len(set(self.identity_risk) | set(self.junk_sector))

    def describe(self, limit: int = 8) -> list[str]:
        lines: list[str] = []
        for column, blurb in _WARNING_COLUMNS.items():
            domains = getattr(self, column)
            if not domains:
                continue
            shown = ", ".join(domains[:limit]) + (" ..." if len(domains) > limit else "")
            lines.append(f"  WARNING  {len(domains)} domain(s) flagged {column}: {blurb}")
            lines.append(f"           {shown}")
        if lines:
            lines.append("           These are NOT skipped -- crawled and classified like "
                         "everything else, per BUILD.md. Read them before trusting a verdict.")
        return lines


def read_domain_warnings(path: Path) -> DomainWarnings:
    """Read advisory flag columns from a CSV domain list, if it has any.

    Exists because `thehamlincompany.com` -- a Boston real-estate firm that
    was never the intended company -- was crawled, extracted, classified and
    hand-reviewed before anyone noticed, and every signal needed to catch it was
    already sitting in a column of the source CSV. 84 of 433 domains share that
    shape (outputs/2026-08-26_design-review.md F8).

    Deliberately tolerant: a list with no such columns, a non-CSV list, or a
    missing file all return empty. This runs alongside `read_domains`, which
    owns the real error reporting, and must never pre-empt its message.
    """
    if path.suffix.lower() != ".csv" or not path.is_file():
        return DomainWarnings()

    reader = csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines())
    fieldnames = [name.strip().lower() for name in (reader.fieldnames or [])]
    present = [column for column in _WARNING_COLUMNS if column in fieldnames]
    if not present:
        return DomainWarnings()

    flagged: dict[str, list[str]] = {column: [] for column in _WARNING_COLUMNS}
    for row in reader:
        domain = (row.get("domain") or "").strip()
        if not domain:
            continue
        for column in present:
            if (row.get(column) or "").strip().lower() in _TRUTHY:
                flagged[column].append(domain)

    return DomainWarnings(
        identity_risk=sorted(flagged["identity_risk"]),
        junk_sector=sorted(flagged["junk_sector"]),
        columns_present=True,
    )


def normalized_domains(raw: Sequence[str]) -> NormalizeResult:
    """Normalize and dedupe, reporting collisions rather than hiding them.

    A parent and a subsidiary that collapse onto one registrable domain are a
    judgement call, not a duplicate, so they are printed where the operator
    running the crawl will see them.
    """
    result = normalize_all(raw)
    for collision in result.subdomain_collapses:
        print(f"  collision: {collision.describe()}", file=sys.stderr)
    for value, error in result.invalid:
        print(f"  dropped {value!r}: {error}", file=sys.stderr)
    return result


# ---------------------------------------------------------------------------
# crawl
# ---------------------------------------------------------------------------


def cmd_crawl(args: argparse.Namespace) -> int:
    settings = load_settings()
    pack = load_pack(args.pack, pack_dir=settings.pack_dir)
    store = get_store(settings)

    raw = read_domains(Path(args.domains))
    result = normalized_domains(raw)
    domains = result.domains[: args.limit] if args.limit else result.domains
    if not domains:
        print("no usable domains in the input list", file=sys.stderr)
        return 1

    crawl_date = date.fromisoformat(args.crawl_date) if args.crawl_date else date.today()

    print(f"pack        : {pack.version}")
    print(f"store       : {store.uri('')} ({store.backend})")
    print(f"crawl date  : {crawl_date.isoformat()}")
    print(f"domains     : {len(domains)} of {len(raw)} input rows ({result.summary()})")

    # Printed before the Fetcher is built, so it lands whether or not this is a
    # dry run and always ahead of the first billed call. Nothing is skipped.
    warnings = read_domain_warnings(Path(args.domains))
    if warnings.total:
        print()
        for line in warnings.describe():
            print(line, file=sys.stderr)

    if args.dry_run:
        return _dry_run(domains, pack, args)

    fetcher = Fetcher.from_settings(
        settings, timeout_ms=args.timeout_ms, rate_limit=args.rate_limit
    )
    balance = fetcher.credit_balance()
    print(f"credits     : {balance if balance is not None else 'unknown'}")
    print(f"rate limit  : {args.rate_limit}/min")
    if balance is not None and args.budget is None:
        # The store is append-only and a half-finished batch needs --force to
        # complete, so the balance is a fact worth acting on rather than
        # printing. Defaulting the budget to it stops the run at the wall
        # instead of past it.
        args.budget = balance
    print(f"budget      : {args.budget if args.budget is not None else 'none'}")
    print()

    header = f"  {'domain':<28}{'pages':>6}{'words':>8}{'credits':>9}  notes"
    print(header)
    print(f"  {'-' * 28}{'-' * 6}{'-' * 8}{'-' * 9}  {'-' * 30}")

    records = crawl_all(
        domains, pack, store, fetcher,
        crawl_date=crawl_date,
        force=args.force,
        concurrency=args.concurrency,
        budget=args.budget,
        on_result=lambda record: print(_row(record), flush=True),
    )

    _print_totals(records)
    if args.report:
        Path(args.report).write_text(
            json.dumps([record.model_dump(mode="json") for record in records], indent=2),
            encoding="utf-8",
        )
        print(f"\nreport written to {args.report}")

    crawled = [record for record in records if not record.skipped]
    return 0 if crawled else 1


def _dry_run(domains: Sequence[str], pack: Any, args: argparse.Namespace) -> int:
    """Discovery only. Free, and the last chance to see the bill before it lands."""
    print("\ndry run: discovery only, no Firecrawl calls, no credits\n")
    results = discover_all(domains, pack, max_workers=6, timeout=20.0)

    print(f"  {'domain':<28}{'pages':>6}  {'sources':<18}mix")
    print(f"  {'-' * 28}{'-' * 6}  {'-' * 18}{'-' * 28}")
    for result in results:
        mix = " ".join(f"{name}:{n}" for name, n in sorted(result.page_type_counts.items()))
        flag = "  FAIL" if result.discovery_failed else ""
        print(
            f"  {result.domain:<28}{len(result.pages):>6}  "
            f"{','.join(result.sources_used):<18}{mix}{flag}"
        )
        if result.discovery_failed:
            print(f"      -> {result.failure_reason}")

    estimate = credit_estimate(results)
    print(
        f"\n  {estimate['pages']} pages to bill  +{estimate['map_calls']} map calls"
        f"  = {estimate['credits']} credits"
    )
    return 0


def _row(record: DomainCrawl) -> str:
    notes: list[str] = []
    if record.exit_step == 0:
        # Zero pages and zero credits here is the ladder working, not a crawl
        # that quietly stored nothing. Say so on the row, with the target, so
        # the requeue is obvious from the report alone.
        notes.append(f"{record.stop_reason} -> {record.referred_domain}")
    if record.skipped:
        notes.append(record.skip_reason or "skipped")
    if record.crawl_failed:
        notes.append(f"NO MANIFEST: {record.crawl_error}")
    if record.rescued:
        notes.append(f"rescued {record.rescue_pages}")
    if record.stealth_pages:
        notes.append(f"STEALTH x{record.stealth_pages}")
    if record.pages_failed:
        notes.append(f"{record.pages_failed} failed")
    if record.discovery_failed:
        notes.append("discovery flagged")
    return (
        f"  {record.domain:<28}{record.pages_stored:>6}{record.word_count:>8}"
        f"{record.credits_billed:>9}  {', '.join(notes)}"
    )


def _print_totals(records: Sequence[DomainCrawl]) -> None:
    crawled = [record for record in records if not record.skipped]
    skipped = [record for record in records if record.skipped]
    # A step-0 exit stores no pages by design, so it is neither a thin crawl
    # nor a retry candidate. Counting it as either buries a resolved domain in
    # a list of problems and invites someone to re-crawl it.
    moved = [record for record in crawled if record.exit_step == 0]
    thin = sorted(
        (record for record in crawled if record.pages_stored < 3 and record.exit_step != 0),
        key=lambda record: (record.pages_stored, record.word_count),
    )

    incomplete = [record for record in crawled if record.crawl_failed]

    print(f"\n  domains crawled     : {len(crawled) - len(incomplete)}")
    print(f"  domains skipped     : {len(skipped)}")
    print(f"  pages stored        : {sum(record.pages_stored for record in crawled)}")
    print(f"  pages failed        : {sum(record.pages_failed for record in crawled)}")
    print(f"  credits billed      : {sum(record.credits_billed for record in crawled)}")
    print(f"  map calls           : {sum(record.map_calls for record in crawled)}")
    print(f"  stealth pages       : {sum(record.stealth_pages for record in crawled)}")
    if moved:
        # Queue these against their target: R12 re-targets the acquirer rather
        # than emitting a verdict on the stub.
        print(f"  moved, requeue      : {len(moved)}")
        for record in moved:
            print(f"    {record.domain} -> {record.referred_domain}")
    if incomplete:
        # No manifest was written for these, so re-running the same command
        # picks them up. That is the point of withholding it.
        print(f"  incomplete, retry   : {len(incomplete)}")
        for record in incomplete:
            print(f"    {record.domain}: {record.crawl_error}")
    if thin:
        # Not the Phase 7a population, and printed so it is not mistaken for
        # it. Under three stored pages is a fetch or discovery problem; the
        # thin-site population is harvested from full manifests by word count.
        print("  under three pages   : " + ", ".join(record.domain for record in thin))
    for record in skipped:
        print(f"    skipped {record.domain}: {record.skip_reason}")


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


def cmd_extract(args: argparse.Namespace) -> int:
    """One LLM call per stored page. Reads the raw store, never the network."""
    settings = load_settings()
    pack = load_pack(args.pack, pack_dir=settings.pack_dir)
    store = get_store(settings)

    crawl_date_str = args.crawl_date or latest_crawl_date(store)
    if not crawl_date_str:
        print("no crawl in the store; run `cleanroom crawl` first", file=sys.stderr)
        return 1
    crawl_date = date.fromisoformat(crawl_date_str)

    domains = _extract_domains(args, store, crawl_date)
    if not domains:
        print(f"no crawled domains for {crawl_date.isoformat()}", file=sys.stderr)
        return 1

    run_id = args.run_id or default_run_id()

    print(f"pack        : {pack.version}")
    print(f"store       : {store.uri('')} ({store.backend})")
    print(f"crawl date  : {crawl_date.isoformat()}")
    print(f"domains     : {len(domains)}")
    print(f"provider    : {settings.llm.provider} / {settings.llm.model_id('extract')}")

    with get_llm_client(settings) as client:
        estimate = cost_estimate(store, domains, pack, client, crawl_date=crawl_date)
        print(f"pages       : {estimate['pages']}")
        print(
            f"system      : {estimate['system_prompt_tokens']} tokens, "
            f"caching {estimate['caching']}"
        )
        print(f"estimate    : ${estimate['cost_usd']:.2f}")
        for domain, reason in estimate["missing"].items():
            print(f"  no pages: {domain}: {reason}", file=sys.stderr)

        if args.dry_run:
            print("\ndry run: no LLM calls, nothing written")
            return 0

        print(f"run id      : {run_id}")
        print(f"budget      : ${args.budget_usd:.2f}" if args.budget_usd else "budget      : none")
        print()
        print(f"  {'domain':<28}{'pages':>6}{'find':>6}{'drop':>6}{'evid':>6}{'cost':>9}  notes")
        print(f"  {'-' * 28}{'-' * 6}{'-' * 6}{'-' * 6}{'-' * 6}{'-' * 9}  {'-' * 24}")

        run = extract_all(
            domains, pack, store, client,
            crawl_date=crawl_date,
            run_id=run_id,
            concurrency=args.concurrency,
            budget_usd=args.budget_usd,
            on_domain=lambda summary: print(_extract_row(summary), flush=True),
        )

    _print_extract_totals(run, store)

    if args.report:
        Path(args.report).write_text(
            json.dumps(run.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        print(f"\nreport written to {args.report}")

    return 0 if run.records_written and not run.stopped_reason else 1


def _extract_domains(args: argparse.Namespace, store: Any, crawl_date: date) -> list[str]:
    """Which domains to extract: an explicit list, or everything crawled."""
    if args.domains:
        raw = read_domains(Path(args.domains))
        domains = normalized_domains(raw).domains
    else:
        domains = crawled_domains(store, crawl_date)
    return domains[: args.limit] if args.limit else domains


def _extract_row(summary: DomainExtraction) -> str:
    notes: list[str] = []
    if summary.pages_failed:
        notes.append(f"{summary.pages_failed} failed")
    if summary.pages_skipped:
        notes.append(f"{summary.pages_skipped} skipped")
    if summary.error:
        notes.append(summary.error)
    return (
        f"  {summary.domain:<28}{summary.pages_extracted:>6}{summary.findings_kept:>6}"
        f"{summary.quotes_dropped:>6}{summary.evidence_pages:>6}"
        f"{summary.cost_usd:>9.4f}  {', '.join(notes)}"
    )


def _print_drops(run: ExtractionRun, limit: int = 8) -> None:
    """Show what verification threw away.

    The most actionable output of the stage. A dropped quote that is obviously
    on the page means the canonical form is too strict; one that reads like a
    summary of the page means the prompt is not being obeyed. The counts alone
    cannot tell those apart, and they need opposite fixes.
    """
    for label, attribute in (
        ("quotes", "dropped_quotes"),
        ("products", "dropped_products"),
        ("trap terms", "dropped_trap_terms"),
    ):
        samples = [
            (page.domain, value)
            for page in run.pages
            for value in getattr(page, attribute)
        ]
        if not samples:
            continue
        print(f"\n  dropped {label} ({len(samples)}):")
        for domain, value in samples[:limit]:
            print(f"    {domain:<24} {value}")
        if len(samples) > limit:
            print(f"    ... and {len(samples) - limit} more, all in the run report")


def _print_extract_totals(run: ExtractionRun, store: Any) -> None:
    print(f"\n  pages extracted     : {run.pages_extracted} of {run.pages_seen}")
    print(f"  pages failed        : {run.pages_failed}")
    if run.pages_skipped:
        print(f"  pages skipped       : {run.pages_skipped}")
    print(f"  findings kept       : {run.findings_kept}")
    # The gate's instrument. A quote that cannot be found on the page is
    # dropped with its claim, so a rising rate between two runs of the same
    # pages is a prompt regression rather than a change in the sites.
    print(f"  quotes dropped      : {run.quotes_dropped} ({run.quote_drop_rate:.1%} of claims)")
    print(f"  products dropped    : {run.products_dropped}")
    print(f"  trap terms dropped  : {run.trap_terms_dropped}")
    print(f"  records written     : {run.records_written} in {len(run.parts)} part(s)")
    print(
        f"  tokens              : {run.input_tokens} in / {run.output_tokens} out"
        f" / {run.cache_read_tokens} cached"
    )
    if run.cache_read_tokens == 0 and run.pages_extracted > 1:
        # Silent on the provider's side: no error, just a bill that never drops.
        print("  WARNING: no cache reads. The system prompt is not caching.")
    print(f"  cost                : ${run.cost_usd:.4f}")
    print(f"  duration            : {run.duration_s}s")
    _print_drops(run)
    if run.stopped_reason:
        print(f"  STOPPED: {run.stopped_reason}")
    failed = [page for page in run.pages if not page.ok and not page.skipped]
    if failed:
        print(f"  retry with --run-id {run.run_id}, which skips what is already stored:")
        for page in failed[:10]:
            print(f"    {page.url}: {page.error}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


def cmd_classify(args: argparse.Namespace) -> int:
    """One LLM call per domain, over the extractions. Reads the store only."""
    settings = load_settings()
    pack = load_pack(args.pack, pack_dir=settings.pack_dir)
    store = get_store(settings)

    extraction_run = args.extraction_run or latest_extraction_run(store, pack.version)
    if not extraction_run:
        print(
            f"no extraction run for pack {pack.version}; run `cleanroom extract` first",
            file=sys.stderr,
        )
        return 1

    domains = _classify_domains(args, store, pack.version, extraction_run)
    if not domains:
        print(f"no extracted domains in run {extraction_run}", file=sys.stderr)
        return 1

    run_id = args.run_id or default_run_id()

    print(f"pack        : {pack.version}")
    print(f"store       : {store.uri('')} ({store.backend})")
    print(f"extractions : {extraction_run}")
    print(f"domains     : {len(domains)}")
    print(f"provider    : {settings.llm.provider} / {settings.llm.model_id('classify')}")

    with get_llm_client(settings) as client:
        estimate = classify_cost_estimate(
            store, domains, pack, client, extraction_run_id=extraction_run
        )
        print(f"pages       : {estimate['pages']} across {estimate['domains']} domains")
        print(
            f"system      : {estimate['system_prompt_tokens']} tokens, "
            f"caching {estimate['caching']}"
        )
        print(f"estimate    : ${estimate['cost_usd']:.2f}")
        for domain, reason in estimate["missing"].items():
            print(f"  no records: {domain}: {reason}", file=sys.stderr)

        if args.dry_run:
            print("\ndry run: no LLM calls, nothing written")
            return 0

        print(f"run id      : {run_id}")
        print(f"budget      : ${args.budget_usd:.2f}" if args.budget_usd else "budget      : none")
        print()
        print(
            f"  {'domain':<28}{'verdict':<12}{'iso':<10}{'conf':>6}"
            f"{'pages':>6}{'evid':>6}{'cost':>9}  notes"
        )
        print(f"  {'-' * 28}{'-' * 12}{'-' * 10}{'-' * 6}{'-' * 6}{'-' * 6}{'-' * 9}  {'-' * 20}")

        run = classify_all(
            domains, pack, store, client,
            extraction_run_id=extraction_run,
            run_id=run_id,
            concurrency=args.concurrency,
            budget_usd=args.budget_usd,
            on_domain=lambda summary: print(_classify_row(summary), flush=True),
        )

    _print_classify_totals(run)
    print(f"\n  verdict CSV         : {store.uri(csv_path(pack.version, run_id))}")

    if args.report:
        Path(args.report).write_text(
            json.dumps(run.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        print(f"  report written to     {args.report}")

    return 0 if run.records_written and not run.stopped_reason else 1


def _classify_domains(
    args: argparse.Namespace, store: Any, pack_version: str, extraction_run: str
) -> list[str]:
    """Which domains to classify: an explicit list, or everything extracted."""
    if args.domains:
        raw = read_domains(Path(args.domains))
        domains = normalized_domains(raw).domains
    else:
        from .classify import group_records
        from .extract import read_records

        domains = sorted(group_records(read_records(store, pack_version, extraction_run)))
    return domains[: args.limit] if args.limit else domains


def _classify_row(summary: DomainClassification) -> str:
    notes: list[str] = []
    if summary.entity_type:
        notes.append(summary.entity_type)
    if summary.needs_manual_review:
        notes.append("REVIEW")
    if summary.evidence_dropped:
        notes.append(f"{summary.evidence_dropped} evidence dropped")
    if summary.skipped:
        notes.append(summary.skip_reason or "skipped")
    if summary.error:
        notes.append(summary.error)
    return (
        f"  {summary.domain:<28}{summary.cleanroom_status or '-':<12}"
        f"{(summary.iso_class or '-'):<10}{summary.confidence:>6.2f}"
        f"{summary.pages_analyzed:>6}{summary.evidence_kept:>6}"
        f"{summary.cost_usd:>9.4f}  {', '.join(notes)}"
    )


def _print_classify_totals(run: ClassificationRun) -> None:
    print(f"\n  domains classified  : {run.domains_classified} of {run.domains_seen}")
    print(f"  domains failed      : {run.domains_failed}")
    if run.domains_skipped:
        print(f"  domains skipped     : {run.domains_skipped}")

    counts = run.label_counts
    total = sum(counts.values()) or 1
    print("\n  verdict            count    share")
    print(f"  {'-' * 18}{'-' * 8}{'-' * 9}")
    for label in ("in_house", "likely_in_house", "outsourced", "no_cleanroom_needed", "unknown"):
        n = counts.get(label, 0)
        print(f"  {label:<18}{n:>6}{n / total:>8.1%}")

    # BUILD.md expects 15 to 20 percent unknown. Outside that band the two
    # labels are probably being collapsed, which is the failure this stage is
    # most likely to have and the one Phase 7b cannot easily see.
    share = run.unknown_share
    if share < 0.10:
        print(f"\n  NOTE: unknown is {share:.1%}, under the 15-20% BUILD.md expects.")
        print("        Check that thin sites are not being resolved to `none`.")
    elif share > 0.30:
        print(f"\n  NOTE: unknown is {share:.1%}, over the 15-20% BUILD.md expects.")
        print("        Check that mixed evidence is not being resolved to `unknown`.")

    print(f"\n  needs manual review : {run.needs_manual_review}")
    print(f"  evidence kept       : {run.evidence_kept}")
    # The same instrument as Phase 5's quote drop rate, one stage down. A quote
    # the classifier returns that is not in the extractions is evidence nobody
    # can trace, so it is dropped and counted here.
    print(
        f"  evidence dropped    : {run.evidence_dropped}"
        f" ({run.evidence_drop_rate:.1%} of returned)"
    )
    print(f"  records written     : {run.records_written} in {len(run.parts)} part(s)")
    print(
        f"  tokens              : {run.input_tokens} in / {run.output_tokens} out"
        f" / {run.cache_read_tokens} cached"
    )
    if run.cache_read_tokens == 0 and run.domains_classified > 1:
        print("  WARNING: no cache reads. The system prompt is not caching.")
    print(f"  cost                : ${run.cost_usd:.4f}")
    print(f"  duration            : {run.duration_s}s")

    dropped = [(row.domain, value) for row in run.domains for value in row.dropped_evidence]
    if dropped:
        print(f"\n  dropped evidence ({len(dropped)}):")
        for domain, value in dropped[:8]:
            print(f"    {domain:<24} {value}")
        if len(dropped) > 8:
            print(f"    ... and {len(dropped) - 8} more, all in the run report")

    if run.stopped_reason:
        print(f"  STOPPED: {run.stopped_reason}")
    failed = [row for row in run.domains if not row.ok and not row.skipped]
    if failed:
        print(f"  retry with --run-id {run.run_id}, which skips what is already stored:")
        for row in failed[:10]:
            print(f"    {row.domain}: {row.error}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")


# ---------------------------------------------------------------------------
# credits
# ---------------------------------------------------------------------------


def cmd_credits(args: argparse.Namespace) -> int:
    """Balance and concurrency. Free, and worth running before any crawl."""
    settings = load_settings()
    fetcher = Fetcher.from_settings(settings)
    balance = fetcher.credit_balance()
    print(f"remaining credits : {balance if balance is not None else 'unknown'}")
    try:
        concurrency = fetcher.client.get_concurrency()
    except Exception as exc:  # noqa: BLE001 - informational only
        log.debug("concurrency check failed: %s", exc)
    else:
        print(f"concurrency       : {getattr(concurrency, 'concurrency', '?')}"
              f" / {getattr(concurrency, 'max_concurrency', '?')}")
    return 0 if balance is not None else 1


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cleanroom", description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="count", default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    crawl = sub.add_parser("crawl", help="fetch and store the pages discovery selects")
    crawl.add_argument("--domains", required=True, help="CSV, JSONL or text file of domains")
    crawl.add_argument("--pack", default=DEFAULT_PACK)
    crawl.add_argument("--crawl-date", help="YYYY-MM-DD, defaults to today")
    crawl.add_argument("--force", action="store_true", help="re-crawl and overwrite today's raw store")
    crawl.add_argument("--dry-run", action="store_true", help="discovery only: free, prints the estimate")
    crawl.add_argument("--limit", type=int, help="crawl only the first N domains")
    crawl.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    crawl.add_argument("--budget", type=int, help="stop once this many credits are spent")
    crawl.add_argument(
        "--rate-limit",
        type=int,
        default=DEFAULT_RATE_LIMIT_PER_MINUTE,
        help="requests per minute, the plan's cap (default %(default)s)",
    )
    crawl.add_argument("--timeout-ms", type=int, help="per-page scrape timeout")
    crawl.add_argument("--report", help="write the per-domain record to this JSON file")
    crawl.set_defaults(func=cmd_crawl)

    extract = sub.add_parser("extract", help="one LLM call per stored page, into the store")
    extract.add_argument("--crawl-date", help="YYYY-MM-DD, defaults to the latest crawl")
    extract.add_argument("--pack", default=DEFAULT_PACK)
    extract.add_argument(
        "--run-id",
        help="reuse a run id to retry its failed pages; already-stored pages are skipped",
    )
    extract.add_argument("--domains", help="restrict to this domain list; defaults to all crawled")
    extract.add_argument("--limit", type=int, help="extract only the first N domains")
    extract.add_argument("--concurrency", type=int, default=DEFAULT_EXTRACT_CONCURRENCY)
    extract.add_argument("--budget-usd", type=float, help="stop once this much has been spent")
    extract.add_argument("--dry-run", action="store_true", help="price it, call nothing")
    extract.add_argument("--report", help="write the run report to this JSON file")
    extract.set_defaults(func=cmd_extract)

    classify = sub.add_parser("classify", help="one LLM call per domain, over the extractions")
    classify.add_argument("--pack", default=DEFAULT_PACK)
    classify.add_argument(
        "--extraction-run",
        help="extraction run id to read; defaults to the newest timestamped run",
    )
    classify.add_argument(
        "--run-id",
        help="reuse a run id to retry its failed domains; already-stored domains are skipped",
    )
    classify.add_argument("--domains", help="restrict to this domain list; defaults to all extracted")
    classify.add_argument("--limit", type=int, help="classify only the first N domains")
    classify.add_argument("--concurrency", type=int, default=DEFAULT_CLASSIFY_CONCURRENCY)
    classify.add_argument("--budget-usd", type=float, help="stop once this much has been spent")
    classify.add_argument("--dry-run", action="store_true", help="price it, call nothing")
    classify.add_argument("--report", help="write the run report to this JSON file")
    classify.set_defaults(func=cmd_classify)

    credits = sub.add_parser("credits", help="Firecrawl balance and concurrency")
    credits.set_defaults(func=cmd_credits)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)-8s %(name)s: %(message)s",
    )
    try:
        return int(args.func(args))
    except (ConfigError, PackValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
