"""Diff two classification runs: which domains changed verdict, and why. Phase 7b.

Reads two run directories off disk and joins them on domain. **No LLM call, no
network, no credits.** This is the other half of the iteration loop BUILD.md
describes:

    edit prompt or pack -> run over goldset (minutes, no re-scrape)
    -> compare to previous run -> repeat

`evaluate.py` says how good a run is. This says what an edit *did*, which is a
different question and the one that tells you whether to keep the edit.

    python eval/compare_runs.py                                  # newest two runs
    python eval/compare_runs.py --run-a 20260810-235224 --run-b 20260811-001815
    python eval/compare_runs.py --run-a X --run-b Y --csv outputs/diff.csv

Three things this prints that a plain verdict diff would not:

- **Whether the input moved.** A verdict can change because the prompt changed
  or because the domain was re-crawled and now has different pages underneath
  it. Those look identical in a verdict column and mean opposite things, so
  every changed row carries `pages_analyzed` and `source_crawl_date` from both
  runs and the row is flagged when they differ. A flagged row is not evidence
  about the prompt.
- **The lead-level delta.** `in_house` is the only label that produces a lead,
  so a `none` -> `outsourced` move costs nothing and an `unknown` -> `in_house`
  move is a new name on the sales list. Both are "changed verdict" and only one
  of them matters commercially.
- **Direction against the labels**, when the compared domains are labelled.
  A change is FIXED, BROKE, or still wrong. A diff with no labels tells you the
  prompt did something; a diff with labels tells you whether it helped.

Exit code is 0 whenever the comparison could be made. A diff is not a failure,
and regressions are reported rather than raised: deciding whether a trade is
worth it is the point of reading the output.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import LABELS, LEAD, read_labels  # noqa: E402

from cleanroom.classify import read_classifications  # noqa: E402
from cleanroom.config import load_pack, load_settings  # noqa: E402
from cleanroom.models import ClassificationRecord  # noqa: E402
from cleanroom.store import Store, get_store  # noqa: E402


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def list_runs(store: Store, pack_version: str) -> list[str]:
    """Every classification run id for a pack, oldest first."""
    prefix = f"classifications/pack={pack_version}/"
    runs = {
        path[len(prefix):].split("/")[0].removeprefix("run=")
        for path in store.list_paths(prefix.rstrip("/"))
        if path.startswith(prefix)
    }
    return sorted(runs)


def load_run(store: Store, pack_version: str, run_id: str) -> dict[str, ClassificationRecord]:
    records = read_classifications(store, pack_version, run_id)
    if not records:
        raise SystemExit(f"run {run_id} holds no classification records for pack {pack_version}")
    return {record.domain: record for record in records}


# ---------------------------------------------------------------------------
# The diff
# ---------------------------------------------------------------------------


def _quotes(record: ClassificationRecord) -> set[str]:
    """Evidence quotes, whitespace-collapsed so re-wrapping is not a change."""
    return {" ".join(item.quote.split()) for item in record.evidence}


def compare(
    a: dict[str, ClassificationRecord], b: dict[str, ClassificationRecord]
) -> list[dict[str, Any]]:
    """One row per shared domain, in domain order.

    Every shared domain gets a row, changed or not. Filtering is the caller's
    job because the unchanged rows still answer a question worth asking: a
    verdict that held while its evidence moved is a verdict that survived for a
    new reason.
    """
    rows: list[dict[str, Any]] = []
    for domain in sorted(set(a) & set(b)):
        ra, rb = a[domain], b[domain]
        quotes_a, quotes_b = _quotes(ra), _quotes(rb)
        rows.append(
            {
                "domain": domain,
                "a": ra,
                "b": rb,
                "verdict_changed": ra.cleanroom_status != rb.cleanroom_status,
                "lead_changed": (ra.cleanroom_status == LEAD) != (rb.cleanroom_status == LEAD),
                "iso_changed": (ra.iso_class or "") != (rb.iso_class or ""),
                "review_changed": ra.needs_manual_review != rb.needs_manual_review,
                "confidence_delta": rb.confidence - ra.confidence,
                "quotes_added": sorted(quotes_b - quotes_a),
                "quotes_removed": sorted(quotes_a - quotes_b),
                # The input moved, so this row says nothing about the prompt.
                "input_changed": (
                    ra.pages_analyzed != rb.pages_analyzed
                    or ra.source_crawl_date != rb.source_crawl_date
                ),
            }
        )
    return rows


def transitions(rows: Sequence[dict[str, Any]]) -> Counter:
    return Counter(
        (row["a"].cleanroom_status, row["b"].cleanroom_status) for row in rows
    )


def verdict_of(row: dict[str, Any], side: str) -> str:
    return row[side].cleanroom_status


def direction(row: dict[str, Any], expected: str | None) -> str:
    """What a change did to the label, when there is a label to judge against."""
    if expected is None:
        return ""
    before, after = verdict_of(row, "a"), verdict_of(row, "b")
    if before == after:
        return "unchanged"
    if after == expected:
        return "FIXED"
    if before == expected:
        return "BROKE"
    return "still wrong"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


#: Smallest number of discordant pairs that can reach p < 0.05 two-sided, when
#: every one of them moves the same way. Below this the test cannot return a
#: significant result no matter how one-sided the change is, which is the single
#: most useful thing a reader of a small diff can know up front.
MIN_DISCORDANT_FOR_SIGNIFICANCE = 6


def mcnemar(*, fixed: int, broke: int) -> dict[str, Any]:
    """Exact two-sided McNemar test on a paired before/after comparison.

    `fixed` and `broke` are the discordant pairs: domains right in run B and
    wrong in run A, and the reverse. Domains that were right in both or wrong in
    both carry no information about whether the change helped, which is exactly
    why a raw accuracy delta misleads at these sample sizes.

    Exact binomial rather than the chi-square approximation because the counts
    here are single digits, where the approximation is not usable.

    This exists because of a specific, repeated error. The 2026-08-25 regression
    run was reported as "17/23 -> 18/23 (73.9% -> 78.3%)" and read as a real
    improvement in three separate documents. It is 2 fixed against 1 regressed:
    p = 1.00. Eight code fixes landed and the accuracy delta measured nothing
    about any of them. Reporting `p` next to the delta is what stops the next
    diff being read the same way.
    """
    n = fixed + broke
    if n == 0:
        return {
            "fixed": 0, "broke": 0, "n_discordant": 0,
            "p_value": 1.0, "significant": False,
            "describe": (
                "no domain changed correctness in either direction, which is not "
                "evidence the two runs are equivalent -- only that this label set "
                "cannot tell them apart"
            ),
        }

    # Two-sided exact binomial against p=0.5, clamped: doubling the smaller
    # tail can exceed 1 when the split is even.
    smaller = min(fixed, broke)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2 ** n)
    p_value = min(1.0, 2 * tail)
    significant = p_value < 0.05

    if significant:
        verdict = f"the change is unlikely to be sampling noise (p = {p_value:.3f})"
    else:
        verdict = (
            f"p = {p_value:.2f}: this is indistinguishable from noise. At this n "
            f"you need {MIN_DISCORDANT_FOR_SIGNIFICANCE} discordant domains all "
            f"moving the same way before the test can clear 0.05"
        )
    return {
        "fixed": fixed, "broke": broke, "n_discordant": n,
        "p_value": p_value, "significant": significant, "describe": verdict,
    }


def clip(text: str, width: int) -> str:
    # ASCII ellipsis on purpose: the Windows console this runs on is cp1252 and
    # renders a real one as a replacement character mid-sentence.
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


def report(
    rows: Sequence[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    only_a: Sequence[str],
    only_b: Sequence[str],
    run_a: str,
    run_b: str,
    show_all: bool,
) -> int:
    print(f"\n  run A   : {run_a}")
    print(f"  run B   : {run_b}")
    print(f"  shared  : {len(rows)} domains")
    if only_a:
        print(f"  only in A ({len(only_a)}): {', '.join(sorted(only_a)[:8])}"
              + (" ..." if len(only_a) > 8 else ""))
    if only_b:
        print(f"  only in B ({len(only_b)}): {', '.join(sorted(only_b)[:8])}"
              + (" ..." if len(only_b) > 8 else ""))
    if not rows:
        print("\n  no domain appears in both runs, so there is nothing to diff.\n")
        return 1

    changed = [row for row in rows if row["verdict_changed"]]
    lead_moves = [row for row in changed if row["lead_changed"]]

    # ---- what moved, in one table ---------------------------------------
    counts_a = Counter(verdict_of(row, "a") for row in rows)
    counts_b = Counter(verdict_of(row, "b") for row in rows)
    print(f"\n  VERDICT MIX. {len(changed)} of {len(rows)} domains changed verdict.\n")
    print(f"  {'label':<12}{'A':>6}{'B':>6}{'delta':>8}")
    print(f"  {'-' * 12}{'-' * 6}{'-' * 6}{'-' * 8}")
    for label in LABELS:
        delta = counts_b[label] - counts_a[label]
        print(f"  {label:<12}{counts_a[label]:>6}{counts_b[label]:>6}{delta:>+8}")

    if changed:
        print("\n  TRANSITIONS (A -> B), changes only\n")
        for (before, after), count in sorted(
            transitions(changed).items(), key=lambda item: (-item[1], item[0])
        ):
            print(f"    {before:<12} -> {after:<12} {count:>3}")

    # ---- the commercial question ----------------------------------------
    gained = [r for r in lead_moves if verdict_of(r, "b") == LEAD]
    lost = [r for r in lead_moves if verdict_of(r, "a") == LEAD]
    print("\n  LEAD DELTA. `in_house` is the only label that sells, so a change")
    print("  inside outsourced/none/unknown costs nothing commercially.\n")
    print(f"    leads in A              : {counts_a[LEAD]}")
    print(f"    leads in B              : {counts_b[LEAD]}")
    print(f"    became a lead           : {len(gained)}"
          + (f"   ({', '.join(r['domain'] for r in gained[:6])}"
             + (" ..." if len(gained) > 6 else "") + ")" if gained else ""))
    print(f"    stopped being a lead    : {len(lost)}"
          + (f"   ({', '.join(r['domain'] for r in lost[:6])}"
             + (" ..." if len(lost) > 6 else "") + ")" if lost else ""))
    print(f"    verdict changed, lead call unchanged : {len(changed) - len(lead_moves)}")

    # ---- did it help ----------------------------------------------------
    scored = [row for row in rows if row["domain"] in labels]
    if scored:
        moves = Counter(
            direction(row, labels[row["domain"]]["expected_label"]) for row in scored
        )
        print(f"\n  AGAINST THE LABELS ({len(scored)} of {len(rows)} domains are labelled)\n")
        print(f"    fixed        : {moves['FIXED']}")
        print(f"    broke        : {moves['BROKE']}")
        print(f"    still wrong  : {moves['still wrong']}   (changed, both verdicts wrong)")
        correct_a = sum(1 for r in scored if verdict_of(r, "a") == labels[r["domain"]]["expected_label"])
        correct_b = sum(1 for r in scored if verdict_of(r, "b") == labels[r["domain"]]["expected_label"])
        print(f"    exact match  : {correct_a} -> {correct_b}   ({correct_b - correct_a:+d})")
        print("    Not an accuracy figure. Per-label precision and recall are")
        print("    evaluate.py's job; this is only the direction of travel.")

        # The delta above is the number that has repeatedly been over-read. The
        # paired test sits directly under it so the two are never quoted apart.
        test = mcnemar(fixed=moves["FIXED"], broke=moves["BROKE"])
        print(f"\n    paired test  : {test['fixed']} fixed vs {test['broke']} broke, "
              f"{test['n_discordant']} discordant")
        print(f"                   {test['describe']}")
        if not test["significant"] and (correct_b - correct_a) != 0:
            print("                   ^ so do NOT report the exact-match delta above")
            print("                     as an improvement or a regression.")

    # ---- every change, with its reason ----------------------------------
    stale_input = [row for row in changed if row["input_changed"]]
    if stale_input:
        print(f"\n  {len(stale_input)} changed row(s) sit on a different page count or crawl")
        print("  date, marked INPUT below. Those say nothing about the prompt.")

    if changed:
        print(f"\n  CHANGED VERDICTS ({len(changed)})\n")
        for row in changed:
            ra, rb = row["a"], row["b"]
            label = labels.get(row["domain"], {}).get("expected_label")
            flags = []
            if label:
                flags.append(f"{direction(row, label)} (expected {label})")
            if row["input_changed"]:
                flags.append("INPUT")
            if row["review_changed"]:
                flags.append(
                    f"review {'set' if rb.needs_manual_review else 'cleared'}"
                )
            if row["iso_changed"]:
                flags.append(f"iso {ra.iso_class or '-'} -> {rb.iso_class or '-'}")
            print(
                f"  {row['domain']}"
                f"\n    {ra.cleanroom_status} -> {rb.cleanroom_status}"
                f"   confidence {ra.confidence:.2f} -> {rb.confidence:.2f}"
                f" ({row['confidence_delta']:+.2f})"
                f"   pages {ra.pages_analyzed} -> {rb.pages_analyzed}"
                f"   evidence {len(ra.evidence)} -> {len(rb.evidence)}"
            )
            if flags:
                print(f"    {'; '.join(flags)}")
            for quote in row["quotes_added"][:2]:
                print(f"    + {clip(quote, 96)}")
            for quote in row["quotes_removed"][:2]:
                print(f"    - {clip(quote, 96)}")
            print(f"    why: {clip(rb.reasoning, 300)}")
            print()

    # ---- verdicts that held on different ground -------------------------
    # A verdict can survive an edit while everything under it moves. That is
    # worth seeing: the next edit may push it over.
    held = [
        row for row in rows
        if not row["verdict_changed"]
        and (row["quotes_added"] or row["quotes_removed"] or abs(row["confidence_delta"]) >= 0.1)
    ]
    if held:
        print(f"  SAME VERDICT, DIFFERENT GROUND ({len(held)})\n")
        print(f"  {'domain':<28}{'verdict':<12}{'conf':>12}{'evidence':>12}")
        print(f"  {'-' * 28}{'-' * 12}{'-' * 12}{'-' * 12}")
        for row in held if show_all else held[:20]:
            ra, rb = row["a"], row["b"]
            print(
                f"  {row['domain']:<28}{rb.cleanroom_status:<12}"
                f"{ra.confidence:>6.2f} ->{rb.confidence:>5.2f}"
                f"{len(ra.evidence):>7} ->{len(rb.evidence):>4}"
            )
        if not show_all and len(held) > 20:
            print(f"  ... {len(held) - 20} more, --all to see them")
        print()

    return 0


# ---------------------------------------------------------------------------
# CSV, for the rows a person wants to sort
# ---------------------------------------------------------------------------

CSV_COLUMNS = (
    "domain", "expected_label", "direction",
    "verdict_a", "verdict_b", "lead_changed", "input_changed",
    "confidence_a", "confidence_b",
    "iso_class_a", "iso_class_b",
    "pages_a", "pages_b", "evidence_a", "evidence_b",
    "review_a", "review_b",
    "quotes_added", "quotes_removed",
    "reasoning_b",
)


def build_csv(rows: Sequence[dict[str, Any]], labels: dict[str, dict[str, Any]]) -> str:
    buffer = io.StringIO()
    # Same reason as the Phase 6 CSV: Excel on Windows reads a bare \n as a new
    # cell, and `reasoning` is multi-sentence.
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        ra, rb = row["a"], row["b"]
        expected = labels.get(row["domain"], {}).get("expected_label")
        writer.writerow([
            row["domain"], expected or "", direction(row, expected),
            ra.cleanroom_status, rb.cleanroom_status,
            "yes" if row["lead_changed"] else "",
            "yes" if row["input_changed"] else "",
            f"{ra.confidence:.2f}", f"{rb.confidence:.2f}",
            ra.iso_class or "", rb.iso_class or "",
            ra.pages_analyzed, rb.pages_analyzed,
            len(ra.evidence), len(rb.evidence),
            "yes" if ra.needs_manual_review else "",
            "yes" if rb.needs_manual_review else "",
            " | ".join(row["quotes_added"]),
            " | ".join(row["quotes_removed"]),
            " ".join(rb.reasoning.split()),
        ])
    return buffer.getvalue()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-a", help="the earlier run; defaults to the second newest")
    parser.add_argument("--run-b", help="the later run; defaults to the newest")
    parser.add_argument("--pack", default="medical_v1")
    parser.add_argument("--labels", default=str(REPO_ROOT / "eval" / "goldset.jsonl"),
                        help="label file, for direction of travel; unlabelled domains still diff")
    parser.add_argument("--csv", help="write one row per shared domain here")
    parser.add_argument("--changed-only", action="store_true",
                        help="restrict the CSV to domains whose verdict changed")
    parser.add_argument("--all", action="store_true",
                        help="print every same-verdict row rather than the first 20")
    args = parser.parse_args(argv)

    settings = load_settings()
    pack = load_pack(args.pack, pack_dir=settings.pack_dir)
    store = get_store(settings)

    runs = list_runs(store, pack.version)
    run_b = args.run_b or (runs[-1] if runs else None)
    run_a = args.run_a or (runs[-2] if len(runs) > 1 else None)
    if not run_a or not run_b:
        raise SystemExit(
            f"need two classification runs for pack {pack.version}; the store holds "
            f"{len(runs)} ({', '.join(runs) or 'none'})"
        )
    if run_a == run_b:
        raise SystemExit(f"run A and run B are the same run ({run_a})")

    a, b = load_run(store, pack.version, run_a), load_run(store, pack.version, run_b)
    labels = read_labels(Path(args.labels)) if Path(args.labels).is_file() else {}

    rows = compare(a, b)
    if args.csv:
        selected = [row for row in rows if row["verdict_changed"]] if args.changed_only else rows
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(build_csv(selected, labels), encoding="utf-8", newline="")
        print(f"\n  wrote {len(selected)} rows to {path}")

    return report(
        rows,
        labels,
        only_a=sorted(set(a) - set(b)),
        only_b=sorted(set(b) - set(a)),
        run_a=run_a,
        run_b=run_b,
        show_all=args.all,
    )


if __name__ == "__main__":
    raise SystemExit(main())
