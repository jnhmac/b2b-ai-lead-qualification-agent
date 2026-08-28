"""Score a stored classification run against labels. Phase 7b.

Reads verdicts off disk and joins them to a label file. **No LLM call, no
network, no credits.** The run being scored has already happened; scoring it is
a join, and that is the whole payoff of keeping the raw store separate.

    python eval/evaluate.py                         # newest run vs goldset
    python eval/evaluate.py --run 20260811-001815
    python eval/evaluate.py --labels eval/batch1_labels.jsonl

**Never reports a single accuracy number**, per BUILD.md. The goldset's base
rate is distorted on purpose by injected positives, so an overall figure would
be flattering and would mean nothing. Precision and recall are reported per
label instead.

Four things this prints that a plain confusion matrix would not:

- **Precision on hard negatives**, which is the Phase 7b gate. The hard
  negatives are companies whose sites are full of cleanroom-adjacent language
  and who run no cleanroom. Calling one of them `in_house` is the specific
  failure this task is prone to.
- **Two lead-level binaries, not one.** A verdict that gets the five-way label
  wrong but the lead call right costs nothing, and the five-way numbers hide
  that. But which lead call? `AUTO-WRITE` is `in_house` only, what goes to the
  sales list with no human, and is what the gate scores. `QUALIFIED` adds
  `likely_in_house`, which is what a salesperson would actually call.
  **Added 2026-08-26 because only the strict binary existed, and it understates
  the tool badly**: on Tier B, 6 of the 9 correct lead calls are
  `likely_in_house`, so the strict number scored every one of them as a miss.
- **95% Wilson intervals on every rate.** Every n here is 13 to 56. 12/13 reads
  as a comfortably passed gate at 92.3% and its interval runs down to 66.7%.
  This project has repeatedly read sampling noise as a result; the interval is
  the correction.
- **A circularity warning.** A label file whose rows were reconciled to match a
  run's own output cannot measure that run. `eval/goldset_tier_c_batch1.jsonl`
  has 13 such rows of 31, which is why it scores 96.8% and means nothing. Rows
  carrying `label_provenance: reconciled_to_pipeline` trigger a banner above
  every number. See `outputs/2026-08-26_design-review.md`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cleanroom.classify import read_classifications  # noqa: E402
from cleanroom.config import load_pack, load_settings  # noqa: E402
from cleanroom.store import Store, get_store  # noqa: E402

LABELS = ("in_house", "likely_in_house", "outsourced", "no_cleanroom_needed", "unknown")

#: The strict binary: the only label that auto-writes a lead without a human.
#: This is what the Phase 7b gate scores. Do not widen it.
LEAD = "in_house"

#: The commercial binary: what a salesperson would actually call. Measured
#: 2026-08-26 on Tier B, 6 of the 9 correct lead calls are `likely_in_house`,
#: so a recall number that excludes it understates the tool by a wide margin.
QUALIFIED_LEAD = ("in_house", "likely_in_house")

#: A label file row whose `expected_label` was reconciled to match a
#: classification run's own output cannot measure that pipeline's accuracy.
#: See eval/goldset_tier_c_batch1.jsonl and outputs/2026-08-26_design-review.md.
CIRCULAR_PROVENANCE = "reconciled_to_pipeline"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def read_labels(path: Path) -> dict[str, dict[str, Any]]:
    """A label file: JSONL with `domain` and `expected_label`.

    Rows carrying no `expected_label` are skipped rather than treated as a
    label, so a half-filled review file scores the part that is filled in. That
    matters because the batch review CSV is meant to be labelled incrementally.
    """
    if not path.is_file():
        raise SystemExit(f"label file not found: {path}")

    labels: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        expected = (row.get("expected_label") or "").strip()
        if not expected:
            continue
        if expected not in LABELS:
            raise SystemExit(f"{path.name}: {row.get('domain')!r} has unknown label {expected!r}")
        labels[row["domain"]] = row
    return labels


def latest_run(store: Store, pack_version: str) -> str | None:
    prefix = f"classifications/pack={pack_version}/"
    runs = {
        path[len(prefix):].split("/")[0].removeprefix("run=")
        for path in store.list_paths(prefix.rstrip("/"))
        if path.startswith(prefix)
    }
    return sorted(runs)[-1] if runs else None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score(
    pairs: Sequence[tuple[str, str, str]]
) -> dict[str, dict[str, float | int]]:
    """Per-label precision and recall from (domain, expected, predicted)."""
    out: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        tp = sum(1 for _, exp, pred in pairs if pred == label and exp == label)
        fp = sum(1 for _, exp, pred in pairs if pred == label and exp != label)
        fn = sum(1 for _, exp, pred in pairs if pred != label and exp == label)
        out[label] = {
            "support": tp + fn,
            "predicted": tp + fp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
        }
    return out


def pct(value: float) -> str:
    return "  n/a " if value != value else f"{value:>6.1%}"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials.

    Wilson rather than normal-approximation because every n in this project is
    small (23 to 56) and the proportions sit near the ends, where the normal
    interval runs past 0 and 1 and is simply wrong. An empty denominator
    returns the whole interval: no observations, no information.

    This exists because the intervals are the finding. 18/23 = 78.3% reads as a
    solid number and is [58.1%, 90.3%]; 12/13 = 92.3% reads as a passed gate and
    is [66.7%, 98.6%]. Printing the point estimate alone has repeatedly led this
    project to treat sampling noise as a result.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def ci(k: int, n: int) -> str:
    if n <= 0:
        return ""
    lo, hi = wilson(k, n)
    return f"[{lo:>5.1%}, {hi:>5.1%}]"


def is_strict_lead(label: str) -> bool:
    """`in_house` only. The Phase 7b gate's definition."""
    return label == LEAD


def is_qualified_lead(label: str) -> bool:
    """`in_house` or `likely_in_house`. What sales would actually call."""
    return label in QUALIFIED_LEAD


def lead_counts(
    pairs: Sequence[tuple[str, str, str]], is_lead: Callable[[str], bool]
) -> dict[str, Any]:
    """Confusion counts plus precision/recall for one lead definition."""
    tp = sum(1 for _, e, p in pairs if is_lead(p) and is_lead(e))
    fp = sum(1 for _, e, p in pairs if is_lead(p) and not is_lead(e))
    fn = sum(1 for _, e, p in pairs if not is_lead(p) and is_lead(e))
    tn = sum(1 for _, e, p in pairs if not is_lead(p) and not is_lead(e))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "base_rate": (tp + fn) / len(pairs) if pairs else float("nan"),
    }


def circular_rows(labels: dict[str, dict[str, Any]]) -> list[str]:
    """Domains whose label was reconciled to a pipeline run's own output.

    Scoring against these measures nothing: the label was moved to match the
    prediction. Label files that predate the `label_provenance` field return
    nothing, which is correct -- absence of the field is not a claim either way.
    """
    return sorted(
        domain for domain, row in labels.items()
        if row.get("label_provenance") == CIRCULAR_PROVENANCE
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(
    pairs: Sequence[tuple[str, str, str]],
    labels: dict[str, dict[str, Any]],
    missing: Iterable[str],
    unlabelled: Iterable[str],
    run_id: str,
    label_path: Path,
) -> int:
    print(f"\n  run     : {run_id}")
    print(f"  labels  : {label_path}")
    print(f"  scored  : {len(pairs)} domains")

    missing = sorted(missing)
    unlabelled = sorted(unlabelled)
    if missing:
        print(f"  labelled but not in the run ({len(missing)}): {', '.join(missing[:8])}"
              + (" ..." if len(missing) > 8 else ""))
    if unlabelled:
        print(f"  in the run but unlabelled ({len(unlabelled)}): {', '.join(unlabelled[:8])}"
              + (" ..." if len(unlabelled) > 8 else ""))
    if not pairs:
        print("\n  nothing to score.")
        return 1

    # ---- circularity warning, before any number ---------------------------
    # A label reconciled to a run's own output cannot measure that run. Tier C
    # is 13 such rows out of 31, which is why it scores 96.8% and means nothing.
    # Printed first and loudly, because the number below it is the misleading
    # part and a reader who scrolls past this will quote it.
    circular = circular_rows({d: labels[d] for d, _, _ in pairs})
    if circular:
        share = len(circular) / len(pairs)
        print(f"\n  {'!' * 62}")
        print(f"  WARNING: {len(circular)} of {len(pairs)} scored labels ({share:.0%}) were")
        print("  RECONCILED TO A PIPELINE RUN'S OWN OUTPUT. Every number below is")
        print("  circular to that extent. This file is a REGRESSION check -- it")
        print("  catches a change that breaks something currently right -- and is")
        print("  not evidence of accuracy. Do not quote its percentages.")
        print(f"  {', '.join(circular[:6])}" + (" ..." if len(circular) > 6 else ""))
        print(f"  {'!' * 62}")

    # ---- per label -------------------------------------------------------
    stats = score(pairs)
    print("\n  PER LABEL. No overall accuracy figure: the base rate is distorted")
    print("  on purpose by injected positives, so one number would mislead.")
    print("  Intervals are 95% Wilson. At these sample sizes they are wide, and")
    print("  that width is the finding, not a footnote.\n")
    print(f"  {'label':<20}{'true':>5}{'pred':>5}{'precision':>10} {'95% CI':<16}"
          f"{'recall':>8} {'95% CI':<16}{'fp':>3} {'fn':>3}")
    print(f"  {'-' * 20}{'-' * 5}{'-' * 5}{'-' * 10} {'-' * 16}{'-' * 8} {'-' * 16}"
          f"{'-' * 3} {'-' * 3}")
    for label in LABELS:
        s = stats[label]
        print(
            f"  {label:<20}{s['support']:>5}{s['predicted']:>5}"
            f"{pct(s['precision']):>10} {ci(s['tp'], s['predicted']):<16}"
            f"{pct(s['recall']):>8} {ci(s['tp'], s['support']):<16}"
            f"{s['fp']:>3} {s['fn']:>3}"
        )

    # ---- the commercial question ----------------------------------------
    # Two binaries, because they answer different questions and quoting only the
    # strict one has understated this tool for months. STRICT is what auto-writes
    # a lead with no human and is what the Phase 7b gate scores. QUALIFIED is
    # what a salesperson would call: on Tier B, 6 of the 9 correct lead calls are
    # `likely_in_house`, so the strict binary scores them all as misses.
    for title, blurb, is_lead in (
        ("QUALIFIED LEAD vs NOT A LEAD",
         "`in_house` or `likely_in_house` -- a name worth calling.",
         is_qualified_lead),
        ("AUTO-WRITE LEAD vs NOT A LEAD",
         "`in_house` only -- writes to the sales list with no human. The gate's binary.",
         is_strict_lead),
    ):
        c = lead_counts(pairs, is_lead)
        print(f"\n  {title}")
        print(f"  {blurb}\n")
        print(f"    leads found correctly   : {c['tp']}")
        print(f"    false positives         : {c['fp']}   <- wasted sales effort")
        print(f"    leads missed            : {c['fn']}   <- lost revenue, invisible without labels")
        print(f"    correctly not a lead    : {c['tn']}")
        print(f"    true base rate          : {pct(c['base_rate'])}  ({c['tp'] + c['fn']} of {len(pairs)})")
        if c["tp"] + c["fp"]:
            print(f"    lead precision          : {pct(c['precision'])}  "
                  f"{ci(c['tp'], c['tp'] + c['fp'])}")
        if c["tp"] + c["fn"]:
            print(f"    lead recall             : {pct(c['recall'])}  "
                  f"{ci(c['tp'], c['tp'] + c['fn'])}")

    # ---- the gate --------------------------------------------------------
    groups = defaultdict(list)
    for domain, expected, predicted in pairs:
        groups[labels[domain].get("group") or "(ungrouped)"].append(
            (domain, expected, predicted)
        )

    if len(groups) > 1:
        print("\n  BY GROUP\n")
        print(f"  {'group':<18}{'n':>4}{'exact':>8}{'lead calls wrong':>18}")
        print(f"  {'-' * 18}{'-' * 4}{'-' * 8}{'-' * 18}")
        for name, rows in sorted(groups.items()):
            exact = sum(1 for _, e, p in rows if e == p)
            lead_wrong = sum(1 for _, e, p in rows if (e == LEAD) != (p == LEAD))
            print(f"  {name:<18}{len(rows):>4}{exact:>8}{lead_wrong:>18}")

    hard = groups.get("Hard negatives", [])
    exit_code = 0
    if hard:
        # THE GATE. A hard negative called `in_house` is the failure this task
        # is built to resist: a site drenched in cleanroom vocabulary that
        # operates no cleanroom.
        false_leads = [(d, e, p) for d, e, p in hard if p == LEAD and e != LEAD]
        print(f"\n  {'=' * 62}")
        print("  THE PHASE 7b GATE: precision on hard negatives")
        print(f"  {'=' * 62}")
        held = len(hard) - len(false_leads)
        print(f"    hard negatives scored : {len(hard)}")
        print(f"    called a lead wrongly : {len(false_leads)}")
        print(f"    precision             : {held / len(hard):.1%}  {ci(held, len(hard))}")
        # 12/13 = 92.3% reads like a comfortably passed gate and its interval
        # runs down to 66.7%. Quote the interval or quote nothing.
        for domain, expected, predicted in false_leads:
            print(f"      FALSE LEAD {domain}: expected {expected}, got {predicted}")
        exit_code = 1 if false_leads else 0

    # ---- every disagreement, so the number can be argued with ------------
    wrong = [(d, e, p) for d, e, p in pairs if e != p]
    if wrong:
        print(f"\n  DISAGREEMENTS ({len(wrong)} of {len(pairs)})\n")
        print(f"  {'domain':<28}{'expected':<12}{'got':<12}{'lead call':<10}group")
        print(f"  {'-' * 28}{'-' * 12}{'-' * 12}{'-' * 10}{'-' * 18}")
        for domain, expected, predicted in sorted(wrong):
            same_lead = "same" if (expected == LEAD) == (predicted == LEAD) else "DIFFERENT"
            print(
                f"  {domain:<28}{expected:<12}{predicted:<12}{same_lead:<10}"
                f"{labels[domain].get('group') or ''}"
            )

    print()
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", default=str(REPO_ROOT / "eval" / "goldset.jsonl"))
    parser.add_argument("--run", help="classification run id; defaults to the newest")
    parser.add_argument("--pack", default="medical_v1")
    args = parser.parse_args(argv)

    settings = load_settings()
    pack = load_pack(args.pack, pack_dir=settings.pack_dir)
    store = get_store(settings)

    run_id = args.run or latest_run(store, pack.version)
    if not run_id:
        raise SystemExit(f"no classification run in the store for pack {pack.version}")

    labels = read_labels(Path(args.labels))
    predictions = {
        record.domain: record.cleanroom_status
        for record in read_classifications(store, pack.version, run_id)
    }

    pairs = [
        (domain, row["expected_label"], predictions[domain])
        for domain, row in sorted(labels.items())
        if domain in predictions
    ]
    return report(
        pairs,
        labels,
        missing=set(labels) - set(predictions),
        unlabelled=set(predictions) - set(labels),
        run_id=run_id,
        label_path=Path(args.labels),
    )


if __name__ == "__main__":
    raise SystemExit(main())
