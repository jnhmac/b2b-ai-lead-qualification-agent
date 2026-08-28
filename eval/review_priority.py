"""Rank flagged domains by how much a reviewer's next hour is worth spending
on each one. Reads a stored classification run — no LLM call, no network, no
credits, same house rule as evaluate.py and compare_runs.py.

    python eval/review_priority.py                    # newest run, medical_v1
    python eval/review_priority.py --run 20260825-batch1-rescore
    python eval/review_priority.py --csv outputs/review_priority.csv

`needs_manual_review` says WHETHER a row is queued; this says in what ORDER
to look at them, targeting `review_minutes_per_qualified_lead` rather than
precision (the goal is minimizing manual work, not eliminating it). Every
weight below is a first-cut starting guess to retune once real review time
has been measured against it, not a final answer — see `review_priority()`.

**This module only became load-bearing on 2026-08-26.** Two things changed
under it, both measured on Tier B's 23 adjudicated rows
(`outputs/2026-08-26_design-review.md`):

1. `ungrounded_negative` and `matcher_disagreement` used to FORCE review. They
   now only rank (`classify.REVIEW_GATES`), because as gates they queued 84% of
   rows to catch 5 errors that confidence alone catches at 61%. The two bonuses
   below are the whole of what those signals now do, so their weights matter in
   a way they did not before.
2. Until 2026-08-26 every stored record had `flag_reasons: []` — the field was
   added the same day and no run had executed since — so
   `UNGROUNDED_NEGATIVE_BONUS` and both `DISAGREEMENT_*` bonuses were dead on
   every real input, and the score collapsed to a categorical re-sort by
   `cleanroom_status`. Any ranking claim made before a run carrying
   `flag_reasons` is a claim about that degenerate score, not this one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cleanroom.classify import read_classifications  # noqa: E402
from cleanroom.config import load_pack, load_settings  # noqa: E402
from cleanroom.models import ClassificationRecord  # noqa: E402
from cleanroom.store import Store, get_store  # noqa: E402


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def latest_run(store: Store, pack_version: str) -> str | None:
    prefix = f"classifications/pack={pack_version}/"
    runs = {
        path[len(prefix):].split("/")[0].removeprefix("run=")
        for path in store.list_paths(prefix.rstrip("/"))
        if path.startswith(prefix)
    }
    return sorted(runs)[-1] if runs else None


# ---------------------------------------------------------------------------
# The score
# ---------------------------------------------------------------------------

#: Where each label sits on a lead-value scale, for scoring disagreement
#: DIRECTION: matcher implying a higher rank than the LLM said is the
#: dangerous direction (a possibly missed lead), not just any disagreement.
_LEAD_RANK: dict[str, int] = {
    "no_cleanroom_needed": 0,
    "outsourced": 1,
    "unknown": 1,
    "likely_in_house": 2,
    "in_house": 3,
}

#: How bad is it if THIS specific flagged row is actually wrong.
#: no_cleanroom_needed highest: wrong here silently deletes a real lead,
#: this pipeline's worst failure mode. in_house/likely_in_house lowest of
#: the flagged set: already being treated as a lead, so review here is a
#: precision check, not a recall rescue.
_STATUS_RISK_WEIGHT: dict[str, float] = {
    "no_cleanroom_needed": 3.0,
    "unknown": 1.0,
    "outsourced": 1.0,
    "likely_in_house": 0.5,
    "in_house": 0.5,
}

UNGROUNDED_NEGATIVE_BONUS = 2.0
DISAGREEMENT_UP_BONUS = 2.0
DISAGREEMENT_DOWN_BONUS = 0.5
CONFIDENCE_WEIGHT = 1.0
EVIDENCE_WEIGHT = 0.1
MAX_EVIDENCE_COUNTED = 5


def review_priority(record: ClassificationRecord) -> float:
    """First-cut priority score for one flagged row. Higher reviews first.

    0.0 for anything not flagged (nothing to queue) and for a `thin_render`/
    `site_moved` row: those need a re-crawl or an identity decision, not a
    closer read of evidence that barely exists — `main()` reports them in a
    separate section rather than mixing action types on one scale.
    """
    if not record.needs_manual_review:
        return 0.0
    if record.run_status not in (None, "ok"):
        return 0.0

    score = _STATUS_RISK_WEIGHT.get(record.cleanroom_status, 1.0)

    if "ungrounded_negative" in record.flag_reasons:
        score += UNGROUNDED_NEGATIVE_BONUS

    if "matcher_disagreement" in record.flag_reasons and record.matcher_verdict:
        llm_rank = _LEAD_RANK.get(record.cleanroom_status, 1)
        matcher_rank = _LEAD_RANK.get(record.matcher_verdict["cleanroom_status"], 1)
        score += DISAGREEMENT_UP_BONUS if matcher_rank > llm_rank else DISAGREEMENT_DOWN_BONUS

    score += CONFIDENCE_WEIGHT * (1 - record.confidence)
    score += EVIDENCE_WEIGHT * min(len(record.evidence), MAX_EVIDENCE_COUNTED)

    return round(score, 3)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_csv(rows: Sequence[tuple[float, ClassificationRecord]]) -> str:
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(["priority", "domain", "cleanroom_status", "confidence", "flag_reasons"])
    for score, record in rows:
        writer.writerow([
            score, record.domain, record.cleanroom_status,
            f"{record.confidence:.2f}", " ".join(record.flag_reasons),
        ])
    return buffer.getvalue()


def report(records: Sequence[ClassificationRecord], run_id: str) -> int:
    print(f"\n  run     : {run_id}")
    print(f"  records : {len(records)}")

    needs_recrawl = sorted(
        (r for r in records if r.run_status not in (None, "ok")),
        key=lambda r: r.domain,
    )
    ranked = sorted(
        ((review_priority(r), r) for r in records if r.needs_manual_review),
        key=lambda pair: -pair[0],
    )
    ranked = [(score, r) for score, r in ranked if score > 0.0]

    if needs_recrawl:
        print(f"\n  NEEDS RE-CRAWL / IDENTITY DECISION, not ranked ({len(needs_recrawl)})\n")
        print(f"  {'domain':<28}{'run_status':<16}")
        print(f"  {'-' * 28}{'-' * 16}")
        for r in needs_recrawl:
            print(f"  {r.domain:<28}{r.run_status:<16}")

    print(f"\n  REVIEW QUEUE, ranked ({len(ranked)} of {len(records)})\n")
    print(f"  {'priority':>8}  {'domain':<28}{'status':<20}{'conf':>5}  flag_reasons")
    print(f"  {'-' * 8}  {'-' * 28}{'-' * 20}{'-' * 5}  {'-' * 28}")
    for score, r in ranked:
        print(
            f"  {score:>8.2f}  {r.domain:<28}{r.cleanroom_status:<20}{r.confidence:>5.2f}  "
            f"{' '.join(r.flag_reasons)}"
        )

    print()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="classification run id; defaults to the newest")
    parser.add_argument("--pack", default="medical_v1")
    parser.add_argument("--csv", help="write the ranked queue here")
    args = parser.parse_args(argv)

    settings = load_settings()
    pack = load_pack(args.pack, pack_dir=settings.pack_dir)
    store = get_store(settings)

    run_id = args.run or latest_run(store, pack.version)
    if not run_id:
        raise SystemExit(f"no classification run in the store for pack {pack.version}")

    records = read_classifications(store, pack.version, run_id)

    if args.csv:
        ranked = sorted(
            ((review_priority(r), r) for r in records if r.needs_manual_review),
            key=lambda pair: -pair[0],
        )
        ranked = [(score, r) for score, r in ranked if score > 0.0]
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(build_csv(ranked), encoding="utf-8", newline="")
        print(f"\n  wrote {len(ranked)} rows to {path}")

    return report(records, run_id)


if __name__ == "__main__":
    raise SystemExit(main())
