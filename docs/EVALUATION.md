# Evaluation

The hard part of this project was not the pipeline. It was building an
instrument honest enough to tell whether the pipeline was getting better, and
then saying plainly what the instrument does not cover.

There are two kinds of evidence here, and they are not interchangeable. The
holdout measured the first version against labels made independently of any
run. The production numbers describe the current version at scale, judged by
the pipeline and reviewed by the project owner.

## The holdout: first version

![Wilson 95% confidence intervals for exact-label accuracy, qualified-lead precision, qualified-lead recall and the true base rate on the 23-company holdout.](intervals.svg)

Measured against 23 companies adjudicated by two independent readers against
the live sites:

| Metric | Value | 95% CI |
|---|---|---|
| Exact-label accuracy | 18/23 = 78.3% | [58.1%, 90.3%] |
| Qualified-lead precision | 9/10 = 90.0% | [59.6%, 98.2%] |
| Qualified-lead recall | 9/12 = 75.0% | [46.8%, 91.1%] |
| True base rate of qualified leads | 12/23 = 52.2% | [33.0%, 70.8%] |
| Errors routed to manual review | 5/5 = 100% | |

The intervals are wide because n is 23. They are printed anyway, and that is
the point: a single accuracy figure on a 23-row holdout is a claim about noise.

The last row matters more than the first. At this sample size accuracy moves
on one company. What does not move on one company is whether the system knows
when it is unsure, and every wrong verdict in the holdout was already flagged
for review.

## Label sets, never pooled

Five labelled sets, kept separate because they were made by different
processes and are not interchangeable evidence:

| Set | Rows | How labelled | What it is for |
|---|---|---|---|
| A | 56 | Confirmed against live sites during development | Development, prompt iteration |
| A, small companies | 51 | The same, restricted to the target size band | Development |
| B | 25 | Two independent readers, independent of any run | **The only accuracy instrument** |
| C | 38 | One reviewer; 12 rows reconciled to pipeline output | Regression detection only |
| Holdout | 30 | Reserved, unlabelled | The next honest test |

Pooling them would produce a bigger n and a worse instrument.

## The circularity check

Set C has 12 rows whose labels were changed to match what the pipeline
already said. Scoring a run against those rows cannot measure accuracy; it
measures whether the run agrees with itself.

So the evaluation harness reads each row's label provenance and prints a
banner naming every row whose label was reconciled to a pipeline verdict. A
test asserts the exact count, with a comment that it must fall as rows are
independently confirmed and must never rise. Set C's apparent 96.8%
exact-match is, correctly, never quoted anywhere.

## Comparing two runs

Every comparison prints the exact-match delta and, directly under it,
McNemar's exact test. This exists because of a specific mistake: a round of
eight fixes moved the holdout from 17/23 to 18/23 and was written up as a
small improvement. McNemar's exact p was 1.00, two companies fixed and one
regressed. The honest report was "no evidence of change". A metric that can
only move up is not a metric.

## Production: list 2, about 1,100 companies

The current version ran on a trade-show exhibitor list of medical device
manufacturers and their suppliers, in six batches from the easiest stratum to
the hardest. Shares are of every company on the list and are rounded.

| | Crawl emitted | After agent review and owner rulings |
|---|---|---|
| Proven qualified lead (`in_house`) | 30% | 28% |
| Likely qualified lead (`likely_in_house`) | 10% | 20% |
| Not a qualified lead | 13% | 31% |
| Not settled, to review (`unknown`) | 45% | 20% |
| No verdict: crawl failed, or the site now belongs to another company | 2% | 1% |
| **Total** | **100%** | **100%** |

About 540 qualified leads for about $72 in total, crawl and inference, audits
and replays included: about seven cents per company and thirteen cents per
qualified lead. The remaining no-verdict rows are domains that now redirect to
a different company, so no verdict about the listed company is possible.

The emitted share of `unknown` climbed from about 12% on the first batch to
73% on the hardest. A replay of the first batch's stored pages on the current
models showed the model change explains 5 to 14 points of that; the rest is
the list getting harder.

On 18 companies with an independent prior label, the crawl emitted `in_house`
on 11 of 12 known positives and on none of 5 known negatives. The positives
are existing customers, chosen for being known and having findable evidence,
so 11/12 is not a recall figure for a cold list. The negatives are the
informative half.

The agent tests, and why they are not an independent measure, are in
[AGENTS.md](AGENTS.md).

## What the numbers do not cover

Stated plainly, because a portfolio that leaves this out is selling something:

- **There is no current independent accuracy figure.** The holdout measured
  the first version. Production numbers are the pipeline's judgment, reviewed
  by the owner whose rulings the expert agent applies.
- **"Qualified lead" is not a sales outcome.** The next measure that matters
  is calls, meetings and orders per qualified lead.
- **Site text caps recall.** Some companies do not describe their facilities
  on their website. No prompt work recovers a fact that was never published,
  which is why "not settled" is a first-class label rather than a failure.
- **The cloud storage backend has never run against a real bucket.** It is
  tested against an in-memory double, which covers paths and framing and
  nothing about auth or permissions.
