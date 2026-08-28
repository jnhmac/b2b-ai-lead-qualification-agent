# Architecture

Six stages, four CLI commands, two swappable seams. Everything is a batch job
writing to a content-addressed store; nothing holds state in memory between
stages.

## Module map

| Module | Lines | Responsibility |
|---|---|---|
| `config.py` | 664 | Environment settings, sector pack loading with loud validation, the model and price table |
| `models.py` | 218 | `Finding`, `ExtractionRecord`, `Evidence`, `ClassificationRecord` |
| `store.py` | 374 | JSONL, markdown and JSON read/write, local and GCS behind one interface |
| `normalize.py` | 375 | Registrable domain extraction, redirect resolution, dedupe with collision reporting |
| `llm.py` | 502 | Structured-output calls against OpenRouter or the Anthropic API, one parse path |
| `discover.py` | 1,695 | Sitemap and nav harvest, URL scoring, per-type caps, crawl manifest |
| `fetch.py` | 1,223 | Firecrawl scrape into the raw store, credit accounting, rate limiting, budget stop |
| `extract.py` | 1,625 | Per-page LLM extraction, verbatim quote verification, run reporting |
| `classify.py` | 2,063 | Per-domain verdict, review gating, confidence bands |
| `matcher.py` | 474 | Deterministic scoring pass over the sector pack, an independent second opinion |
| `ladder.py` | 323 | Conditional crawl ladder: escalate only when the cheap pass is inconclusive |
| `cli.py` | 809 | `crawl`, `extract`, `classify`, `credits` |

Roughly 10,000 lines of source against 9,900 lines of tests.

## Data flow

```
eval/goldset.jsonl ──┐
a CSV, JSONL or      │
plain text list      │
                     v
              normalize.py ────────────► registrable domains, deduped
                     │
                     v
              discover.py  ──── free ──► sitemap.xml, robots.txt, homepage nav
                     │                   scored, capped per page type
                     │
                     v                   raw/{domain}/{crawl_date}/manifest.json
              fetch.py ──── metered ───► raw/{domain}/{crawl_date}/{url_hash}.md
                     │                   one Firecrawl credit per page
                     v
              extract.py ── metered ───► extractions/pack={pack}/run={run_id}/part-*.jsonl
                     │                   one LLM call per page, quotes verified
                     v
              classify.py ─ metered ───► classifications/pack={pack}/run={run_id}/part-*.jsonl
                     │                   one LLM call per domain
                     v
              eval/evaluate.py ────────► accuracy, Wilson intervals, review-queue stats
```

The stage boundaries are where the money is. Discovery is free and decides what
gets paid for. Fetch is the expensive stage in credits, extract is the expensive
stage in tokens, and both write everything they buy to disk so that no later
change forces a re-buy.

## Store layout

Identical under a local root or a bucket:

```
raw/{domain}/{crawl_date}/manifest.json
raw/{domain}/{crawl_date}/{url_hash}.md
extractions/pack={pack}/run={run_id}/part-*.jsonl
classifications/pack={pack}/run={run_id}/part-*.jsonl
```

Two properties are load-bearing:

**`raw/` is keyed by crawl date only, never by pack version.** Scraping and
prompting version independently. A new pack or a rewritten prompt reads the
pages already on disk, which is why a prompt iteration costs minutes instead of
hours.

**Writes refuse to overwrite unless the caller passes `overwrite=True`.** A
re-run that rewrites history invalidates every evaluation that cited it, so the
default is refusal and the exception is explicit.

## The storage seam

`store.py` exposes one interface over a local filesystem root and a GCS bucket.
Nothing above it knows which is in use; `STORE_ROOT=./data` and
`STORE_ROOT=gs://bucket/prefix` are the entire difference.

It is tested against an in-memory double of the `google.cloud.storage` client,
so both backends run in CI with no bucket and no credentials. That covers path
construction, encoding and JSONL framing, and it does not cover auth, network
failures or bucket permissions, which need a real bucket. The gate test asserts
the two backends produce byte-identical output.

## The provider seam

`llm.py` is the only module that speaks to a model API. Two providers, one
parse path:

- OpenRouter takes `response_format`; the Anthropic API takes
  `output_config.format`. `llm.py` varies the request and then always returns
  raw JSON text that Pydantic validates. Swapping providers is an
  `LLM_PROVIDER` change and nothing above `llm.py` is touched.
- Both providers reject schemas carrying `minimum`, `maxLength` and similar
  constraints, so those are stripped from the outbound request and re-checked by
  Pydantic on the response, which is where validation belongs anyway.
- Model IDs are never inlined. The same model is spelled
  `anthropic/claude-haiku-4.5` on OpenRouter and `claude-haiku-4-5` on the
  Anthropic API. Both live in `STAGE_MODELS` in `config.py`, alongside the
  per-token prices each run bills itself against.

## The sector pack

The vocabulary is configuration, not code. `packs/medical_v1.yaml` and
`packs/medical_v2.2.yaml` hold roughly 1,100 terms across positive products,
negative products, trap terms, outsourcing cues and URL path patterns, plus the
scoring weights and verdict bands.

Loading validates loudly. A surface form appearing in two lists is a hard error,
because a term in two lists has no defined direction. Weight names are checked
against an enum, and the `ignore` weight is rejected anywhere except the trap
term list.

A second sector means a second YAML file, not a second codebase.

## Two independent opinions

`classify.py` produces the verdict from an LLM call over the extracted evidence.
`matcher.py` scores the same evidence deterministically against the pack, with
anchored regex for the numeric ISO and class patterns, and never produces the
primary verdict.

The value is in the disagreement. Where the two agree the row can be
auto-accepted; where they disagree it goes to the review queue. That is what
lets the auto-accept rate rise without leaking errors, and it is measured rather
than assumed: see [EVALUATION.md](EVALUATION.md).
