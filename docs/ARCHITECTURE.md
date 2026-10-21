# Architecture

A free discovery pass, a two-step crawl with four guards, an agent review and
a workbook build. Every stage is a batch job that writes to a date-keyed store
before the next one reads, and nothing holds state in memory between stages.

## Stages

```
input list (CSV)
      │
      v
normalize ─────────────────► registrable domains, deduped, collisions reported
      │
      v
discover ────── free ──────► 10 pages shortlisted, nothing bought
      │
      v
step 1 ──────── 2 credits ─► homepage + best product page, one model read
      │                      exit A: own cleanroom    ─┐
      │                      exit B: cannot need one  ─┼─ four guards, then stop
      │                      exit C: not settled      ─┘  a refused guard falls through
      v
step 2 ──────── 8 credits ─► the rest of the shortlist, verdict settled
      │                      raw/{domain}/{crawl_date}/manifest.json
      v
agent review ── LLM only ──► cleanroom expert (blind), diagnostician, approval sheet
      │
      v
owner approval ────────────► rulings applied by script
      │
      v
hand-over ───── free ──────► integrity check, then one workbook per source list
```

The stage boundaries are where the money is. Discovery is free and decides
what gets paid for. Step 1 and step 2 cost crawl credits and inference. The
agent review costs inference only, because it reads pages already on disk.
The hand-over costs nothing and can be rebuilt at any time.

## The private codebase

| Module | Responsibility |
|---|---|
| `config` | Settings, sector pack loading with loud validation, the model and price table |
| `models` | The schemas every model answer is parsed into |
| `store` | JSONL, markdown and JSON read and write, local disk and Google Cloud Storage behind one interface |
| `normalize` | Registrable domains, redirect resolution, dedupe with collision reporting |
| `llm` | Structured-output calls through OpenRouter or the Anthropic API, one parse path, pinned temperature and seed |
| `discover` | Sitemap and nav harvest, URL scoring, per-type caps, priority for pages that name a cleanroom |
| `fetch` | Firecrawl scrape, the two-page checkpoint, per-company credit accounting, rate limit, budget stop |
| `ladder` | Step 1, exits A, B and C, the four guards, step 2 and the verdict |
| `manifest` | The per-company crawl record that carries the verdict and its evidence |
| `extract` | Per-page extraction and verbatim quote verification |
| `classify` | Review gating and confidence bands |
| `matcher` | Deterministic scoring over the sector pack, an independent second opinion |

About 15,000 lines of source and 863 offline tests. Around the engine sit the
batch runner with its spend gate, the integrity check, the workbook builder,
and the three agents.

## Model routing

| Stage | Model | Why |
|---|---|---|
| Step 1 | `openai/gpt-5.4-nano` | Reads two pages and picks an exit; the cheap model is enough |
| Step 2 | `openai/gpt-5.6-luna-pro` | Attribution and persistence, where a weaker model degrades most |
| Exit confirmation | `openai/gpt-5.6-luna-pro` | Exit B is the only irreversible call, so it is confirmed on the strong model before it is emitted |
| Cleanroom expert | `claude-opus-5-5` | On Sonnet it rejected 7 of the owner's qualified leads |
| Diagnostician | `claude-sonnet-5-5` | Scored 10/10 on Opus and on Sonnet |

The crawl models were chosen after a four-family bakeoff on stored pages.
Model IDs are never written inline; they live in one table with the per-token
prices each run bills itself against.

## Independent readers

The verdict comes from the crawl models. The exit confirmation is the same
model family as the crawl, so it is a second read, not an independent one.
Independence comes from two places: the cleanroom expert, which re-reads the
company blind on a different model family, and the deterministic matcher,
which scores the same evidence against the sector pack without a model. A row
where these disagree is the only kind that reaches the owner.

## Store layout

Identical under a local root or a bucket:

```
raw/{domain}/{crawl_date}/manifest.json
raw/{domain}/{crawl_date}/{url_hash}.md
runs/{project}/{run_id}/results.csv
runs/{project}/{run_id}/stats.md
```

Three properties carry the design:

- **`raw/` is keyed by crawl date only, never by pack version.** Crawling and
  prompting version independently. A new pack or a rewritten prompt replays
  the pages already on disk, which is why a fix is tested for about a dollar
  of inference and no credits.
- **Writes refuse to overwrite** unless the caller says so. A re-run that
  rewrites history invalidates every evaluation that cited it.
- **Every reader pins the crawl date.** A company has one folder per crawl,
  and anything that globs across dates mixes runs. Two cost reports once
  disagreed by 76 credits because ten companies were charged an earlier run's
  bill.

## The storage seam

One interface over a local root and a Google Cloud Storage bucket. Nothing
above it knows which is in use, and the two are tested against each other for
byte-identical output, using an in-memory double of the storage client so no
bucket or credentials are needed.

## The provider seam

One module speaks to model APIs. OpenRouter takes `response_format`; the
Anthropic API takes `output_config.format`. The module varies the request and
always returns raw JSON that Pydantic validates, so swapping providers is a
settings change. Both providers reject schemas that carry `minimum`,
`maxLength` and similar constraints, so those are stripped from the request
and re-checked on the response, which is where validation belongs.

## The sector pack

The vocabulary is configuration: about 1,100 terms across positive products,
negative products, trap terms, seller cues, outsourcing cues, careers titles
and URL patterns, plus the scoring weights and verdict bands. Loading
validates loudly. A term in two lists is a hard error, because a term in two
lists has no defined direction.

A second market means a second pack. See the excerpt in
[samples/demo-pack-food-processing.yaml](../samples/demo-pack-food-processing.yaml).
