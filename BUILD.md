# Cleanroom classifier: core engine build spec

> **Read this as history.** This is the build spec as written before any
> code, kept as it was apart from removing client detail. The design moved on:
> the ten-page crawl became a two-step read, the labels changed, and three
> agents now review every batch. The current design is in
> [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**What this builds.** A batch pipeline that takes a list of medical device
manufacturer domains, crawls a handful of pages from each, and returns a
four-way verdict on whether the company operates a cleanroom, with verbatim
evidence quotes.

**What this does NOT build.** No spreadsheet export, no CRM sync, no webhooks,
no visitor-intent integration, no API endpoint, no queue, no managed deploy.
Those come later and are explicitly out of scope. Do not add them, do not stub
them, do not write adapters "for later."

**Language.** Python 3.11+.

---

## The problem in one paragraph

The input list is already filtered to medical device manufacturers by an
industry classification code.
That label is too broad: a wheelchair manufacturer and a catheter manufacturer
both carry it, but only one needs a cleanroom. Keyword matching fails because
the certification every med device firm advertises (ISO 13485) says nothing
about facilities. The signal that works is the product catalogue plus explicit
contamination-control language, and the hard case is a company that sells
sterile product but outsources all manufacturing. That is why classification is
an LLM call over extracted evidence rather than a regex.

---

## Locked decisions

| Decision | Value |
|---|---|
| Scraper | Firecrawl. Free tier for goldset, Hobby for full run |
| Raw store | Local filesystem for dev, GCS for the full run. Same interface |
| Output labels | `in_house` / `outsourced` / `none` / `unknown` |
| LLM | Anthropic API, structured output |
| Optimize for | Precision over recall |
| Sector | `medical_v1` only. The pack is config, not code |

**No exclusion filters.** Enterprise multi-division companies and non-English
sites are crawled and classified like everything else. Flag them in the output
for manual review. Do not skip them at normalize or discovery time.

---

## Repo structure

```
cleanroom-classifier/
  pyproject.toml
  .env.example
  README.md
  packs/
    medical_v1.yaml        # provided, drop in as-is
  src/cleanroom/
    __init__.py
    config.py              # env + pack loading and validation
    models.py              # pydantic schemas
    normalize.py           # domain cleanup and dedupe
    discover.py            # sitemap + nav harvest + URL scoring
    fetch.py               # Firecrawl client
    store.py               # local + GCS backends, one interface
    extract.py             # per-page LLM extraction
    classify.py            # per-domain LLM classification
    pipeline.py            # orchestration
    cli.py                 # entry points
  eval/
    goldset.jsonl          # provided, 47 labeled domains
    evaluate.py            # precision/recall per label
    compare_runs.py        # diff two run_ids
  tests/
```

## Store layout

Identical paths under a local root or a GCS bucket. The backend is a config
switch, nothing above `store.py` knows which is in use.

```
raw/{domain}/{crawl_date}/manifest.json
raw/{domain}/{crawl_date}/{url_hash}.md
extractions/pack={pack}/run={run_id}/part-*.jsonl
classifications/pack={pack}/run={run_id}/part-*.jsonl
```

**The rule that makes iteration cheap.** `raw/` is keyed by crawl date only,
never by pack version. Scraping and prompting version independently. Everything
append-only. Never overwrite a run.

## Environment

```
FIRECRAWL_API_KEY=
ANTHROPIC_API_KEY=
STORE_BACKEND=local          # local | gcs
STORE_ROOT=./data            # local path, or gs://bucket for gcs
```

Never hardcode keys. Never commit `.env`.

---

# Phase 1: Scaffold

**Build**

1. Repo, `pyproject.toml`, pinned dependencies.
2. `config.py`: load env, load and validate `packs/medical_v1.yaml`. Fail loud
   on a malformed pack.
3. `models.py`: the pydantic schemas below.
4. `store.py`: read/write JSONL, read/write markdown, existence check. Local
   and GCS backends behind one interface.
5. `normalize.py`: strip to registrable domain with `tldextract`, resolve
   redirects, dedupe. Input may be full URLs (`https://www.example.com/`) or
   bare domains.

**Done when** the 47 goldset domains normalize cleanly and a JSONL file
round-trips through both store backends.

**Watch for** parent/subsidiary collisions during dedupe. Log them, do not
silently merge.

---

# Phase 2: Sector pack

The pack is provided. Build the loader and validator.

**Done when** `medical_v1.yaml` loads, validates, and every list is populated.

**Watch for** the trap list being treated as negative signal. It is neutral.
The classifier must see these terms and know to discount them, not subtract
points. This is the single most common way to get this wrong.

---

# Phase 3: Discovery

**Build** `discover.py`. Returns a ranked URL list, capped at 10.

1. Try `/sitemap.xml`, then `/sitemap_index.xml`, then a `Sitemap:` directive
   in `robots.txt`.
2. Fallback: fetch homepage, harvest internal links from header and footer.
3. Score URLs against pack patterns, matching **both URL path and anchor text**.
   Anchor text matters: a page at `/company/heritage` matches no path pattern,
   but a nav link reading "Who We Are" catches it.
4. Select: homepage always, then top scorers respecting per-type caps.
5. Write a manifest recording attempted URLs, 404s, and which page types came
   back empty.

**Done when** discovery over 20 sample domains returns sensible page mixes with
no single type dominating.

**Watch for**
- Deep catalogues eating every slot. Per-type caps prevent this.
- JS-rendered nav returning zero links. Firecrawl's map endpoint handles
  rendering; confirm it is on.
- Any domain yielding fewer than three pages. Flag as a discovery failure. Do
  not pass it through as a thin-site `unknown`: those are different problems.

---

# Phase 4: Fetch and raw store

**Build** `fetch.py` using Firecrawl.

1. Use the map endpoint where a sitemap is missing.
2. Scrape selected URLs, markdown format.
3. Write each page to `raw/{domain}/{crawl_date}/{url_hash}.md`.
4. Write the manifest alongside.
5. Read credits billed from response metadata, log per domain.

**Done when** all target domains are crawled and stored, with a per-domain
credit log.

**Watch for**
- Stealth mode at 5 credits per page instead of 1. Rare on these sites. Alert
  if it fires.
- Do not use Firecrawl's `/extract`. Billed separately, and it re-couples
  scraping to prompting.
- Idempotency: skip domains already crawled on the same date unless forced.

**Budget.** 500 domains at ~10 pages plus map calls lands near 6,000 credits.
Hobby gives 5,000. Expect one extra credit pack or a split across two months.
The 47-domain goldset is ~500 pages and fits the free tier.

---

# Phase 5: Extraction

**Build** `extract.py`. One LLM call per page. Structured output.

```python
class Finding(BaseModel):
    claim: str
    quote: str          # verbatim, mandatory
    confidence: float

class ExtractionRecord(BaseModel):
    domain: str
    url: str
    page_type: str
    crawl_date: date
    pack_version: str
    page_word_count: int
    iso_class_mentions: list[Finding]
    cleanroom_terms: list[Finding]
    sterilization_methods: list[Finding]
    products_named: list[str]
    manufacturing_language: list[Finding]
    trap_terms_present: list[str]
```

**Two rules that carry the design.**

1. Every claim carries a verbatim quote. No quote, the field stays empty.
2. The extractor decides nothing. It reports what the page says. All judgment
   lives in classification.

`products_named` stays uninterpreted. Taxonomy matching happens at classify
time against the pack, so changing the taxonomy never requires re-scraping.

**Required fields** (`domain`, `url`, `page_type`, `crawl_date`,
`pack_version`, `page_word_count`) must always be populated. Empty means a
parsing failure, not a finding. The list fields are allowed to be empty: most
pages have nothing to say about sterilisation.

---

# Phase 6: Classification

**Build** `classify.py`. One LLM call per domain over the aggregated
extractions.

```python
class Evidence(BaseModel):
    quote: str
    source_url: str

class ClassificationRecord(BaseModel):
    domain: str
    cleanroom_status: Literal["in_house", "outsourced", "none", "unknown"]
    iso_class: str | None
    confidence: float
    reasoning: str
    evidence: list[Evidence]
    pages_analyzed: int
    page_types_found: list[str]
    needs_manual_review: bool      # enterprise multi-division, non-English
    run_id: str
    pack_version: str
    source_crawl_date: date
```

**Label definitions, verbatim into the prompt:**

- `in_house`: operates its own cleanroom. ISO class stated, cleanroom square
  footage, gowning procedures, cleanroom job postings, facility described as a
  controlled production environment. Do NOT use this when the evidence only
  shows the product is sterile.
- `outsourced`: sells sterile product, manufactures none of it. Contract
  manufacturer named, manufacturing-partner language, design-and-marketing only
  with no facility mentioned anywhere.
- `none`: the product line does not require a cleanroom. Catalogue is entirely
  durable equipment, furniture, reusable instruments, or software. This is a
  **positive determination from the product taxonomy**, not an absence of
  evidence.
- `unknown`: the site is too thin to judge. Fewer than three pages retrieved,
  no products listed, no facility or manufacturing language at all.

**Edge cases, also verbatim:**

- **Hybrid** (some in-house, some outsourced) resolves to `in_house`. The
  question is whether a cleanroom exists at all.
- **Controlled but unclassified** (a controlled environment with no ISO class
  stated) resolves to `in_house` with `iso_class: null`.
- **Distributor only** resolves to `none`.
- **Parent/subsidiary, recently acquired, non-English**: classify normally but
  set `needs_manual_review: true`.

**The distinction to enforce hardest.** `none` is a determination. `unknown` is
absence of evidence. Collapsing them destroys the re-run population. Expect 15
to 20 percent `unknown`.

**Weight semantics.** Weight is signal *reliability*, not direction. Direction
comes from which pack list a term sits in. A `strong` negative product pushes
toward `none`. `ignore` appears on trap terms only.

---

# Phase 7: Evaluation harness

**This is the gate. Do not run the full 500 before this exists.**

**Build**

1. `evaluate.py`: run classification over `eval/goldset.jsonl` only, report
   precision and recall **per label**.
2. `compare_runs.py`: diff two `run_id`s, list domains whose verdict changed
   and why.

**Iteration loop**

```
edit prompt or pack -> run over goldset (minutes, no re-scrape)
-> compare to previous run -> repeat
```

**Never report a single accuracy number.** The base rate is deliberately
distorted by injecting positives.

**Goldset composition as provided.** 46 domains: 19 confirmed yes, 14 confirmed
no, 13 hard negatives. The thin-sites group is empty, so the `unknown` label has
almost no test coverage.

**Thin sites are harvested from the crawl, not sourced by hand.** They are a
byproduct of Phase 4, so the evaluation gate splits in two:

- **Gate 7a: after Phase 4.** Sort the manifests by page count and word count
  ascending. The bottom of that list is the thin-site population. Spot-check the
  stored markdown for the lowest 15 and separate two cases that look identical
  in the numbers: genuinely thin sites (real text, just not much of it) go into
  the goldset labeled `unknown`; empty or nav-only pages are discovery failures
  and belong in the Phase 3 bug list, not the goldset. Target 8 to 10 additions.
- **Gate 7b: the real gate.** Precision on hard negatives holds, and `unknown`
  now has coverage.

If Phase 7 needs to start before the crawl finishes, a stopgap: take 8 domains
already in the goldset, restrict extraction to the homepage only, and label
those `unknown`. This tests whether the classifier abstains under thin evidence.
It is synthetic and does not test real brochureware phrasing, so replace it with
harvested thin sites once available.

**Holdout.** Reserve 30 fresh random domains from the 500, untouched during
tuning. Run once at the end. If precision holds there it is real. If it drops
sharply, the prompt was fitted to the goldset.

**Done when** precision on hard negatives is acceptable and the holdout
confirms it.

---

# CLI

```
crawl     --domains file.csv [--force]
extract   --crawl-date YYYY-MM-DD
classify  --pack medical_v1
evaluate  --goldset
compare   --run-a X --run-b Y
```

Each stage reads from the store and writes back to it. No stage calls the next
one implicitly.

---

# Build order

| Phase | Gate before proceeding |
|---|---|
| 1 Scaffold | JSONL round-trips through both store backends |
| 2 Pack | Pack loads and validates |
| 3 Discovery | Sensible page mix on 20 samples |
| 4 Fetch | Goldset crawled and stored once |
| 5 Extraction | Verbatim quote present on every claim |
| 6 Classification | Runs clean over the goldset |
| 7a Thin sites | 8 to 10 harvested from crawl manifests, labeled `unknown` |
| **7b Evaluation** | **Precision on hard negatives holds. Hard gate.** |

---

## The two rules that matter most

**Scrape once, classify many times.** Everything above the raw store runs once.
Everything below it runs fifty times. Keeping them separated turns a prompt
revision from a four-hour re-crawl into a three-minute re-run.

**No goldset, no trust.** A fluent classifier and a correct one look identical
without labeled data. Phase 7 is not optional.
