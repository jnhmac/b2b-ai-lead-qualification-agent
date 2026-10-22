# Design decisions

The decisions that took the longest to reach, including the ones that were wrong
first. Most of these are comments in the private codebase; they are collected here
because the reasoning is the transferable part.

## Verification is a layer, not a prompt instruction

Every claim the extractor makes carries a verbatim quote. Pydantic enforces that
a `Finding` cannot be constructed without one, and Pydantic has no idea whether
that quote is on the page.

So after parsing, both sides are canonicalised (markdown syntax and list markers
removed, image alt text kept and its URL dropped, typography folded, whitespace
collapsed, case folded) and the quote must appear in the text the model was
actually shown. Findings that fail are dropped along with their claim.

Measured on the goldset, this catches elision (a sentence quoted with three
words removed from the middle), stitching across unrelated lines, pronoun edits,
and certifications supplied from sector knowledge rather than from the page.

**The run report records the dropped text, not just the count.** A count tells
you the drop rate changed. Only the text tells you which way to fix it: a quote
plainly on the page means the canonical form is too strict, and one that reads
like a summary means the prompt is not being obeyed.

## Two failures that look identical and are not

Fewer than three discovered pages sets `discovery_failed`. That is a bug report
against `discover.py`, and it must never reach the classifier as `unknown`.

A thin site and a broken crawler produce the same empty result, and they demand
opposite responses. The thin-site population is the only test coverage the
`unknown` label has, so a discovery bug mixed into it corrupts the instrument.
`failure_reason` names which of four causes fired: unreachable homepage, bot
wall answering 200, client-rendered nav, or links that all failed the URL
filters.

The same distinction runs through the crawl stage. **A domain that stored
nothing writes no manifest at all**, because a manifest asserts that a domain
was crawled and every later phase reads it that way. Recording an account rate
limit as a fact about a company is the one failure this stage cannot leave open:
the thin-site harvest would collect it, and the `unknown` population would be
built on it.

## The bug I reported as a fact about the web, twice

A sweep found five discovery failures out of 46 domains, and I reported an 11%
rate of sites blocking us. Twice. Before reading a single attempt record.

When I finally read them, three distinct bugs fell out:

| Bug | Effect | Fix |
|---|---|---|
| The `www.` host was never tried | One domain answers 404 on the bare host, another fails cert verification there, a third 403s. All three serve fine on `www` | Try both, for the homepage and the sitemap seeds |
| The sitemap parser refused any file declaring a DOCTYPE | Two large sites serve valid sitemaps behind one, and both were discarded | Strip the DOCTYPE and parse. Entities can only be *defined* there, so an XML bomb now dies as an undefined entity: same protection, no collateral |
| `dedupe_key` folds `www.` away | Introduced by the first fix: the `www` sitemap seeds collapsed onto the bare-domain key and were never fetched | Dedupe the sitemap queue on the URL itself. Right identity for a page, wrong one for a fetch target |

Five failures became two, one of which is intermittent. The real rate is 2 to
4%, not 11%, and the remedy was free rather than roughly 55 credits per 500
domains.

The rule that came out of it is in the source: do not treat a single
`discovery_failed` as settled, re-run before triaging one, and read the attempt
records before blaming the site.

## Weight is reliability, not direction

Direction comes from which pack list a term sits in. A `strong` negative product
pushes toward `no_cleanroom_needed` exactly as hard as a `strong` positive one
pushes toward `in_house`. The validator enforces that a surface form lives in
exactly one list, because a term in two lists has no defined direction.

## Trap terms are neutral, not negative

ISO 13485, FDA registered and ISO 9001 are carried by nearly every medical
device firm, including wheelchair manufacturers. The classifier must see them
and discount them, never subtract points for them. `ignore` is the only weight
reserved to a single list, and the validator rejects it anywhere but
`trap_terms`.

The related tuning failure is worth stating: as bare literals, the numeric ISO
and class rows matched *inside* "ISO 13485" and "class 100000", awarding direct
cleanroom weight to almost every site. They are anchored regex now.

## `none` and `unknown` are different answers

`no_cleanroom_needed` is a positive determination from the product taxonomy.
`unknown` is absence of evidence. Collapsing them destroys the re-run
population, which is the set of domains worth revisiting when the site changes
or the prompt improves.

## The extractor is not allowed to decide anything

Three deliberate omissions:

- **It never sees the product taxonomy.** `positive_products` and
  `negative_products` are held out of the prompt, because a model shown a list of
  "products that indicate a cleanroom" starts filtering to that list.
  `products_named` stays uninterpreted, so the taxonomy can change without
  re-extracting a single page.
- **It reports, it does not judge.** Whether the company operates a cleanroom is
  the classifier's call, made from the quotes.
- **It is never asked for what the manifest already knows.** Domain, URL, page
  type, crawl date, pack version and word count are assembled locally. Cheaper in
  output tokens, and a URL that is never generated cannot be hallucinated.

## A long system prompt is the cheap one

Haiku 4.5 caches nothing below 4,096 tokens. No error, no warning, just
`cache_read_tokens` stuck at zero on the stage that is 90% of all calls.

Above that line the whole prefix bills at a tenth of the input rate, so the
detailed prompt is *cheaper per page* than a terse one that never caches.
`llm.py` warns once per client if the prefix ever drops back under the
threshold.

## Per-type caps, not scoring, stop a catalogue winning

A device catalogue can publish four hundred URLs under `/products/`. The pack's
per-type `max_pages` values sum to 14 against a total cap of 10, deliberately, so
the scorer fills slots from whichever page types a site actually has rather than
from whichever type it has most of.

## Editorial content never takes a typed slot

Three rules, because sites hide articles in three places:

- A section index (`/blog`, `/news`, `/resources/downloads`) is dropped
  outright. It is navigation plus titles and dates, with no prose to quote.
- Posts underneath it are demoted to the unmatched fill, not excluded. "We
  opened a Class 7 cleanroom" is exactly the evidence this pipeline looks for,
  and a press release is where a company says it.
- Articles published at the site root with no `/blog/` prefix are caught
  structurally: a final slug over five words, or a four-digit year anywhere in
  the path. A company page is named (`/capabilities`, `/about-us`); an article is
  a headline in a URL.

The third rule removes *wrong* type assignments, not merely weak ones. A URL
reading `/plant-cell-model-magnified-500k-to-100m-times.html` matched the
`plant` pattern and was filed as a facilities page, reporting a facility on a
site that has none. Measured over the sample, the split is clean at five words.

## English wins among locale twins

The target market is the US, so when a manufacturer publishes one page in a
dozen languages, the English copy takes the slot. This is a scoring penalty
rather than a filter: a site published only in German is still crawled,
classified and flagged for review, because dropping it would delete a real
supplier from the population.

## Nothing is filtered out at normalize time

Enterprise multi-division companies and non-English sites normalize like
everything else and get flagged at classification. Parent and subsidiary hosts
that collapse onto one registrable domain are logged at warning level rather
than merged silently.

Discovery's URL filters are a different thing entirely: they drop PDFs, shopping
carts and cookie policies, which are not company content in any language.

## Credits are tallied per domain, not per process

The batch runs two domains concurrently against a plan whose concurrency limit
is two, and a single process-level counter would attribute one domain's spend to
whichever one finished last. Stealth-proxy pages bill five credits instead of
one; they are rare on these sites, which is exactly why every one is logged and
counted in the manifest.

## Two pages first, all ten only if unsure

The first version fetched ten pages for every company and extracted each one.
The current crawl buys two, the homepage and the best product page, decides
once, and either stops or buys the other eight. There is no check after every
page; one checkpoint was enough. A company costs 2 or 10 credits, and the
page counts per batch are bimodal exactly as that predicts.

The early-exit rate is set by the list, not the code. The first batch of the
trade-show list, drawn from companies whose homepage already named a
cleanroom, exited early about half the time at under six credits per company.
The harder batches that followed exited early 11 to 21% of the time at eight
to nine credits. Credits follow roughly `10 - 8 x early-exit rate`.

## A refused guard is not a "no"

Six checks stand between a quote and a proven qualified lead: not an image
filename, describes a controlled environment, verbatim on the page, the
company as the subject, not seller wording, and attributed to the company
itself by a second model. A quote that passes is then graded: thin wording,
with no class or room named, makes a likely qualified lead rather than proof.
A refusal cancels the early exit and sends
the company to the full read. It never settles the row, because the cost of a
refused guard should be eight credits, not a lost lead.

The attribution guard is the only one that sees every stored page. In the
first version it confirmed exit B but not exit A, so proven rows shipped
unconfirmed for four batches before the gate was wired into the path the
crawl actually takes.

## The auditor was blinder than the pipeline

The first audit after the crawl gave a second model the quote, the URL and
the domain, and nothing else. The median quote was 34 characters. "Class 7
Clean Rooms" as a bare bullet reads as ambiguous; on the page it sits under
"when you partner with us, you benefit from". Seven of eight ambiguous rows
read by hand were first-party. Independence is worth less than context, which
is why the cleanroom expert reads the company's whole site rather than the
quote.

## Reason per company, never by category

Every attempt to write a verdict rule as a category, such as "molders are
qualified leads" or "polymer producers are not", failed on the next batch.
Rulings are written as the reason a company does or does not buy the
consumables, and when in doubt between "likely" and a rejection, the lead is
kept.

A one-shot stopping rule applies to wording work. A prompt written for
injection molders scored 31/40 on an unseen sample against 32/40 for the
original crawl, and it was dropped rather than iterated.

## Outsourced is not a qualified lead; the contract manufacturer is

A company whose cleanroom belongs to a partner buys no consumables. The
partner does. The expert records the contract manufacturer so it can become
the qualified lead.

## Agents propose, the owner adopts

No agent edits a verdict, a rule or the engine, and none can spend. The run
operator stops at a spend gate and writes a sheet of at most 20 decisions.
This is slower than applying the expert's answers automatically, and it is why
the rule book can be trusted: every ruling in it was made by a person.

## Agreements stop at a script

The expert re-judges every company blind, but no person reads its answers
row by row. A script compares them with the current hand-over, and only the
rows where the two differ go on: to the diagnostician, which names the cause
and the smallest fix, and then to the owner's sheet, which also skips rows the
owner has already ruled. The owner's time is the scarce input, so it is spent
only where two independent reads disagree. On a 98-company test batch the
sheet went from 29 items to 3.

## Locations come from pages already bought

The target is US companies, so sales needs to know where each qualified lead
has a site. The answer was mostly on disk already. A script scans the stored
pages at the pinned crawl date for postal addresses, with no model and no
credits, and writes every site per company into one splittable cell plus a
long file with one row per location. A foreign company with a US plant counts
as a US lead. Location ranks leads and never decides a verdict. On a checked
batch it agreed with the expert on US against non-US 92% of the time; a third
of companies stayed unknown because their pages carry no address, and those
are left unknown rather than guessed.

## Gate on regression, not on totals

At this sample size a percentage moves on one row. A fix ships only if it
loses no confirmed row. Six fixes raised agreement on a test batch from 74 to
79 of 98 and lost 9 confirmed rows; four of the six were held back.

## Measure cost from the invoice, not the rate card

Three compounding cost bugs survived until the provider's invoice was
compared with the rate card: one reconstructed inference figure ran 3.78
times low. Every batch now snapshots provider usage before and after the
crawl, and billing that posts late is carried as its own row.

## Known omissions

- **Discovery does not enforce `robots.txt` Disallow.** It reads `robots.txt`
  only for its `Sitemap:` directive; the fetch provider applies its own policy.
- **The prompts and product lists are written for medical device
  manufacturers.** A second market needs its own pack and prompt wording; the
  knowledge file is written to carry over.
- **Sampling is pinned, but not proven honoured.** Temperature 0 and a seed
  are set. The reasoning models may not respect them, and an earlier 22%
  run-to-run churn predates the pin.
- **Page selection can squeeze out the facilities page.** Higher-priority page
  types can fill nine of ten slots on a deep catalogue. Pages that name a
  cleanroom now get priority, which covers the common case but not all of it.
