# The agent team

Three agents on top of a scripted pipeline. Scripts do the bookkeeping, agents
do the judgment, and the project owner adopts or rejects every change.

## Why agents

After the first production list, accuracy was no longer limited by how well
the crawler read a page. It was limited by three things a pipeline does badly:

- **Applying a growing rule book.** The owner's rulings piled up faster than
  prompts could absorb them: an outsourced cleanroom is not a qualified lead,
  but the contract manufacturer that runs it is; a stated ISO class is a
  cleanroom whatever the page calls the room.
- **Explaining its own mistakes.** A wrong verdict could come from page
  selection, vocabulary, the judge prompt, attribution, the quote check, the
  audit, a missing rule or bad data. Finding which took an afternoon per row.
- **The owner's time.** Every batch produced a long disagreement list, most of
  it already ruled on.

## The three agents

| Agent | Model | Can | Cannot |
|---|---|---|---|
| **Cleanroom expert** | `claude-opus-5-5`, medium effort | Read the company's stored pages and live site, the rulings and the knowledge file | See the crawler's verdict or any audit, edit, spend |
| **Diagnostician** | `claude-sonnet-5-5`, medium effort | Read the crawl records pinned to their date, the stored pages, the run files, the code and the sector pack | Edit, re-run, spend |
| **Run operator** | The main Claude Code session, as a skill | Run the free parts of the pipeline, launch the other two agents, write the approval sheet | Spend without the owner's yes, change a verdict, a rule or the engine |

The run operator is a skill rather than a subagent because a subagent cannot
launch other agents.

The agents run on a different model family from the crawl, which runs on
OpenAI models. When the expert agrees with the crawler, that is a second
opinion, not the crawler confirming itself.

### Cleanroom expert

It asks one question per company: does this company operate a cleanroom, so
that it buys the consumables? It reasons per company, never by category, and
when it is in doubt between "likely" and a rejection, it keeps the qualified
lead. Its answer is one line of JSON: the verdict, a verbatim quote and its
URL, the ruling it applied, a draft ruling if none fits, and the contract
manufacturer if the work is outsourced.

### Diagnostician

It names one cause per wrong verdict and the smallest fix, such as a product
term, a prompt sentence or a discovery pattern, plus which other rows the fix
would likely move. It may disagree with the expert or the owner rather than
invent a crawler fault.

### Run operator

It plans the batch for free, stops at the spend gate, crawls, runs the expert
in chunks of 12 three at a time, sends disagreements to the diagnostician, and
writes one approval sheet. The sheet holds only what needs the owner:
disagreements between a qualified lead and a rejection, differences on proven
rows with their cause and fix, cases no ruling covers, and integrity failures.
It has at most 20 items, skips rows the owner has already ruled on, and groups
the rest by cause, one yes or no each.

## How they were tested

| Test | Result |
|---|---|
| Expert, blind, against the owner's labels | 36/37, then 35/37 on a relabelled key |
| Expert on Sonnet instead of Opus | Rejected 7 of the owner's qualified leads (0 on Opus) and skipped required reading; stays on Opus |
| Diagnostician, 10 past rows with known causes | 10/10 on Opus and on Sonnet; moved to Sonnet |
| Full batch replay, 98 companies, owner-ruled answer key | Approval sheet cut from 29 items to 3 |
| The diagnostician's three ladder fixes, replayed on 67 stored companies | 25 moved, about $1 of inference, no crawl credits; shipped |
| A later set of six fixes | 79/98 agreement with the owner against 74, but 9 confirmed rows lost; four of the six held back |

The last row is the rule working: a fix that makes any confirmed row worse
does not ship, however good the total looks.

**These numbers are not an independent accuracy measure.** The owner labels
the test sets, and the owner's rulings are the expert's rule book: every
ruling goes into the knowledge file the expert reads each session. A high
match shows that the expert applies the owner's rules consistently, not that
the rules are right. The diagnostician's 10/10 is against causes the project
had already found. No current test set is labelled independently of the
owner; the only one, in [EVALUATION.md](EVALUATION.md), measured the first
version of the crawler.

## How it learns

Every row the owner confirms joins the known-answer set. Every ruling is
logged, and its reasoning goes into a knowledge file written independently of
any one market, so a semiconductor list starts from what is already known.
When the expert matches the owner over enough batches, a class of decision can
move from "owner's yes" to "applied, owner informed", with the owner's
agreement.

## Rules that hold throughout

- No paid run without the owner's yes.
- Agents propose; only the owner adopts. Nothing is applied automatically.
- A verdict needs a verbatim sentence from the company's own pages. The word
  "cleanroom" alone never decides, and an image filename is never evidence.
- When the work is outsourced, the contract manufacturer is the qualified
  lead, not the company that uses it.
