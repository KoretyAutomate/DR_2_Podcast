# Research-basis defects found during script audit — 2026-07-19
### Updated 2026-07-20 with B0, the root cause — read that first

Found while auditing `research_outputs/2026-05-05_05-58-00/` (sleep-habits episode).
These are defects in the **research basis**, not the script. They cannot be fixed by
editing dialogue, and they will corrupt any future episode that draws on this SOT or
on a SOT built the same way. Each was verified directly against the files cited.

Severity note: B0, B1 and B3 are the dangerous ones — they don't look wrong. A careful
reviewer reading the basis in good faith will be confidently misled. B3 misled the
2026-07-19 audit in real time; B0 invalidated that audit's central assumption.

---

## B0 — The raw records themselves are mispaired: title ≠ content (CRITICAL, root cause)

**Added 2026-07-20** after an independent memory-free expert review traced all 40 records
in `research_sources.json`.

Roughly **12 of 40 records carry a title that does not describe their own extracted
content.** The full-text fetcher appears to retrieve the wrong document while keeping the
search-result title. Verified directly:

| PMID | Title claims | Content actually is |
|---|---|---|
| 26899133 | "Coffee, caffeine, and sleep: A systematic review" | Antihistamines & sleep, cross-sectional, n=385, Palestine |
| 32475359 | "Glucose control / morning caffeinated coffee" | NASA dynamic-lighting RCT, n=16 |
| 19588402 | "Social norms interventions for student alcohol misuse" | Sulforaphane, macrophages, wax-worm larvae (bench study) |
| 10586387 | "Caffeine withdrawal, blinded pilot" | HUNT headache cross-sectional, n=50,483 |
| 35108682 | "[Sleep and Body Temperature]" | Seasonal solar terms & PSQI, n=25,428 |
| 39631226 | "Alcohol & sleep: SR & meta-analysis" | "Watershed Framework" mental-health theory paper |
| 38228408 | "Energy drinks: systematic review" | Cross-sectional descriptive, 1,250 Spanish adolescents |
| 26458258 | "2-Year RCT on moderate alcohol intake" | Narrative review |
| 39097006 | "Alcohol warning labels, Chile" | Texas/Florida pregnancy warning diff-in-diff |
| 35422151 | "Smartphone addiction meta-analysis" | Single cross-sectional survey, n=823, China |

**Why this is the root cause.** It explains B1 and B2 as symptoms: the "study
characteristics" and the Abstract's headline numbers were extracted from whichever
document was actually fetched, then filed under the title that was searched for. Nothing
downstream can be right if this is wrong.

**Concrete consequence for this episode.** The caffeine evidence base is effectively
**empty**. Both records that look like caffeine-and-sleep systematic reviews are other
papers. The only genuine caffeine-and-sleep record (PMID 40362813) is a **null result**:
no significant difference in objective sleep duration (p=0.183), no architecture change,
self-assignment rather than randomization, and it dosed 240mg only 3h before bed. Any
episode ranking caffeine as strongly-evidenced from this basis is unsupported.

**This invalidates the verification rule stated in the 2026-07-19 version of this
document.** That version said to verify numbers against `research_sources.json` as "the
most reliable layer." That advice is withdrawn — the raw layer is corrupted too, just
differently from the derived layers.

**The rule that actually works:** check **title-vs-content agreement within a single
record.** Read the summary bullets and ask whether they describe the paper the title
names. If they don't, the record is unusable no matter which file it lives in. Then
cross-check design/N/population against the SOT bibliography, which has been correct
everywhere it was checked.

---

## B1 — Fabricated study characteristics in `source_of_truth.md` (HIGH)

**PMID 35108682** is Ishihara et al., *"[Sleep and Body Temperature]"*, *Brain and Nerve*
(2022) — a Japanese-language **review article**. `source_of_truth.md:693` cites it
correctly as such.

But `source_of_truth.md:323` presents the same PMID as:

> retrospective observational study | N=25428 | 25,428 chronic insomnia patients; 76.7% … | 6 years (2018–2023) with biweekly assessments

A review article is not a 25,428-patient six-year retrospective cohort. The design, N,
population, and follow-up window are all invented. The row even contradicts itself: its
own citation-count (2) and FWCI (0.1) columns are consistent with a minor review, not a
25k-patient study.

This matters beyond bookkeeping: **this is the SOT's only large-scale temperature
evidence.** Remove it and bedroom-temperature guidance rests on a single N=10 trial
(PMID 33863439) that found no significant sleep-quality effect.

`accuracy_audit.md` case 2 already flagged the script for citing this study by name.
The remediation pass responded by deleting the *name* and keeping the *claim* — which
made the problem harder to audit rather than fixing it.

## B2 — Wrong study backing the SOT Abstract's headline number (HIGH)

Ref 3 (**PMID 32475359**) is a coffee/glucose nutrition trial. The SOT body uses it to
support circadian **light** physiology (a 45-day NASA dynamic-lighting protocol,
acrophase RR 3.46). The same PMID supplies the Abstract's headline **NNT = 1.7**.

That number has no valid anchor. It was not spoken in this episode, but it is the most
quotable figure in the document and sits in the Abstract, where a future episode would
reach for it first.

## B3 — `grade_synthesis.md` §8 table: columns decoupled from PMIDs (HIGH)

In the §8 synthesis table (lines 86–98), the design / N / effect columns do not
correspond to the PMID column. Verified example:

| Source | Says |
|---|---|
| `grade_synthesis.md:90` | `21323679 \| RCT \| 10 \| Cooling Device \| … \| Low (No Effect)` |
| `source_of_truth.md:309` | PMID 21323679 = *"Sleep following alcohol intoxication…"*, crossover RCT, **N=93** |

So the table's only quantitative cooling-device result is actually an alcohol study.
The real cooling trial is PMID **33863439** (Embr Wave, N=10), which appears in
`research_sources.json` but not in that row.

**This defect actively misled the 2026-07-19 audit.** A correction was written into the
episode citing "the N=10 cooling trial with no significant effect" on the strength of
row 90. The claim happened to survive — the real Embr study independently is N=10 with
a null sleep-quality result — but it was right by coincidence, not by sourcing. Anyone
reading this table is being handed wrong attributions that look authoritative.

## B4 — `Study N` is not a stable identifier (MED)

The same label denotes different papers in different sections. Verified:

- `source_of_truth.md:635` — Study 19 = "Caffeine and Headache Association Study"
- `source_of_truth.md:323` — Study 19 = Ishihara, *[Sleep and Body Temperature]*

At least five labels (2, 13, 15, 17, 18, 19) collide between the affirmative and
falsification sections, and the GRADE section merges both namespaces without
disambiguation. Any cross-reference by "Study N" is therefore unreliable.

## B5 — Degenerate NNT row narrated as a finding (MED)

`grade_synthesis.md:41` — PMID 40748681: `CER=0.000, EER=1.000, ARR=-1.0000,
RRR="+0.00%", NNT=1.0, "Severe Harm: 100% failure rate"`.

The RRR is a division by zero printed silently as `+0.00%`; the whole row is a
degenerate artifact of the deterministic math step, not a result.

`grade_synthesis.md:47` then narrates it in prose as evidence that **commercial sleep
aids** may be "actively detrimental." PMID 40748681 is Thivierge — an alcohol /
cardiovascular crossover RCT. It is not a commercial aid, and its outcome is not a
sleep measure.

Suggested fix: have `clinical_math.py` emit `undefined` rather than a formatted number
when CER or EER is 0 or 1, and exclude degenerate rows from the narrative synthesis.

## B6 — GRADE rating assigned to an empty evidence set (MED)

`grade_synthesis.md:26` assigns blue-light-blocking glasses **LOW to MODERATE**,
downgraded for "inconsistency." The basis contains **zero** trials of blue-light glasses.
There are no results to be inconsistent with.

The correct output is "no evidence identified," which is a materially different claim
from "weak evidence of an effect" — the latter implies trials exist and disagree.

---

## Suggested pipeline actions

0. **Fix the fetcher mispairing first (B0).** Everything else is downstream of it.
   Likely site: `research/fulltext_fetcher.py` — the title from the search result is being
   attached to a document retrieved under a different identifier. A cheap guard: after
   fetch, ask the Fast model whether the extracted content plausibly belongs to the stated
   title, and drop or re-fetch the record on mismatch. ~12/40 records failing this is a
   30% corruption rate, high enough that it should have a hard gate rather than a warning.

1. **Validate study characteristics against the fetched record.** B1 is a case where the
   extraction invented design/N/population for a paper whose real metadata was available.
   A publication-type check (review vs trial vs cohort) would catch it.
2. **Key the GRADE synthesis table to PMIDs, not to positional rows.** B3 is a join bug;
   it should be structurally impossible rather than caught by review.
3. **Guard the deterministic math** against CER/EER of 0 or 1 (B5).
4. **Make "no evidence identified" a first-class GRADE outcome** distinct from LOW (B6).
5. **Namespace study labels per track** (`A17` / `F17`) or drop the label scheme (B4).

## Scope not yet checked

Only the `2026-05-05_05-58-00` basis was examined. The sibling sleep-week runs were not.
If these are systemic to the SOT builder rather than specific to this run, every episode
in the week carries them.
