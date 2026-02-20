# PLAN: Evidence-Based Research Pipeline Revamp

## Overview

Replace the current iterative gap-filling search loop in `deep_research_agent.py` with a structured 8-step clinical research pipeline modeled on systematic review methodology. The new pipeline uses PICO framework, MeSH-powered Boolean searches, a wide-net-then-screen approach (500 → 20), full-text PDF extraction, deterministic math (ARR/NNT), and GRADE synthesis.

**Goals:** Higher evidence quality, reproducible methodology, faster execution (one exhaustive search pass instead of iterative loops), and elimination of LLM math hallucinations.

---

## Architecture: Current vs. New

### Current Flow (deep_research_agent.py)
```
Lead Researcher ─┐
  (3 iterations: │  parallel  ─→ Auditor Synthesis
   plan→search→  │
   fetch→summarize→evaluate)
Counter Researcher┘
```

### New Flow
```
┌─ AFFIRMATIVE TRACK ────────────────────────────────────────┐
│ Step 1: PICO/MeSH/Boolean (Smart)                          │
│ Step 2: Wide Net — 500 results (Fast)                      │
│ Step 3: Screen → top 20 (Smart)                            │
│ Step 4: Deep Extraction — full text PDFs (Fast)            │
│ Step 5: Affirmative Case (Smart)                           │
├────────────────────────────────────────────────────────────┤
│              ↕ both tracks run in parallel                  │
├─ FALSIFICATION TRACK ─────────────────────────────────────┤
│ Step 1': PICO/MeSH/Boolean — adversarial framing (Smart)   │
│ Step 2': Wide Net — 500 results (Fast)                     │
│ Step 3': Screen → top 20 (Smart)                           │
│ Step 4': Deep Extraction — full text PDFs (Fast)           │
│ Step 6: Falsification Case (Smart)                         │
└────────────────────────────────────────────────────────────┘
                           ↓
              Step 7: Deterministic Math (Python)
              ARR = CER − EER,  NNT = 1 / ARR
                           ↓
              Step 8: Final GRADE Synthesis (Smart)
```

---

## Files to Create / Modify

| File | Action | Purpose |
|------|--------|---------|
| `deep_research_agent.py` | **Major rewrite** | Replace iterative loop with 8-step pipeline |
| `clinical_math.py` | **New file** | Deterministic ARR/NNT calculator (Python, no LLM) |
| `fulltext_fetcher.py` | **New file** | PMC/EuropePMC/Unpaywall full-text PDF downloader |
| `search_agent.py` | Minor edits | Add Cochrane engine to SearXNG config |
| `podcast_crew.py` | Minor edits | Update integration points (same return type) |
| `podcast_web_ui.py` | Minor edits | Update progress phase names |

The `Orchestrator.run()` return signature (`Dict[str, ResearchReport]`) stays identical, so `podcast_crew.py` and `podcast_web_ui.py` require only cosmetic updates (phase label names).

---

## Step-by-Step Implementation

### Step 1: Search Strategy Formulation (Smart Model)

**What:** The Lead Researcher translates the user's topic into a structured PICO framework, generates MeSH terms, and writes Boolean search strings for PubMed and Cochrane.

**Implementation:**

New method `ResearchAgent._formulate_search_strategy()`:

```python
async def _formulate_search_strategy(self, topic: str, role_instructions: str) -> SearchStrategy:
```

**Smart model prompt:**
```
You are a medical librarian and systematic review specialist.

Given the research topic below, produce a search strategy:

1. PICO FRAMEWORK:
   - P (Population): [target population]
   - I (Intervention): [intervention/exposure]
   - C (Comparison): [control/comparator]
   - O (Outcome): [primary outcome measures]

2. MeSH TERMS (for PubMed):
   - Population MeSH: [term1[MeSH], term2[MeSH], ...]
   - Intervention MeSH: [term1[MeSH], term2[MeSH], ...]
   - Outcome MeSH: [term1[MeSH], term2[MeSH], ...]

3. BOOLEAN SEARCH STRINGS:
   - PubMed query: (Population MeSH OR synonyms) AND (Intervention MeSH OR synonyms) AND (Outcome terms)
   - Filters: Humans[MeSH], English[la], ("2010"[dp] : "3000"[dp])
   - Study type filters: Randomized Controlled Trial[pt] OR Meta-Analysis[pt] OR Systematic Review[pt] OR Clinical Trial[pt]
   - Cochrane query: [adapted Boolean for Cochrane CENTRAL]
   - General scholar query: [plain-language version for Google Scholar]

Return as JSON:
{
  "pico": {"population": "...", "intervention": "...", "comparison": "...", "outcome": "..."},
  "mesh_terms": {"population": [...], "intervention": [...], "outcome": [...]},
  "search_strings": {
    "pubmed_primary": "...",       // strict Boolean with MeSH + filters
    "pubmed_broad": "...",         // relaxed Boolean (fewer filters, more recall)
    "cochrane": "...",             // adapted for Cochrane CENTRAL
    "scholar": "..."              // plain-language for Google Scholar
  }
}
```

**Adversarial variant (Step 1'):** Same PICO but the smart model is prompted to add adversarial terms:
- Add `"adverse effects"[Subheading]`, `"toxicity"[MeSH]`, `"drug-related side effects"[MeSH]`
- Add `"no significant difference"`, `"lack of efficacy"`, `"negative results"`
- Search for retraction notices: `"Retracted Publication"[pt]`
- Add funding bias terms: `"conflict of interest"`, `"industry-funded"`

**New dataclass:**
```python
@dataclass
class SearchStrategy:
    pico: Dict[str, str]                    # P, I, C, O
    mesh_terms: Dict[str, List[str]]        # population/intervention/outcome MeSH
    search_strings: Dict[str, str]          # pubmed_primary, pubmed_broad, cochrane, scholar
    role: str                               # "affirmative" or "adversarial"
```

**Estimated time:** ~15s (one smart model call)

---

### Step 2: The Wide Net (Fast Model)

**What:** Use the Boolean search strings to query PubMed, Cochrane CENTRAL (via PubMed subset), and Google Scholar. Collect up to 500 results. The fast model summarizes each result into a lightweight JSON record from the title + abstract (no full-text fetching yet).

**Implementation:**

#### 2a. Enhanced PubMed Client

Modify `PubMedClient.search()` to support:
- Boolean queries with MeSH terms (already supported by E-utilities `term` param)
- `retmax=500` (increase from current 10)
- `sort=relevance` parameter
- Extract additional fields from XML: `PublicationType`, `MeshHeadingList`, `AbstractText` (structured with labels like BACKGROUND, METHODS, RESULTS, CONCLUSIONS)
- Return `PubMedArticle` dataclass instead of raw dict

**New dataclass for wide-net results:**
```python
@dataclass
class WideNetRecord:
    """Lightweight screening record — no full text, just title + abstract metadata."""
    pmid: Optional[str]
    doi: Optional[str]
    title: str
    abstract: str                          # full abstract text
    study_type: str                        # from PublicationType XML element
    sample_size: Optional[str]             # extracted by fast model from abstract
    primary_objective: str                 # extracted by fast model
    year: Optional[int]
    journal: Optional[str]
    authors: Optional[str]
    url: str                              # PubMed URL or DOI URL
    source_db: str                        # "pubmed", "cochrane_central", "scholar"
    relevance_score: Optional[float]      # search engine relevance score if available
```

#### 2b. Cochrane CENTRAL via PubMed

Cochrane CENTRAL trials are indexed in PubMed. Add a filter to the PubMed query:
```
{boolean_query} AND "Cochrane Database Syst Rev"[Journal]
```
Or use the subset filter: `cochrane[sb]` for Cochrane reviews.

This avoids needing a separate Cochrane API (which requires Wiley subscription).

#### 2c. Google Scholar via SearXNG

Use existing `SearxngClient` with `engines=['google scholar']` and the plain-language query from the search strategy. Scrape additional metadata from snippet text.

#### 2d. Fast Model Abstract Screening

For each of the 500 results, the fast model extracts a lightweight JSON from the abstract:
```python
async def _screen_abstract(self, record: dict) -> WideNetRecord:
    """Fast model extracts structured fields from title + abstract."""
```

**Prompt:**
```
Extract from this abstract:
- study_type: RCT | meta-analysis | systematic-review | cohort | case-control | cross-sectional | case-report | in-vitro | animal-model | review | guideline | other
- sample_size: "n=X" or null
- primary_objective: one sentence
Return JSON only.
```

**Optimization:** PubMed XML already contains `<PublicationType>` — use it directly for study_type without LLM. Only call fast model for sample_size and primary_objective extraction from abstract text. This reduces fast model calls by ~50%.

**Parallelism:** Process in batches of 50 with `asyncio.gather()` and semaphore (max 10 concurrent).

**Termination:** Stop at 500 results or when all databases exhausted, whichever comes first.

**Estimated time:** ~90s (PubMed API: 5s for 500 results, fast model screening: 500 calls × ~0.5s each with 10-way parallelism = ~25s, but the API call is 180ms each so more like 50-90s)

---

### Step 3: Prioritization / Screening (Smart Model)

**What:** The smart model scans the 500 lightweight JSON records and selects the top 20 most rigorous human clinical studies. Filters out weak methodologies (in vitro, small samples, animal models, case reports).

**Implementation:**

New method `ResearchAgent._screen_and_prioritize()`:

```python
async def _screen_and_prioritize(
    self, records: List[WideNetRecord], strategy: SearchStrategy, max_select: int = 20
) -> List[WideNetRecord]:
```

**Smart model prompt:**
```
You are a systematic review screener performing title/abstract screening.

INCLUSION CRITERIA:
- Human clinical studies (RCTs, meta-analyses, systematic reviews, large cohort studies)
- Sample size ≥ 30 participants (prefer ≥ 100)
- Published in peer-reviewed journals
- Directly relevant to the PICO: {pico}

EXCLUSION CRITERIA:
- Animal models / in vitro studies
- Case reports (n < 5)
- Conference abstracts without full data
- Non-English
- Retracted publications
- Duplicate reports of the same study

From the {N} studies below, select the TOP 20 most rigorous.
Rank by: meta-analyses first, then RCTs (by sample size), then large cohort studies.

Return a JSON array of the selected PMIDs/indices in ranked order:
[{"index": 0, "reason": "Meta-analysis of 45 RCTs, n=12,000"}, ...]
```

**Input format:** Compact JSON array of the 500 `WideNetRecord` objects (title, abstract snippet, study_type, sample_size, year). At ~200 chars per record, 500 records ≈ 100K chars ≈ 25K tokens. This fits within the 32K context window of Qwen2.5-32B. If it exceeds context, split into chunks of 250 and have the smart model select top 20 from each chunk, then merge.

**Chunking strategy for context overflow:**
- If total tokens > 28K: split into chunks of 200 records
- Smart model selects top 20 from each chunk
- Merge selections, re-rank if > 20 total, final cut to 20

**Estimated time:** ~20s (one or two smart model calls)

---

### Step 4: Deep Extraction (Fast Model)

**What:** Download full-text PDFs/HTML of the top 20 studies using PMIDs/DOIs. The fast model reads each full text and extracts heavy variables.

**Implementation:**

#### 4a. Full-Text Fetcher — New file: `fulltext_fetcher.py`

**Sources (tried in order per article):**
1. **PubMed Central OA API**: `https://www.ncbi.nlm.nih.gov/pmc/utils/oa.cgi?id=PMID` → returns XML/PDF link
2. **Europe PMC REST API**: `https://www.ebi.ac.uk/europepmc/webservices/rest/{PMID}/fullTextXML` → returns full-text XML for OA articles
3. **Unpaywall API**: `https://api.unpaywall.org/v2/{DOI}?email=your@email.com` → returns OA PDF URL
4. **Publisher page scrape**: Fall back to scraping the publisher's HTML page (existing `ContentFetcher` logic)

```python
class FullTextFetcher:
    """Fetch full-text content for studies identified by PMID/DOI."""

    async def fetch_fulltext(self, record: WideNetRecord) -> FullTextArticle:
        """Try PMC → Europe PMC → Unpaywall → publisher scrape."""

    async def fetch_all(self, records: List[WideNetRecord]) -> List[FullTextArticle]:
        """Parallel fetch with semaphore (max 5 concurrent for rate limits)."""
```

**New dataclass:**
```python
@dataclass
class FullTextArticle:
    pmid: Optional[str]
    doi: Optional[str]
    title: str
    full_text: str                 # full article text (stripped of HTML/XML tags)
    source: str                    # "pmc", "europepmc", "unpaywall", "scrape"
    word_count: int
    url: str
    error: Optional[str] = None
```

**Rate limiting:**
- PMC: 3 requests/second (with API key: 10/s)
- Europe PMC: 20 requests/second
- Unpaywall: 100K/day, 10/second
- Use `asyncio.Semaphore(5)` to stay well under limits

#### 4b. Deep Extraction Prompt

For each full-text article, the fast model extracts the heavy clinical variables:

```python
async def _deep_extract(self, article: FullTextArticle, pico: Dict) -> DeepExtraction:
```

**Fast model prompt:**
```
You are a clinical data extraction specialist. Read this full-text study and extract ALL of the following variables. Use "null" for any field not found in the text.

EXTRACTION TEMPLATE (return as JSON):
{
  "attrition_pct": "exact dropout/attrition percentage",
  "effect_size": "primary effect size with CI (e.g., 'HR 0.76, 95% CI 0.65-0.89')",
  "demographics": "age range, sex ratio, ethnicity, population description",
  "follow_up_period": "duration of follow-up (e.g., '5.2 years median')",
  "funding_source": "exact funding source and any declared conflicts",
  "conflicts_of_interest": "any declared COI or 'None declared'",
  "biological_mechanism": "described mechanism/pathway if mentioned",
  "control_event_rate": "CER - event rate in control group as decimal (e.g., 0.15)",
  "experimental_event_rate": "EER - event rate in experimental group as decimal (e.g., 0.10)",
  "primary_outcome": "exact primary endpoint as defined in methods",
  "secondary_outcomes": "list of secondary endpoints",
  "blinding": "single-blind, double-blind, open-label, or null",
  "randomization_method": "described randomization technique",
  "intention_to_treat": true/false/null,
  "sample_size_total": integer,
  "sample_size_intervention": integer or null,
  "sample_size_control": integer or null,
  "study_design": "parallel RCT | crossover RCT | meta-analysis | cohort | etc.",
  "risk_of_bias": "low | some concerns | high | unclear"
}
```

**New dataclass:**
```python
@dataclass
class DeepExtraction:
    pmid: Optional[str]
    doi: Optional[str]
    title: str
    url: str
    attrition_pct: Optional[str]
    effect_size: Optional[str]
    demographics: Optional[str]
    follow_up_period: Optional[str]
    funding_source: Optional[str]
    conflicts_of_interest: Optional[str]
    biological_mechanism: Optional[str]
    control_event_rate: Optional[float]      # CER — needed for Step 7
    experimental_event_rate: Optional[float]  # EER — needed for Step 7
    primary_outcome: Optional[str]
    secondary_outcomes: Optional[List[str]]
    blinding: Optional[str]
    randomization_method: Optional[str]
    intention_to_treat: Optional[bool]
    sample_size_total: Optional[int]
    sample_size_intervention: Optional[int]
    sample_size_control: Optional[int]
    study_design: Optional[str]
    risk_of_bias: Optional[str]
    raw_facts: str                           # free-text summary of key findings
```

**Parallelism:** 10-way parallel with semaphore (fast model can handle concurrent requests).

**Estimated time:** ~120s (20 articles × ~2KB fast model input each, 10-way parallel)

---

### Step 5: The Affirmative Case (Smart Model)

**What:** The Lead Researcher analyzes the deep-extraction data and drafts a comprehensive summary arguing FOR the hypothesis.

**Implementation:**

New method `ResearchAgent._build_case()`:

```python
async def _build_case(
    self, topic: str, strategy: SearchStrategy,
    extractions: List[DeepExtraction], case_type: str  # "affirmative" or "falsification"
) -> str:
```

**Smart model prompt (affirmative):**
```
You are a Lead Researcher writing the AFFIRMATIVE case for the following hypothesis.

PICO: {pico}

You have deeply extracted data from {N} clinical studies. Analyze this evidence and write a comprehensive argument FOR the hypothesis.

Structure:
1. Clinical Significance: How large are the observed effects? Are they clinically meaningful (not just statistically significant)?
2. Biological Plausibility: What mechanisms support the intervention's efficacy?
3. Consistency: Do multiple independent studies converge on the same finding?
4. Dose-Response: Is there evidence of a dose-response relationship?
5. Strength of Evidence: Rate the overall supporting evidence as STRONG / MODERATE / WEAK / INSUFFICIENT
6. Evidence Table:
   | Study | Design | N | Effect Size | CER | EER | Follow-up | Bias Risk |
7. Key Supporting Citations (Author et al. (Year) format)

Be precise. Cite specific numbers. Do not speculate beyond the data.
```

**Estimated time:** ~30s (one smart model call with ~20K token input)

---

### Step 6: The Falsification Case (Adversarial Researcher)

**What:** Runs Steps 1-5 in parallel with the affirmative track, but with adversarial framing. Same pipeline, different prompts.

**Differences from affirmative:**
- Step 1': PICO is the same population/intervention, but search terms emphasize adverse effects, null results, toxicity, funding bias
- Step 2': Same wide net but with adversarial Boolean queries
- Step 3': Screening prioritizes studies showing harm, null effects, or methodological concerns
- Step 4': Same deep extraction template
- Step 5' → Step 6: Falsification case prompt

**Smart model prompt (falsification):**
```
You are an Adversarial Researcher writing the FALSIFICATION case against the following hypothesis.

Your mandate: Find every reason this intervention may NOT work, may cause harm, or may be overstated.

Structure:
1. Adverse Effects: What harms have been documented?
2. Null Results: Which studies found NO significant effect?
3. Methodological Concerns: Poor blinding, high attrition, small samples, short follow-up
4. Funding Bias: Industry-funded studies vs. independent results
5. Publication Bias: Evidence of selective reporting or p-hacking
6. Biological Implausibility: Any mechanistic concerns?
7. Evidence Table (same format as affirmative)
8. Strength of Counter-Evidence: STRONG / MODERATE / WEAK / INSUFFICIENT
```

**Parallelism:** The entire affirmative track (Steps 1-5) and falsification track (Steps 1'-4', 6) run in parallel via `asyncio.gather()`, exactly like the current Lead + Counter researcher pattern.

**Estimated time:** Same as affirmative (~4 min), but runs in parallel so no added wall-clock time.

---

### Step 7: Deterministic Math (Python Script)

**What:** A hardcoded Python function (no LLM) calculates real-world clinical impact from the CER and EER values extracted in Step 4.

**Implementation — New file: `clinical_math.py`**

```python
"""
Deterministic clinical statistics calculator.
No LLM involvement — pure arithmetic to prevent hallucinated math.
"""

@dataclass
class ClinicalImpact:
    study_id: str                    # PMID or title
    cer: float                       # Control Event Rate
    eer: float                       # Experimental Event Rate
    arr: float                       # Absolute Risk Reduction = CER - EER
    rrr: float                       # Relative Risk Reduction = ARR / CER
    nnt: float                       # Number Needed to Treat = 1 / ARR
    nnt_interpretation: str          # "Treat 10 patients to prevent 1 event"
    direction: str                   # "benefit" | "harm" | "no_effect"

def calculate_impact(study_id: str, cer: float, eer: float) -> Optional[ClinicalImpact]:
    """
    Calculate ARR, RRR, NNT from CER and EER.

    ARR = CER - EER        (positive = benefit, negative = harm)
    RRR = ARR / CER        (relative measure)
    NNT = 1 / |ARR|        (patients needed to treat for one outcome)
    """
    if cer is None or eer is None:
        return None
    arr = cer - eer
    if abs(arr) < 1e-10:
        return ClinicalImpact(
            study_id=study_id, cer=cer, eer=eer,
            arr=0.0, rrr=0.0, nnt=float('inf'),
            nnt_interpretation="No measurable difference between groups",
            direction="no_effect"
        )
    rrr = arr / cer if cer != 0 else 0.0
    nnt = 1.0 / abs(arr)
    direction = "benefit" if arr > 0 else "harm"
    verb = "prevent" if direction == "benefit" else "cause"
    interp = f"Treat {nnt:.0f} patients to {verb} 1 additional event"
    return ClinicalImpact(
        study_id=study_id, cer=cer, eer=eer,
        arr=round(arr, 6), rrr=round(rrr, 4), nnt=round(nnt, 1),
        nnt_interpretation=interp, direction=direction
    )

def batch_calculate(extractions: List[DeepExtraction]) -> List[ClinicalImpact]:
    """Calculate clinical impact for all studies that have CER and EER."""
    results = []
    for ex in extractions:
        if ex.control_event_rate is not None and ex.experimental_event_rate is not None:
            impact = calculate_impact(
                study_id=ex.pmid or ex.title,
                cer=ex.control_event_rate,
                eer=ex.experimental_event_rate
            )
            if impact:
                results.append(impact)
    return results

def format_math_report(impacts: List[ClinicalImpact]) -> str:
    """Format a deterministic math report for the Auditor."""
    if not impacts:
        return "No studies provided both CER and EER. NNT calculation not possible.\n"
    lines = ["## Deterministic Clinical Impact Calculations\n",
             "| Study | CER | EER | ARR | RRR | NNT | Direction |",
             "|-------|-----|-----|-----|-----|-----|-----------|"]
    for i in impacts:
        lines.append(
            f"| {i.study_id} | {i.cer:.3f} | {i.eer:.3f} | "
            f"{i.arr:+.4f} | {i.rrr:+.2%} | {i.nnt:.1f} | {i.direction} |"
        )
    lines.append("")
    for i in impacts:
        lines.append(f"- **{i.study_id}**: {i.nnt_interpretation}")
    return "\n".join(lines)
```

**Key principle:** Zero LLM involvement. This is pure Python arithmetic. The only input is the CER/EER floats already extracted by the fast model in Step 4.

**Estimated time:** <1ms (pure computation)

---

### Step 8: Final GRADE Synthesis (The Auditor)

**What:** The Auditor reads both the affirmative case and the falsification case, reviews the Python-calculated NNT table, and issues a final GRADE-framework synthesis.

**GRADE (Grading of Recommendations, Assessment, Development, and Evaluations) framework:**
- **High quality:** Further research very unlikely to change confidence
- **Moderate quality:** Further research likely to change confidence
- **Low quality:** Further research very likely to change confidence
- **Very low quality:** Very uncertain about the estimate

**Smart model prompt:**
```
You are The Auditor — an independent scientific arbiter.

You have received:
1. The AFFIRMATIVE CASE (arguing FOR the intervention)
2. The FALSIFICATION CASE (arguing AGAINST the intervention)
3. DETERMINISTIC MATH (Python-calculated ARR, RRR, NNT — these numbers are EXACT, not LLM-generated)

Your task: Issue a GRADE-framework synthesis.

Structure:
1. Executive Summary (3-4 sentences)
2. Evidence Profile
   - Study designs: [list study types included]
   - Total participants across key studies: N = X
   - Risk of bias assessment: [summary]
   - Consistency: [do studies agree?]
   - Directness: [do studies directly measure the outcome of interest?]
   - Precision: [are confidence intervals narrow?]
   - Publication bias: [any evidence of selective reporting?]

3. GRADE Assessment
   Start at HIGH for RCTs, LOW for observational. Then apply modifiers:
   DOWNGRADE for:
   - Risk of bias (serious -1, very serious -2)
   - Inconsistency (-1 or -2)
   - Indirectness (-1 or -2)
   - Imprecision (-1 or -2)
   - Publication bias (-1)
   UPGRADE for:
   - Large effect (+1 or +2)
   - Dose-response (+1)
   - Plausible confounders would reduce effect (+1)

   FINAL GRADE: ⊕⊕⊕⊕ HIGH | ⊕⊕⊕○ MODERATE | ⊕⊕○○ LOW | ⊕○○○ VERY LOW

4. Clinical Impact (from deterministic math)
   - Include the NNT table directly (do NOT recalculate — use the exact numbers provided)
   - Interpret the NNT in clinical context

5. Balanced Verdict
   - What does the weight of evidence actually say?
   - What are the key caveats?
   - What would change the conclusion?

6. Recommendations for Further Research

7. PRISMA Flow Diagram (text-based)
   Records identified → Screened → Eligible → Included

8. Consolidated Evidence Table
   | Study | Design | N | Effect | CER | EER | ARR | NNT | Bias Risk | GRADE Impact |

9. Full Reference List

CRITICAL RULES:
- NEVER recalculate ARR or NNT — use the Python-provided numbers exactly
- Be heavily caveated — acknowledge uncertainty
- Flag any potential conflicts of interest
- Distinguish between statistical significance and clinical significance
- Note that absence of evidence is not evidence of absence
```

**Input to auditor:** Concatenation of:
1. Affirmative case report (from Step 5)
2. Falsification case report (from Step 6)
3. Deterministic math report (from Step 7)
4. Search methodology metadata (PRISMA counts)

**Estimated time:** ~45s (one smart model call with large input)

---

## Data Model Changes

### New dataclasses (add to `deep_research_agent.py` or new `models.py`)

```python
@dataclass
class SearchStrategy:
    pico: Dict[str, str]
    mesh_terms: Dict[str, List[str]]
    search_strings: Dict[str, str]
    role: str  # "affirmative" or "adversarial"

@dataclass
class WideNetRecord:
    pmid: Optional[str]
    doi: Optional[str]
    title: str
    abstract: str
    study_type: str
    sample_size: Optional[str]
    primary_objective: Optional[str]
    year: Optional[int]
    journal: Optional[str]
    authors: Optional[str]
    url: str
    source_db: str
    relevance_score: Optional[float] = None

@dataclass
class FullTextArticle:
    pmid: Optional[str]
    doi: Optional[str]
    title: str
    full_text: str
    source: str  # "pmc", "europepmc", "unpaywall", "scrape"
    word_count: int
    url: str
    error: Optional[str] = None

@dataclass
class DeepExtraction:
    pmid: Optional[str]
    doi: Optional[str]
    title: str
    url: str
    attrition_pct: Optional[str]
    effect_size: Optional[str]
    demographics: Optional[str]
    follow_up_period: Optional[str]
    funding_source: Optional[str]
    conflicts_of_interest: Optional[str]
    biological_mechanism: Optional[str]
    control_event_rate: Optional[float]
    experimental_event_rate: Optional[float]
    primary_outcome: Optional[str]
    secondary_outcomes: Optional[List[str]]
    blinding: Optional[str]
    randomization_method: Optional[str]
    intention_to_treat: Optional[bool]
    sample_size_total: Optional[int]
    sample_size_intervention: Optional[int]
    sample_size_control: Optional[int]
    study_design: Optional[str]
    risk_of_bias: Optional[str]
    raw_facts: str
```

### Existing dataclasses preserved
- `SummarizedSource` — still used for compatibility with podcast_crew.py research library tools
- `SearchMetrics` — updated with new fields (wide_net_total, screened_in, fulltext_retrieved)
- `ResearchReport` — unchanged (same return type from Orchestrator.run())

---

## Modified Orchestrator.run() Flow

```python
async def run(self, topic, framing_context="", progress_callback=None):
    # Affirmative track (Steps 1-5)
    async def affirmative_track():
        strategy = await self.lead._formulate_search_strategy(topic, "affirmative")  # Step 1
        records = await self.lead._wide_net_search(strategy)                         # Step 2
        top_20 = await self.lead._screen_and_prioritize(records, strategy)           # Step 3
        fulltexts = await self.fulltext_fetcher.fetch_all(top_20)                    # Step 4a
        extractions = await self.lead._deep_extract_batch(fulltexts)                 # Step 4b
        case_report = await self.lead._build_case(topic, strategy, extractions, "affirmative")  # Step 5
        return strategy, extractions, case_report

    # Falsification track (Steps 1'-4', 6)
    async def falsification_track():
        strategy = await self.counter._formulate_search_strategy(topic, "adversarial")  # Step 1'
        records = await self.counter._wide_net_search(strategy)                          # Step 2'
        top_20 = await self.counter._screen_and_prioritize(records, strategy)            # Step 3'
        fulltexts = await self.fulltext_fetcher.fetch_all(top_20)                        # Step 4'a
        extractions = await self.counter._deep_extract_batch(fulltexts)                  # Step 4'b
        case_report = await self.counter._build_case(topic, strategy, extractions, "falsification")  # Step 6
        return strategy, extractions, case_report

    # Run both tracks in parallel
    (aff_strategy, aff_extractions, aff_case), \
    (fal_strategy, fal_extractions, fal_case) = await asyncio.gather(
        affirmative_track(), falsification_track()
    )

    # Step 7: Deterministic math
    all_extractions = aff_extractions + fal_extractions
    impacts = clinical_math.batch_calculate(all_extractions)
    math_report = clinical_math.format_math_report(impacts)

    # Step 8: GRADE synthesis
    audit_report = await self._grade_synthesis(
        topic, aff_case, fal_case, math_report, aff_strategy, metrics
    )

    # Return same Dict[str, ResearchReport] for backward compatibility
    return {"lead": lead_report, "counter": counter_report, "audit": audit_report}
```

---

## Integration Points with podcast_crew.py

### What stays the same
- `Orchestrator.run()` returns `Dict[str, ResearchReport]` — no change
- Output files: `deep_research_lead.md`, `deep_research_counter.md`, `deep_research_audit.md`
- `deep_research_sources.json` — populated from WideNetRecord + DeepExtraction data

### What changes
- `deep_research_sources.json` entries will have richer metadata (CER, EER, NNT, GRADE)
- Audit report will include GRADE assessment and NNT table
- Progress callback messages will reflect new phase names

### New output files
- `deep_research_math.md` — deterministic math report (Step 7)
- `deep_research_strategy_aff.json` — affirmative PICO/MeSH/Boolean strategy
- `deep_research_strategy_neg.json` — adversarial PICO/MeSH/Boolean strategy
- `deep_research_screening.json` — screening decisions (500 → 20, with reasons)

---

## Progress Callback Updates for Web UI

Replace current phase labels in `podcast_web_ui.py`:

| Old Phase Label | New Phase Label |
|----------------|-----------------|
| "PHASE 1+2: LEAD & COUNTER RESEARCHERS (PARALLEL)" | "PHASE 1: SEARCH STRATEGY FORMULATION" |
| "Iteration 1/3" | "PHASE 2: WIDE NET SEARCH (0/500 records)" |
| "Iteration 2/3" | "PHASE 3: SCREENING (500 → top 20)" |
| "Iteration 3/3" | "PHASE 4: DEEP EXTRACTION (0/20 articles)" |
| N/A | "PHASE 5-6: BUILDING CASES (PARALLEL)" |
| N/A | "PHASE 7: DETERMINISTIC MATH" |
| "PHASE 3: AUDITOR SYNTHESIS" | "PHASE 8: GRADE SYNTHESIS" |

---

## Timing Estimates (Wall Clock)

| Step | Smart Calls | Fast Calls | IO Calls | Est. Time |
|------|-------------|------------|----------|-----------|
| 1. Strategy | 1 | 0 | 0 | 15s |
| 2. Wide Net | 0 | ~250* | 3 API | 90s |
| 3. Screening | 1-2 | 0 | 0 | 20s |
| 4. Deep Extract | 0 | 20 | 20 fetch | 120s |
| 5/6. Build Case | 1 | 0 | 0 | 30s |
| **Per track** | **3-4** | **~270** | **~23** | **~275s** |
| Both tracks (parallel) | — | — | — | **~275s** |
| 7. Math | 0 | 0 | 0 | <1s |
| 8. GRADE | 1 | 0 | 0 | 45s |
| **Total** | **8** | **~540** | **~46** | **~320s (~5.3 min)** |

*\* Only ~250 of 500 need fast model; the rest get study_type from PubMed XML directly.*

**Current system:** ~6-10 minutes (3 iterations × 2 tracks × search/fetch/summarize)
**New system:** ~5.3 minutes — faster due to batch API calls and no iterative loops.

---

## Implementation Order

### Phase A: Foundation (New files + data models)
1. Create `clinical_math.py` — self-contained, no dependencies, unit-testable
2. Create `fulltext_fetcher.py` — PMC/EuropePMC/Unpaywall clients
3. Add new dataclasses to `deep_research_agent.py` (SearchStrategy, WideNetRecord, DeepExtraction, FullTextArticle)

### Phase B: Enhanced PubMed Client
4. Extend `PubMedClient` to support Boolean queries, `retmax=500`, structured abstract parsing, `PublicationType` extraction, DOI extraction from XML

### Phase C: Pipeline Steps
5. Implement `_formulate_search_strategy()` (Step 1)
6. Implement `_wide_net_search()` (Step 2) — PubMed bulk + Scholar + lightweight screening
7. Implement `_screen_and_prioritize()` (Step 3)
8. Implement `_deep_extract_batch()` (Step 4) — uses fulltext_fetcher + fast model
9. Implement `_build_case()` (Steps 5 & 6)

### Phase D: Orchestrator Rewrite
10. Rewrite `Orchestrator.run()` with new flow (parallel tracks + math + GRADE)
11. Update `SearchMetrics` to track new pipeline stages
12. Ensure backward-compatible `Dict[str, ResearchReport]` return

### Phase E: Integration & UI
13. Update `podcast_crew.py` to handle enriched `deep_research_sources.json`
14. Update `podcast_web_ui.py` progress labels
15. Add new output files to artifact list

### Phase F: Testing
16. Unit tests for `clinical_math.py`
17. Integration test with a known medical topic (e.g., "omega-3 and cardiovascular disease")
18. Verify podcast_crew.py phases still work end-to-end

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Full-text PDF not available (paywalled) | 4-tier fallback: PMC → Europe PMC → Unpaywall → publisher scrape. If still unavailable, use abstract-only data (degrade gracefully) |
| 500 results exceeds smart model context for screening | Chunk into batches of 200, screen each, merge top results |
| PubMed rate limit (3 req/sec without key) | Add `PUBMED_API_KEY` env var for 10 req/sec; add 0.4s delay between calls |
| Fast model timeout on deep extraction | Existing 180s timeout per call; add retry with truncated input on failure |
| Not all topics are clinical/medical | Add fallback: if PICO generation returns low-confidence or topic is non-medical, fall back to current iterative search pipeline |
| CER/EER not reported in many studies | Step 7 gracefully handles nulls; NNT table only includes studies with both values |
| Cochrane requires subscription | Use Cochrane CENTRAL via PubMed subset filter (`cochrane[sb]`), not Wiley API |

---

## Environment / Dependencies

**No new pip packages required.** Everything uses:
- `httpx` (already installed) — for PMC/Europe PMC/Unpaywall API calls
- `xml.etree.ElementTree` (stdlib) — for PubMed XML parsing
- `openai` (already installed) — for LLM calls
- `asyncio` (stdlib) — for parallelism
- `json` (stdlib) — for data serialization

**New env vars (optional):**
- `UNPAYWALL_EMAIL` — required for Unpaywall API (free, just needs an email)
- `PUBMED_API_KEY` — already supported, increases rate limit to 10 req/sec
