"""
Source-of-Truth (IMRaD) builder for deep research outputs.

Extracted from pipeline.py (T4.1).
Contains: build_imrad_sot, _build_social_science_sot,
and supporting helpers (_extract_conclusion_status, _parse_grade_sections,
_format_study_characteristics_table, _format_references).
"""

import re
from dataclasses import dataclass
from typing import Any

from dr2_podcast.config import SMART_MODEL
from dr2_podcast.research.clinical_math import format_rrr
from dr2_podcast.utils import strip_think_blocks


def _smart_model_display() -> str:
    return SMART_MODEL.split("/", 1)[-1] if SMART_MODEL else "Smart LLM"


#: The record's enum, spelled the way the status map and the prose have always spelled it.
_GRADE_RECORD_LEVELS = {"high": "High", "moderate": "Moderate", "low": "Low", "very_low": "Very Low"}


def _extract_conclusion_status(
    grade_report: str,
    domain: str = "clinical",
    language: str = "en",
    grade_record: dict | None = None,
) -> tuple:
    """Extract evidence level, conclusion status, and executive summary.

    Supports both GRADE (clinical) and Evidence Quality (social science) levels.
    Uses i18n status_map when language != 'en'.

    ``grade_record`` is the structured record step 7 now produces, and when it is present the level
    is READ from it rather than scraped out of the prose. The regex below survives for runs that
    predate the record — but it is why the record exists: a pattern that misses yields
    "Not Determined", which the status map turns into a mild "Under Evaluation" and the episode goes
    out speaking a confidence nobody computed. A structured record cannot miss; it either says what
    the level is or it fails validation upstream where someone can see it.
    """
    from dr2_podcast.sot_i18n import get_templates

    tmpl = get_templates(language)
    tmpl_status = tmpl["status_map"]

    if grade_record and domain != "social_science":
        grade = _GRADE_RECORD_LEVELS.get(str(grade_record.get("level", "")).lower(), "Not Determined")
        status = tmpl_status.get("clinical", {}).get(grade, tmpl_status.get("default_status", "Under Evaluation"))
        m2 = re.search(r"Executive\s+Summary[#\s:]*\n+(.+?)(?:\n\n|\n#)", grade_report, re.DOTALL)
        return grade, status, (m2.group(1).strip() if m2 else "")

    if domain == "social_science":
        # Social science evidence quality levels
        m = re.search(
            r"Final\s+Evidence\s+Quality[:\s]*\*{0,2}(STRONG|MODERATE_STRONG|MODERATE_WEAK|MODERATE|WEAK|VERY_WEAK)\*{0,2}",
            grade_report,
            re.IGNORECASE,
        )
        grade = m.group(1).strip().upper() if m else "Not Determined"
        status_map = tmpl_status.get("social_science", {})
    else:
        # Clinical GRADE levels
        m = re.search(
            r"Final\s+(?:GRADE|Grade)[:\s]*\*{0,2}(High|Moderate|Low|Very\s+Low)\*{0,2}", grade_report, re.IGNORECASE
        )
        grade = m.group(1).strip() if m else "Not Determined"
        status_map = tmpl_status.get("clinical", {})
    status = status_map.get(grade, tmpl_status.get("default_status", "Under Evaluation"))

    m2 = re.search(r"Executive\s+Summary[#\s:]*\n+(.+?)(?:\n\n|\n#)", grade_report, re.DOTALL)
    summary = m2.group(1).strip() if m2 else ""

    return grade, status, summary


def _parse_grade_sections(audit_text: str) -> dict:
    """Split GRADE synthesis text into named subsections by ### headers."""
    sections = {}
    current_key = None
    current_lines = []
    for line in audit_text.split("\n"):
        if line.startswith("### "):
            if current_key is not None:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = line.lstrip("#").strip().lower()
            current_lines = []
        else:
            current_lines.append(line)
    if current_key is not None:
        sections[current_key] = "\n".join(current_lines).strip()
    return sections


def _format_study_characteristics_table(extractions: list) -> str:
    """Build a study characteristics table from DeepExtraction objects."""
    if not extractions:
        return "*No studies with full extraction data available.*\n"
    # Check if any extraction has enrichment metadata
    has_metadata = any(getattr(ext, "paper_metadata", None) for ext in extractions)
    if has_metadata:
        rows = [
            "| # | Study | Design | N | Demographics | Follow-up | Funding | Bias Risk | Citations | FWCI | Tier |",
            "|---|-------|--------|---|--------------|-----------|---------|-----------|-----------|------|------|",
        ]
    else:
        rows = [
            "| # | Study | Design | N | Demographics | Follow-up | Funding | Bias Risk | Tier |",
            "|---|-------|--------|---|--------------|-----------|---------|-----------|------|",
        ]
    seen = set()
    idx = 0
    for ext in extractions:
        key = ext.pmid or ext.doi or ext.title
        if key in seen:
            continue
        seen.add(key)
        idx += 1
        _ellipsis = "\u2026" if len(ext.title) > 50 else ""
        label = f"{ext.title[:50]}{_ellipsis}"
        if ext.pmid:
            label += f" ([PMID:{ext.pmid}](https://pubmed.ncbi.nlm.nih.gov/{ext.pmid}/))"
        tier_label = f"T{ext.research_tier}" if getattr(ext, "research_tier", None) else "N/A"
        base = (
            f"| {idx} "
            f"| {label} "
            f"| {ext.study_design or 'N/A'} "
            f"| {ext.sample_size_total or 'N/A'} "
            f"| {(ext.demographics or 'N/A')[:40]} "
            f"| {ext.follow_up_period or 'N/A'} "
            f"| {_funding_cell(ext)} "
            f"| {ext.risk_of_bias or 'N/A'} "
        )
        if has_metadata:
            pm = getattr(ext, "paper_metadata", None)
            cite_str = str(pm.citation_count) if pm and pm.citation_count is not None else "N/A"
            fwci_str = f"{pm.fwci:.1f}" if pm and pm.fwci is not None else "N/A"
            base += f"| {cite_str} | {fwci_str} "
        base += f"| {tier_label} |"
        rows.append(base)
    return "\n".join(rows) + "\n"


def _funding_cell(ext) -> str:
    """The funding column: category, disclosure state, and whether anyone can check it.

    undisclosed (the paper is silent) is NOT unknown (we failed to extract) — Ep09's thesis makes
    that distinction the finding, so the two never collapse into one 'N/A'. The API-derived variant
    is flagged because it exists nowhere in the paper and cannot be verified against it.
    """
    funding = getattr(ext, "funding", None)
    if funding is None or funding.funding_disclosure == "unknown":
        legacy = getattr(ext, "funding_source", None)
        return (legacy or "unknown")[:30]
    if funding.funding_disclosure == "undisclosed":
        return "undisclosed (paper silent)"
    flag = "" if funding.funding_source_type == "extracted_text" else " (API, unverified)"
    return f"{(funding.funding_raw or '')[:30]} — {funding.funding_category}{flag}"


def _format_references(extractions: list, wide_net_records: list) -> str:
    """Build a numbered reference list from extraction metadata enriched by WideNetRecords."""
    wnr_by_pmid = {r.pmid: r for r in wide_net_records if r.pmid}
    wnr_by_title = {r.title.lower().strip(): r for r in wide_net_records if r.title}
    refs = []
    seen = set()
    idx = 0
    for ext in extractions:
        key = ext.pmid or ext.doi or ext.title
        if key in seen:
            continue
        seen.add(key)
        idx += 1
        wnr = wnr_by_pmid.get(ext.pmid) or wnr_by_title.get((ext.title or "").lower().strip())
        authors = (wnr.authors if wnr and wnr.authors else "").strip() or "Unknown authors"
        journal = (wnr.journal if wnr and wnr.journal else "").strip()
        year = wnr.year if wnr and wnr.year else ""
        title = ext.title or "Untitled"
        parts = [f"{idx}. {authors}."]
        parts.append(f"*{title}*.")
        if journal:
            parts.append(f"{journal}.")
        if year:
            parts.append(f"({year}).")
        if ext.pmid:
            parts.append(f"PMID: [{ext.pmid}](https://pubmed.ncbi.nlm.nih.gov/{ext.pmid}/).")
        if ext.doi:
            parts.append(f"DOI: {ext.doi}.")
        refs.append(" ".join(parts))
    return "\n".join(refs) + "\n" if refs else "*No references available.*\n"


@dataclass
class _SocialCtx:
    """Inputs for the social-science SOT sections.

    Only the fields the document actually reads. The old positional signature
    also took ev_quality, aff_cand, all_wide and fal_strategy \u2014 none of them
    were ever referenced in the body.
    """

    topic: str
    pd: dict
    audit_text: str
    aff_case_text: str
    fal_case_text: str
    all_extractions: list
    impacts: list
    framing: str
    search_date: str
    peco: dict
    grade_level: str
    conclusion_status: str
    exec_summary: str
    total_wide: int
    total_screened: int
    total_ft_ok: int


def _social_peco(aff_strategy) -> dict:
    """PECO frame, whether the strategy is an object or a raw dict."""
    if aff_strategy and hasattr(aff_strategy, "peco"):
        return aff_strategy.peco if isinstance(aff_strategy.peco, dict) else getattr(aff_strategy, "peco", {})
    if isinstance(aff_strategy, dict):
        return aff_strategy.get("peco", {})
    return {}


def _social_dedup_extractions(all_extractions: list):
    """Yield (index, extraction) skipping repeats of the same DOI/title."""
    seen = set()
    idx = 0
    for ext in all_extractions:
        key = getattr(ext, "doi", None) or getattr(ext, "title", "")
        if key in seen:
            continue
        seen.add(key)
        idx += 1
        yield idx, ext


def _social_abstract(c: _SocialCtx) -> list[str]:
    from dr2_podcast.research.effect_size_math import EffectSizeImpact

    out = [f"# Source of Truth: {c.topic}\n", "## 1. Abstract\n"]

    # Research question (PECO)
    if c.peco:
        out.append(
            f"**Research Question (PECO):** In {c.peco.get('P', 'the target population')}, "
            f"does exposure to {c.peco.get('E', 'the intervention')} compared to "
            f"{c.peco.get('C', 'no exposure')} affect {c.peco.get('O', 'outcomes')}?\n"
        )

    out.append(
        f"**Methods:** Systematic search of OpenAlex, ERIC, and Google Scholar identified "
        f"{c.total_wide} records. After screening, {c.total_screened} studies were selected and "
        f"{c.total_ft_ok} were fully extracted using the PECO framework.\n"
    )

    # Key finding (effect sizes)
    if c.impacts:
        es_list = [i for i in c.impacts if isinstance(i, EffectSizeImpact)]
        if es_list:
            avg_d = sum(abs(i.cohens_d or 0) for i in es_list) / len(es_list)
            magnitude = (
                "negligible" if avg_d < 0.2 else "small" if avg_d < 0.5 else "medium" if avg_d < 0.8 else "large"
            )
            out.append(
                f"**Key Finding:** Across {len(es_list)} studies with reported effect sizes, "
                f"the average magnitude was {magnitude} (mean |d| = {avg_d:.3f}).\n"
            )

    out.append(f"**Evidence Quality:** {c.grade_level} \u2014 {c.conclusion_status}\n")
    if c.exec_summary:
        out.append(f"**Executive Summary:** {c.exec_summary}\n")
    return out


def _social_introduction(c: _SocialCtx) -> list[str]:
    out = ["\n## 2. Introduction\n"]
    if c.framing:
        out.append(f"{c.framing}\n")
    else:
        out.append(f"This review examines the evidence for: *{c.topic}*.\n")
    out.append(
        "This review employs a dual-hypothesis design with parallel affirmative and "
        "falsification research tracks, using the PECO (Population, Exposure, Comparison, Outcome) framework.\n"
    )
    return out


def _social_methods(c: _SocialCtx) -> list[str]:
    out = [
        "\n## 3. Methods\n",
        "### 3.1 Search Strategy\n",
        "**Framework:** PECO (Population, Exposure, Comparison, Outcome)\n",
    ]
    if c.peco:
        out.append(f"- **P (Population):** {c.peco.get('P', 'Not specified')}\n")
        out.append(f"- **E (Exposure):** {c.peco.get('E', 'Not specified')}\n")
        out.append(f"- **C (Comparison):** {c.peco.get('C', 'Not specified')}\n")
        out.append(f"- **O (Outcome):** {c.peco.get('O', 'Not specified')}\n")

    out.append("\n### 3.2 Data Collection\n")
    out.append("**Databases:** OpenAlex, ERIC (IES), Google Scholar\n")
    out.append(f"**Search date:** {c.search_date}\n")
    out.append(f"**Records identified:** {c.total_wide}\n")
    out.append(f"**Screened:** {c.total_screened}\n")
    out.append(f"**Extracted:** {c.total_ft_ok}\n")

    out.append("\n### 3.3 Statistical Analysis\n")
    out.append(
        "Effect sizes were standardized to Cohen's d using deterministic Python calculations. "
        "Hedges' g correction was applied where sample sizes were available. "
        "Odds ratios and correlation coefficients were converted to d for comparability.\n"
    )
    return out


def _social_study_table(c: _SocialCtx) -> list[str]:
    if not c.all_extractions:
        return ["*No studies with full extraction data available.*\n"]
    rows = [
        "| # | Study | Design | N | Setting | Demographics | Effect Size | Follow-up | Tier |",
        "|---|-------|--------|---|---------|--------------|-------------|-----------|------|",
    ]
    for idx, ext in _social_dedup_extractions(c.all_extractions):
        title_str = (getattr(ext, "title", "") or "")[:50]
        es_val = getattr(ext, "effect_size_value", None)
        es_type = getattr(ext, "effect_size_type", None)
        es_str = f"{es_type}={es_val}" if es_val is not None else "N/A"
        setting = (getattr(ext, "setting", None) or "N/A")[:30]
        demo = (getattr(ext, "demographics", None) or "N/A")[:30]
        design = getattr(ext, "study_design", None) or "N/A"
        n = getattr(ext, "sample_size_total", None) or "N/A"
        fu = getattr(ext, "follow_up_period", None) or "N/A"
        tier = f"T{getattr(ext, 'research_tier', 'N/A')}" if getattr(ext, "research_tier", None) else "N/A"
        rows.append(f"| {idx} | {title_str} | {design} | {n} | {setting} | {demo} | {es_str} | {fu} | {tier} |")
    return ["\n".join(rows) + "\n"]


def _social_results(c: _SocialCtx) -> list[str]:
    out = ["\n## 4. Results\n", "### 4.1 Study Characteristics\n"]
    out += _social_study_table(c)

    out.append("\n### 4.2 Effect Size Analysis\n")
    math_report = c.pd.get("math_report", "")
    if math_report:
        out.append(f"{math_report}\n")
    else:
        out.append("*No effect sizes calculated.*\n")
    return out


def _social_discussion(c: _SocialCtx) -> list[str]:
    out = ["\n## 5. Discussion\n", "### 5.1 Affirmative Case\n"]
    if c.aff_case_text:
        out.append(f"{c.aff_case_text}\n")
    out.append("\n### 5.2 Falsification Case\n")
    if c.fal_case_text:
        out.append(f"{c.fal_case_text}\n")

    out.append("\n### 5.3 Evidence Quality Synthesis\n")
    if c.audit_text:
        out.append(f"{c.audit_text}\n")
    return out


def _social_references(c: _SocialCtx) -> list[str]:
    out = ["\n## 6. References\n"]
    if not c.all_extractions:
        out.append("*No references available.*\n")
        return out
    for idx, ext in _social_dedup_extractions(c.all_extractions):
        title = getattr(ext, "title", "Untitled") or "Untitled"
        doi = getattr(ext, "doi", None)
        parts = [f"{idx}. *{title}*."]
        if doi:
            parts.append(f"DOI: {doi}.")
        url = getattr(ext, "url", None)
        if url:
            parts.append(f"URL: {url}")
        out.append(" ".join(parts))
    out.append("")
    return out


def _build_social_science_sot(ctx: _SocialCtx) -> str:
    """Build IMRaD SOT for social science topics (PECO, effect sizes, evidence quality)."""
    out: list[str] = []
    out += _social_abstract(ctx)
    out += _social_introduction(ctx)
    out += _social_methods(ctx)
    out += _social_results(ctx)
    out += _social_discussion(ctx)
    out += _social_references(ctx)
    return "\n".join(out)


@dataclass
class _ImradCounts:
    """PRISMA counters, unpacked once from pipeline_data["metrics"]."""

    aff_wide: int = 0
    fal_wide: int = 0
    aff_screened: int = 0
    fal_screened: int = 0
    aff_ft_ok: int = 0
    fal_ft_ok: int = 0
    aff_ft_err: int = 0
    fal_ft_err: int = 0

    @classmethod
    def from_metrics(cls, m: dict) -> "_ImradCounts":
        return cls(
            aff_wide=m.get("aff_wide_net_total", 0),
            fal_wide=m.get("fal_wide_net_total", 0),
            aff_screened=m.get("aff_screened_in", 0),
            fal_screened=m.get("fal_screened_in", 0),
            aff_ft_ok=m.get("aff_fulltext_ok", 0),
            fal_ft_ok=m.get("fal_fulltext_ok", 0),
            aff_ft_err=m.get("aff_fulltext_err", 0),
            fal_ft_err=m.get("fal_fulltext_err", 0),
        )

    @property
    def total_wide(self) -> int:
        return self.aff_wide + self.fal_wide

    @property
    def total_screened(self) -> int:
        return self.aff_screened + self.fal_screened

    @property
    def total_ft_ok(self) -> int:
        return self.aff_ft_ok + self.fal_ft_ok

    @property
    def total_ft_err(self) -> int:
        return self.aff_ft_err + self.fal_ft_err


@dataclass
class _ImradCtx:
    """Everything the IMRaD section builders read.

    Assembled once by build_imrad_sot so each section builder takes a single
    argument instead of re-threading fifteen locals.
    """

    tmpl: dict
    t: Any  # sot_i18n.t — imported inside build_imrad_sot to avoid a cycle
    pd: dict
    topic: str
    aff_strategy: Any
    fal_strategy: Any
    aff_extractions: list
    fal_extractions: list
    all_extractions: list
    all_wide: list
    impacts: list
    framing: str
    search_date: str
    counts: _ImradCounts
    grade_level: str
    conclusion_status: str
    exec_summary: str
    grade_sections: dict
    audit_text: str
    aff_case_text: str
    fal_case_text: str
    output_dir: Any = None
    output_path_fn: Any = None


def _imrad_abstract(c: _ImradCtx) -> list[str]:
    tmpl, t, n = c.tmpl, c.t, c.counts

    # Summarize PICO for abstract
    pico_summary = ""
    if c.aff_strategy and hasattr(c.aff_strategy, "pico"):
        p = c.aff_strategy.pico
        pico_summary = tmpl["pico_summary_template"].format(
            population=p.get("population", "N/A"),
            intervention=p.get("intervention", "N/A"),
            comparison=p.get("comparison", "N/A"),
            outcome=p.get("outcome", "N/A"),
        )

    # Determine representative NNT for abstract
    nnt_summary = ""
    if c.impacts:
        benefit = [i for i in c.impacts if i.direction == "benefit"]
        ref_impact = benefit[0] if benefit else c.impacts[0]
        nnt_summary = tmpl["nnt_summary_template"].format(
            nnt=ref_impact.nnt, direction=ref_impact.direction, arr=ref_impact.arr
        )

    out = [t(tmpl, "title", "prefix", topic=c.topic)]
    out.append(t(tmpl, "abstract", "header"))
    if pico_summary:
        out.append(t(tmpl, "abstract", "pico_label", pico_summary=pico_summary))
    out.append(
        t(
            tmpl,
            "abstract",
            "methods",
            total_wide=n.total_wide,
            total_screened=n.total_screened,
            total_ft_ok=n.total_ft_ok,
        )
    )
    if nnt_summary:
        out.append(t(tmpl, "abstract", "key_finding", nnt_summary=nnt_summary))
    out.append(
        t(tmpl, "abstract", "evidence_quality", grade_level=c.grade_level, conclusion_status=c.conclusion_status)
    )
    if c.exec_summary:
        out.append(f"\n{c.exec_summary}\n")
    return out


def _imrad_introduction(c: _ImradCtx) -> list[str]:
    tmpl, t = c.tmpl, c.t
    out = [t(tmpl, "introduction", "header")]
    if c.framing:
        out.append(c.framing.strip() + "\n")
    else:
        out.append(t(tmpl, "introduction", "default_framing", topic=c.topic))
    out.append(t(tmpl, "introduction", "dual_hypothesis"))
    if c.aff_strategy and hasattr(c.aff_strategy, "pico"):
        p = c.aff_strategy.pico
        out.append(
            t(
                tmpl,
                "introduction",
                "aff_hypothesis",
                population=p.get("population", "the target population"),
                intervention=p.get("intervention", "the intervention"),
                outcome=p.get("outcome", "the primary outcome"),
                comparison=p.get("comparison", "control"),
            )
        )
    if c.fal_strategy and hasattr(c.fal_strategy, "pico"):
        fp = c.fal_strategy.pico
        out.append(
            t(
                tmpl,
                "introduction",
                "fal_hypothesis",
                intervention=fp.get("intervention", "the intervention"),
                outcome=fp.get("outcome", "the primary outcome"),
                population=fp.get("population", "the target population"),
            )
        )
    return out


def _imrad_search_strategy_track(c: _ImradCtx, label_key: str, strategy) -> list[str]:
    """Methods §2.1 for a single track. Returns [] when the track has no strategy."""
    if not strategy or not hasattr(strategy, "pico"):
        return []
    tmpl, t = c.tmpl, c.t
    label = tmpl["track_labels"][label_key]
    out = [t(tmpl, "methods", "track_header", label=label)]
    p = strategy.pico
    out.append(
        t(
            tmpl,
            "methods",
            "pico_framework",
            population=p.get("population", "N/A"),
            intervention=p.get("intervention", "N/A"),
            comparison=p.get("comparison", "N/A"),
            outcome=p.get("outcome", "N/A"),
        )
    )
    # Tiered keyword plan (new architecture)
    if hasattr(strategy, "tier1"):
        tier_label_list = tmpl["methods"]["tier_labels"]
        tier_map = [
            (tier_label_list[0], strategy.tier1),
            (tier_label_list[1], strategy.tier2),
            (tier_label_list[2], strategy.tier3),
        ]
        out.append(t(tmpl, "methods", "three_tier_header"))
        for tier_label, tier_kw in tier_map:
            if hasattr(tier_kw, "intervention") and tier_kw.intervention:
                out.append(f"\n*{tier_label}*\n")
                out.append(t(tmpl, "methods", "intervention_label", terms=", ".join(tier_kw.intervention)))
                out.append(t(tmpl, "methods", "outcome_label", terms=", ".join(tier_kw.outcome)))
                if tier_kw.population:
                    out.append(t(tmpl, "methods", "population_label", terms=", ".join(tier_kw.population)))
                out.append(t(tmpl, "methods", "rationale_label", rationale=tier_kw.rationale))
        if strategy.auditor_approved:
            out.append(t(tmpl, "methods", "auditor_approved", revision_count=strategy.revision_count))
        else:
            out.append(t(tmpl, "methods", "auditor_not_approved", notes=strategy.auditor_notes[:200]))
    # Legacy: Boolean search strings (old architecture -- kept for backward compat)
    elif hasattr(strategy, "mesh_terms") and strategy.mesh_terms:
        mt = strategy.mesh_terms
        out.append(t(tmpl, "methods", "mesh_terms_header"))
        for cat, terms in mt.items():
            if terms:
                out.append(f"- *{cat.capitalize()}*: {', '.join(terms)}\n")
    if hasattr(strategy, "search_strings") and strategy.search_strings:
        ss = strategy.search_strings
        out.append(t(tmpl, "methods", "boolean_search_header"))
        for db, query in ss.items():
            if query:
                out.append(f"- **{db.replace('_', ' ').title()}**: `{query}`\n")
    out.append("\n")
    return out


def _imrad_methods(c: _ImradCtx) -> list[str]:
    tmpl, t, n = c.tmpl, c.t, c.counts
    out = [t(tmpl, "methods", "header")]

    # 2.1 Search Strategy
    out.append(t(tmpl, "methods", "search_strategy_header"))
    for label_key, strategy in [("affirmative", c.aff_strategy), ("falsification", c.fal_strategy)]:
        out += _imrad_search_strategy_track(c, label_key, strategy)

    # 2.2 Data Collection
    out.append(t(tmpl, "methods", "data_collection_header"))
    aff_tier = c.pd.get("aff_highest_tier", 1)
    fal_tier = c.pd.get("fal_highest_tier", 1)
    tier_cascade = tmpl["methods"]["tier_cascade_labels"]
    out.append(
        t(
            tmpl,
            "methods",
            "data_collection_body",
            search_date=c.search_date,
            aff_tier_label=tier_cascade.get(aff_tier, str(aff_tier)),
            fal_tier_label=tier_cascade.get(fal_tier, str(fal_tier)),
        )
    )
    if aff_tier == 3 or fal_tier == 3:
        out.append(t(tmpl, "methods", "tier3_warning"))
    out.append(t(tmpl, "methods", "track_records", aff_wide=n.aff_wide, fal_wide=n.fal_wide, total_wide=n.total_wide))

    # 2.3 Screening & Selection
    out.append(t(tmpl, "methods", "screening_header"))
    out.append(
        t(
            tmpl,
            "methods",
            "screening_body",
            aff_screened=n.aff_screened,
            fal_screened=n.fal_screened,
            total_screened=n.total_screened,
            smart_model=_smart_model_display(),
        )
    )

    # 2.4 Data Extraction
    out.append(t(tmpl, "methods", "extraction_header"))
    out.append(
        t(
            tmpl,
            "methods",
            "extraction_body",
            aff_ft_ok=n.aff_ft_ok,
            aff_ft_err=n.aff_ft_err,
            fal_ft_ok=n.fal_ft_ok,
            fal_ft_err=n.fal_ft_err,
            total_ft_ok=n.total_ft_ok,
        )
    )

    # 2.5 Statistical Analysis
    out.append(t(tmpl, "methods", "stats_header"))
    out.append(t(tmpl, "methods", "stats_body"))
    return out


def _imrad_clinical_impact(c: _ImradCtx) -> list[str]:
    """Results §3.3 — prefers the on-disk clinical_math.md over recomputing."""
    tmpl, t = c.tmpl, c.t
    out = [t(tmpl, "results", "clinical_impact_header")]
    # Try to read clinical_math.md from output directory
    _math_content = None
    if c.output_dir is not None and c.output_path_fn is not None:
        math_file_path = c.output_path_fn(c.output_dir, "clinical_math.md")
        if math_file_path.exists():
            _math_content = math_file_path.read_text().strip()
    if _math_content:
        out.append(_math_content + "\n")
    elif c.impacts:
        rows = [tmpl["results"]["impact_table_header"]]
        for i in c.impacts:
            # row_label, not study_id: one paper can contribute a row per endpoint, and keying the
            # table by the study alone renders them as duplicate rows nobody can tell apart.
            # EffectSizeImpact has no such split, so it falls back to its study_id.
            label = getattr(i, "row_label", None) or i.study_id
            rows.append(
                f"| {label} | {i.cer:.3f} | {i.eer:.3f} | "
                f"{i.arr:+.4f} | {format_rrr(i.rrr)} | {i.nnt:.1f} | {i.direction} |"
            )
        out.append("\n".join(rows) + "\n\n")
        for i in c.impacts:
            out.append(f"- **{getattr(i, 'row_label', None) or i.study_id}**: {i.nnt_interpretation}\n")
    else:
        out.append(t(tmpl, "results", "no_impact_data"))
    return out


def _imrad_results(c: _ImradCtx) -> list[str]:
    tmpl, t, n = c.tmpl, c.t, c.counts
    out = [t(tmpl, "results", "header")]

    # 3.1 Study Selection (PRISMA)
    out.append(t(tmpl, "results", "study_selection_header"))
    prisma_from_grade = c.grade_sections.get("prisma flow diagram", "")
    prisma_rows = tmpl["results"]["prisma_rows"]
    out.append(
        t(tmpl, "results", "prisma_label")
        + tmpl["results"]["prisma_table_header"]
        + prisma_rows["identified"].format(aff=n.aff_wide, fal=n.fal_wide, total=n.total_wide)
        + prisma_rows["screened"].format(aff=n.aff_screened, fal=n.fal_screened, total=n.total_screened)
        + prisma_rows["fulltext"].format(aff=n.aff_ft_ok, fal=n.fal_ft_ok, total=n.total_ft_ok)
        + prisma_rows["errors"].format(aff=n.aff_ft_err, fal=n.fal_ft_err, total=n.total_ft_err)
        + prisma_rows["included"].format(
            aff=len(c.aff_extractions), fal=len(c.fal_extractions), total=len(c.all_extractions)
        )
    )
    if prisma_from_grade:
        out.append(f"\n{prisma_from_grade}\n")

    # 3.2 Study Characteristics
    out.append(t(tmpl, "results", "study_chars_header"))
    out.append(_format_study_characteristics_table(c.all_extractions))

    # 3.3 Clinical Impact
    out += _imrad_clinical_impact(c)
    return out


def _imrad_discussion(c: _ImradCtx) -> list[str]:
    tmpl, t = c.tmpl, c.t
    out = [t(tmpl, "discussion", "header")]

    # 4.1 Affirmative Case
    out.append(t(tmpl, "discussion", "aff_case_header"))
    out.append(c.aff_case_text.strip() + "\n")

    # 4.2 Falsification Case
    out.append(t(tmpl, "discussion", "fal_case_header"))
    out.append(c.fal_case_text.strip() + "\n")

    # 4.3 GRADE Evidence Assessment
    out.append(t(tmpl, "discussion", "grade_header"))
    ep = c.grade_sections.get("evidence profile", "")
    ga = c.grade_sections.get("grade assessment", "")
    if ep:
        out.append(t(tmpl, "discussion", "evidence_profile_label", text=ep))
    if ga:
        out.append(t(tmpl, "discussion", "grade_assessment_label", text=ga))
    if not ep and not ga:
        # Fallback: include the full audit text minus already-extracted sections
        out.append(c.audit_text.strip() + "\n")

    # 4.4 Balanced Verdict
    out.append(t(tmpl, "discussion", "verdict_header"))
    bv = c.grade_sections.get("balanced verdict", "")
    if bv:
        out.append(bv + "\n")
    else:
        out.append(
            t(
                tmpl,
                "discussion",
                "verdict_fallback",
                grade_level=c.grade_level,
                conclusion_status=c.conclusion_status,
            )
        )

    # 4.5 Limitations
    out.append(t(tmpl, "discussion", "limitations_header"))
    out.append(t(tmpl, "discussion", "limitations_body"))

    # 4.6 Recommendations
    out.append(t(tmpl, "discussion", "recs_header"))
    recs = c.grade_sections.get("recommendations for further research", "")
    if recs:
        out.append(recs + "\n")
    else:
        out.append(t(tmpl, "discussion", "recs_fallback"))
    return out


def _imrad_references(c: _ImradCtx) -> list[str]:
    return [
        c.t(c.tmpl, "references", "header"),
        _format_references(c.all_extractions, c.all_wide),
    ]


def build_imrad_sot(
    topic: str,
    reports: dict,
    domain: str = "clinical",
    output_dir=None,
    output_path_fn=None,
    language: str = "en",
) -> str:
    """Assemble the Source of Truth document in IMRaD scientific paper format.

    Args:
        domain: "clinical" or "social_science" -- controls framework terminology
        output_dir: Path to the current output directory (for reading clinical_math.md).
        output_path_fn: Callable(run_dir, filename) -> Path.
        language: "en" or "ja" -- selects pre-translated boilerplate templates.

    The evidence-quality label and affirmative-candidate count used to be
    parameters here. Neither document branch ever read them — they were
    forwarded to _build_social_science_sot, which ignored them — so they are
    gone. pipeline_flow still computes both; it uses them for its own
    evidence-limited notice.
    """
    from dr2_podcast.sot_i18n import get_templates, t

    tmpl = get_templates(language)

    pd = reports.get("pipeline_data", {})
    # Auto-detect domain from pipeline_data if not explicitly set
    if pd.get("domain") == "social_science":
        domain = "social_science"
    aff_strategy = pd.get("aff_strategy")
    fal_strategy = pd.get("fal_strategy")
    aff_extractions = pd.get("aff_extractions", [])
    fal_extractions = pd.get("fal_extractions", [])
    aff_top = pd.get("aff_top", [])
    fal_top = pd.get("fal_top", [])
    impacts = pd.get("impacts", [])
    framing = pd.get("framing_context", "")
    search_date = pd.get("search_date", "")
    metrics = pd.get("metrics", {})
    all_extractions = aff_extractions + fal_extractions
    all_wide = aff_top + fal_top

    _empty_rpt = type(
        "_E", (), {"report": "", "total_summaries": 0, "total_urls_fetched": 0, "duration_seconds": 0, "sources": []}
    )()
    audit_text = strip_think_blocks(reports.get("audit", _empty_rpt).report)
    aff_case_text = strip_think_blocks(reports.get("lead", _empty_rpt).report)
    fal_case_text = strip_think_blocks(reports.get("counter", _empty_rpt).report)

    # Dispatch to domain-specific SOT builder
    if domain == "social_science":
        ss_grade, ss_status, ss_summary = _extract_conclusion_status(audit_text, domain="social_science")
        return _build_social_science_sot(
            _SocialCtx(
                topic=topic,
                pd=pd,
                audit_text=audit_text,
                aff_case_text=aff_case_text,
                fal_case_text=fal_case_text,
                all_extractions=all_extractions,
                impacts=impacts,
                framing=framing,
                search_date=search_date,
                peco=_social_peco(aff_strategy),
                grade_level=ss_grade,
                conclusion_status=ss_status,
                exec_summary=ss_summary,
                total_wide=metrics.get("aff_wide_net_total", 0) + metrics.get("fal_wide_net_total", 0),
                total_screened=metrics.get("aff_screened_in", 0) + metrics.get("fal_screened_in", 0),
                total_ft_ok=metrics.get("aff_fulltext_ok", 0) + metrics.get("fal_fulltext_ok", 0),
            )
        )

    grade_level, conclusion_status, exec_summary = _extract_conclusion_status(
        audit_text, language=language, grade_record=pd.get("grade_record")
    )

    ctx = _ImradCtx(
        tmpl=tmpl,
        t=t,
        pd=pd,
        topic=topic,
        aff_strategy=aff_strategy,
        fal_strategy=fal_strategy,
        aff_extractions=aff_extractions,
        fal_extractions=fal_extractions,
        all_extractions=all_extractions,
        all_wide=all_wide,
        impacts=impacts,
        framing=framing,
        search_date=search_date,
        counts=_ImradCounts.from_metrics(metrics),
        grade_level=grade_level,
        conclusion_status=conclusion_status,
        exec_summary=exec_summary,
        grade_sections=_parse_grade_sections(audit_text),
        audit_text=audit_text,
        aff_case_text=aff_case_text,
        fal_case_text=fal_case_text,
        output_dir=output_dir,
        output_path_fn=output_path_fn,
    )

    out: list[str] = []
    out += _imrad_abstract(ctx)
    out += _imrad_introduction(ctx)
    out += _imrad_methods(ctx)
    out += _imrad_results(ctx)
    out += _imrad_discussion(ctx)
    out += _imrad_references(ctx)
    return "\n".join(out)
