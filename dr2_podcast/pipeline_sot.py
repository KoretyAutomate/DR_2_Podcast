"""
Source-of-Truth (IMRaD) builder for deep research outputs.

Extracted from pipeline.py (T4.1).
Contains: build_imrad_sot, _build_social_science_sot,
and supporting helpers (_extract_conclusion_status, _parse_grade_sections,
_format_study_characteristics_table, _format_references).
"""

import re
from pathlib import Path
from dr2_podcast.utils import strip_think_blocks


def _extract_conclusion_status(grade_report: str, domain: str = "clinical",
                               language: str = "en") -> tuple:
    """Extract evidence level, conclusion status, and executive summary.

    Supports both GRADE (clinical) and Evidence Quality (social science) levels.
    Uses i18n status_map when language != 'en'.
    """
    from dr2_podcast.sot_i18n import get_templates
    tmpl = get_templates(language)
    tmpl_status = tmpl["status_map"]

    if domain == "social_science":
        # Social science evidence quality levels
        m = re.search(
            r'Final\s+Evidence\s+Quality[:\s]*\*{0,2}(STRONG|MODERATE_STRONG|MODERATE_WEAK|MODERATE|WEAK|VERY_WEAK)\*{0,2}',
            grade_report, re.IGNORECASE)
        grade = m.group(1).strip().upper() if m else "Not Determined"
        status_map = tmpl_status.get("social_science", {})
    else:
        # Clinical GRADE levels
        m = re.search(
            r'Final\s+(?:GRADE|Grade)[:\s]*\*{0,2}(High|Moderate|Low|Very\s+Low)\*{0,2}',
            grade_report, re.IGNORECASE)
        grade = m.group(1).strip() if m else "Not Determined"
        status_map = tmpl_status.get("clinical", {})
    status = status_map.get(grade, tmpl_status.get("default_status", "Under Evaluation"))

    m2 = re.search(r'Executive\s+Summary[#\s:]*\n+(.+?)(?:\n\n|\n#)',
                   grade_report, re.DOTALL)
    summary = m2.group(1).strip() if m2 else ""

    return grade, status, summary

def _parse_grade_sections(audit_text: str) -> dict:
    """Split GRADE synthesis text into named subsections by ### headers."""
    sections = {}
    current_key = None
    current_lines = []
    for line in audit_text.split('\n'):
        if line.startswith('### '):
            if current_key is not None:
                sections[current_key] = '\n'.join(current_lines).strip()
            current_key = line.lstrip('#').strip().lower()
            current_lines = []
        else:
            current_lines.append(line)
    if current_key is not None:
        sections[current_key] = '\n'.join(current_lines).strip()
    return sections

def _format_study_characteristics_table(extractions: list) -> str:
    """Build a study characteristics table from DeepExtraction objects."""
    if not extractions:
        return "*No studies with full extraction data available.*\n"
    # Check if any extraction has enrichment metadata
    has_metadata = any(getattr(ext, 'paper_metadata', None) for ext in extractions)
    if has_metadata:
        rows = ["| # | Study | Design | N | Demographics | Follow-up | Funding | Bias Risk | Citations | FWCI | Tier |",
                "|---|-------|--------|---|--------------|-----------|---------|-----------|-----------|------|------|"]
    else:
        rows = ["| # | Study | Design | N | Demographics | Follow-up | Funding | Bias Risk | Tier |",
                "|---|-------|--------|---|--------------|-----------|---------|-----------|------|"]
    seen = set()
    idx = 0
    for ext in extractions:
        key = ext.pmid or ext.doi or ext.title
        if key in seen:
            continue
        seen.add(key)
        idx += 1
        _ellipsis = '\u2026' if len(ext.title) > 50 else ''
        label = f"{ext.title[:50]}{_ellipsis}"
        if ext.pmid:
            label += f" ([PMID:{ext.pmid}](https://pubmed.ncbi.nlm.nih.gov/{ext.pmid}/))"
        tier_label = f"T{ext.research_tier}" if getattr(ext, 'research_tier', None) else "N/A"
        base = (
            f"| {idx} "
            f"| {label} "
            f"| {ext.study_design or 'N/A'} "
            f"| {ext.sample_size_total or 'N/A'} "
            f"| {(ext.demographics or 'N/A')[:40]} "
            f"| {ext.follow_up_period or 'N/A'} "
            f"| {(ext.funding_source or 'N/A')[:30]} "
            f"| {ext.risk_of_bias or 'N/A'} "
        )
        if has_metadata:
            pm = getattr(ext, 'paper_metadata', None)
            cite_str = str(pm.citation_count) if pm and pm.citation_count is not None else "N/A"
            fwci_str = f"{pm.fwci:.1f}" if pm and pm.fwci is not None else "N/A"
            base += f"| {cite_str} | {fwci_str} "
        base += f"| {tier_label} |"
        rows.append(base)
    return '\n'.join(rows) + '\n'

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
    return '\n'.join(refs) + '\n' if refs else "*No references available.*\n"

def _build_social_science_sot(
    topic, pd, audit_text, aff_case_text, fal_case_text,
    ev_quality, aff_cand, all_extractions, all_wide, impacts, metrics,
    framing, search_date, aff_strategy, fal_strategy,
) -> str:
    """Build IMRaD SOT for social science topics (PECO, effect sizes, evidence quality)."""
    from dr2_podcast.research.effect_size_math import EffectSizeImpact

    grade_level, conclusion_status, exec_summary = _extract_conclusion_status(
        audit_text, domain="social_science"
    )

    # Extract metrics
    aff_wide = metrics.get("aff_wide_net_total", 0)
    fal_wide = metrics.get("fal_wide_net_total", 0)
    total_wide = aff_wide + fal_wide
    total_screened = metrics.get("aff_screened_in", 0) + metrics.get("fal_screened_in", 0)
    total_ft_ok = metrics.get("aff_fulltext_ok", 0) + metrics.get("fal_fulltext_ok", 0)

    out = []

    # --- Abstract ---
    out.append(f"# Source of Truth: {topic}\n")
    out.append("## 1. Abstract\n")

    # Research question (PECO)
    peco = {}
    if aff_strategy and hasattr(aff_strategy, 'peco'):
        peco = aff_strategy.peco if isinstance(aff_strategy.peco, dict) else getattr(aff_strategy, 'peco', {})
    elif isinstance(aff_strategy, dict):
        peco = aff_strategy.get("peco", {})
    if peco:
        out.append(f"**Research Question (PECO):** In {peco.get('P', 'the target population')}, "
                   f"does exposure to {peco.get('E', 'the intervention')} compared to "
                   f"{peco.get('C', 'no exposure')} affect {peco.get('O', 'outcomes')}?\n")

    out.append(f"**Methods:** Systematic search of OpenAlex, ERIC, and Google Scholar identified "
               f"{total_wide} records. After screening, {total_screened} studies were selected and "
               f"{total_ft_ok} were fully extracted using the PECO framework.\n")

    # Key finding (effect sizes)
    if impacts:
        es_list = [i for i in impacts if isinstance(i, EffectSizeImpact)]
        if es_list:
            avg_d = sum(abs(i.cohens_d or 0) for i in es_list) / len(es_list)
            magnitude = "negligible" if avg_d < 0.2 else "small" if avg_d < 0.5 else "medium" if avg_d < 0.8 else "large"
            out.append(f"**Key Finding:** Across {len(es_list)} studies with reported effect sizes, "
                       f"the average magnitude was {magnitude} (mean |d| = {avg_d:.3f}).\n")

    out.append(f"**Evidence Quality:** {grade_level} \u2014 {conclusion_status}\n")
    if exec_summary:
        out.append(f"**Executive Summary:** {exec_summary}\n")

    # --- Introduction ---
    out.append("\n## 2. Introduction\n")
    if framing:
        out.append(f"{framing}\n")
    else:
        out.append(f"This review examines the evidence for: *{topic}*.\n")
    out.append("This review employs a dual-hypothesis design with parallel affirmative and "
               "falsification research tracks, using the PECO (Population, Exposure, Comparison, Outcome) framework.\n")

    # --- Methods ---
    out.append("\n## 3. Methods\n")
    out.append("### 3.1 Search Strategy\n")
    out.append(f"**Framework:** PECO (Population, Exposure, Comparison, Outcome)\n")
    if peco:
        out.append(f"- **P (Population):** {peco.get('P', 'Not specified')}\n")
        out.append(f"- **E (Exposure):** {peco.get('E', 'Not specified')}\n")
        out.append(f"- **C (Comparison):** {peco.get('C', 'Not specified')}\n")
        out.append(f"- **O (Outcome):** {peco.get('O', 'Not specified')}\n")

    out.append(f"\n### 3.2 Data Collection\n")
    out.append(f"**Databases:** OpenAlex, ERIC (IES), Google Scholar\n")
    out.append(f"**Search date:** {search_date}\n")
    out.append(f"**Records identified:** {total_wide}\n")
    out.append(f"**Screened:** {total_screened}\n")
    out.append(f"**Extracted:** {total_ft_ok}\n")

    out.append(f"\n### 3.3 Statistical Analysis\n")
    out.append("Effect sizes were standardized to Cohen's d using deterministic Python calculations. "
               "Hedges' g correction was applied where sample sizes were available. "
               "Odds ratios and correlation coefficients were converted to d for comparability.\n")

    # --- Results ---
    out.append("\n## 4. Results\n")
    out.append("### 4.1 Study Characteristics\n")
    if all_extractions:
        has_metadata = any(getattr(ext, 'paper_metadata', None) for ext in all_extractions)
        rows = ["| # | Study | Design | N | Setting | Demographics | Effect Size | Follow-up | Tier |",
                "|---|-------|--------|---|---------|--------------|-------------|-----------|------|"]
        seen = set()
        idx = 0
        for ext in all_extractions:
            key = getattr(ext, 'doi', None) or getattr(ext, 'title', '')
            if key in seen:
                continue
            seen.add(key)
            idx += 1
            title_str = (getattr(ext, 'title', '') or '')[:50]
            es_val = getattr(ext, 'effect_size_value', None)
            es_type = getattr(ext, 'effect_size_type', None)
            es_str = f"{es_type}={es_val}" if es_val is not None else "N/A"
            setting = (getattr(ext, 'setting', None) or "N/A")[:30]
            demo = (getattr(ext, 'demographics', None) or "N/A")[:30]
            design = getattr(ext, 'study_design', None) or "N/A"
            n = getattr(ext, 'sample_size_total', None) or "N/A"
            fu = getattr(ext, 'follow_up_period', None) or "N/A"
            tier = f"T{getattr(ext, 'research_tier', 'N/A')}" if getattr(ext, 'research_tier', None) else "N/A"
            rows.append(f"| {idx} | {title_str} | {design} | {n} | {setting} | {demo} | {es_str} | {fu} | {tier} |")
        out.append('\n'.join(rows) + '\n')
    else:
        out.append("*No studies with full extraction data available.*\n")

    out.append("\n### 4.2 Effect Size Analysis\n")
    math_report = pd.get("math_report", "")
    if math_report:
        out.append(f"{math_report}\n")
    else:
        out.append("*No effect sizes calculated.*\n")

    # --- Discussion ---
    out.append("\n## 5. Discussion\n")
    out.append("### 5.1 Affirmative Case\n")
    if aff_case_text:
        out.append(f"{aff_case_text}\n")
    out.append("\n### 5.2 Falsification Case\n")
    if fal_case_text:
        out.append(f"{fal_case_text}\n")

    out.append("\n### 5.3 Evidence Quality Synthesis\n")
    if audit_text:
        out.append(f"{audit_text}\n")

    # --- References ---
    out.append("\n## 6. References\n")
    if all_extractions:
        seen = set()
        idx = 0
        for ext in all_extractions:
            key = getattr(ext, 'doi', None) or getattr(ext, 'title', '')
            if key in seen:
                continue
            seen.add(key)
            idx += 1
            title = getattr(ext, 'title', 'Untitled') or 'Untitled'
            doi = getattr(ext, 'doi', None)
            parts = [f"{idx}. *{title}*."]
            if doi:
                parts.append(f"DOI: {doi}.")
            url = getattr(ext, 'url', None)
            if url:
                parts.append(f"URL: {url}")
            out.append(" ".join(parts))
        out.append("")
    else:
        out.append("*No references available.*\n")

    return '\n'.join(out)

def build_imrad_sot(
    topic: str,
    reports: dict,
    ev_quality: str,
    aff_cand: int,
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

    _empty_rpt = type('_E', (), {'report': '', 'total_summaries': 0, 'total_urls_fetched': 0, 'duration_seconds': 0, 'sources': []})()
    audit_text = strip_think_blocks(reports.get("audit", _empty_rpt).report)
    aff_case_text = strip_think_blocks(reports.get("lead", _empty_rpt).report)
    fal_case_text = strip_think_blocks(reports.get("counter", _empty_rpt).report)

    # Dispatch to domain-specific SOT builder
    if domain == "social_science":
        return _build_social_science_sot(
            topic, pd, audit_text, aff_case_text, fal_case_text,
            ev_quality, aff_cand, all_extractions, all_wide, impacts, metrics,
            framing, search_date, aff_strategy, fal_strategy,
        )

    grade_level, conclusion_status, exec_summary = _extract_conclusion_status(
        audit_text, language=language)
    grade_sections = _parse_grade_sections(audit_text)

    m = metrics
    aff_wide = m.get("aff_wide_net_total", 0)
    fal_wide = m.get("fal_wide_net_total", 0)
    aff_screened = m.get("aff_screened_in", 0)
    fal_screened = m.get("fal_screened_in", 0)
    aff_ft_ok = m.get("aff_fulltext_ok", 0)
    fal_ft_ok = m.get("fal_fulltext_ok", 0)
    aff_ft_err = m.get("aff_fulltext_err", 0)
    fal_ft_err = m.get("fal_fulltext_err", 0)
    total_wide = aff_wide + fal_wide
    total_screened = aff_screened + fal_screened
    total_ft_ok = aff_ft_ok + fal_ft_ok
    total_ft_err = aff_ft_err + fal_ft_err

    # Summarize PICO for abstract
    pico_summary = ""
    if aff_strategy and hasattr(aff_strategy, 'pico'):
        p = aff_strategy.pico
        pico_summary = tmpl["pico_summary_template"].format(
            population=p.get('population', 'N/A'),
            intervention=p.get('intervention', 'N/A'),
            comparison=p.get('comparison', 'N/A'),
            outcome=p.get('outcome', 'N/A'))

    # Determine representative NNT for abstract
    nnt_summary = ""
    if impacts:
        benefit = [i for i in impacts if i.direction == "benefit"]
        ref_impact = benefit[0] if benefit else impacts[0]
        nnt_summary = tmpl["nnt_summary_template"].format(
            nnt=ref_impact.nnt, direction=ref_impact.direction, arr=ref_impact.arr)

    track_labels = tmpl["track_labels"]

    # -- ABSTRACT --
    out = [t(tmpl, "title", "prefix", topic=topic)]
    out.append(t(tmpl, "abstract", "header"))
    if pico_summary:
        out.append(t(tmpl, "abstract", "pico_label", pico_summary=pico_summary))
    out.append(t(tmpl, "abstract", "methods",
                 total_wide=total_wide, total_screened=total_screened,
                 total_ft_ok=total_ft_ok))
    if nnt_summary:
        out.append(t(tmpl, "abstract", "key_finding", nnt_summary=nnt_summary))
    out.append(t(tmpl, "abstract", "evidence_quality",
                 grade_level=grade_level, conclusion_status=conclusion_status))
    if exec_summary:
        out.append(f"\n{exec_summary}\n")

    # -- 1. INTRODUCTION --
    out.append(t(tmpl, "introduction", "header"))
    if framing:
        out.append(framing.strip() + "\n")
    else:
        out.append(t(tmpl, "introduction", "default_framing", topic=topic))
    out.append(t(tmpl, "introduction", "dual_hypothesis"))
    if aff_strategy and hasattr(aff_strategy, 'pico'):
        p = aff_strategy.pico
        out.append(t(tmpl, "introduction", "aff_hypothesis",
                     population=p.get('population', 'the target population'),
                     intervention=p.get('intervention', 'the intervention'),
                     outcome=p.get('outcome', 'the primary outcome'),
                     comparison=p.get('comparison', 'control')))
    if fal_strategy and hasattr(fal_strategy, 'pico'):
        fp = fal_strategy.pico
        out.append(t(tmpl, "introduction", "fal_hypothesis",
                     intervention=fp.get('intervention', 'the intervention'),
                     outcome=fp.get('outcome', 'the primary outcome'),
                     population=fp.get('population', 'the target population')))

    # -- 2. METHODS --
    out.append(t(tmpl, "methods", "header"))

    # 2.1 Search Strategy
    out.append(t(tmpl, "methods", "search_strategy_header"))
    for label_key, strategy in [("affirmative", aff_strategy), ("falsification", fal_strategy)]:
        if not strategy or not hasattr(strategy, 'pico'):
            continue
        label = track_labels[label_key]
        out.append(t(tmpl, "methods", "track_header", label=label))
        p = strategy.pico
        out.append(t(tmpl, "methods", "pico_framework",
                     population=p.get('population', 'N/A'),
                     intervention=p.get('intervention', 'N/A'),
                     comparison=p.get('comparison', 'N/A'),
                     outcome=p.get('outcome', 'N/A')))
        # Tiered keyword plan (new architecture)
        if hasattr(strategy, 'tier1'):
            tier_label_list = tmpl["methods"]["tier_labels"]
            tier_map = [
                (tier_label_list[0], strategy.tier1),
                (tier_label_list[1], strategy.tier2),
                (tier_label_list[2], strategy.tier3),
            ]
            out.append(t(tmpl, "methods", "three_tier_header"))
            for tier_label, tier_kw in tier_map:
                if hasattr(tier_kw, 'intervention') and tier_kw.intervention:
                    out.append(f"\n*{tier_label}*\n")
                    out.append(t(tmpl, "methods", "intervention_label",
                                 terms=', '.join(tier_kw.intervention)))
                    out.append(t(tmpl, "methods", "outcome_label",
                                 terms=', '.join(tier_kw.outcome)))
                    if tier_kw.population:
                        out.append(t(tmpl, "methods", "population_label",
                                     terms=', '.join(tier_kw.population)))
                    out.append(t(tmpl, "methods", "rationale_label",
                                 rationale=tier_kw.rationale))
            if strategy.auditor_approved:
                out.append(t(tmpl, "methods", "auditor_approved",
                             revision_count=strategy.revision_count))
            else:
                out.append(t(tmpl, "methods", "auditor_not_approved",
                             notes=strategy.auditor_notes[:200]))
        # Legacy: Boolean search strings (old architecture -- kept for backward compat)
        elif hasattr(strategy, 'mesh_terms') and strategy.mesh_terms:
            mt = strategy.mesh_terms
            out.append(t(tmpl, "methods", "mesh_terms_header"))
            for cat, terms in mt.items():
                if terms:
                    out.append(f"- *{cat.capitalize()}*: {', '.join(terms)}\n")
        if hasattr(strategy, 'search_strings') and strategy.search_strings:
            ss = strategy.search_strings
            out.append(t(tmpl, "methods", "boolean_search_header"))
            for db, query in ss.items():
                if query:
                    out.append(f"- **{db.replace('_', ' ').title()}**: `{query}`\n")
        out.append("\n")

    # 2.2 Data Collection
    out.append(t(tmpl, "methods", "data_collection_header"))
    aff_tier = pd.get("aff_highest_tier", 1)
    fal_tier = pd.get("fal_highest_tier", 1)
    tier_cascade = tmpl["methods"]["tier_cascade_labels"]
    out.append(t(tmpl, "methods", "data_collection_body",
                 search_date=search_date,
                 aff_tier_label=tier_cascade.get(aff_tier, str(aff_tier)),
                 fal_tier_label=tier_cascade.get(fal_tier, str(fal_tier))))
    if aff_tier == 3 or fal_tier == 3:
        out.append(t(tmpl, "methods", "tier3_warning"))
    out.append(t(tmpl, "methods", "track_records",
                 aff_wide=aff_wide, fal_wide=fal_wide, total_wide=total_wide))

    # 2.3 Screening & Selection
    out.append(t(tmpl, "methods", "screening_header"))
    out.append(t(tmpl, "methods", "screening_body",
                 aff_screened=aff_screened, fal_screened=fal_screened,
                 total_screened=total_screened))

    # 2.4 Data Extraction
    out.append(t(tmpl, "methods", "extraction_header"))
    out.append(t(tmpl, "methods", "extraction_body",
                 aff_ft_ok=aff_ft_ok, aff_ft_err=aff_ft_err,
                 fal_ft_ok=fal_ft_ok, fal_ft_err=fal_ft_err,
                 total_ft_ok=total_ft_ok))

    # 2.5 Statistical Analysis
    out.append(t(tmpl, "methods", "stats_header"))
    out.append(t(tmpl, "methods", "stats_body"))

    # -- 3. RESULTS --
    out.append(t(tmpl, "results", "header"))

    # 3.1 Study Selection (PRISMA)
    out.append(t(tmpl, "results", "study_selection_header"))
    prisma_from_grade = grade_sections.get("prisma flow diagram", "")
    prisma_rows = tmpl["results"]["prisma_rows"]
    out.append(
        t(tmpl, "results", "prisma_label")
        + tmpl["results"]["prisma_table_header"]
        + prisma_rows["identified"].format(aff=aff_wide, fal=fal_wide, total=total_wide)
        + prisma_rows["screened"].format(aff=aff_screened, fal=fal_screened, total=total_screened)
        + prisma_rows["fulltext"].format(aff=aff_ft_ok, fal=fal_ft_ok, total=total_ft_ok)
        + prisma_rows["errors"].format(aff=aff_ft_err, fal=fal_ft_err, total=total_ft_err)
        + prisma_rows["included"].format(aff=len(aff_extractions), fal=len(fal_extractions),
                                          total=len(all_extractions))
    )
    if prisma_from_grade:
        out.append(f"\n{prisma_from_grade}\n")

    # 3.2 Study Characteristics
    out.append(t(tmpl, "results", "study_chars_header"))
    out.append(_format_study_characteristics_table(all_extractions))

    # 3.3 Clinical Impact
    out.append(t(tmpl, "results", "clinical_impact_header"))
    # Try to read clinical_math.md from output directory
    _math_content = None
    if output_dir is not None and output_path_fn is not None:
        math_file_path = output_path_fn(output_dir, "clinical_math.md")
        if math_file_path.exists():
            _math_content = math_file_path.read_text().strip()
    if _math_content:
        out.append(_math_content + "\n")
    elif impacts:
        rows = [tmpl["results"]["impact_table_header"]]
        for i in impacts:
            rows.append(
                f"| {i.study_id} | {i.cer:.3f} | {i.eer:.3f} | "
                f"{i.arr:+.4f} | {i.rrr:+.2%} | {i.nnt:.1f} | {i.direction} |"
            )
        out.append('\n'.join(rows) + "\n\n")
        for i in impacts:
            out.append(f"- **{i.study_id}**: {i.nnt_interpretation}\n")
    else:
        out.append(t(tmpl, "results", "no_impact_data"))

    # -- 4. DISCUSSION --
    out.append(t(tmpl, "discussion", "header"))

    # 4.1 Affirmative Case
    out.append(t(tmpl, "discussion", "aff_case_header"))
    out.append(aff_case_text.strip() + "\n")

    # 4.2 Falsification Case
    out.append(t(tmpl, "discussion", "fal_case_header"))
    out.append(fal_case_text.strip() + "\n")

    # 4.3 GRADE Evidence Assessment
    out.append(t(tmpl, "discussion", "grade_header"))
    ep = grade_sections.get("evidence profile", "")
    ga = grade_sections.get("grade assessment", "")
    if ep:
        out.append(t(tmpl, "discussion", "evidence_profile_label", text=ep))
    if ga:
        out.append(t(tmpl, "discussion", "grade_assessment_label", text=ga))
    if not ep and not ga:
        # Fallback: include the full audit text minus already-extracted sections
        out.append(audit_text.strip() + "\n")

    # 4.4 Balanced Verdict
    out.append(t(tmpl, "discussion", "verdict_header"))
    bv = grade_sections.get("balanced verdict", "")
    if bv:
        out.append(bv + "\n")
    else:
        out.append(t(tmpl, "discussion", "verdict_fallback",
                     grade_level=grade_level, conclusion_status=conclusion_status))

    # 4.5 Limitations
    out.append(t(tmpl, "discussion", "limitations_header"))
    out.append(t(tmpl, "discussion", "limitations_body"))

    # 4.6 Recommendations
    out.append(t(tmpl, "discussion", "recs_header"))
    recs = grade_sections.get("recommendations for further research", "")
    if recs:
        out.append(recs + "\n")
    else:
        out.append(t(tmpl, "discussion", "recs_fallback"))

    # -- 5. REFERENCES --
    out.append(t(tmpl, "references", "header"))
    out.append(_format_references(all_extractions, all_wide))

    return '\n'.join(out)
