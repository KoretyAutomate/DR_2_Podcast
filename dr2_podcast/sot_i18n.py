"""
Internationalized templates for Source-of-Truth (IMRaD) documents.

Provides EN/JA template dictionaries with {variable} placeholders for
.format() interpolation. Eliminates LLM translation errors by using
pre-translated boilerplate for Japanese SoT output.

Key JA translations (correcting known LLM errors):
  - "Falsification" → "反証" (NOT 偽証 = perjury)
  - "confounders" → "交絡因子" (NOT 混雑因子 = congestion)
  - "screening" → "スクリーニング" (NOT 篩査 = Chinese)
  - "Affirmative Track" → "肯定的トラック" (NOT トレック = trek)
  - "randomized controlled trials" → "無作為化比較試験" (NOT 随機化 = Chinese)
"""

SOT_TEMPLATES = {
    "en": {
        "title": {
            "prefix": "# Source of Truth: {topic}\n",
        },
        "abstract": {
            "header": "## Abstract\n",
            "pico_label": "**Clinical Question (PICO):**  \n{pico_summary}\n",
            "methods": (
                "**Methods:** Dual-track systematic review using parallel Affirmative and "
                "Falsification search strategies. A total of {total_wide} records were identified "
                "across PubMed and Google Scholar; {total_screened} were screened to top candidates "
                "per track; {total_ft_ok} full-text articles were retrieved and deeply extracted. "
                "Absolute Risk Reduction (ARR) and Number Needed to Treat (NNT) were calculated "
                "deterministically in Python (no LLM). Evidence quality was assessed using the "
                "GRADE framework.\n"
            ),
            "key_finding": "**Key Finding:** {nnt_summary}\n",
            "evidence_quality": (
                "**Evidence Quality (GRADE):** {grade_level}  \n"
                "**Conclusion:** {conclusion_status}\n"
            ),
        },
        "introduction": {
            "header": "\n---\n\n## 1. Introduction\n",
            "default_framing": (
                "This systematic review was initiated to evaluate the scientific evidence "
                "for and against the following research question: **{topic}**\n"
            ),
            "dual_hypothesis": (
                "\nThis review employs a dual-hypothesis design. Two parallel search tracks "
                "were run simultaneously:\n\n"
                "- **Affirmative Track**: Seeks evidence supporting the hypothesis.\n"
                "- **Falsification Track**: Adversarially seeks evidence of null results, harms, "
                "methodological flaws, and confounders.\n"
            ),
            "aff_hypothesis": (
                "\n**Affirmative Hypothesis:** In {population}, "
                "does {intervention} improve "
                "{outcome} compared to "
                "{comparison}?\n"
            ),
            "fal_hypothesis": (
                "\n**Falsification Hypothesis:** Does {intervention} "
                "fail to improve, or actively harm, {outcome} "
                "in {population}?\n"
            ),
        },
        "methods": {
            "header": "\n---\n\n## 2. Methods\n",
            "search_strategy_header": "### 2.1 Search Strategy\n",
            "track_header": "#### {label} Track\n",
            "pico_framework": (
                "**PICO Framework:**  \n"
                "- **P** (Population): {population}  \n"
                "- **I** (Intervention): {intervention}  \n"
                "- **C** (Comparison): {comparison}  \n"
                "- **O** (Outcome): {outcome}\n"
            ),
            "three_tier_header": "\n**Three-Tier Keyword Plan:**\n",
            "tier_labels": [
                "Tier 1 \u2014 Established evidence (exact folk terms)",
                "Tier 2 \u2014 Supporting evidence (canonical synonyms)",
                "Tier 3 \u2014 Speculative extrapolation (compound class)",
            ],
            "intervention_label": "- Intervention: {terms}\n",
            "outcome_label": "- Outcome: {terms}\n",
            "population_label": "- Population: {terms}\n",
            "rationale_label": "- *Rationale: {rationale}*\n",
            "auditor_approved": "\n\u2705 *Auditor approved after {revision_count} revision(s).*\n",
            "auditor_not_approved": "\n\u26a0 *Auditor not approved (proceeded after max revisions). Notes: {notes}*\n",
            "mesh_terms_header": "\n**MeSH Terms:**\n",
            "boolean_search_header": "\n**Boolean Search Strings:**\n",
            "data_collection_header": "### 2.2 Data Collection\n",
            "tier_cascade_labels": {
                1: "Tier 1 (established \u2014 exact folk terms)",
                2: "Tier 2 (supporting \u2014 canonical synonyms)",
                3: "Tier 3 (speculative \u2014 compound class)",
            },
            "data_collection_body": (
                "- **Databases searched:** PubMed (NCBI E-utilities), Google Scholar (via SearXNG)\n"
                "- **Search date:** {search_date}\n"
                "- **Search architecture:** Three-tier cascading keyword search. "
                "Tier 1 runs first (exact folk terms). If pool < 50 records, Tier 2 runs "
                "(canonical synonyms). If still < 50, Tier 3 runs (compound class/mechanism \u2014 "
                "results require inference and are flagged as speculative extrapolation).\n"
                "- **Affirmative track cascade reached:** {aff_tier_label}\n"
                "- **Falsification track cascade reached:** {fal_tier_label}\n"
            ),
            "tier3_warning": (
                "- \u26a0 **Note:** One or both tracks reached Tier 3. Tier 3 evidence involves the "
                "active compound class (e.g., caffeine from any source, not coffee specifically). "
                "These results require an inference step to apply to the original substance and "
                "are presented as speculative extrapolation in this review.\n"
            ),
            "track_records": (
                "- **Affirmative track records identified:** {aff_wide}\n"
                "- **Falsification track records identified:** {fal_wide}\n"
                "- **Total records identified:** {total_wide}\n"
            ),
            "screening_header": "\n### 2.3 Screening & Selection\n",
            "screening_body": (
                "Title and abstract screening was performed by the Smart Model (Qwen3-32B-AWQ) "
                "using structured inclusion/exclusion criteria:\n\n"
                "**Inclusion criteria:** Human clinical studies (RCTs, meta-analyses, systematic reviews, "
                "large cohort studies); sample size \u2265 30 participants; published in peer-reviewed journals; "
                "directly relevant to the PICO question.\n\n"
                "**Exclusion criteria:** Animal models; in vitro studies; case reports (n < 5); "
                "conference abstracts without full data; non-English publications; retracted publications.\n\n"
                "- **Affirmative track screened to top candidates:** {aff_screened}\n"
                "- **Falsification track screened to top candidates:** {fal_screened}\n"
                "- **Total articles selected for full-text retrieval:** {total_screened}\n"
            ),
            "extraction_header": "\n### 2.4 Data Extraction\n",
            "extraction_body": (
                "Full-text articles were retrieved using a 4-tier cascade:\n\n"
                "1. **PMC EFetch** (NCBI EUtils `elink` + `efetch`): Open-access full XML\n"
                "2. **Europe PMC REST API**: Full-text XML for OA articles\n"
                "3. **Unpaywall API**: Open-access PDF/HTML links via DOI\n"
                "4. **NCBI Abstract EFetch**: Official abstract XML (fallback for paywalled articles)\n\n"
                "- **Affirmative track full-text retrieved:** {aff_ft_ok} "
                "(errors: {aff_ft_err})\n"
                "- **Falsification track full-text retrieved:** {fal_ft_ok} "
                "(errors: {fal_ft_err})\n"
                "- **Total full texts successfully retrieved:** {total_ft_ok}\n\n"
                "Clinical variables were extracted from each full text by the Fast Model "
                "(llama3.2:1b) using a structured extraction template capturing: "
                "study design, sample sizes, demographics, follow-up period, "
                "Control Event Rate (CER), Experimental Event Rate (EER), "
                "effect size with confidence intervals, blinding, randomization method, "
                "intention-to-treat analysis, funding source, and risk of bias.\n"
            ),
            "stats_header": "\n### 2.5 Statistical Analysis\n",
            "stats_body": (
                "**Deterministic Clinical Math (Step 6):** ARR, RRR, and NNT were calculated "
                "using pure Python arithmetic from extracted CER and EER values \u2014 no LLM involvement:\n\n"
                "- ARR (Absolute Risk Reduction) = CER \u2212 EER\n"
                "- RRR (Relative Risk Reduction) = ARR / CER\n"
                "- NNT (Number Needed to Treat) = 1 / |ARR|\n\n"
                "**GRADE Framework (Step 7):** Evidence quality was assessed by the Smart Model "
                "using the Grading of Recommendations, Assessment, Development, and Evaluations "
                "(GRADE) framework. Starting quality was HIGH for RCTs and LOW for observational "
                "studies, then adjusted for: risk of bias, inconsistency, indirectness, imprecision, "
                "and publication bias (downgrade factors); and large effect size, dose-response "
                "gradient, and plausible confounders (upgrade factors).\n"
            ),
        },
        "results": {
            "header": "\n---\n\n## 3. Results\n",
            "study_selection_header": "### 3.1 Study Selection\n",
            "prisma_label": "**PRISMA Flow:**\n\n",
            "prisma_table_header": (
                "| Stage | Affirmative | Falsification | Total |\n"
                "|-------|-------------|---------------|-------|\n"
            ),
            "prisma_rows": {
                "identified": "| Records identified | {aff} | {fal} | {total} |\n",
                "screened": "| Screened (top candidates) | {aff} | {fal} | {total} |\n",
                "fulltext": "| Full-text retrieved | {aff} | {fal} | {total} |\n",
                "errors": "| Full-text errors | {aff} | {fal} | {total} |\n",
                "included": "| Included in synthesis | {aff} | {fal} | {total} |\n",
            },
            "study_chars_header": "\n### 3.2 Study Characteristics\n",
            "clinical_impact_header": "\n### 3.3 Clinical Impact (Deterministic Math)\n",
            "impact_table_header": (
                "| Study | CER | EER | ARR | RRR | NNT | Direction |\n"
                "|-------|-----|-----|-----|-----|-----|-----------|"
            ),
            "no_impact_data": "*No studies provided both CER and EER \u2014 NNT calculation not available.*\n",
        },
        "discussion": {
            "header": "\n---\n\n## 4. Discussion\n",
            "aff_case_header": "### 4.1 Affirmative Case\n",
            "fal_case_header": "\n### 4.2 Falsification Case\n",
            "grade_header": "\n### 4.3 GRADE Evidence Assessment\n",
            "evidence_profile_label": "**Evidence Profile:**\n\n{text}\n",
            "grade_assessment_label": "\n**GRADE Assessment:**\n\n{text}\n",
            "verdict_header": "\n### 4.4 Balanced Verdict\n",
            "verdict_fallback": (
                "**Evidence Quality (GRADE):** {grade_level}  \n"
                "**Conclusion:** {conclusion_status}\n"
            ),
            "limitations_header": "\n### 4.5 Limitations\n",
            "limitations_body": (
                "The following pipeline-specific limitations apply to this synthesis:\n\n"
                "- The Fast Model (llama3.2:1b) may have misclassified study designs or "
                "misextracted clinical variables from complex full-text articles.\n"
                "- Articles not available via PMC, Europe PMC, or Unpaywall were reduced to "
                "abstract-level data, limiting extraction depth for paywalled literature.\n"
                "- The 500-result cap per track may exclude relevant studies not retrieved "
                "in the top results from PubMed or Google Scholar.\n"
                "- CER/EER extraction relies on the Fast Model correctly identifying "
                "event rates in text; studies reporting outcomes without explicit event "
                "rates could not be included in NNT calculations.\n"
                "- Non-English language publications were excluded.\n"
            ),
            "recs_header": "\n### 4.6 Recommendations for Further Research\n",
            "recs_fallback": (
                "- Conduct large-scale, long-term randomized controlled trials with "
                "rigorous methodologies to address identified evidence gaps.\n"
                "- Investigate outcomes in under-represented populations.\n"
                "- Address potential biases and ensure transparency in study design "
                "and reporting.\n"
            ),
        },
        "references": {
            "header": "\n---\n\n## 5. References\n",
        },
        "status_map": {
            "clinical": {
                "High": "Scientifically Supported",
                "Moderate": "Partially Supported \u2014 Further Research Recommended",
                "Low": "Insufficient Evidence \u2014 More Research Needed",
                "Very Low": "Not Supported by Current Evidence",
            },
            "social_science": {
                "STRONG": "Scientifically Supported",
                "MODERATE_STRONG": "Well Supported \u2014 Robust Evidence",
                "MODERATE": "Partially Supported \u2014 Further Research Recommended",
                "MODERATE_WEAK": "Tentatively Supported \u2014 More Rigorous Studies Needed",
                "WEAK": "Insufficient Evidence \u2014 More Research Needed",
                "VERY_WEAK": "Not Supported by Current Evidence",
            },
            "default_status": "Under Evaluation",
        },
        "fallbacks": {
            "no_studies": "*No studies with full extraction data available.*\n",
            "no_references": "*No references available.*\n",
        },
        "pico_labels": {
            "P": "Population",
            "I": "Intervention",
            "C": "Comparison",
            "O": "Outcome",
        },
        "track_labels": {
            "affirmative": "Affirmative",
            "falsification": "Falsification",
        },
        "nnt_summary_template": (
            "The primary quantitative finding is NNT = **{nnt:.1f}** "
            "({direction}; ARR = {arr:+.4f})."
        ),
        "pico_summary_template": (
            "**P** (Population): {population}  \n"
            "**I** (Intervention): {intervention}  \n"
            "**C** (Comparison): {comparison}  \n"
            "**O** (Outcome): {outcome}"
        ),
    },

    "ja": {
        "title": {
            "prefix": "# 情報源ドキュメント: {topic}\n",
        },
        "abstract": {
            "header": "## 要約\n",
            "pico_label": "**臨床的疑問 (PICO):**  \n{pico_summary}\n",
            "methods": (
                "**方法:** 肯定的トラックと反証トラックを並行して用いたデュアルトラック系統的レビュー。"
                "PubMedおよびGoogle Scholarから合計{total_wide}件のレコードが同定され、"
                "各トラックから{total_screened}件が候補としてスクリーニングされ、"
                "{total_ft_ok}件の全文論文が取得・詳細抽出された。"
                "絶対リスク減少率（ARR）および治療必要数（NNT）は"
                "Pythonによる決定論的計算で算出（LLM不使用）。"
                "エビデンスの質はGRADEフレームワークで評価された。\n"
            ),
            "key_finding": "**主要所見:** {nnt_summary}\n",
            "evidence_quality": (
                "**エビデンスの質 (GRADE):** {grade_level}  \n"
                "**結論:** {conclusion_status}\n"
            ),
        },
        "introduction": {
            "header": "\n---\n\n## 1. 序論\n",
            "default_framing": (
                "本系統的レビューは、以下の研究課題に関する科学的エビデンスを"
                "賛否両面から評価するために実施された: **{topic}**\n"
            ),
            "dual_hypothesis": (
                "\n本レビューはデュアル仮説デザインを採用している。"
                "2つの並行検索トラックが同時に実行された:\n\n"
                "- **肯定的トラック**: 仮説を支持するエビデンスを探索する。\n"
                "- **反証トラック**: 帰無結果、有害性、方法論的欠陥、"
                "および交絡因子のエビデンスを対抗的に探索する。\n"
            ),
            "aff_hypothesis": (
                "\n**肯定的仮説:** {population}において、"
                "{intervention}は{comparison}と比較して"
                "{outcome}を改善するか？\n"
            ),
            "fal_hypothesis": (
                "\n**反証仮説:** {intervention}は"
                "{population}における{outcome}を"
                "改善できないか、または積極的に害するか？\n"
            ),
        },
        "methods": {
            "header": "\n---\n\n## 2. 方法\n",
            "search_strategy_header": "### 2.1 検索戦略\n",
            "track_header": "#### {label}トラック\n",
            "pico_framework": (
                "**PICOフレームワーク:**  \n"
                "- **P** (対象集団): {population}  \n"
                "- **I** (介入): {intervention}  \n"
                "- **C** (比較対照): {comparison}  \n"
                "- **O** (アウトカム): {outcome}\n"
            ),
            "three_tier_header": "\n**3段階キーワード計画:**\n",
            "tier_labels": [
                "Tier 1 \u2014 確立されたエビデンス（正確な一般用語）",
                "Tier 2 \u2014 補足的エビデンス（標準的な同義語）",
                "Tier 3 \u2014 推論的外挿（化合物クラス）",
            ],
            "intervention_label": "- 介入: {terms}\n",
            "outcome_label": "- アウトカム: {terms}\n",
            "population_label": "- 対象集団: {terms}\n",
            "rationale_label": "- *根拠: {rationale}*\n",
            "auditor_approved": "\n\u2705 *監査者が{revision_count}回の修正後に承認。*\n",
            "auditor_not_approved": "\n\u26a0 *監査者未承認（最大修正回数到達後に続行）。備考: {notes}*\n",
            "mesh_terms_header": "\n**MeSH用語:**\n",
            "boolean_search_header": "\n**ブール検索文字列:**\n",
            "data_collection_header": "### 2.2 データ収集\n",
            "tier_cascade_labels": {
                1: "Tier 1（確立 \u2014 正確な一般用語）",
                2: "Tier 2（補足 \u2014 標準的な同義語）",
                3: "Tier 3（推論 \u2014 化合物クラス）",
            },
            "data_collection_body": (
                "- **検索データベース:** PubMed (NCBI E-utilities)、Google Scholar (SearXNG経由)\n"
                "- **検索日:** {search_date}\n"
                "- **検索アーキテクチャ:** 3段階カスケードキーワード検索。"
                "Tier 1が最初に実行される（正確な一般用語）。プールが50件未満の場合、Tier 2が実行される"
                "（標準的な同義語）。それでも50件未満の場合、Tier 3が実行される（化合物クラス/メカニズム — "
                "結果は推論を必要とし、推論的外挿としてフラグされる）。\n"
                "- **肯定的トラックのカスケード到達レベル:** {aff_tier_label}\n"
                "- **反証トラックのカスケード到達レベル:** {fal_tier_label}\n"
            ),
            "tier3_warning": (
                "- \u26a0 **注意:** 一方または両方のトラックがTier 3に到達した。"
                "Tier 3のエビデンスは活性化合物クラス（例：コーヒーに限らず、あらゆる供給源からの"
                "カフェイン）に関連する。これらの結果は元の物質に適用するために推論ステップが必要であり、"
                "本レビューでは推論的外挿として提示される。\n"
            ),
            "track_records": (
                "- **肯定的トラック同定レコード数:** {aff_wide}\n"
                "- **反証トラック同定レコード数:** {fal_wide}\n"
                "- **総同定レコード数:** {total_wide}\n"
            ),
            "screening_header": "\n### 2.3 スクリーニングと選択\n",
            "screening_body": (
                "タイトルおよび抄録のスクリーニングは、Smart Model (Qwen3-32B-AWQ)により"
                "構造化された組み入れ/除外基準を用いて実施された:\n\n"
                "**組み入れ基準:** ヒト臨床研究（無作為化比較試験、メタアナリシス、系統的レビュー、"
                "大規模コホート研究）；サンプルサイズ≧30人；査読付きジャーナルに掲載；"
                "PICO課題に直接関連。\n\n"
                "**除外基準:** 動物モデル；in vitro研究；症例報告（n < 5）；"
                "完全なデータのない学会抄録；英語以外の出版物；撤回された出版物。\n\n"
                "- **肯定的トラック候補スクリーニング数:** {aff_screened}\n"
                "- **反証トラック候補スクリーニング数:** {fal_screened}\n"
                "- **全文取得対象として選択された総論文数:** {total_screened}\n"
            ),
            "extraction_header": "\n### 2.4 データ抽出\n",
            "extraction_body": (
                "全文論文は4段階カスケードにより取得された:\n\n"
                "1. **PMC EFetch** (NCBI EUtils `elink` + `efetch`): オープンアクセス全文XML\n"
                "2. **Europe PMC REST API**: OA論文の全文XML\n"
                "3. **Unpaywall API**: DOI経由のオープンアクセスPDF/HTMLリンク\n"
                "4. **NCBI Abstract EFetch**: 公式抄録XML（有料壁論文のフォールバック）\n\n"
                "- **肯定的トラック全文取得数:** {aff_ft_ok} "
                "(エラー: {aff_ft_err})\n"
                "- **反証トラック全文取得数:** {fal_ft_ok} "
                "(エラー: {fal_ft_err})\n"
                "- **全文取得成功総数:** {total_ft_ok}\n\n"
                "臨床変数は、Fast Model (llama3.2:1b)により構造化された抽出テンプレートを用いて"
                "各全文から抽出された。抽出項目: 研究デザイン、サンプルサイズ、人口統計学的特性、"
                "追跡期間、対照群イベント率（CER）、実験群イベント率（EER）、"
                "効果量と信頼区間、盲検化、無作為化方法、"
                "intent-to-treat分析、資金源、およびバイアスリスク。\n"
            ),
            "stats_header": "\n### 2.5 統計解析\n",
            "stats_body": (
                "**決定論的臨床数学（ステップ6）:** ARR、RRR、およびNNTは、"
                "抽出されたCERおよびEER値からPythonの純粋な算術演算で計算された — LLMの関与なし:\n\n"
                "- ARR（絶対リスク減少率）= CER \u2212 EER\n"
                "- RRR（相対リスク減少率）= ARR / CER\n"
                "- NNT（治療必要数）= 1 / |ARR|\n\n"
                "**GRADEフレームワーク（ステップ7）:** エビデンスの質はSmart Modelにより"
                "GRADE（Grading of Recommendations, Assessment, Development, and Evaluations）"
                "フレームワークを用いて評価された。初期品質は無作為化比較試験ではHIGH、"
                "観察研究ではLOWとし、以下で調整された: バイアスリスク、非一貫性、非直接性、不精確さ、"
                "および出版バイアス（グレードダウン因子）；大きな効果量、用量反応勾配、"
                "および妥当な交絡因子（グレードアップ因子）。\n"
            ),
        },
        "results": {
            "header": "\n---\n\n## 3. 結果\n",
            "study_selection_header": "### 3.1 研究選択\n",
            "prisma_label": "**PRISMAフロー:**\n\n",
            "prisma_table_header": (
                "| 段階 | 肯定的トラック | 反証トラック | 合計 |\n"
                "|------|----------------|--------------|------|\n"
            ),
            "prisma_rows": {
                "identified": "| 同定レコード数 | {aff} | {fal} | {total} |\n",
                "screened": "| スクリーニング済み（候補） | {aff} | {fal} | {total} |\n",
                "fulltext": "| 全文取得 | {aff} | {fal} | {total} |\n",
                "errors": "| 全文エラー | {aff} | {fal} | {total} |\n",
                "included": "| 統合に組み入れ | {aff} | {fal} | {total} |\n",
            },
            "study_chars_header": "\n### 3.2 研究特性\n",
            "clinical_impact_header": "\n### 3.3 臨床的インパクト（決定論的数学）\n",
            "impact_table_header": (
                "| 研究 | CER | EER | ARR | RRR | NNT | 方向性 |\n"
                "|------|-----|-----|-----|-----|-----|--------|"
            ),
            "no_impact_data": "*CERとEERの両方を提供した研究がなく、NNT計算は利用できません。*\n",
        },
        "discussion": {
            "header": "\n---\n\n## 4. 考察\n",
            "aff_case_header": "### 4.1 肯定的ケース\n",
            "fal_case_header": "\n### 4.2 反証ケース\n",
            "grade_header": "\n### 4.3 GRADEエビデンス評価\n",
            "evidence_profile_label": "**エビデンスプロファイル:**\n\n{text}\n",
            "grade_assessment_label": "\n**GRADE評価:**\n\n{text}\n",
            "verdict_header": "\n### 4.4 総合的判定\n",
            "verdict_fallback": (
                "**エビデンスの質 (GRADE):** {grade_level}  \n"
                "**結論:** {conclusion_status}\n"
            ),
            "limitations_header": "\n### 4.5 限界\n",
            "limitations_body": (
                "本統合には以下のパイプライン固有の限界が適用される:\n\n"
                "- Fast Model (llama3.2:1b)が複雑な全文論文から研究デザインを"
                "誤分類または臨床変数を誤抽出した可能性がある。\n"
                "- PMC、Europe PMC、またはUnpaywallで利用できない論文は"
                "抄録レベルのデータに限定され、有料壁文献の抽出深度が制限された。\n"
                "- トラックごとの500件上限により、PubMedまたはGoogle Scholarの"
                "上位結果に含まれない関連研究が除外された可能性がある。\n"
                "- CER/EER抽出はFast Modelがテキスト内のイベント率を正しく"
                "同定することに依存しており、明示的なイベント率なしにアウトカムを報告した"
                "研究はNNT計算に含めることができなかった。\n"
                "- 英語以外の出版物は除外された。\n"
            ),
            "recs_header": "\n### 4.6 今後の研究への提言\n",
            "recs_fallback": (
                "- 同定されたエビデンスギャップに対処するため、厳密な方法論による"
                "大規模・長期の無作為化比較試験を実施する。\n"
                "- 十分に代表されていない集団におけるアウトカムを調査する。\n"
                "- 潜在的なバイアスに対処し、研究デザインと報告の透明性を確保する。\n"
            ),
        },
        "references": {
            "header": "\n---\n\n## 5. 参考文献\n",
        },
        "status_map": {
            "clinical": {
                "High": "科学的に支持されている",
                "Moderate": "部分的に支持 — さらなる研究が推奨される",
                "Low": "エビデンス不十分 — さらなる研究が必要",
                "Very Low": "現在のエビデンスでは支持されていない",
            },
            "social_science": {
                "STRONG": "科学的に支持されている",
                "MODERATE_STRONG": "十分に支持 — 堅固なエビデンス",
                "MODERATE": "部分的に支持 — さらなる研究が推奨される",
                "MODERATE_WEAK": "暫定的に支持 — より厳密な研究が必要",
                "WEAK": "エビデンス不十分 — さらなる研究が必要",
                "VERY_WEAK": "現在のエビデンスでは支持されていない",
            },
            "default_status": "評価中",
        },
        "fallbacks": {
            "no_studies": "*全文抽出データを持つ研究はありません。*\n",
            "no_references": "*利用可能な参考文献はありません。*\n",
        },
        "pico_labels": {
            "P": "対象集団",
            "I": "介入",
            "C": "比較対照",
            "O": "アウトカム",
        },
        "track_labels": {
            "affirmative": "肯定的",
            "falsification": "反証",
        },
        "nnt_summary_template": (
            "主要な定量的所見はNNT = **{nnt:.1f}** "
            "（{direction}; ARR = {arr:+.4f}）。"
        ),
        "pico_summary_template": (
            "**P** (対象集団): {population}  \n"
            "**I** (介入): {intervention}  \n"
            "**C** (比較対照): {comparison}  \n"
            "**O** (アウトカム): {outcome}"
        ),
    },
}


def get_templates(language: str, domain: str = "clinical") -> dict:
    """Return template dict for the given language. Falls back to 'en' for unknown languages."""
    return SOT_TEMPLATES.get(language, SOT_TEMPLATES["en"])


def t(templates: dict, section: str, key: str, **kwargs) -> str:
    """Look up a template string and interpolate placeholders.

    Usage: t(tmpl, "abstract", "header")
           t(tmpl, "abstract", "methods", total_wide=100, total_screened=20, total_ft_ok=10)
    """
    tmpl_str = templates[section][key]
    if kwargs:
        return tmpl_str.format(**kwargs)
    return tmpl_str
