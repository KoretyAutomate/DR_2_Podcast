"""Externalized LLM prompt strings — keyed by (section, language).

Structural labels stay in English for parser compatibility.
Descriptive text and examples are localized.

Created as part of Episode Blueprint & Script Inflation restructure (2026-03-04).
Only prompts being actively changed are externalized here; stable prompts remain
inline in their original modules.
"""

# ---------------------------------------------------------------------------
# BLUEPRINT PROMPTS — Section 5 (Narrative Arc with inline Discussion Points)
# ---------------------------------------------------------------------------

BLUEPRINT_PROMPTS: dict[str, dict[str, str]] = {

    "section5_intro": {
        "en": (
            "## 5. Narrative Arc (4 Acts) with Discussion Points\n\n"
            "For EVERY discussion point below, write the question as if asked by a\n"
            "{core_target_or_default} --- in their voice, from their life context.\n"
            "Write the answer as if the host is explaining directly to that person.\n"
        ),
        "ja": (
            "## 5. Narrative Arc (4 Acts) with Discussion Points\n\n"
            "以下のすべてのディスカッションポイントについて、\n"
            "{core_target_or_default}の視点から質問を書いてください。\n"
            "回答は、ホストがその人に直接説明するように書いてください。\n"
        ),
    },

    "act1_header": {
        "en": "### Act 1 --- The Claim\n",
        "ja": "### Act 1 --- The Claim\n",
    },

    "act1_description": {
        "en": (
            "What people believe. The folk wisdom or common assumption. "
            "Why this matters personally.\n"
            "Key points to cover: [3-4 bullets]\n"
        ),
        "ja": (
            "人々が信じていること。常識や思い込み。なぜそれが個人的に重要なのか。\n"
            "カバーすべきポイント: [3-4項目]\n"
        ),
    },

    "act1_discussion": {
        "en": (
            "**Discussion Points (generate 5-8):**\n"
            "- Q: [question from {core_target_or_default}'s perspective about this common belief]\n"
            "  A: [80-150 word answer with specific details from the research]\n"
            "[repeat for each point]\n"
        ),
        "ja": (
            "**Discussion Points (generate 5-8):**\n"
            "- Q: [{core_target_or_default}の視点からの、この常識に関する質問]\n"
            "  A: [80-150語の回答、研究からの具体的な詳細を含む]\n"
            "[各ポイントについて繰り返す]\n"
        ),
    },

    "act2_header": {
        "en": "### Act 2 --- Evidence & Nuance (above 50% of the episode)\n",
        "ja": "### Act 2 --- Evidence & Nuance (above 50% of the episode)\n",
    },

    "act2_description": {
        "en": (
            "**Structure:** Start by stating the overall conclusion upfront. Then walk through\n"
            "each study individually, interleaving its findings with its limitations and nuances.\n"
            "End with a bridge sentence leading to Act 3.\n\n"
            "For EACH study cited in the research, generate 1-2 questions that reference\n"
            "the study specifically. Questions must name the study or its key finding.\n"
        ),
        "ja": (
            "**構成:** まず全体的な結論を先に述べてください。その後、各研究を個別に\n"
            "取り上げ、その知見と限界・ニュアンスを交互に提示してください。\n"
            "Act 3へのブリッジ文で締めくくってください。\n\n"
            "研究で引用された各研究について、その研究を具体的に参照する\n"
            "1-2の質問を生成してください。質問にはその研究名や主要な知見を含めてください。\n"
        ),
    },

    "act2_bad_example": {
        "en": 'BAD: "Is there evidence that coffee improves focus?"',
        "ja": 'BAD: 「コーヒーが集中力を向上させるエビデンスはありますか？」',
    },

    "act2_good_example": {
        "en": (
            'GOOD: "The 2020 Journal of Occupational Health RCT found 100-200mg improves\n'
            'alertness --- but how does that translate to cups for a busy parent?"'
        ),
        "ja": (
            'GOOD: 「2020年のJournal of Occupational HealthのRCTでは100-200mgが覚醒度を\n'
            '向上させると報告されましたが、忙しい親にとってカップ数ではどうなりますか？」'
        ),
    },

    "act2_sub_structure": {
        "en": (
            "**Sub-structure per study:**\n"
            "  Opener: State the conclusion upfront\n"
            "  Study 1: [finding] -> [listener question] -> [nuance/limitation]\n"
            "  Study 2: [finding] -> [listener question] -> [nuance/limitation]\n"
            "  ...\n"
            '  Bridge: "So across all these studies, what\'s the takeaway?"\n\n'
            "Supporting evidence: [2-3 key studies with how to frame them]\n"
            "Contradicting evidence: [1-2 key studies]\n"
            "Key numbers to cite: [NNT, ARR, sample sizes if available]\n"
        ),
        "ja": (
            "**各研究ごとの構成:**\n"
            "  冒頭: 結論を先に述べる\n"
            "  研究1: [知見] -> [リスナーの質問] -> [ニュアンス/限界]\n"
            "  研究2: [知見] -> [リスナーの質問] -> [ニュアンス/限界]\n"
            "  ...\n"
            "  ブリッジ: 「これらすべての研究を通して、何が分かるのでしょうか？」\n\n"
            "支持するエビデンス: [2-3の主要研究とフレーミング方法]\n"
            "反するエビデンス: [1-2の主要研究]\n"
            "引用すべき数値: [NNT, ARR, サンプルサイズ（利用可能な場合）]\n"
        ),
    },

    "act2_discussion": {
        "en": (
            "**Discussion Points (generate 5-8, at least 1 per study):**\n"
            "- Q: [question referencing a specific study's finding, from {core_target_or_default}'s voice]\n"
            "  A: [80-150 word answer including the study's nuance/limitation]\n"
            "[repeat for each point]\n"
        ),
        "ja": (
            "**Discussion Points (generate 5-8, at least 1 per study):**\n"
            "- Q: [{core_target_or_default}の声で、特定の研究の知見を参照する質問]\n"
            "  A: [80-150語の回答、その研究のニュアンス/限界を含む]\n"
            "[各ポイントについて繰り返す]\n"
        ),
    },

    "act3_header": {
        "en": "### Act 3 --- Holistic Conclusion\n",
        "ja": "### Act 3 --- Holistic Conclusion\n",
    },

    "act3_description": {
        "en": (
            "Synthesize ALL evidence into a unified takeaway. This is NOT new evidence ---\n"
            "it connects the dots across all studies discussed in Act 2.\n"
            "- Restate the conclusion, now with the listener equipped with the evidence\n"
            "- What does the totality of evidence mean for {core_target_or_default}?\n"
            "- GRADE confidence level and what it means in plain language\n"
        ),
        "ja": (
            "すべてのエビデンスを統一的なテイクアウェイに統合してください。\n"
            "これは新しいエビデンスではなく、Act 2で議論したすべての研究の\n"
            "点と点をつなげるものです。\n"
            "- 結論を再度述べる（リスナーがエビデンスを理解した上で）\n"
            "- エビデンス全体が{core_target_or_default}にとって何を意味するか\n"
            "- GRADEの確信度とそれが平易な言葉で何を意味するか\n"
        ),
    },

    "act3_discussion": {
        "en": (
            "**Discussion Points (generate 3-5):**\n"
            "- Q: [synthesis question from {core_target_or_default}'s perspective]\n"
            "  A: [80-150 word answer connecting multiple studies]\n"
            "[repeat for each point]\n"
        ),
        "ja": (
            "**Discussion Points (generate 3-5):**\n"
            "- Q: [{core_target_or_default}の視点からの統合的な質問]\n"
            "  A: [80-150語の回答、複数の研究をつなげる]\n"
            "[各ポイントについて繰り返す]\n"
        ),
    },

    "act4_header": {
        "en": "### Act 4 --- The Protocol\n",
        "ja": "### Act 4 --- The Protocol\n",
    },

    "act4_description": {
        "en": (
            "Actionable translation to daily life.\n"
            "'One Action' for the ending: [specific, memorable, doable this week]\n\n"
            "Questions MUST focus on real-world barriers for {core_target_or_default}.\n"
            'Frame as: "I understand the science now, but I face this challenge..."\n'
        ),
        "ja": (
            "科学を日常生活に変換する実践的な内容。\n"
            "エンディングの「One Action」: [具体的で記憶に残る、今週できること]\n\n"
            "質問は{core_target_or_default}の現実の障壁に焦点を当てなければなりません。\n"
            "フレーミング: 「科学は理解しましたが、こんな課題があります...」\n"
        ),
    },

    "act4_bad_example": {
        "en": 'BAD: "What is the recommended daily intake?"',
        "ja": 'BAD: 「推奨される一日の摂取量は？」',
    },

    "act4_good_example": {
        "en": (
            'GOOD: "I get that I should stop caffeine by 3pm, but my toddler wakes me\n'
            'at 5am and I have a 2pm meeting --- how do I survive the afternoon?"'
        ),
        "ja": (
            'GOOD: 「午後3時までにカフェインをやめるべきだと分かりましたが、\n'
            '幼児が朝5時に起きて午後2時に会議があります --- 午後をどう乗り切ればいいですか？」'
        ),
    },

    "act4_discussion": {
        "en": (
            "**Discussion Points (generate 5-8):**\n"
            "- Q: [barrier/challenge from {core_target_or_default}'s daily life]\n"
            "  A: [80-150 word practical answer addressing the specific barrier]\n"
            "[repeat for each point]\n"
        ),
        "ja": (
            "**Discussion Points (generate 5-8):**\n"
            "- Q: [{core_target_or_default}の日常生活からの障壁/課題]\n"
            "  A: [80-150語の実践的な回答、具体的な障壁に対処する]\n"
            "[各ポイントについて繰り返す]\n"
        ),
    },
}


# ---------------------------------------------------------------------------
# SCRIPT PROMPTS — Act descriptions for the script_task
# ---------------------------------------------------------------------------

SCRIPT_PROMPTS: dict[str, dict[str, str]] = {

    "act1": {
        "en": (
            "  3. ACT 1 --- THE CLAIM:\n"
            "     What people believe. The folk wisdom. Why this matters personally.\n"
            "     - Presenter sets up the common belief or question\n"
            "     - Questioner validates: 'Right, I've heard that too' / 'That's what everyone says'\n"
            "     - Establish emotional stakes: why should the listener care?\n\n"
        ),
        "ja": (
            "  3. ACT 1 --- THE CLAIM:\n"
            "     人々が信じていること。常識。なぜ個人的に重要なのか。\n"
            "     - プレゼンターが一般的な信念や疑問を提示\n"
            "     - 質問者が共感: 「確かに、私もそう聞いていました」\n"
            "     - 感情的なステークスを確立: なぜリスナーが気にすべきか？\n\n"
        ),
    },

    "act2": {
        "en": (
            "  4. ACT 2 --- EVIDENCE & NUANCE (above 50%% of the episode --- at least {act2_min} {target_unit_plural}, expand freely):\n"
            "     Start by stating the episode's conclusion upfront.\n"
            "     Then walk through EACH study individually:\n"
            "     - Present the study's finding with GRADE-informed framing from the Blueprint\n"
            "     - Questioner asks a study-specific question (from Coverage Checklist)\n"
            "     - Presenter answers, including the study's limitations and nuances\n"
            "     - Include specific numbers (NNT, ARR, sample sizes) where available\n"
            "     For EACH study, the conversation should be: finding -> question -> answer with nuance.\n"
            "     Do NOT lump all evidence together then all nuance together. Interleave them.\n"
            '     End with: "So across all these studies, here\'s what we see..."\n\n'
        ),
        "ja": (
            "  4. ACT 2 --- EVIDENCE & NUANCE (above 50%% of the episode --- at least {act2_min} {target_unit_plural}, expand freely):\n"
            "     まずエピソードの結論を先に述べてください。\n"
            "     次に各研究を個別に取り上げます:\n"
            "     - Blueprintからの GRADE に基づくフレーミングで研究の知見を提示\n"
            "     - 質問者が研究固有の質問をする（Coverage Checklistから）\n"
            "     - プレゼンターがその研究の限界やニュアンスを含めて回答\n"
            "     - 利用可能な場合、具体的な数値（NNT, ARR, サンプルサイズ）を含める\n"
            "     各研究ごとに: 知見 -> 質問 -> ニュアンスを含む回答 の流れにしてください。\n"
            "     エビデンスをまとめて述べてからニュアンスをまとめるのではなく、交互に提示してください。\n"
            "     最後に: 「これらすべての研究を通して、見えてくることは...」\n\n"
        ),
    },

    "act3": {
        "en": (
            "  5. ACT 3 --- HOLISTIC CONCLUSION:\n"
            "     Synthesize what ALL the evidence means together.\n"
            "     - Restate the conclusion --- the listener is now equipped with the evidence\n"
            "     - What does the totality of evidence mean for {core_target_or_default}?\n"
            "     - GRADE confidence level in plain language\n"
            "     - Questioner: 'So if I had to summarize everything we just discussed...'\n"
            "     - Presenter ties it all together, building confidence in the conclusion\n\n"
        ),
        "ja": (
            "  5. ACT 3 --- HOLISTIC CONCLUSION:\n"
            "     すべてのエビデンスが合わせて何を意味するかを統合してください。\n"
            "     - 結論を再度述べる --- リスナーはエビデンスを理解した上で\n"
            "     - エビデンス全体が{core_target_or_default}にとって何を意味するか\n"
            "     - GRADEの確信度を平易な言葉で\n"
            "     - 質問者: 「では、今まで話したことをまとめると...」\n"
            "     - プレゼンターがすべてをまとめ、結論への信頼を構築\n\n"
        ),
    },

    "act4": {
        "en": (
            "  6. ACT 4 --- THE PROTOCOL:\n"
            "     Translate science into daily life, focusing on real-world barriers.\n"
            "     - Questioner voices listener challenges: 'I get the science, but...'\n"
            "     - Presenter addresses each barrier with practical solutions\n"
            "     - Specific, practical recommendations tailored to {core_target_or_default}\n"
            "     - 'In practical terms, this means...'\n"
            "     - Each barrier gets its own mini-conversation (challenge -> solution -> encouragement)\n\n"
        ),
        "ja": (
            "  6. ACT 4 --- THE PROTOCOL:\n"
            "     科学を日常生活に変換し、現実の障壁に焦点を当てる。\n"
            "     - 質問者がリスナーの課題を代弁: 「科学は分かりましたが...」\n"
            "     - プレゼンターが各障壁に実践的な解決策で対応\n"
            "     - {core_target_or_default}に合わせた具体的で実践的な推奨事項\n"
            "     - 「実際には、これは...を意味します」\n"
            "     - 各障壁ごとにミニ会話（課題 -> 解決策 -> 励まし）\n\n"
        ),
    },

    "length_note": {
        "en": (
            "  NOTE: Only Act 2 has a minimum length target (above 50%). Acts 1, 3, and 4\n"
            "  should be as long as needed to cover their discussion points fully.\n"
            "  The TOTAL script length of {target_script} {target_unit_plural} is the hard target.\n"
            "  The polish phase will rebalance if needed.\n\n"
        ),
        "ja": (
            "  注意: Act 2のみが最低長さ目標（50%以上）を持ちます。Acts 1, 3, 4は\n"
            "  ディスカッションポイントを完全にカバーするのに必要な長さにしてください。\n"
            "  合計スクリプト長{target_script} {target_unit_plural}がハードターゲットです。\n"
            "  ポリッシュフェーズで必要に応じてリバランスします。\n\n"
        ),
    },
}


# ---------------------------------------------------------------------------
# POLISH PROMPTS — Act names for the 8-part structure verification
# ---------------------------------------------------------------------------

POLISH_PROMPTS: dict[str, dict[str, str]] = {

    "structure_acts": {
        "en": (
            "  3. Act 1 --- The Claim\n"
            "  4. Act 2 --- Evidence & Nuance\n"
            "  5. Act 3 --- Holistic Conclusion\n"
            "  6. Act 4 --- The Protocol\n"
        ),
        "ja": (
            "  3. Act 1 --- The Claim\n"
            "  4. Act 2 --- Evidence & Nuance\n"
            "  5. Act 3 --- Holistic Conclusion\n"
            "  6. Act 4 --- The Protocol\n"
        ),
    },

    "expected_output_structure": {
        "en": (
            "8-part structure with [TRANSITION] markers between acts. "
            "Acts: Claim, Evidence & Nuance, Holistic Conclusion, Protocol. "
            "One Action ending present."
        ),
        "ja": (
            "8-part structure with [TRANSITION] markers between acts. "
            "Acts: Claim, Evidence & Nuance, Holistic Conclusion, Protocol. "
            "One Action ending present."
        ),
    },
}


# ---------------------------------------------------------------------------
# CONDENSE PROMPTS — Rewrite-based condensing instructions
# ---------------------------------------------------------------------------

CONDENSE_PROMPTS: dict[str, dict[str, str]] = {

    "system": {
        "en": (
            "You are condensing a two-host science podcast script about \"{topic_name}\" "
            "from {current_count} {length_unit} down to approximately {target_count} {length_unit}.\n"
            "Hosts: {presenter} (presenter) and {questioner} (questioner).\n\n"
            "CONDENSING RULES:\n"
            "- Shorten verbose passages by tightening language\n"
            "- Merge overlapping discussion points\n"
            "- Reduce conversational filler\n"
            "- Do NOT delete entire discussion topics --- condense them\n"
            "- Within each act, condense later items first (earlier items set up context)\n"
            "- Prefer shortening analogies/examples before mechanism explanations\n"
            "- Preserve ALL [TRANSITION] markers exactly as-is\n"
            "- Preserve the One Action ending\n"
            "- Preserve speaker labels: {presenter}: and {questioner}:\n"
            "- Do NOT condense below {floor_count} {length_unit}\n"
            "- Return ONLY the condensed script dialogue, no commentary\n"
            "{target_instruction}"
        ),
        "ja": (
            "「{topic_name}」についての2人のホストによる科学ポッドキャストスクリプトを "
            "{current_count} {length_unit}から約{target_count} {length_unit}に凝縮してください。\n"
            "ホスト: {presenter}（プレゼンター）と{questioner}（質問者）。\n\n"
            "凝縮ルール:\n"
            "- 冗長な箇所を簡潔な表現に書き換える\n"
            "- 重複するディスカッションポイントを統合する\n"
            "- 会話のフィラーを減らす\n"
            "- ディスカッショントピック全体を削除しない --- 凝縮する\n"
            "- 各アクト内で、後半の項目から先に凝縮する（前半の項目は文脈を設定する）\n"
            "- メカニズムの説明よりも比喩/例を先に短縮する\n"
            "- すべての[TRANSITION]マーカーをそのまま保持する\n"
            "- One Actionエンディングを保持する\n"
            "- スピーカーラベルを保持: {presenter}: と {questioner}:\n"
            "- {floor_count} {length_unit}未満に凝縮しない\n"
            "- 凝縮されたスクリプトの対話のみを返す、コメントなし\n"
            "{target_instruction}"
        ),
    },

    "user": {
        "en": (
            "SCRIPT TO CONDENSE ({current_count} {length_unit}):\n\n"
            "{script_text}\n\n"
            "Return the condensed script at approximately {target_count} {length_unit}."
        ),
        "ja": (
            "凝縮するスクリプト ({current_count} {length_unit}):\n\n"
            "{script_text}\n\n"
            "約{target_count} {length_unit}に凝縮したスクリプトを返してください。"
        ),
    },
}


# ---------------------------------------------------------------------------
# SECTION GENERATION PROMPTS — Per-section system/user prompts for sectional
# script drafting (replaces the monolithic single-call script prompt).
# ---------------------------------------------------------------------------

SECTION_GEN_PROMPTS: dict[str, dict[str, str]] = {

    # Speakability rules vary by language:
    # EN: ~25 words/sentence ≈ 10s at 150 wpm
    # JA: ~60 chars/sentence ≈ 7s at 500 chars/min
    "speakability_rule": {
        "en": "Max 25 words per sentence (for speakability — hosts must read this aloud without running out of breath)",
        "ja": "1文あたり最大60文字（句点「。」まで）（読みやすさ — ホストが息切れせずに読めるように）",
    },

    "system": {
        "en": (
            "You are writing one section of a two-host science podcast about \"{topic}\".\n\n"
            "CHARACTER ROLES:\n"
            "  - {presenter} (Presenter): {presenter_personality}\n"
            "  - {questioner} (Questioner): {questioner_personality}\n\n"
            "WRITING RULES:\n"
            "- Write conversationally — contractions, everyday vocabulary, natural fillers\n"
            "- {speakability_rule}\n"
            "- One idea per sentence — if a sentence covers two ideas, split it\n"
            "- Every claim needs: explanation -> real-world analogy -> host reaction -> nuance\n"
            "- Vary energy: surprising findings get excitement, nuances get thoughtful pauses\n"
            "- Include natural fillers: 'Hm, that's interesting', 'Right, right', 'Oh wow'\n"
            "- Brief banter between hosts — a shared laugh, a relatable admission\n"
            "- Maintain consistent roles — {presenter} explains, {questioner} asks and reacts\n\n"
            "FORMAT: Dialogue only. No markdown headers, no stage directions, no commentary.\n"
            "{presenter}: [dialogue]\n"
            "{questioner}: [dialogue]\n"
        ),
        "ja": (
            "「{topic}」についての2人のホストによる科学ポッドキャストの1セクションを書いてください。\n\n"
            "キャラクター設定:\n"
            "  - {presenter}（プレゼンター）: {presenter_personality}\n"
            "  - {questioner}（質問者）: {questioner_personality}\n\n"
            "執筆ルール:\n"
            "- 会話調で書く — 口語表現、日常的な語彙、自然なフィラー\n"
            "- {speakability_rule}\n"
            "- 1文1アイデア — 2つのアイデアが含まれる場合は文を分割する\n"
            "- 各主張には: 説明 -> 現実世界の例え -> ホストの反応 -> ニュアンス\n"
            "- エネルギーを変化させる: 驚きの発見には興奮、ニュアンスには落ち着いた間\n"
            "- 自然なフィラーを含む: 「へー、面白いですね」「確かに」「えっ、本当に？」\n"
            "- ホスト間の軽いやり取り — 共感の笑い、親しみやすいエピソード\n"
            "- 一貫した役割を維持 — {presenter}が説明し、{questioner}が質問・反応する\n\n"
            "形式: 対話のみ。マークダウンヘッダー、ト書き、コメントなし。\n"
            "{presenter}: [対話]\n"
            "{questioner}: [対話]\n"
        ),
    },

    "user_opening": {
        "en": (
            "SECTION: Opening (Channel Intro + Hook + Act 1 — The Claim)\n"
            "WORD BUDGET: Write approximately {word_budget} {length_unit} for this section.\n"
            "This is {budget_pct}% of a {target_min}-minute episode.\n\n"
            "1. CHANNEL INTRO (~25 words):\n"
            "   {channel_intro_directive}\n\n"
            "2. HOOK (~40 words):\n"
            "   {presenter}: [Provocative question that makes listeners care personally]\n"
            "   {questioner}: [Engaged reaction: 'Oh wow' / 'Hmm, I had no idea']\n\n"
            "3. ACT 1 — THE CLAIM:\n"
            "   What people believe. The folk wisdom. Why this matters personally.\n"
            "   - {presenter} sets up the common belief\n"
            "   - {questioner} validates: 'Right, I've heard that too'\n"
            "   - Establish emotional stakes: why should the listener care?\n\n"
            "PACING: {pacing}\n\n"
            "COVERAGE CHECKLIST — address EACH of these in Act 1:\n"
            "{checklist_block}\n\n"
            "Write this section now. Target {word_budget} {length_unit}.\n"
            "Writing more is fine. Writing less will cause production to FAIL."
        ),
        "ja": (
            "セクション: オープニング（チャンネルイントロ + フック + Act 1 — The Claim）\n"
            "文字数目標: このセクションは約{word_budget} {length_unit}で書いてください。\n"
            "{target_min}分エピソードの{budget_pct}%です。\n\n"
            "1. チャンネルイントロ（約50文字）:\n"
            "   {channel_intro_directive}\n\n"
            "2. フック（約80文字）:\n"
            "   {presenter}: [リスナーが個人的に気になる挑発的な質問]\n"
            "   {questioner}: [興味を引かれた反応: 「えっ、本当に？」「それは知らなかった」]\n\n"
            "3. ACT 1 — THE CLAIM:\n"
            "   人々が信じていること。常識。なぜ個人的に重要なのか。\n"
            "   - {presenter}が一般的な信念を提示\n"
            "   - {questioner}が共感: 「確かに、私もそう聞いていました」\n"
            "   - 感情的なステークスを確立: なぜリスナーが気にすべきか？\n\n"
            "ペーシング: {pacing}\n\n"
            "カバレッジチェックリスト — Act 1で以下のそれぞれに対応してください:\n"
            "{checklist_block}\n\n"
            "このセクションを書いてください。目標{word_budget} {length_unit}。\n"
            "多めに書いても問題ありません。少なすぎると制作が失敗します。"
        ),
    },

    "user_evidence": {
        "en": (
            "SECTION: Act 2 — Evidence & Nuance (the core of the episode)\n"
            "WORD BUDGET: Write approximately {word_budget} {length_unit} for this section.\n"
            "This is {budget_pct}% of a {target_min}-minute episode — the longest section.\n\n"
            "STRUCTURE:\n"
            "- Start by stating the episode's conclusion upfront\n"
            "- Walk through EACH study individually:\n"
            "  Study -> finding -> {questioner} asks a study-specific question -> {presenter} answers with nuance/limitations\n"
            "- Include specific numbers (NNT, ARR, sample sizes) where available\n"
            "- Do NOT lump all evidence then all nuance — interleave them per study\n"
            "- End with: 'So across all these studies, here's what we see...'\n\n"
            "PACING: {pacing}\n\n"
            "COVERAGE CHECKLIST — address EACH of these:\n"
            "{checklist_block}\n\n"
            "CONTINUITY — The previous section ended with:\n---\n{lead_in}\n---\n"
            "Continue naturally. Do not repeat what was already said.\n\n"
            "Write this section now. Target {word_budget} {length_unit}.\n"
            "Writing more is fine. Writing less will cause production to FAIL."
        ),
        "ja": (
            "セクション: Act 2 — Evidence & Nuance（エピソードの核心）\n"
            "文字数目標: このセクションは約{word_budget} {length_unit}で書いてください。\n"
            "{target_min}分エピソードの{budget_pct}% — 最も長いセクションです。\n\n"
            "構成:\n"
            "- まずエピソードの結論を先に述べる\n"
            "- 各研究を個別に取り上げる:\n"
            "  研究 -> 知見 -> {questioner}が研究固有の質問 -> {presenter}がニュアンス/限界を含めて回答\n"
            "- 利用可能な場合、具体的な数値（NNT, ARR, サンプルサイズ）を含める\n"
            "- エビデンスとニュアンスを交互に — まとめて述べない\n"
            "- 最後に:「これらすべての研究を通して、見えてくることは...」\n\n"
            "ペーシング: {pacing}\n\n"
            "カバレッジチェックリスト — 以下のそれぞれに対応してください:\n"
            "{checklist_block}\n\n"
            "前のセクションの終わり:\n---\n{lead_in}\n---\n"
            "自然に続けてください。すでに述べたことを繰り返さないでください。\n\n"
            "このセクションを書いてください。目標{word_budget} {length_unit}。\n"
            "多めに書いても問題ありません。少なすぎると制作が失敗します。"
        ),
    },

    "user_synthesis": {
        "en": (
            "SECTION: Act 3 — Holistic Conclusion\n"
            "WORD BUDGET: Write approximately {word_budget} {length_unit} for this section.\n"
            "This is {budget_pct}% of a {target_min}-minute episode.\n\n"
            "STRUCTURE:\n"
            "- Synthesize ALL evidence into a unified takeaway — this is NOT new evidence\n"
            "- Restate the conclusion — the listener now has the evidence\n"
            "- What does the totality of evidence mean practically?\n"
            "- GRADE confidence level in plain language\n"
            "- {questioner}: 'So if I had to summarize everything we just discussed...'\n"
            "- {presenter} ties it all together\n\n"
            "PACING: {pacing}\n\n"
            "COVERAGE CHECKLIST — address EACH of these:\n"
            "{checklist_block}\n\n"
            "CONTINUITY — The previous section ended with:\n---\n{lead_in}\n---\n"
            "Continue naturally. Do not repeat what was already said.\n\n"
            "Write this section now. Target {word_budget} {length_unit}.\n"
            "Writing more is fine. Writing less will cause production to FAIL."
        ),
        "ja": (
            "セクション: Act 3 — Holistic Conclusion\n"
            "文字数目標: このセクションは約{word_budget} {length_unit}で書いてください。\n"
            "{target_min}分エピソードの{budget_pct}%です。\n\n"
            "構成:\n"
            "- すべてのエビデンスを統一的なテイクアウェイに統合 — 新しいエビデンスではない\n"
            "- 結論を再度述べる — リスナーはエビデンスを理解した\n"
            "- エビデンス全体が実際に何を意味するか\n"
            "- GRADEの確信度を平易な言葉で\n"
            "- {questioner}: 「では、今まで話したことをまとめると...」\n"
            "- {presenter}がすべてをまとめる\n\n"
            "ペーシング: {pacing}\n\n"
            "カバレッジチェックリスト — 以下のそれぞれに対応してください:\n"
            "{checklist_block}\n\n"
            "前のセクションの終わり:\n---\n{lead_in}\n---\n"
            "自然に続けてください。すでに述べたことを繰り返さないでください。\n\n"
            "このセクションを書いてください。目標{word_budget} {length_unit}。\n"
            "多めに書いても問題ありません。少なすぎると制作が失敗します。"
        ),
    },

    "user_closing": {
        "en": (
            "SECTION: Closing (Act 4 — The Protocol + Wrap-up + One Action)\n"
            "WORD BUDGET: Write approximately {word_budget} {length_unit} for this section.\n"
            "This is {budget_pct}% of a {target_min}-minute episode.\n\n"
            "1. ACT 4 — THE PROTOCOL:\n"
            "   Translate science into daily life, focusing on real-world barriers.\n"
            "   - {questioner} voices listener challenges: 'I get the science, but...'\n"
            "   - {presenter} addresses each barrier with practical solutions\n"
            "   - Each barrier gets its own mini-conversation (challenge -> solution -> encouragement)\n\n"
            "2. WRAP-UP (~60 words):\n"
            "   Three-sentence summary of the most important takeaways.\n\n"
            "3. ONE ACTION ENDING (~40 words):\n"
            "   {presenter}: 'If you take ONE thing from today — [specific, doable action].'\n"
            "   {questioner}: [Brief agreement + sign-off]\n\n"
            "PACING: {pacing}\n\n"
            "COVERAGE CHECKLIST — address EACH of these in Act 4:\n"
            "{checklist_block}\n\n"
            "CONTINUITY — The previous section ended with:\n---\n{lead_in}\n---\n"
            "Continue naturally. Do not repeat what was already said.\n\n"
            "Write this section now. Target {word_budget} {length_unit}.\n"
            "Writing more is fine. Writing less will cause production to FAIL."
        ),
        "ja": (
            "セクション: クロージング（Act 4 — The Protocol + まとめ + One Action）\n"
            "文字数目標: このセクションは約{word_budget} {length_unit}で書いてください。\n"
            "{target_min}分エピソードの{budget_pct}%です。\n\n"
            "1. ACT 4 — THE PROTOCOL:\n"
            "   科学を日常生活に変換し、現実の障壁に焦点を当てる。\n"
            "   - {questioner}がリスナーの課題を代弁: 「科学は分かりましたが...」\n"
            "   - {presenter}が各障壁に実践的な解決策で対応\n"
            "   - 各障壁ごとにミニ会話（課題 -> 解決策 -> 励まし）\n\n"
            "2. まとめ（約120文字）:\n"
            "   最も重要なポイントを3文でまとめる。\n\n"
            "3. ONE ACTIONエンディング（約80文字）:\n"
            "   {presenter}: 「今日のエピソードから一つだけ覚えて帰るなら — [具体的なアクション]」\n"
            "   {questioner}: [短い同意 + サインオフ]\n\n"
            "ペーシング: {pacing}\n\n"
            "カバレッジチェックリスト — Act 4で以下のそれぞれに対応してください:\n"
            "{checklist_block}\n\n"
            "前のセクションの終わり:\n---\n{lead_in}\n---\n"
            "自然に続けてください。すでに述べたことを繰り返さないでください。\n\n"
            "このセクションを書いてください。目標{word_budget} {length_unit}。\n"
            "多めに書いても問題ありません。少なすぎると制作が失敗します。"
        ),
    },

    "retry_feedback": {
        "en": (
            "Your previous attempt was only {actual_count} {length_unit} — "
            "need at least {floor_count} {length_unit}. "
            "Expand the conversation: add deeper explanations, real-world analogies, "
            "and more host back-and-forth for EACH checklist item."
        ),
        "ja": (
            "前回の試みは{actual_count} {length_unit}のみでした — "
            "少なくとも{floor_count} {length_unit}が必要です。"
            "会話を拡張してください: 各チェックリスト項目について、より深い説明、"
            "現実世界の例え、ホスト間のやり取りを追加してください。"
        ),
    },
}


# ---------------------------------------------------------------------------
# Lookup helper
# ---------------------------------------------------------------------------

_SECTION_MAP = {
    "blueprint": BLUEPRINT_PROMPTS,
    "script": SCRIPT_PROMPTS,
    "polish": POLISH_PROMPTS,
    "condense": CONDENSE_PROMPTS,
    "section_gen": SECTION_GEN_PROMPTS,
}


def get_prompt(section: str, key: str, language: str = "en", **kwargs) -> str:
    """Look up a prompt string by section/key/language, with .format() interpolation.

    Falls back to English if the requested language is not available.
    Raises KeyError if the section or key doesn't exist.
    """
    prompts = _SECTION_MAP[section]
    lang_dict = prompts[key]
    template = lang_dict.get(language, lang_dict["en"])
    return template.format(**kwargs) if kwargs else template
