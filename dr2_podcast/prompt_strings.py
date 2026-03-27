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
            "     - Questioner connects the claim to something they encountered in preparation: 'I read that this traces back to...' or 'The common argument seems to be...'\n"
            "     - Establish emotional stakes: why should the listener care?\n\n"
        ),
        "ja": (
            "  3. ACT 1 --- THE CLAIM:\n"
            "     人々が信じていること。常識。なぜ個人的に重要なのか。\n"
            "     - プレゼンターが一般的な信念や疑問を提示\n"
            "     - 質問者が準備で調べた内容と関連づける: 「調べてみたところ、これは〜に由来するらしいですね」\n"
            "     - 感情的なステークスを確立: なぜリスナーが気にすべきか？\n\n"
        ),
    },

    "act2": {
        "en": (
            "  4. ACT 2 --- EVIDENCE & NUANCE (above 50%% of the episode --- at least {act2_min} {target_unit_plural}, expand freely):\n"
            "     Start by stating the episode's conclusion upfront.\n"
            "     Then walk through EACH study individually:\n"
            "     - Present the study's finding with GRADE-informed framing from the Blueprint\n"
            "     - Questioner asks a study-specific question (from Coverage Checklist) --- focus on methodology: sample size, study design, confounders, effect sizes\n"
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
            "     - 質問者が研究固有の質問をする（Coverage Checklistから）--- 方法論に焦点: サンプルサイズ、研究デザイン、交絡因子、効果量\n"
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
            "     - Questioner offers their own synthesis attempt for the Presenter to refine: 'So if I had to pull this together, it seems like...'\n"
            "     - Presenter ties it all together, building confidence in the conclusion\n\n"
        ),
        "ja": (
            "  5. ACT 3 --- HOLISTIC CONCLUSION:\n"
            "     すべてのエビデンスが合わせて何を意味するかを統合してください。\n"
            "     - 結論を再度述べる --- リスナーはエビデンスを理解した上で\n"
            "     - エビデンス全体が{core_target_or_default}にとって何を意味するか\n"
            "     - GRADEの確信度を平易な言葉で\n"
            "     - 質問者が自分なりの統合を試み、プレゼンターに洗練を求める: 「ここまでの話をまとめると、つまり…ということでしょうか？」\n"
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

    "hook": {
        "en": (
            "  2. THE HOOK (~40 {target_unit_plural}, ~15 seconds):\n"
            "     Based on the hook question from the Episode Blueprint.\n"
            "     {presenter}: [Provocative question from Blueprint --- must be a question, NOT a statement]\n"
            "     {questioner}: [Engaged reaction: 'Oh, that's a great question!' or 'Hmm, I actually have no idea...']\n\n"
        ),
        "ja": (
            "  2. THE HOOK (~40 {target_unit_plural}, ~15秒):\n"
            "     エピソードブループリントのフック質問に基づく。\n"
            "     {presenter}: [ブループリントからの挑発的な質問 --- 文ではなく質問であること]\n"
            "     {questioner}: [興味津々の反応: 「えっ、それ本当ですか？」や「うーん、全然知らなかった...」]\n\n"
        ),
    },

    "wrapup": {
        "en": (
            "  7. WRAP-UP (~60 {target_unit_plural}, ~25 seconds):\n"
            "     Three-sentence summary of the most important takeaways.\n\n"
        ),
        "ja": (
            "  7. まとめ (~60 {target_unit_plural}, ~25秒):\n"
            "     最も重要なポイントを3文で要約。\n\n"
        ),
    },

    "one_action": {
        "en": (
            "  8. THE 'ONE ACTION' ENDING (~40 {target_unit_plural}, ~15 seconds):\n"
            "     {presenter}: 'If you take ONE thing from today --- [action {one_action_tail}].'\n"
            "     {questioner}: [Brief agreement + sign-off]\n\n"
        ),
        "ja": (
            "  8. 「ワンアクション」エンディング (~40 {target_unit_plural}, ~15秒):\n"
            "     {presenter}: 「今日の放送から一つだけ持ち帰るなら --- [{one_action_tail}]」\n"
            "     {questioner}: [短い同意 + サインオフ]\n\n"
        ),
    },

    "personality": {
        "en": (
            "PERSONALITY DIRECTIVES:\n"
            "- ENERGY: Vary vocal energy --- excited for surprising findings, thoughtful pauses for nuance, urgency for practical advice\n"
            "- REACTIONS: Questioner reacts authentically per their character role --- with informed curiosity, reasoned skepticism, and occasional humor\n"
            "- RESPONSE STYLE: When the Questioner makes a point, the Presenter always ADVANCES the conversation. "
            "Every Presenter response after a Questioner contribution must introduce new information, a new angle, or a qualification. "
            "Pure agreement ('Exactly!', 'Correct!', 'Spot on!', 'That\\'s right!') adds nothing --- replace with: "
            "'And building on that...', 'That\\'s an interesting angle --- actually...', 'Let me add to that...', "
            "'That makes me think of another angle entirely...', 'There\\'s actually a deeper layer to that...', "
            "'Mostly, yes --- though there\\'s one important caveat...'\n"
            "- QUESTION DEPTH: Questioner asks research-informed questions, not surface reactions.\n"
            "  For science/medical topics --- GOOD: 'What was the sample size?', 'Did they control for socioeconomic status?', 'What\\'s the effect size --- is that clinically meaningful?'\n"
            "  For social science/policy topics --- GOOD: 'Has that effect been replicated in other countries?', 'How was the comparison group set up?', 'What\\'s the cost-benefit ratio of that intervention?'\n"
            "  BAD (always): 'Wow, really?', 'That\\'s amazing!', 'Tell me more about that'\n"
            "- BANTER: Include brief moments of friendly banter between hosts --- a shared laugh, a playful jab, a relatable personal admission\n"
            "- FILLERS: Natural conversational fillers: 'Hm, that\\'s interesting', 'Right, right', 'Oh wow', 'Okay so let me get this straight...'\n"
            "- EMPHASIS: Dramatic pauses via ellipses: 'And here\\'s where it gets interesting...'\n"
            "- STORYTELLING: After each key finding, paint a picture: 'Imagine you\\'re...' or 'Think about your morning routine...'\n"
            "- PERSONAL: Brief personal connections: 'I actually tried this myself and...' or 'My partner always says...'\n"
            "- MOMENTUM: Each act builds energy --- start curious, peak at the most surprising finding, resolve with practical clarity\n\n"
        ),
        "ja": (
            "パーソナリティ指示:\n"
            "- 語調ガイド: 基本はです・ます調の知的な会話体。\n"
            "  許可: 「なるほど」「確かに」「面白い」「ちょっと意外」「そうなんですよ」「〜ですよね」「〜かもしれないですね」\n"
            "  禁止: 「やばい」「マジ」「ウケる」「やだ」「すげー」「めっちゃ」「ヤバくない？」\n"
            "  知的で落ち着いた語調を保ちつつ、二人の友人が話しているような温かさを維持する。\n"
            "- エネルギー: 声のエネルギーを変化させる --- 驚きの発見には興奮、ニュアンスには考え深い間、実践的アドバイスには緊迫感\n"
            "- リアクション: 質問者はキャラクター設定に沿った本物のリアクションをする --- 知的な好奇心（「え、そうなんですか？」）、根拠のある懐疑（「うーん、出来すぎた話に聞こえるけど...」）、ユーモア（「あー、つまり私ずっと間違ってたってこと？」）\n"
            "- 応答スタイル: 質問者が意見を述べた後、プレゼンターは必ず対話を前進させる。"
            "新しい情報、別の角度、または補足を加えること。"
            "単純な同意（「正解です」「完全にその通り」「その通り！」「まさにそれ」）は禁止。代わりに: "
            "「それに加えて…」「面白い視点ですね、実は…」「そこをもう少し掘ると…」"
            "「その観点から言うと、もう一つ重要なのが…」「実はそこにはもう一層深い話があって…」"
            "「概ねそうなんですが、一つ注意点があって…」\n"
            "- 質問の深さ: 質問者は表面的なリアクション（「へー」「すごい」）ではなく、研究の核心に迫る質問をする。\n"
            "  科学・医学トピックの場合 --- 良い例: 「そのメタ分析のサンプルサイズはどれくらいですか？」「交絡因子はコントロールされていましたか？」「効果量はどの程度ですか？」\n"
            "  社会科学・政策トピックの場合 --- 良い例: 「その知見は日本の文脈にも当てはまりますか？」「先行研究との整合性はどうでしょう？」「そのデータの解釈には別の可能性もありませんか？」\n"
            "  悪い例（常に）: 「へー、それってどんな研究だったんですか？」「すごいですね」\n"
            "- 掛け合い: ホスト間の気軽な掛け合いを含める --- 共有する笑い、軽いツッコミ、共感できる個人的な告白\n"
            "- フィラー: 自然な会話のフィラー: 「へー、面白いですね」「確かに」「え、そうなんですか？」「ちょっと待って、整理させて...」\n"
            "- 強調: 省略記号で劇的な間: 「で、ここからが面白いんですけど...」\n"
            "- ストーリーテリング: 重要な発見の後に情景を描く: 「想像してみてください...」「あなたの朝のルーティンを思い浮かべて...」\n"
            "- 個人的つながり: 短い個人的エピソード: 「実は私も試してみたんですけど...」「うちのパートナーがいつも言うんですけど...」\n"
            "- モメンタム: 各Actでエネルギーを高める --- 好奇心から始まり、最も驚きの発見でピーク、実践的な明確さで解決\n\n"
        ),
    },

    "character_roles": {
        "en": (
            "CHARACTER ROLES:\n"
            "  - {presenter} (Presenter): presents evidence and explains the topic, {presenter_personality}\n"
            "  - {questioner} (Questioner): asks informed, probing questions based on preparation; offers hypotheses for the presenter to refine; draws out specifics for the listener, {questioner_personality}\n\n"
        ),
        "ja": (
            "キャラクター役割:\n"
            "  - {presenter} (プレゼンター): エビデンスを提示しトピックを解説、{presenter_personality}\n"
            "  - {questioner} (質問者): 事前リサーチに基づく的確な質問で専門家の知見を深掘りし、仮説を投げかけながらリスナーの理解を導く、{questioner_personality}\n\n"
        ),
    },

    "target_length": {
        "en": (
            "TARGET LENGTH: AT LEAST {target_script} {target_unit_plural} (= {target_min} minutes). "
            "Aim for {aim_target} {target_unit_plural}. "
            "Writing more than the target is fine --- it will be condensed during polish. "
            "Writing less will cause the production to FAIL. Cover ALL items in the Coverage Checklist above.\n"
            "ACT CHECKLIST: You must write all 4 acts plus Hook, Channel Intro, Wrap-up, and One Action. Count them as you write.\n"
            "TO REACH THIS LENGTH: You must be extremely detailed and conversational. For every single claim or mechanism, you MUST provide:\n"
            "  1. A deep-dive explanation of the specific scientific mechanism\n"
            "  2. A real-world analogy or metaphor that lasts several lines\n"
            "  3. A practical, relatable example or case study\n"
            "  4. A counter-argument or nuance followed by a rebuttal\n"
            "  5. Interactive host dialogue (e.g., 'Wait, let me make sure I've got this right...', 'That's fascinating, tell me more about...')\n"
            "Expand the conversation. Do not just list facts. Have the hosts explore the 'So what?' and 'What now?' for the audience.\n"
            "Maintain consistent roles throughout. NO role switching mid-conversation. "
        ),
        "ja": (
            "目標長さ: 最低 {target_script} {target_unit_plural} (= {target_min}分). "
            "{aim_target} {target_unit_plural}を目指してください。 "
            "目標より長くても構いません --- ポリッシュ段階で調整します。 "
            "目標より短いと制作が失敗します。上記のCoverage Checklistのすべての項目をカバーしてください。\n"
            "ACTチェックリスト: 4つのAct + Hook、Channel Intro、まとめ、ワンアクションをすべて書いてください。書きながら数えてください。\n"
            "この長さに到達するために: 非常に詳細かつ会話的に書いてください。すべての主張やメカニズムについて、以下を必ず含めてください:\n"
            "  1. 具体的な科学的メカニズムの深掘り説明\n"
            "  2. 数行にわたる現実世界のアナロジーや比喩\n"
            "  3. 実践的で共感できる具体例やケーススタディ\n"
            "  4. 反論やニュアンス、それに対する反駁\n"
            "  5. ホスト間のインタラクティブな対話（例: 「ちょっと待って、ここまでの理解が合っているか確認させて...」「それ面白い、もっと詳しく教えて...」）\n"
            "会話を拡張してください。事実を列挙するだけでなく、ホストがリスナーのために「だから何？」「次はどうする？」を探ってください。\n"
            "一貫した役割を維持してください。会話の途中で役割を入れ替えないでください。"
        ),
    },
}


# ---------------------------------------------------------------------------
# POLISH PROMPTS — Masters-level polish & length control
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
            "8-part structure with [INTRO_END] after Channel Intro and [TRANSITION] markers between acts. "
            "Acts: Claim, Evidence & Nuance, Holistic Conclusion, Protocol. "
            "One Action ending present."
        ),
        "ja": (
            "8-part structure with [INTRO_END] after Channel Intro and [TRANSITION] markers between acts. "
            "Acts: Claim, Evidence & Nuance, Holistic Conclusion, Protocol. "
            "One Action ending present."
        ),
    },

    "masters_level": {
        "en": (
            "MASTERS-LEVEL REQUIREMENTS:\n"
            "- Remove ALL definitions of basic scientific concepts (DNA, peer review, RCT, meta-analysis)\n"
            "- Ensure the questioner's questions feel natural and audience-aligned\n"
            "- Keep technical language intact - NO dumbing down\n"
        ),
        "ja": (
            "修士レベル要件:\n"
            "- 基本的な科学概念の定義をすべて削除（DNA、ピアレビュー、RCT、メタアナリシス）\n"
            "- 質問者の質問が自然でリスナー目線であることを確認\n"
            "- 専門用語はそのまま維持 — 平易化しない\n"
        ),
    },

    "length_section": {
        "en": (
            "- LENGTH: Target {target_script} {target_unit_plural} "
            "(acceptable range: {range_low}–{range_high} {target_unit_plural}).\n"
            "  First, estimate how many {target_unit_plural} the input draft contains.\n"
            "  - If input is MORE THAN 20% over target: trim repetition, redundant examples, "
            "and filler to bring it toward target --- preserve all factual claims and the 8-part structure.\n"
            "  - If input is 20% or less over target: focus on improving quality, flow, and verbal "
            "clarity --- do NOT aggressively cut content. Minor trimming of true filler is fine.\n"
            "  - If input is AT or UNDER target: do NOT shorten. Expand thin sections, add depth, "
            "improve transitions.\n"
            "  ABSOLUTE FLOOR: Your output MUST contain at least "
            "{range_low} {target_unit_plural}. "
            "Cutting below this floor is a critical failure. If the input is already near or "
            "below target, focus on improving flow, transitions, and verbal clarity --- do NOT shorten it.\n"
        ),
        "ja": (
            "- 長さ: 目標 {target_script} {target_unit_plural} "
            "(許容範囲: {range_low}–{range_high} {target_unit_plural}).\n"
            "  まず、入力ドラフトの{target_unit_plural}数を推定してください。\n"
            "  - 目標より20%以上長い場合: 繰り返し、冗長な例、フィラーを削って目標に近づける "
            "--- 事実に基づく主張と8部構成は必ず保持。\n"
            "  - 目標より20%以下の超過の場合: 品質、流れ、口語的な明瞭さの改善に集中 "
            "--- 積極的にカットしない。本当のフィラーの軽微なトリミングは可。\n"
            "  - 目標以下の場合: 短くしない。薄いセクションを拡張し、深みを加え、"
            "トランジションを改善。\n"
            "  絶対最低ライン: 出力は最低 "
            "{range_low} {target_unit_plural}を含むこと。 "
            "この最低ラインを下回ることは致命的な失敗です。入力が目標付近または目標以下の場合は、"
            "流れ、トランジション、口語的明瞭さの改善に集中してください --- 短くしないでください。\n"
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
            "- Preserve ALL audio markers ([TRANSITION], [INTRO_END], [PAUSE], [BEAT]) exactly as-is\n"
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
            "- すべてのオーディオマーカー（[TRANSITION]、[INTRO_END]、[PAUSE]、[BEAT]）をそのまま保持する\n"
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
            "- Maintain consistent roles — {presenter} provides expert depth, {questioner} asks informed questions and offers hypotheses\n"
            "- Despite these intellectual standards, the conversation must feel like two friends talking — not a panel discussion. Include genuine laughter, moments of wonder, and personal vulnerability. Intelligence and warmth are not opposites.\n\n"
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
            "- 一貫した役割を維持 — {presenter}が専門知識を提供し、{questioner}が準備に基づく質問と仮説で対話を深める\n"
            "- 知的で落ち着いた語調を保つ。スラング（やばい、マジ等）禁止。\n"
            "- 知的水準を保ちつつも、二人の友人が話しているような温かさを維持する。笑い、驚き、個人的な共感を忘れない。\n\n"
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
            "■ 文字数ノルマ: 最低{word_budget}文字（{target_min}分エピソードの{budget_pct}%）。\n"
            "  これは約{turn_count}回の発言に相当します。必ずこの分量を書いてください。\n\n"
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
            "【重要】このセクションは最低{word_budget}文字です。{word_budget}文字未満の出力は自動的に不合格になります。\n"
            "各チェックリスト項目について十分に深く掘り下げ、具体例・ホスト間のやり取りを豊富に含めてください。"
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
            "■ 文字数ノルマ: 最低{word_budget}文字（{target_min}分エピソードの{budget_pct}% — 最も長いセクション）。\n"
            "  これは約{turn_count}回の発言に相当します。必ずこの分量を書いてください。\n\n"
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
            "【重要】このセクションは最低{word_budget}文字です。{word_budget}文字未満の出力は自動的に不合格になります。\n"
            "各チェックリスト項目について十分に深く掘り下げ、具体例・ホスト間のやり取りを豊富に含めてください。"
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
            "■ 文字数ノルマ: 最低{word_budget}文字（{target_min}分エピソードの{budget_pct}%）。\n"
            "  これは約{turn_count}回の発言に相当します。必ずこの分量を書いてください。\n\n"
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
            "【重要】このセクションは最低{word_budget}文字です。{word_budget}文字未満の出力は自動的に不合格になります。\n"
            "各チェックリスト項目について十分に深く掘り下げ、具体例・ホスト間のやり取りを豊富に含めてください。"
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
            "■ 文字数ノルマ: 最低{word_budget}文字（{target_min}分エピソードの{budget_pct}%）。\n"
            "  これは約{turn_count}回の発言に相当します。必ずこの分量を書いてください。\n\n"
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
            "【重要】このセクションは最低{word_budget}文字です。{word_budget}文字未満の出力は自動的に不合格になります。\n"
            "各チェックリスト項目について十分に深く掘り下げ、具体例・ホスト間のやり取りを豊富に含めてください。"
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
            "【不合格】前回の出力は{actual_count}文字でした。最低{floor_count}文字が必要です。\n"
            "この基準を満たさない出力は使用できません。以下を実行して文字数を増やしてください:\n"
            "- 各チェックリスト項目について2〜3往復の会話を追加\n"
            "- 研究結果に対する具体的な日常生活の例えを追加\n"
            "- {questioner}の「なぜ？」「どういうこと？」のフォローアップ質問を追加\n"
            "- {presenter}の回答をより詳細に展開\n"
            "最初から書き直してください。{floor_count}文字以上を必ず書いてください。"
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
