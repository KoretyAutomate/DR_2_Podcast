# Script Quality Improvement Plan — Dialogue Naturalness, Intelligence & Audio Transition

> **Origin:** User review of run `2026-03-24_20-56-42` (Japanese, childcare/daycare topic).
> Three issues identified: (1) too casual/frank language, (2) abrupt audio transition after channel intro, (3) hosts sound unintelligent.
>
> **Reviewed by:** Prompt engineering agent + technical risk agent. Findings incorporated below.

---

## Fix A: Japanese Register Floor — Tiered Whitelist/Blacklist (Not Blanket Ban)

**Files:** `dr2_podcast/prompt_strings.py`

### Root Cause

The JA personality directives (line 394) give example reactions like 「えっ、マジですか？！」and fillers like 「えー！」. The model escalates these into heavy slang — Host 1 uses 「やばい」17+ times, plus 「やだ」「わー」. No register floor is specified, so the model defaults to youth-casual Japanese instead of intellectual-casual.

### Changes

1. **`prompt_strings.py` line 391-401 (JA personality):**
   - Replace reaction examples with educated-casual alternatives:
     - Before: 「えっ、マジですか？！」→ After: 「えっ、本当ですか？」
     - Before: 「えー！」→ keep as 「え、そうなんですか？」(natural educated surprise)
     - Before: 「うーん、出来すぎた話に聞こえるけど...」→ keep (this is fine register)
   - Add **tiered register guide** instead of blanket ban:
     ```
     語調ガイド: 基本はです・ます調の知的な会話体。
       許可: 「なるほど」「確かに」「面白い」「ちょっと意外」「そうなんですよ」「〜ですよね」「〜かもしれないですね」
       禁止: 「やばい」「マジ」「ウケる」「やだ」「すげー」「めっちゃ」「ヤバくない？」
     知的で落ち着いた語調を保ちつつ、二人の友人が話しているような温かさを維持する。
     ```

2. **`prompt_strings.py` line 630-647 (JA section_gen system):**
   - Update filler examples at line 641: 「へー、面白いですね」→ keep, 「確かに」→ keep, 「えっ、本当に？」→ keep (these are all fine register; the problem is in personality directives, not here)
   - Add after fillers: 「知的で落ち着いた語調を保つ。スラング（やばい、マジ等）禁止。」

### Files Changed
- `dr2_podcast/prompt_strings.py` — personality JA (lines 391-401), section_gen JA system (line ~641)

---

## Fix B: Audio Transition After Channel Intro — Add Breathing Room

**Files:** `dr2_podcast/audio/engine.py`, `dr2_podcast/prompt_strings.py`, `dr2_podcast/pipeline_crew.py`, `dr2_podcast/pipeline_script.py`, `dr2_podcast/pipeline.py`

### Root Cause

After the channel intro, the conversation starts with only a 1.5s `[TRANSITION]` gap. This feels abrupt because the tonal shift from formal intro → conversation needs more breathing room.

### Changes

> **Technical review identified 6 cascade risks.** All addressed below.

1. **`audio/engine.py` line 675 (AUDIO_MARKERS):** Add `[INTRO_END]` placeholder:
   ```python
   AUDIO_MARKERS = {
       '[TRANSITION]': '___TRANSITION___',
       '[PAUSE]': '___PAUSE___',
       '[BEAT]': '___BEAT___',
       '[INTRO_END]': '___INTRO_END___',
   }
   ```

2. **`audio/engine.py` line 678-682 (MARKER_SILENCE):** Add `[INTRO_END]` with 2.5s:
   ```python
   MARKER_SILENCE = {
       '[TRANSITION]': 1.5,
       '[INTRO_END]': 2.5,
       '[PAUSE]': 0.8,
       '[BEAT]': 0.3,
   }
   ```

3. **`audio/engine.py` line 688 (docstring):** Update to mention `[INTRO_END]`.

4. **`pipeline_crew.py` line ~449-457 (polish_task transition instructions):** Change first bullet:
   - Before: `"  - After Channel Intro, before Act 1\n"`
   - After: `"  - After Channel Intro, before Act 1: use [INTRO_END] (not [TRANSITION])\n"`

5. **`pipeline_crew.py` line ~481 (polish expected_output):** Update to mention both marker types:
   - Before: `"8-part structure with [TRANSITION] markers between acts."`
   - After: `"8-part structure with [INTRO_END] after Channel Intro and [TRANSITION] markers between acts."`

6. **`pipeline_script.py` line 204 (transition count validation):** Count both marker types:
   ```python
   transition_count = script_text.count('[TRANSITION]') + script_text.count('[INTRO_END]')
   ```

7. **`pipeline.py` line 1716 (section assembly):** Post-process to replace the first `[TRANSITION]` with `[INTRO_END]`:
   ```python
   assembled = '\n\n[TRANSITION]\n\n'.join(generated)
   assembled = assembled.replace('[TRANSITION]', '[INTRO_END]', 1)  # first occurrence only
   ```

8. **`prompt_strings.py` line 554 (condense EN) and line 572 (condense JA):** Generalize marker preservation:
   - EN: `"Preserve ALL audio markers ([TRANSITION], [INTRO_END], [PAUSE], [BEAT]) exactly as-is\n"`
   - JA: `"すべてのオーディオマーカー（[TRANSITION]、[INTRO_END]、[PAUSE]、[BEAT]）をそのまま保持する\n"`

9. **`pipeline.py` lines 3305, 3332 (accuracy correction):** Generalize to "Preserve all audio markers" instead of only `[TRANSITION]`.

10. **`pipeline_script.py` line 258 (deduplication skip):** Add `[INTRO_END]` to the skip condition:
    ```python
    if all(l.startswith('[TRANSITION]') or l.startswith('[INTRO_END]') or l.startswith('## [') for l in block):
    ```

11. **BGM transition tracking (engine.py lines 341-344, 527-530):** Intentionally NOT adding `[INTRO_END]` to transition_positions_ms — no BGM bump desired after intro (already has BGM intro).

### Files Changed
- `dr2_podcast/audio/engine.py` — AUDIO_MARKERS, MARKER_SILENCE, docstring
- `dr2_podcast/pipeline_crew.py` — polish_task instructions + expected_output
- `dr2_podcast/pipeline_script.py` — validation count + dedup skip
- `dr2_podcast/pipeline.py` — section assembly + accuracy correction
- `dr2_podcast/prompt_strings.py` — condense prompts (EN+JA)

---

## Fix C: Questioner Persona Upgrade — Analytically Trained Generalist

**Files:** `dr2_podcast/pipeline.py`

### Root Cause (3a)

The questioner personality is: "Curious and sharp interviewer who reacts with genuine surprise, playful skepticism, and humor." This gives no intellectual foundation — the model interprets "curious" as naive.

> **Prompt review:** "Ivy League" is culturally American and adds a translation layer for Japanese output. Use trait-based description instead — specify the cognitive skills directly.

### Changes

1. **`pipeline.py` line 639-643 (ROLE_PERSONALITIES["questioner"]):** Rewrite personality (~45 words, trimmed to match presenter length):
   ```python
   "questioner": (
       "Well-educated science generalist with strong analytical training (statistics, "
       "experimental design, causal reasoning) who has researched today's topic beforehand. "
       "Asks informed, probing questions to draw out specifics for the listener. Connects "
       "findings to adjacent domains, notices methodological gaps, and pushes back with "
       "reasoned skepticism when evidence is weak."
   ),
   ```

2. **`pipeline.py` line 666, 671 (stance values):** Update for consistency (cosmetic — not injected into prompts, but aligns metadata):
   - `"stance": "teaching"` → `"stance": "expert"`
   - `"stance": "curious"` → `"stance": "informed"`

### Files Changed
- `dr2_podcast/pipeline.py` — ROLE_PERSONALITIES["questioner"], stance values

---

## Fix D: Dialogue Dynamic — Expert + Prepared Generalist (Not Teacher/Student)

**Files:** `dr2_podcast/pipeline_crew.py`, `dr2_podcast/prompt_strings.py`

### Root Cause (3b)

The current dialogue rules enforce a rigid teacher→student dynamic. The target: presenter is a domain expert, questioner is an informed generalist who has prepared.

> **Prompt review:** Risk of questioner becoming a second lecturer. Add dominance ratio constraint. Use "仮説を投げかけながら" (tossing out hypotheses) not "自らの解釈も交えながら" (mixing in interpretations) — hypotheses invite correction, interpretations assert authority.

### Changes

1. **`pipeline_crew.py` lines 221-226 (producer backstory CRITICAL RULES #4):** Rewrite:
   ```
   4. DIALOGUE DYNAMIC: The Presenter is a domain expert who explains systematically.
      The Questioner is a well-read generalist who has prepared for this episode:
      - Asks methodologically informed questions (sample size, control groups, effect sizes)
      - Connects findings to related fields or common knowledge
      - Pushes back with reasoned skepticism, not just surprise
      - Occasionally tosses out a hypothesis for the Presenter to confirm, refine, or correct
      - Does NOT simply react with emotions — every response adds intellectual substance
      BALANCE: The Questioner contributes ~20-30% of substantive content per exchange.
      They open doors; the Presenter walks through them. If the Questioner's turn runs
      longer than 3 sentences of analysis, redirect to a question.
   ```

2. **`pipeline_crew.py` line 260 (editor backstory):** Update:
   - Before: `"  - Teaching flow: presenter explains, questioner bridges gaps for listeners\n"`
   - After: `"  - Dialogue flow: presenter provides expert depth, questioner contributes informed perspective and probing questions\n"`

3. **`prompt_strings.py` line 407-408 (EN character_roles):** Update:
   - Before: `"asks questions the audience would ask, bridges gaps"`
   - After: `"asks informed, probing questions based on preparation; offers hypotheses for the presenter to refine; draws out specifics for the listener"`

4. **`prompt_strings.py` line 412-413 (JA character_roles):** Update:
   - Before: `"リスナーが聞きたい質問を代弁し、理解のギャップを埋める"`
   - After: `"事前リサーチに基づく的確な質問で専門家の知見を深掘りし、仮説を投げかけながらリスナーの理解を導く"`

5. **`prompt_strings.py` line 625 (EN section_gen system):** Update:
   - Before: `"Maintain consistent roles — {presenter} explains, {questioner} asks and reacts"`
   - After: `"Maintain consistent roles — {presenter} provides expert depth, {questioner} asks informed questions and offers hypotheses"`

6. **`prompt_strings.py` line 643 (JA section_gen system):** Update:
   - Before: `"一貫した役割を維持 — {presenter}が説明し、{questioner}が質問・反応する"`
   - After: `"一貫した役割を維持 — {presenter}が専門知識を提供し、{questioner}が準備に基づく質問と仮説で対話を深める"`

### Files Changed
- `dr2_podcast/pipeline_crew.py` — producer backstory (CRITICAL RULES #4), editor backstory
- `dr2_podcast/prompt_strings.py` — character_roles (EN+JA), section_gen system (EN+JA)

---

## Fix E: Replace Condescending Validation With Substantive Engagement

**Files:** `dr2_podcast/prompt_strings.py`, `dr2_podcast/pipeline_crew.py`

### Root Cause (3c)

Host 2 (presenter) constantly says 「完全にその通り！」「正解です！」「まさにそれ！」— a quiz-show dynamic.

> **Prompt review:** Frame as positive "RESPONSE STYLE" guide first, ban second. Expand replacement patterns to 5-6 per language for reliable model compliance.

### Changes

1. **`prompt_strings.py` EN personality directives (line 380-389):** Add response style guide:
   ```
   - RESPONSE STYLE: When the Questioner makes a point, the Presenter always ADVANCES the conversation.
     Every Presenter response after a Questioner contribution must introduce new information, a new angle, or a qualification.
     Pure agreement ('Exactly!', 'Correct!', 'Spot on!', 'That's right!') adds nothing — replace with:
     'And building on that...', 'That's an interesting angle — actually...', 'Let me add to that...',
     'That makes me think of another angle entirely...', 'There's actually a deeper layer to that...',
     'Mostly, yes — though there's one important caveat...'
   ```

2. **`prompt_strings.py` JA personality directives (line 391-401):** Add equivalent:
   ```
   - 応答スタイル: 質問者が意見を述べた後、プレゼンターは必ず対話を前進させる。
     新しい情報、別の角度、または補足を加えること。
     単純な同意（「正解です」「完全にその通り」「その通り！」「まさにそれ」）は禁止。代わりに:
     「それに加えて…」「面白い視点ですね、実は…」「そこをもう少し掘ると…」
     「その観点から言うと、もう一つ重要なのが…」「実はそこにはもう一層深い話があって…」
     「概ねそうなんですが、一つ注意点があって…」
   ```

3. **`pipeline_crew.py` producer backstory:** Add to CRITICAL RULES:
   ```
   6. NO QUIZ-SHOW: The Presenter must NEVER validate the Questioner with grading phrases
      like "Exactly!", "Correct!", "That's right!". Every response must advance the conversation
      with new information, a new angle, or a qualification.
   ```

### Files Changed
- `dr2_podcast/prompt_strings.py` — personality directives (EN+JA)
- `dr2_podcast/pipeline_crew.py` — producer backstory CRITICAL RULES

---

## Fix F: Deeper Question Examples — Domain-Adaptive Probing

**Files:** `dr2_podcast/prompt_strings.py`

### Root Cause (3d)

No examples of "good" deep-dive questions in either language. The model defaults to surface-level reactions.

> **Prompt review:** Clinical-only examples will misfire on social science topics. Add domain-adaptive examples. Use natural Japanese indirect questioning style for JA examples.

### Changes

1. **`prompt_strings.py` EN personality directives:** Add domain-adaptive question depth:
   ```
   - QUESTION DEPTH: Questioner asks research-informed questions, not surface reactions.
     For science/medical topics:
       GOOD: 'What was the sample size?', 'Did they control for socioeconomic status?', 'What's the effect size — is that clinically meaningful?'
     For social science/policy topics:
       GOOD: 'Has that effect been replicated in other countries?', 'How was the comparison group set up?', 'What's the cost-benefit ratio of that intervention?'
     BAD (always): 'Wow, really?', 'That's amazing!', 'Tell me more about that'
   ```

2. **`prompt_strings.py` JA personality directives:** Add equivalent with natural JA probing style:
   ```
   - 質問の深さ: 質問者は表面的なリアクション（「へー」「すごい」）ではなく、研究の核心に迫る質問をする。
     科学・医学トピックの場合:
       良い例: 「そのメタ分析のサンプルサイズはどれくらいですか？」「交絡因子はコントロールされていましたか？」「効果量はどの程度ですか？」
     社会科学・政策トピックの場合:
       良い例: 「その知見は日本の文脈にも当てはまりますか？」「先行研究との整合性はどうでしょう？」「そのデータの解釈には別の可能性もありませんか？」
     悪い例（常に）: 「へー、それってどんな研究だったんですか？」「すごいですね」
   ```

3. **`prompt_strings.py` act-specific instructions (lines 239-322):** Strengthen per-act questioner behavior (both EN and JA):

   **Act 1 (lines 239-253):**
   - EN line 244: Before: `"Questioner validates: 'Right, I've heard that too'"` → After: `"Questioner connects the claim to something they encountered in preparation: 'I read that this traces back to...' or 'The common argument seems to be...'"`
   - JA line 251: Before: `"質問者が共感: 「確かに、私もそう聞いていました」"` → After: `"質問者が準備で調べた内容と関連づける: 「調べてみたところ、これは〜に由来するらしいですね」"`

   **Act 2 (lines 256-280):** Already says "Questioner asks a study-specific question" — add methodological depth:
   - EN line 262: append `"— focus on methodology: sample size, study design, confounders, effect sizes"`
   - JA line 274: append `"— 方法論に焦点: サンプルサイズ、研究デザイン、交絡因子、効果量"`

   **Act 3 (lines 283-301):**
   - EN line 290: Before: `"Questioner: 'So if I had to summarize everything we just discussed...'"` → After: `"Questioner offers their own synthesis attempt for the Presenter to refine: 'So if I had to pull this together, it seems like...'"`
   - JA line 299: Before: `"質問者: 「では、今まで話したことをまとめると...」"` → After: `"質問者が自分なりの統合を試み、プレゼンターに洗練を求める: 「ここまでの話をまとめると、つまり…ということでしょうか？」"`

   **Act 4 (lines 304-322):** Already good ("Questioner voices listener challenges") — no change needed.

### Files Changed
- `dr2_podcast/prompt_strings.py` — personality directives (EN+JA), act instructions (EN+JA: Act 1, 2, 3)

---

## Warmth Safety Valve

> **Prompt review flagged:** Fixes A+C+D+E+F all push toward more intellectual, controlled output. Without a counterbalance, the podcast may become emotionally flat.

Add to **both EN and JA section_gen system prompts** (`prompt_strings.py` lines 611-648):

- EN: `"Despite these intellectual standards, the conversation must feel like two friends talking — not a panel discussion. Include genuine laughter, moments of wonder, and personal vulnerability. Intelligence and warmth are not opposites.\n"`
- JA: `"知的水準を保ちつつも、二人の友人が話しているような温かさを維持する。笑い、驚き、個人的な共感を忘れない。\n"`

---

## Instruction Budget Management

> **Prompt review flagged:** ~400-500 tokens of new instructions added. Existing instructions should be trimmed where they overlap with new fixes.

1. **Shorten existing REACTIONS directive** (lines 383/394) — Fix C/D/F now define how the questioner reacts. The REACTIONS bullet can be condensed to a single line referencing "authentic reactions per the character role" instead of listing examples that conflict with the register floor.

2. **Update FILLERS examples** (lines 385/396) to align with Fix A register — already addressed in Fix A.

3. **The BANTER, EMPHASIS, STORYTELLING, PERSONAL, MOMENTUM** bullets remain unchanged — they don't conflict with new fixes and provide valuable variety guidance.

---

## Implementation Order

| Fix | Priority | Scope | Files Changed |
|-----|----------|-------|---------------|
| C — Questioner persona | Critical | EN+JA | pipeline.py |
| D — Dialogue dynamic | Critical | EN+JA | pipeline_crew.py, prompt_strings.py |
| E — Response style | Critical | EN+JA | prompt_strings.py, pipeline_crew.py |
| F — Deep questions | Critical | EN+JA | prompt_strings.py |
| A — Register floor | High | JA | prompt_strings.py |
| Warmth + Budget | High | EN+JA | prompt_strings.py |
| B — Audio transition | High | Audio | engine.py, pipeline_crew.py, pipeline_script.py, pipeline.py, prompt_strings.py |

**Recommended order:** C → D → E → F → A → Warmth/Budget → B

---

## Verification

After all changes:
1. Syntax-check all modified `.py` files with `ast.parse()`
2. Run `python -m pytest tests/ --tb=short -q` to ensure no regressions
3. Spot-check prompt token counts — section_gen system prompt should stay under 800 tokens
4. Run a test generation with the same topic to compare output quality
5. Verify `[INTRO_END]` marker produces 2.5s silence in audio
6. Spot-check generated script for:
   - No 「やばい」/「マジ」/「正解です」/「完全にその通り」
   - Questioner asks methodological questions (not just "へー")
   - Presenter builds on points instead of grading them
   - Conversation maintains warmth despite intellectual depth
