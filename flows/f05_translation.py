"""
flows/f05_translation.py — Translation of research documents for non-English languages.

No-op for English. For Japanese, uses the smart LLM to translate
the source-of-truth and key research docs.
"""

from pathlib import Path

from crewai import Agent, Task, Crew

from shared.models import PipelineParams, TranslationResult
from shared.config import SUPPORTED_LANGUAGES, build_llm_instances


def run_translation(
    params: PipelineParams,
    source_of_truth: str,
    supporting_research: str,
    output_dir: Path,
    session_roles: dict,
) -> TranslationResult:
    """Translate research documents if language != 'en'.

    For English, returns inputs unchanged (no-op).
    For other languages, uses smart LLM to translate.
    """
    result = TranslationResult(output_dir=output_dir)
    language = params.language
    language_config = SUPPORTED_LANGUAGES[language]

    if language == "en":
        # No-op: return original text
        result.translated_source_of_truth = source_of_truth
        result.translated_supporting = supporting_research
        return result

    print(f"\n{'='*70}")
    print(f"TRANSLATION PHASE: Translating to {language_config['name']}")
    print(f"{'='*70}")

    target_instruction = language_config['instruction']

    # Build LLM for translation (use creative for natural output)
    _, creative_llm = build_llm_instances()

    presenter = session_roles['presenter']['character']
    questioner = session_roles['questioner']['character']

    translator = Agent(
        role='Research Translator',
        goal=f'Translate research documents into {language_config["name"]} for podcast production.',
        backstory=(
            f'Expert bilingual translator specializing in scientific communication. '
            f'You translate for natural spoken delivery, not literal translation. '
            f'Keep proper nouns, study names, journal names in English. '
            f'Maintain scientific terminology accuracy.'
            + (f'\nCRITICAL: Output MUST be in Japanese (\u65e5\u672c\u8a9e) only. Do NOT switch to Chinese (\u4e2d\u6587). '
               f'Use katakana for host names: \u30ab\u30ba and \u30a8\u30ea\u30ab (NOT \u5361\u5179/\u57c3\u91cc\u5361). '
               f'Avoid Kanji only used in Chinese (e.g., use \u6c17 instead of \u6c14, \u697d instead of \u4e50).'
               if language == 'ja' else '')
        ),
        llm=creative_llm,
        verbose=True,
    )

    translate_task = Task(
        description=(
            f"Translate the following Source-of-Truth research document "
            f"about the podcast topic into {language_config['name']}.\n\n"
            f"RULES:\n"
            f"- Preserve {presenter}: / {questioner}: format exactly\n"
            f"- Preserve scientific terminology accuracy\n"
            f"- Translate for natural spoken delivery, not literal translation\n"
            f"- Keep proper nouns, study names, journal names in English\n"
            f"- Maintain teaching structure and conversational flow\n\n"
            f"DOCUMENT TO TRANSLATE:\n{source_of_truth[:8000]}\n"
            f"--- END DOCUMENT ---\n\n"
            f"{target_instruction}"
        ),
        expected_output=f"Complete translated document in {language_config['name']}.",
        agent=translator,
    )

    crew = Crew(
        agents=[translator],
        tasks=[translate_task],
        verbose=True,
        process='sequential',
    )

    try:
        crew.kickoff()
        translated = translate_task.output.raw if hasattr(translate_task, 'output') and translate_task.output else ""
        result.translated_source_of_truth = translated if translated else source_of_truth

        # Save translated doc
        outfile = output_dir / "source_of_truth_translated.md"
        with open(outfile, 'w') as f:
            f.write(result.translated_source_of_truth)
        print(f"Translation saved: {outfile} ({len(result.translated_source_of_truth)} chars)")

    except Exception as e:
        print(f"Translation failed: {e}")
        print("Falling back to original language documents")
        result.translated_source_of_truth = source_of_truth

    result.translated_supporting = supporting_research
    return result


if __name__ == "__main__":
    print("Usage: This module is called by the orchestrator.")
    print("  from flows.f05_translation import run_translation")
