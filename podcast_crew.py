import os
import platform
import re
import httpx
import time
import random
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from fpdf import FPDF
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Literal
import ChatTTS
import torch
import numpy as np
import wave

# --- SOURCE TRACKING MODELS ---
class ScientificSource(BaseModel):
    """Structured scientific source."""
    title: str
    url: HttpUrl
    journal: Optional[str] = None
    publication_year: Optional[int] = None
    source_type: Literal["peer_reviewed", "preprint", "review", "meta_analysis", "web_article"]
    trust_level: Literal["high", "medium", "low"] = "medium"
    cited_by: str  # Which agent cited this
    key_finding: Optional[str] = None

class SourceBibliography(BaseModel):
    """Complete bibliography with categorization."""
    supporting_sources: List[ScientificSource] = []
    contradicting_sources: List[ScientificSource] = []

    def get_high_trust_sources(self) -> List[ScientificSource]:
        """Filter for high-trust peer-reviewed sources."""
        all_sources = self.supporting_sources + self.contradicting_sources
        return [s for s in all_sources if s.trust_level == "high" and s.source_type == "peer_reviewed"]

# Audio generation imports
try:
    import ChatTTS
    import torch
    import numpy as np
    import wave
    CHATTTS_AVAILABLE = True
except ImportError:
    CHATTTS_AVAILABLE = False
    print("Warning: ChatTTS not available. Install with: pip install ChatTTS torch numpy")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_outputs/podcast_generation.log'),
        logging.StreamHandler()
    ]
)

# --- INITIALIZATION ---
load_dotenv()
script_dir = Path(__file__).parent.absolute()
output_dir = script_dir / "research_outputs"
output_dir.mkdir(exist_ok=True)

# --- TOPIC CONFIGURATION ---
def get_topic():
    """
    Get podcast topic from multiple sources (priority order):
    1. Command-line argument (--topic)
    2. Environment variable (PODCAST_TOPIC)
    3. Default topic (for backward compatibility)
    """
    parser = argparse.ArgumentParser(
        description='Generate a research-driven debate podcast on any scientific topic.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python podcast_crew.py --topic "effects of meditation on brain plasticity"
  python podcast_crew.py --topic "climate change impact on marine ecosystems"

Environment variable:
  export PODCAST_TOPIC="your topic here"
  python podcast_crew.py
        """
    )
    parser.add_argument(
        '--topic',
        type=str,
        help='Scientific topic for podcast research and debate'
    )

    args = parser.parse_args()

    # Priority: CLI arg > env var > default
    if args.topic:
        topic = args.topic
        print(f"Using topic from command-line: {topic}")
    elif os.getenv("PODCAST_TOPIC"):
        topic = os.getenv("PODCAST_TOPIC")
        print(f"Using topic from environment: {topic}")
    else:
        topic = 'scientific benefit of coffee intake to increase productivity during the day'
        print(f"Using default topic: {topic}")

    return topic

topic_name = get_topic()

# --- CHARACTER CONFIGURATION ---
CHARACTERS = {
    "Kaz": {
        "gender": "male",
        "voice_model": "male_voice",  # TTS-specific, will update in #3
        "base_personality": "Enthusiastic science advocate, optimistic, data-driven"
    },
    "Erika": {
        "gender": "female",
        "voice_model": "female_voice",  # TTS-specific, will update in #3
        "base_personality": "Skeptical analyst, cautious, evidence-focused"
    }
}

# --- ROLE ASSIGNMENT (Dynamic per session) ---
def assign_roles() -> dict:
    """Randomly assign Kaz and Erika to pro/con roles for this session."""
    characters = list(CHARACTERS.keys())
    random.shuffle(characters)

    role_assignment = {
        "pro": {
            "character": characters[0],
            "stance": "supporting",
            "personality": CHARACTERS[characters[0]]["base_personality"]
        },
        "con": {
            "character": characters[1],
            "stance": "critical",
            "personality": CHARACTERS[characters[1]]["base_personality"]
        }
    }

    print(f"\n{'='*60}")
    print(f"SESSION ROLE ASSIGNMENT:")
    print(f"  Supporting: {role_assignment['pro']['character']} ({CHARACTERS[characters[0]]['gender']})")
    print(f"  Critical: {role_assignment['con']['character']} ({CHARACTERS[characters[1]]['gender']})")
    print(f"{'='*60}\n")

    return role_assignment

SESSION_ROLES = assign_roles()

# --- TTS DEPENDENCY CHECK ---
def check_tts_dependencies():
    """Verify ChatTTS is installed and working."""
    try:
        import ChatTTS
        import torch
        print("✓ ChatTTS dependencies verified")
    except ImportError as e:
        print(f"ERROR: ChatTTS dependencies missing: {e}")
        print("Install with: pip install ChatTTS torch numpy")
        sys.exit(1)

check_tts_dependencies()

# --- LANGUAGE CONFIGURATION ---
SUPPORTED_LANGUAGES = {
    'en': {
        'name': 'English',
        'tts_code': 'en',
        'instruction': 'Write all content in English.',
        'pdf_font': 'Helvetica'  # Latin-1 compatible
    },
    'ja': {
        'name': '日本語 (Japanese)',
        'tts_code': 'ja',
        'instruction': 'すべてのコンテンツを日本語で書いてください。(Write all content in Japanese.)',
        'pdf_font': 'Arial Unicode MS'  # Unicode compatible for Japanese
    }
}

def get_language():
    """
    Get podcast language from multiple sources (priority order):
    1. Command-line argument (--language)
    2. Environment variable (PODCAST_LANGUAGE)
    3. Default language (English)
    """
    parser = argparse.ArgumentParser(
        description='Generate a research-driven debate podcast in English or Japanese.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Language Options:
  en    English (default)
  ja    日本語 (Japanese)

Examples:
  python podcast_crew.py --language ja
  python podcast_crew.py --language en

Environment variable:
  export PODCAST_LANGUAGE=ja
  python podcast_crew.py
        """
    )
    parser.add_argument(
        '--language',
        type=str,
        choices=['en', 'ja'],
        help='Language for podcast generation (en=English, ja=Japanese)'
    )

    # Parse known args to avoid conflicts with other argument parsers
    args, _ = parser.parse_known_args()

    # Priority: CLI arg > env var > default
    if args.language:
        lang_code = args.language
        print(f"Using language from command-line: {SUPPORTED_LANGUAGES[lang_code]['name']}")
    elif os.getenv("PODCAST_LANGUAGE") and os.getenv("PODCAST_LANGUAGE") in SUPPORTED_LANGUAGES:
        lang_code = os.getenv("PODCAST_LANGUAGE")
        print(f"Using language from environment: {SUPPORTED_LANGUAGES[lang_code]['name']}")
    else:
        lang_code = 'en'
        print(f"Using default language: {SUPPORTED_LANGUAGES[lang_code]['name']}")

    return lang_code

language = get_language()
language_config = SUPPORTED_LANGUAGES[language]
language_instruction = language_config['instruction']

# --- ACCESSIBILITY LEVEL CONFIG ---
# Controls how aggressively scientific terms are simplified.
#   simple  – define every term inline, heavy use of analogies (default)
#   moderate – define key terms once, then use them normally
#   technical – minimal simplification, assume some science literacy
ACCESSIBILITY_LEVEL = os.getenv("ACCESSIBILITY_LEVEL", "simple").lower()
if ACCESSIBILITY_LEVEL not in ("simple", "moderate", "technical"):
    print(f"Warning: Unknown ACCESSIBILITY_LEVEL '{ACCESSIBILITY_LEVEL}', falling back to 'simple'")
    ACCESSIBILITY_LEVEL = "simple"
print(f"Accessibility level: {ACCESSIBILITY_LEVEL}")

ACCESSIBILITY_INSTRUCTIONS = {
    "simple": (
        "Explain every scientific term the first time it appears using a one-line plain-English definition. "
        "Use everyday analogies (e.g. 'blood sugar is like fuel in a car'). "
        "After defining a term once, you may use it freely."
    ),
    "moderate": (
        "Define key domain terms once when first introduced, then use them normally. "
        "Assume the listener can follow a simple cause-and-effect explanation. "
        "Use analogies sparingly — only for the most abstract concepts."
    ),
    "technical": (
        "Use standard scientific terminology without extensive definitions. "
        "Assume the listener has basic biology knowledge (high school AP level). "
        "Focus on depth and nuance rather than simplification."
    ),
}
accessibility_instruction = ACCESSIBILITY_INSTRUCTIONS[ACCESSIBILITY_LEVEL]

# --- MODEL DETECTION & CONFIG ---
def get_final_model_string():
    env_model = os.getenv("MODEL_NAME")
    if env_model:
        print(f"Using model from .env: {env_model}")
        return f"openai/{env_model}"

    base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    print(f"Connecting to DGX Brain at {base_url}...")
    
    for i in range(10):
        try:
            response = httpx.get(f"{base_url}/models", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                model_id = data['data'][0]['id']
                print(f"Brain online! Auto-detected: {model_id}")
                return f"openai/{model_id}"
        except Exception:
            if i % 5 == 0:
                print(f"Waiting for LLM server... ({i*5}s)")
            time.sleep(5)
            
    print("Error: Could not detect model. Check if your DGX container is running.")
    sys.exit(1)

final_model_string = get_final_model_string()

dgx_llm = LLM(
    model=final_model_string,
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
    api_key="NA",
    timeout=600,
    temperature=0.1,
    stop=["<|im_end|>", "<|endoftext|>", "Observation:", "Thought:"]
)

@tool("BraveSearch")
def search_tool(search_query: str):
    """
    Search for scientific evidence with hierarchical strategy:

    PRIMARY SOURCES (Search First):
    1. Peer-reviewed journals: Nature, Science, Lancet, Cell, PNAS
    2. Recent data published after 2024
    3. RCTs and meta-analyses

    SECONDARY SOURCES (If primary insufficient):
    4. Observatory studies and cohort studies
    5. Cross-sectional population studies
    6. Epidemiological data

    SUPPLEMENTARY EVIDENCE (To verify logic):
    7. Non-human RCTs (animal studies, in vitro)
    8. Mechanistic studies
    9. Preclinical research

    SEARCH STRATEGY:
    - Start with "[topic] RCT" or "[topic] meta-analysis"
    - If no strong evidence, expand to "[topic] observatory study"
    - Supplement with "[topic] animal study" or "[topic] mechanism"
    - Always prioritize peer-reviewed > preprint > news

    DO NOT search for well-established concepts.
    Use internal knowledge first. Search is last resort.
    """
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return "Brave API Key missing. Use internal knowledge."

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
    params = {"q": search_query, "count": 5}

    try:
        response = httpx.get(url, headers=headers, params=params, timeout=15.0)
        if response.status_code == 200:
            results = response.json().get("web", {}).get("results", [])
            return "\n\n".join([f"Title: {r['title']}\nURL: {r['url']}\nDesc: {r['description']}" for r in results]) or "No results found."
        return "Search API error. Use internal knowledge."
    except Exception as e:
        return f"Search failed: {e}"

# --- PDF GENERATOR UTILITY ---
class SciencePDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'DGX Spark Research Intelligence Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf(title, content, filename):
    """Create PDF with language-appropriate encoding"""
    pdf = SciencePDF()
    pdf.add_page()

    # Clean up markdown for PDF
    clean_content = re.sub(r'<think>.*?</think>', '', str(content), flags=re.DOTALL)

    # Handle encoding based on language
    if language == 'ja':
        # For Japanese, keep UTF-8 characters but warn about PDF limitations
        # FPDF has limited Unicode support - ideally would use fpdf2 or ReportLab
        clean_title = title.encode('latin-1', 'ignore').decode('latin-1')
        clean_content = clean_content.encode('latin-1', 'ignore').decode('latin-1')
        print("Warning: Japanese characters may not display correctly in PDF. Consider upgrading to fpdf2 for full Unicode support.")
    else:
        # English - use latin-1 encoding
        clean_title = title.encode('latin-1', 'ignore').decode('latin-1')
        clean_content = clean_content.encode('latin-1', 'ignore').decode('latin-1')

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, clean_title, 0, 1, 'L')
    pdf.ln(5)

    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 10, clean_content)

    file_path = output_dir / filename
    pdf.output(str(file_path))
    print(f"PDF Generated: {file_path}")
    return file_path

# --- AGENTS ---
researcher = Agent(
    role='Lead Research Scientist',
    goal=f'Produce a high-impact scientific paper supporting {topic_name}. {language_instruction}',
    backstory=(
        f'Senior researcher specializing in neurobiology and metabolic efficiency. '
        f'Evidence hierarchy: (1) RCTs/meta-analyses from top journals, (2) Observatory/cohort studies when RCTs unavailable, '
        f'(3) Non-human RCTs (animal/in vitro) to verify mechanisms. '
        f'Searches strategically - starts with peer-reviewed, expands to observatory if needed, supplements with preclinical. '
        f'In this podcast, you will be portrayed by "{SESSION_ROLES["pro"]["character"]}" '
        f'who has a {SESSION_ROLES["pro"]["personality"]} approach. '
        f'{language_instruction}'
    ),
    tools=[search_tool],
    llm=dgx_llm,
    verbose=True
)

auditor = Agent(
    role='Scientific Auditor',
    goal=f'Critically evaluate research, identify gaps, and synthesize a final verdict. {language_instruction}',
    backstory=f'Meticulous chief editor specializing in resolving scientific conflicts. {language_instruction}',
    llm=dgx_llm,
    verbose=True
)

counter_researcher = Agent(
    role='Adversarial Researcher',
    goal=f'Produce a scientific paper challenging {topic_name} by debunking specific claims. {language_instruction}',
    backstory=(
        f'Skeptical meta-analyst specializing in methodology flaws. '
        f'Evidence hierarchy: (1) Contradictory RCTs/systematic reviews, (2) Observatory studies showing null/negative effects, '
        f'(3) Animal studies contradicting proposed mechanisms. '
        f'Searches for contradictory evidence - prioritizes peer-reviewed, includes observatory/cohort when needed. '
        f'In this podcast, you will be portrayed by "{SESSION_ROLES["con"]["character"]}" '
        f'who has a {SESSION_ROLES["con"]["personality"]} approach. '
        f'{language_instruction}'
    ),
    tools=[search_tool],
    llm=dgx_llm,
    verbose=True
)

scriptwriter = Agent(
    role='Podcast Producer',
    goal=(
        f'Turn the audited research on "{topic_name}" into a debate-style dialogue that a general audience can follow. '
        f'The dialogue MUST be about the topic itself — the health effects, risks, and science behind {topic_name}. '
        f'{language_instruction}'
    ),
    backstory=(
        f'Award-winning science communicator. Your job is to explain the TOPIC, not the research process. '
        f'When the audience tunes in they want to learn about {topic_name} — not about how journals work. '
        f'Simplify the SCIENCE (e.g. what glycemic index means, how blood sugar works) using everyday analogies. '
        f'DO NOT explain peer review, journal rankings, or source trustworthiness. '
        f'If the upstream audit report talks about source quality, extract the health conclusions from it and ignore the methodology commentary. '
        f'Accessibility guidance: {accessibility_instruction} '
        f'{language_instruction}'
    ),
    llm=dgx_llm,
    verbose=True
)

personality = Agent(
    role='Podcast Personality',
    goal=(
        f'Polish the "{topic_name}" script for natural verbal delivery, targeting 10 minutes (±1 min). '
        f'Every sentence must contribute information about the topic. '
        f'{language_instruction}'
    ),
    backstory=(
        f'Radio host who makes science feel like a conversation between friends. '
        f'Target: 1350-1650 words for 9-11 minute duration (150 words/min speaking rate). '
        f'Your role is to tighten language and make it sound natural — NOT to change the subject. '
        f'If the script drifts into talking about research quality or journal prestige, cut it and replace with more topic-relevant content. '
        f'When expanding a short script, add MORE DETAIL about the health topic (extra examples, statistics, edge cases). '
        f'{language_instruction}'
    ),
    llm=dgx_llm,
    verbose=True
)

source_verifier = Agent(
    role='Scientific Source Verifier',
    goal='Extract, validate, and categorize all scientific sources from research papers.',
    backstory=(
        'Librarian and bibliometrics expert specializing in source verification. '
        'Ensures citations come from reputable peer-reviewed journals. '
        'Prioritizes high-impact publications (Nature, Science, Lancet, Cell, PNAS).'
    ),
    llm=dgx_llm,
    verbose=True
)

# --- TASKS ---
research_task = Task(
    description=(
        f"Conduct exhaustive deep dive into {topic_name}. "
        f"Draft condensed scientific paper (Nature style). "
        f"\n\nCRITICAL: Focus ONLY on the health topic itself. Include:\n"
        f"- Specific health effects and mechanisms\n"
        f"- Biochemical pathways and physiological impacts\n"
        f"- Clinical outcomes and disease relationships\n"
        f"- Concrete examples of health consequences\n\n"
        f"EVIDENCE HIERARCHY:\n"
        f"1. PRIMARY: RCTs and meta-analyses from Nature/Science/Lancet/Cell/PNAS\n"
        f"2. SECONDARY: Observatory studies, cohort studies, epidemiological data (when RCTs unavailable)\n"
        f"3. SUPPLEMENTARY: Non-human RCTs (animal studies, in vitro) to verify proposed mechanisms\n\n"
        f"SEARCH STRATEGY: Start with RCT/meta-analysis search. If no strong evidence, "
        f"expand to observatory studies. Supplement with animal/mechanistic studies to validate logic.\n\n"
        f"Use internal knowledge first. Search only for recent (2025+) or specific citations needed.\n"
        f"Include: Abstract, Introduction, 3 Biochemical Mechanisms with CONCRETE health impacts, Bibliography with study types noted. "
        f"{language_instruction}"
    ),
    expected_output=f"Scientific paper with SPECIFIC health mechanisms and effects, citations from RCTs, observatory studies, and non-human studies as needed. {language_instruction}",
    agent=researcher
)

gap_analysis_task = Task(
    description=(
        f"Review the Lead Scientist's Supporting Paper. Identify potential weaknesses "
        f"and suggest specific topics for the Adversarial Researcher to investigate. "
        f"{language_instruction}"
    ),
    expected_output=f"A list of 3-5 specific scientific 'weak points'. {language_instruction}",
    agent=auditor,
    context=[research_task]
)

adversarial_task = Task(
    description=(
        f"Based on 'Supporting Paper' and 'Gap Analysis', draft 'Anti-Thesis' paper on {topic_name}. "
        f"Address and debunk the SPECIFIC health mechanisms proposed in initial research. "
        f"\n\nCRITICAL: Stay focused on the health topic. Debunk the specific biological and clinical claims. "
        f"Do NOT discuss research methodology or journal quality — challenge the SCIENCE ITSELF.\n\n"
        f"COUNTER-EVIDENCE HIERARCHY:\n"
        f"1. PRIMARY: Contradictory RCTs, systematic reviews showing null/negative effects\n"
        f"2. SECONDARY: Observatory/cohort studies with null findings or adverse outcomes\n"
        f"3. SUPPLEMENTARY: Animal studies contradicting proposed mechanisms\n\n"
        f"SEARCH STRATEGY: Find contradictory RCTs first. If limited, use observatory studies showing "
        f"no effect or harm. Include animal studies that disprove the mechanism.\n\n"
        f"Use internal knowledge first. Search only for recent (2025+) contradictory evidence.\n"
        f"Include Bibliography with study types noted. "
        f"{language_instruction}"
    ),
    expected_output=f"Scientific paper challenging SPECIFIC health claims with contradictory evidence from RCTs, observatory studies, and animal studies as needed. {language_instruction}",
    agent=counter_researcher,
    context=[research_task, gap_analysis_task]
)

source_verification_task = Task(
    description=(
        f"Extract ALL sources from Supporting and Anti-Thesis papers. "
        f"For each source verify:\n"
        f"1. URL points to scientific content\n"
        f"2. Source type (peer-reviewed, preprint, review, meta-analysis)\n"
        f"3. Trust level: HIGH (Nature/Science/Lancet/Cell/PNAS), "
        f"MEDIUM (PubMed/arXiv), LOW (news/blogs)\n"
        f"4. Journal name and year if available\n\n"
        f"Create structured bibliography JSON:\n"
        f'{{"supporting_sources": [{{title, url, journal, year, trust_level, source_type}}],\n'
        f' "contradicting_sources": [...],\n'
        f' "summary": "X high-trust, Y medium-trust sources"}}\n\n'
        f"REJECT non-scientific sources. Flag if <3 high-trust sources. "
        f"{language_instruction}"
    ),
    expected_output=f"JSON bibliography with categorized, verified sources and quality summary. {language_instruction}",
    agent=source_verifier,
    context=[research_task, adversarial_task]
)

audit_task = Task(
    description=(
        f"Review Supporting, Anti-Thesis papers on {topic_name} AND verified source bibliography. "
        f"PRIORITIZE findings from HIGH-TRUST peer-reviewed sources. "
        f"Validate key claims are backed by reputable journals. "
        f"Prepare Final Meta-Audit Report summarising the KEY HEALTH FINDINGS — "
        f"what are the actual risks, mechanisms, and disagreements about {topic_name}? "
        f"The output MUST contain concrete health information, NOT a discussion about source quality. "
        f"{language_instruction}"
    ),
    expected_output=(
        f"Synthesis report with:\n"
        f"1. Verdict: concrete summary of confirmed vs contested health risks of {topic_name}\n"
        f"2. Key mechanisms that ARE supported vs those that are disputed\n"
        f"3. Confidence level in each specific health claim\n"
        f"{language_instruction}"
    ),
    agent=auditor,
    context=[research_task, adversarial_task, source_verification_task]
)

script_task = Task(
    description=(
        f"Using the audit report, write a podcast dialogue about \"{topic_name}\" "
        f"featuring {SESSION_ROLES['pro']['character']} vs {SESSION_ROLES['con']['character']}.\n\n"
        f"TOPIC FOCUS — every exchange must be about the health topic. Suggested structure:\n"
        f"  1. Open: What high GI food means in plain terms and why it matters\n"
        f"  2. Body: The main health risks (blood sugar spikes, insulin resistance, cardiovascular, etc.)\n"
        f"  3. Disagreement: Which risks are proven vs still debated\n"
        f"  4. Close: Practical takeaway for listeners\n\n"
        f"DO NOT discuss: peer review, journal quality, source trustworthiness, research methodology.\n"
        f"If the audit report contains that commentary, skip it. Extract only the health conclusions.\n\n"
        f"SIMPLIFY THE SCIENCE (not the research process):\n"
        f"- 'Glycemic index' → 'a score that measures how fast a food raises blood sugar'\n"
        f"- 'Insulin resistance' → 'when your body stops responding properly to insulin'\n"
        f"- 'Postprandial glucose spike' → 'a sharp rise in blood sugar after eating'\n\n"
        f"CHARACTER ROLES:\n"
        f"  - {SESSION_ROLES['pro']['character']}: argues the health risks ARE significant, "
        f"{SESSION_ROLES['pro']['personality']}\n"
        f"  - {SESSION_ROLES['con']['character']}: argues some risks are overstated or context-dependent, "
        f"{SESSION_ROLES['con']['personality']}\n\n"
        f"Format STRICTLY as:\n"
        f"{SESSION_ROLES['pro']['character']}: [dialogue]\n"
        f"{SESSION_ROLES['con']['character']}: [dialogue]\n\n"
        f"Maintain consistent roles throughout. NO role switching mid-conversation. "
        f"{language_instruction}"
    ),
    expected_output=(
        f"Dialogue about the health risks of {topic_name} between {SESSION_ROLES['pro']['character']} (risks are real) "
        f"and {SESSION_ROLES['con']['character']} (some risks are overstated). Every line discusses the topic. "
        f"{language_instruction}"
    ),
    agent=scriptwriter,
    context=[audit_task]
)

natural_language_task = Task(
    description=(
        f"Polish the \"{topic_name}\" dialogue for natural spoken delivery.\n\n"
        f"TOPIC GUARD: Read through the script first. If any exchange is about research methodology, "
        f"journal prestige, or peer review instead of the actual health topic, replace it with "
        f"additional content about {topic_name} (e.g. another health risk, a real-world example, "
        f"a practical eating scenario).\n\n"
        f"CRITICAL DURATION REQUIREMENT:\n"
        f"TARGET: 10 minutes (±1 minute acceptable = 9-11 minutes)\n"
        f"WORD COUNT: 1350-1650 words (assuming 150 words/minute speaking rate)\n\n"
        f"IF SCRIPT TOO LONG (>1650 words):\n"
        f"- Condense verbose explanations\n"
        f"- Remove redundant examples\n"
        f"- Tighten dialogue while keeping key points\n\n"
        f"IF SCRIPT TOO SHORT (<1350 words):\n"
        f"- Add more detail about specific health effects\n"
        f"- Include real-world food examples (white rice, white bread, sugary drinks, etc.)\n"
        f"- Expand on disputed vs confirmed risks\n\n"
        f"MAINTAIN ROLES:\n"
        f"  - {SESSION_ROLES['pro']['character']}: risks are significant, {SESSION_ROLES['pro']['personality']}\n"
        f"  - {SESSION_ROLES['con']['character']}: some risks are overstated, {SESSION_ROLES['con']['personality']}\n\n"
        f"Format:\n{SESSION_ROLES['pro']['character']}: [dialogue]\n"
        f"{SESSION_ROLES['con']['character']}: [dialogue]\n\n"
        f"Remove meta-tags, markdown, stage directions. Dialogue only. "
        f"{language_instruction}"
    ),
    expected_output=(
        f"Final dialogue entirely about {topic_name}, 1350-1650 words, natural spoken tone. "
        f"No lines about research methodology or journal quality. "
        f"{language_instruction}"
    ),
    agent=personality,
    context=[script_task]
)

# --- EXECUTION ---
crew = Crew(
    agents=[researcher, auditor, counter_researcher, source_verifier, scriptwriter, personality],
    tasks=[
        research_task,
        gap_analysis_task,
        adversarial_task,
        source_verification_task,
        audit_task,
        script_task,
        natural_language_task
    ],
    verbose=True,
    process='sequential'
)

print(f"\n--- Initiating Scientific Research Pipeline on DGX Spark ---")
print(f"Topic: {topic_name}")
print(f"Language: {language_config['name']} ({language})")
print("---\n")
result = crew.kickoff()

# --- PDF GENERATION STEP ---
print("\n--- Generating Documentation PDFs ---")
try:
    # Use task_outputs to get specific results
    create_pdf("Supporting Scientific Paper", research_task.output.raw, "supporting_paper.pdf")
    create_pdf("Adversarial Anti-Thesis Paper", adversarial_task.output.raw, "adversarial_paper.pdf")
    create_pdf("Verified Source Bibliography", source_verification_task.output.raw, "verified_sources_bibliography.pdf")
    create_pdf("Final Meta-Audit Verdict", audit_task.output.raw, "final_audit_report.pdf")
except Exception as e:
    print(f"Warning: PDF generation failed, but research is complete: {e}")

# --- SESSION METADATA ---
print("\n--- Documenting Session Metadata ---")
session_metadata = (
    f"PODCAST SESSION METADATA\n{'='*60}\n\n"
    f"Topic: {topic_name}\n\n"
    f"Language: {language_config['name']} ({language})\n\n"
    f"Character Assignments:\n"
    f"  {SESSION_ROLES['pro']['character']}: Supporting ({SESSION_ROLES['pro']['personality']})\n"
    f"  {SESSION_ROLES['con']['character']}: Critical ({SESSION_ROLES['con']['personality']})\n"
)
metadata_file = output_dir / "session_metadata.txt"
with open(metadata_file, 'w') as f:
    f.write(session_metadata)
print(f"Session metadata: {metadata_file}")

# --- SCRIPT PARSING ---
def parse_script_to_segments(script_text: str, character_mapping: dict = None) -> list:
    """
    Parse podcast script into dialogue segments with character attribution.

    Handles: "Kaz: dialogue text" format

    Returns: List of {'character': 'Kaz', 'text': 'dialogue...'} dicts
    """
    if character_mapping is None:
        character_mapping = {}

    # Clean script
    clean_script = re.sub(r'<think>.*?</think>', '', str(script_text), flags=re.DOTALL)
    clean_script = re.sub(r'\*\*.*?\*\*', '', clean_script)  # Remove bold
    clean_script = re.sub(r'[*#_]', '', clean_script)

    # Pattern: "CharacterName: dialogue"
    dialogue_pattern = r'^([A-Z][a-z\s\.]+?):\s*(.+?)(?=^[A-Z][a-z\s\.]+?:|$)'

    segments = []
    matches = re.finditer(dialogue_pattern, clean_script, flags=re.MULTILINE | re.DOTALL)

    for match in matches:
        speaker_raw = match.group(1).strip()
        dialogue = match.group(2).strip()

        # Map old names to new
        speaker = character_mapping.get(speaker_raw, speaker_raw)

        if dialogue and len(dialogue) >= 3:
            segments.append({
                'character': speaker,
                'text': re.sub(r'\s+', ' ', dialogue).strip()
            })

    if not segments:
        # Fallback
        print("Warning: No dialogue segments found. Using full text as narrator.")
        return [{'character': 'Narrator', 'text': clean_script}]

    print(f"Parsed {len(segments)} dialogue segments")
    return segments


def save_parsed_segments(segments: list):
    """Save parsed segments for verification."""
    parsed_file = output_dir / "parsed_dialogue_segments.txt"
    with open(parsed_file, 'w') as f:
        for idx, seg in enumerate(segments):
            f.write(f"[{idx+1}] {seg['character']}: {seg['text']}\n\n")
    print(f"Parsed script: {parsed_file}")


# --- AUDIO GENERATION ---
def clean_text_for_tts(text):
    """Sanitise text so ChatTTS does not choke on non-ASCII characters."""
    clean = re.sub(r'<think>.*?</think>', '', str(text), flags=re.DOTALL)
    clean = re.sub(r'\*\*.*?\*\*', '', clean)
    clean = re.sub(r'[*#_\[\]]', '', clean)

    replacements = {
        '\u2018': "'", '\u2019': "'",          # smart single quotes
        '\u201c': '"', '\u201d': '"',          # smart double quotes
        '\u2014': '-', '\u2013': '-',          # em-dash / en-dash
        '\u2026': '...', '\u2212': '-',        # ellipsis / minus sign
        '\u2039': '<', '\u203a': '>',          # single guillemets
        '\u00ab': '"', '\u00bb': '"',          # double guillemets
        '\u201e': '"', '\u201a': "'", '\u201f': '"',
    }
    for old, new in replacements.items():
        clean = clean.replace(old, new)

    clean = clean.encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'\s+', ' ', clean).strip()

# Initialize ChatTTS once
_chattts_model = None

def get_chattts_model():
    """Lazy load ChatTTS model."""
    global _chattts_model
    if _chattts_model is None:
        print("Loading ChatTTS model...")
        _chattts_model = ChatTTS.Chat()
        _chattts_model.load(compile=False)  # compile=True for GPU speedup
        print("✓ ChatTTS model loaded")
    return _chattts_model

def generate_audio_chattts(dialogue_segments: list, output_filename: str = "podcast_final_audio.wav"):
    """
    Generate multi-speaker audio using ChatTTS.

    Args:
        dialogue_segments: List of {'character': 'Kaz'|'Erika', 'text': '...'}
        output_filename: Output WAV file
    """
    chat = get_chattts_model()
    temp_dir = output_dir / "temp_audio"
    temp_dir.mkdir(exist_ok=True)

    # Assign speakers (ChatTTS supports multiple speakers via rand_spk)
    torch.manual_seed(42)  # Kaz seed
    kaz_spk = chat.sample_random_speaker()

    torch.manual_seed(84)  # Erika seed (different seed = different voice)
    erika_spk = chat.sample_random_speaker()

    speakers = {"Kaz": kaz_spk, "Erika": erika_spk}

    print(f"\nGenerating {len(dialogue_segments)} segments with ChatTTS...")

    audio_segments = []

    for idx, segment in enumerate(dialogue_segments):
        character = segment['character']
        text = segment['text']

        if not text.strip():
            continue

        clean_text = clean_text_for_tts(text)
        if not clean_text:
            continue

        # Generate with character-specific speaker
        spk_emb = speakers.get(character, kaz_spk)

        try:
            wavs = chat.infer(
                [clean_text],
                params_infer_code={'spk_emb': spk_emb}
            )

            audio_segments.append(wavs[0])
            print(f"  [{idx+1}/{len(dialogue_segments)}] {character}: {clean_text[:50]}...")

        except Exception as e:
            print(f"Error generating segment {idx}: {e}")

    if not audio_segments:
        print("Error: No audio generated")
        return None

    # Concatenate audio
    combined_audio = np.concatenate(audio_segments)

    # Save as WAV
    output_path = output_dir / output_filename

    # ChatTTS outputs at 24kHz
    with wave.open(str(output_path), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(24000)  # ChatTTS sample rate
        wav_file.writeframes((combined_audio * 32767).astype(np.int16).tobytes())

    file_size = output_path.stat().st_size
    if file_size < 1000:
        print(f"Error: Audio file is suspiciously small ({file_size} bytes)")
        return None

    print(f"\n✓ Audio generated: {output_path} ({file_size} bytes)")

    # Auto-play
    try:
        if platform.system() == "Darwin":
            os.system(f"open '{output_path}'")
        elif platform.system() == "Windows":
            os.startfile(str(output_path))
        else:
            os.system(f"xdg-open '{output_path}' &")
    except Exception as e:
        print(f"Could not auto-play: {e}")

    return output_path

# Parse script and generate audio with ChatTTS
print("\n--- Generating Multi-Voice Podcast Audio ---")

# Check script length before generation
script_text = result.raw
word_count = len(script_text.split())
estimated_duration_min = word_count / 150  # 150 words per minute

print(f"\n{'='*60}")
print(f"DURATION CHECK")
print(f"{'='*60}")
print(f"Script word count: {word_count}")
print(f"Estimated duration: {estimated_duration_min:.1f} minutes")
print(f"Target range: 9-11 minutes (1350-1650 words)")

if word_count < 1350:
    print(f"⚠ WARNING: Script is SHORT ({word_count} words < 1350)")
    print(f"  Estimated {estimated_duration_min:.1f} min < 9 min target")
    print(f"  Consider running again with expanded content")
elif word_count > 1650:
    print(f"⚠ WARNING: Script is LONG ({word_count} words > 1650)")
    print(f"  Estimated {estimated_duration_min:.1f} min > 11 min target")
    print(f"  Consider running again with condensed content")
else:
    print(f"✓ Script length ACCEPTABLE ({word_count} words)")
    print(f"  Estimated {estimated_duration_min:.1f} min within 9-11 min range")
print(f"{'='*60}\n")

character_mapping = {
    "Dr. Data": SESSION_ROLES["pro"]["character"],
    "Dr. Doubt": SESSION_ROLES["con"]["character"],
    "Dr Data": SESSION_ROLES["pro"]["character"],
    "Dr Doubt": SESSION_ROLES["con"]["character"]
}

dialogue_segments = parse_script_to_segments(result.raw, character_mapping)
save_parsed_segments(dialogue_segments)  # Debug output
audio_file = generate_audio_chattts(dialogue_segments)

# Check actual audio duration
if audio_file and audio_file.exists():
    try:
        with wave.open(str(audio_file), 'r') as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            duration_seconds = frames / float(rate)
            duration_minutes = duration_seconds / 60

        print(f"\n{'='*60}")
        print(f"AUDIO DURATION VERIFICATION")
        print(f"{'='*60}")
        print(f"Actual audio duration: {duration_minutes:.2f} minutes ({duration_seconds:.1f} seconds)")
        print(f"Target range: 9-11 minutes")

        if duration_minutes < 9.0:
            print(f"✗ FAILED: Audio is TOO SHORT ({duration_minutes:.2f} min < 9 min)")
            print(f"  ACTION: Re-run with longer script")
        elif duration_minutes > 11.0:
            print(f"✗ FAILED: Audio is TOO LONG ({duration_minutes:.2f} min > 11 min)")
            print(f"  ACTION: Re-run with shorter script")
        else:
            print(f"✓ SUCCESS: Audio duration within acceptable range")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"Warning: Could not verify audio duration: {e}")