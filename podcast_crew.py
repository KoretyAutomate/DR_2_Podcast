import os
import platform
import re
import httpx
import time
import random
import sys
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

# --- INITIALIZATION ---
load_dotenv()
script_dir = Path(__file__).parent.absolute()
output_dir = script_dir / "research_outputs"
output_dir.mkdir(exist_ok=True)

topic_name = 'scientific benefit of coffee intake to increase productivity during the day'

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
    Search for scientific data. ONLY use when you need:
    1. Recent data published after 2024 (your knowledge cutoff)
    2. Specific study citations not in your training
    3. Real-time statistics or ongoing clinical trials
    4. Verification of controversial claims

    DO NOT search for well-established concepts (e.g., caffeine metabolism).
    Use internal knowledge first. Search is expensive - last resort only.
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
    pdf = SciencePDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, 0, 1, 'L')
    pdf.ln(5)
    
    # Clean up markdown for PDF
    clean_content = re.sub(r'<think>.*?</think>', '', str(content), flags=re.DOTALL)
    clean_content = clean_content.encode('latin-1', 'ignore').decode('latin-1')
    
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 10, clean_content)
    
    file_path = output_dir / filename
    pdf.output(str(file_path))
    print(f"PDF Generated: {file_path}")
    return file_path

# --- AGENTS ---
researcher = Agent(
    role='Lead Research Scientist',
    goal=f'Produce a high-impact scientific paper supporting {topic_name}',
    backstory=(
        f'Senior researcher specializing in neurobiology and metabolic efficiency. '
        f'Relies on deep scientific knowledge. Only searches for recent publications (2025+) or specific citations. '
        f'In this podcast, you will be portrayed by "{SESSION_ROLES["pro"]["character"]}" '
        f'who has a {SESSION_ROLES["pro"]["personality"]} approach.'
    ),
    tools=[search_tool],
    llm=dgx_llm,
    verbose=True
)

auditor = Agent(
    role='Scientific Auditor',
    goal='Critically evaluate research, identify gaps, and synthesize a final verdict.',
    backstory='Meticulous chief editor specializing in resolving scientific conflicts.',
    llm=dgx_llm,
    verbose=True
)

counter_researcher = Agent(
    role='Adversarial Researcher',
    goal=f'Produce a scientific paper challenging {topic_name} by debunking specific claims.',
    backstory=(
        f'Skeptical meta-analyst specializing in methodology flaws. '
        f'Leverages extensive knowledge. Only searches for recent contradictory evidence or specific debunking studies. '
        f'In this podcast, you will be portrayed by "{SESSION_ROLES["con"]["character"]}" '
        f'who has a {SESSION_ROLES["con"]["personality"]} approach.'
    ),
    tools=[search_tool],
    llm=dgx_llm,
    verbose=True
)

scriptwriter = Agent(
    role='Podcast Producer',
    goal='Translate audited papers into a balanced dialogue.',
    backstory='Award-winning science communicator.',
    llm=dgx_llm,
    verbose=True
)

personality = Agent(
    role='Podcast Personality',
    goal='Polish the script for natural verbal delivery.',
    backstory='Radio host expert in humanizing technical data.',
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
        "Draft condensed scientific paper (Nature style). "
        "IMPORTANT: Use existing scientific knowledge as primary source. "
        "Only use BraveSearch for recent studies (2025+) or specific citations. "
        "Include: Abstract, Introduction, 3 Mechanisms, Bibliography with URLs."
    ),
    expected_output="A formal, condensed scientific paper with citations supporting the benefits.",
    agent=researcher
)

gap_analysis_task = Task(
    description=(
        "Review the Lead Scientist's Supporting Paper. Identify potential weaknesses "
        "and suggest specific topics for the Adversarial Researcher to investigate."
    ),
    expected_output="A list of 3-5 specific scientific 'weak points'.",
    agent=auditor,
    context=[research_task]
)

adversarial_task = Task(
    description=(
        f"Based on 'Supporting Paper' and 'Gap Analysis', draft 'Anti-Thesis' paper. "
        "Address and debunk mechanisms proposed in initial research. "
        "IMPORTANT: Use existing knowledge as primary source. "
        "Only use BraveSearch for recent contradictory studies (2025+) or specific citations. "
        "Include Bibliography with URLs."
    ),
    expected_output="A formal, condensed scientific paper challenging the findings.",
    agent=counter_researcher,
    context=[research_task, gap_analysis_task]
)

source_verification_task = Task(
    description=(
        "Extract ALL sources from Supporting and Anti-Thesis papers. "
        "For each source verify:\n"
        "1. URL points to scientific content\n"
        "2. Source type (peer-reviewed, preprint, review, meta-analysis)\n"
        "3. Trust level: HIGH (Nature/Science/Lancet/Cell/PNAS), "
        "MEDIUM (PubMed/arXiv), LOW (news/blogs)\n"
        "4. Journal name and year if available\n\n"
        "Create structured bibliography JSON:\n"
        '{"supporting_sources": [{title, url, journal, year, trust_level, source_type}],\n'
        ' "contradicting_sources": [...],\n'
        ' "summary": "X high-trust, Y medium-trust sources"}\n\n'
        "REJECT non-scientific sources. Flag if <3 high-trust sources."
    ),
    expected_output="JSON bibliography with categorized, verified sources and quality summary.",
    agent=source_verifier,
    context=[research_task, adversarial_task]
)

audit_task = Task(
    description=(
        "Review Supporting, Anti-Thesis papers AND verified source bibliography. "
        "PRIORITIZE findings from HIGH-TRUST peer-reviewed sources. "
        "Validate key claims are backed by reputable journals. "
        "Prepare Final Meta-Audit Report weighing evidence quality and source credibility."
    ),
    expected_output=(
        "Synthesis report with:\n"
        "1. Verdict based on evidence quality\n"
        "2. Source assessment (high-trust vs low-trust count)\n"
        "3. Confidence level in conclusions"
    ),
    agent=auditor,
    context=[research_task, adversarial_task, source_verification_task]
)

script_task = Task(
    description=(
        f"Write a technical podcast script featuring {SESSION_ROLES['pro']['character']} "
        f"vs {SESSION_ROLES['con']['character']}. "
        f"\n\nCHARACTER ROLES:\n"
        f"  - {SESSION_ROLES['pro']['character']}: SUPPORTING perspective, "
        f"{SESSION_ROLES['pro']['personality']}\n"
        f"  - {SESSION_ROLES['con']['character']}: CRITICAL perspective, "
        f"{SESSION_ROLES['con']['personality']}\n\n"
        f"Format STRICTLY as:\n"
        f"{SESSION_ROLES['pro']['character']}: [dialogue]\n"
        f"{SESSION_ROLES['con']['character']}: [dialogue]\n\n"
        f"Maintain consistent roles throughout. NO role switching mid-conversation."
    ),
    expected_output=(
        f"Conversational dialogue between {SESSION_ROLES['pro']['character']} (supporting) "
        f"and {SESSION_ROLES['con']['character']} (critical)."
    ),
    agent=scriptwriter,
    context=[audit_task]
)

natural_language_task = Task(
    description=(
        f"Rewrite {SESSION_ROLES['pro']['character']} vs {SESSION_ROLES['con']['character']} "
        f"dialogue for natural verbal delivery.\n\n"
        f"MAINTAIN ROLES:\n"
        f"  - {SESSION_ROLES['pro']['character']}: SUPPORTING, {SESSION_ROLES['pro']['personality']}\n"
        f"  - {SESSION_ROLES['con']['character']}: CRITICAL, {SESSION_ROLES['con']['personality']}\n\n"
        f"Format:\n{SESSION_ROLES['pro']['character']}: [dialogue]\n"
        f"{SESSION_ROLES['con']['character']}: [dialogue]\n\n"
        f"Remove meta-tags, markdown, stage directions. Dialogue only."
    ),
    expected_output=f"Final dialogue between {SESSION_ROLES['pro']['character']} and {SESSION_ROLES['con']['character']}.",
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

print(f"\n--- Initiating Scientific Research Pipeline on DGX Spark ---\n")
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

        # Clean text
        clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        clean_text = re.sub(r'[*#_]', '', clean_text).strip()

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

    print(f"\n✓ Audio generated: {output_path}")

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
character_mapping = {
    "Dr. Data": SESSION_ROLES["pro"]["character"],
    "Dr. Doubt": SESSION_ROLES["con"]["character"],
    "Dr Data": SESSION_ROLES["pro"]["character"],
    "Dr Doubt": SESSION_ROLES["con"]["character"]
}

dialogue_segments = parse_script_to_segments(result.raw, character_mapping)
save_parsed_segments(dialogue_segments)  # Debug output
generate_audio_chattts(dialogue_segments)