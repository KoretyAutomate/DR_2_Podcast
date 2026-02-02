import os
import platform
import re
import httpx
import time
import random
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from gtts import gTTS
from fpdf import FPDF

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
    """Search the web for real-time scientific data and peer-reviewed studies."""
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
    backstory='Senior researcher specializing in neurobiology and metabolic efficiency.',
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
    backstory='Skeptical meta-analyst who specializes in identifying methodology flaws.',
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

# --- TASKS ---
research_task = Task(
    description=(
        f"Conduct an exhaustive deep dive into {topic_name}. "
        "Draft a condensed scientific paper (Nature style). "
        "Include: Abstract, Introduction, 3 Biochemical Mechanisms, Bibliography with URLs."
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
        f"Based on the 'Supporting Paper' and 'Gap Analysis', draft an 'Anti-Thesis' paper. "
        "Address and debunk the mechanisms proposed in the initial research. Include Bibliography with URLs."
    ),
    expected_output="A formal, condensed scientific paper challenging the findings.",
    agent=counter_researcher,
    context=[research_task, gap_analysis_task]
)

audit_task = Task(
    description=(
        "Review both the Supporting and Anti-Thesis papers. Validate sources. "
        "Prepare a Final Meta-Audit Report weighing evidence from both sides."
    ),
    expected_output="A high-level synthesis report providing a definitive verdict.",
    agent=auditor,
    context=[research_task, adversarial_task]
)

script_task = Task(
    description="Write a technical podcast script based on the meta-report (Dr. Data vs Dr. Doubt).",
    expected_output="A conversational but technical dialogue script.",
    agent=scriptwriter,
    context=[audit_task]
)

natural_language_task = Task(
    description="Rewrite for verbal delivery. Ensure natural flow and remove meta-tags.",
    expected_output="A final dialogue-only script.",
    agent=personality,
    context=[script_task]
)

# --- EXECUTION ---
crew = Crew(
    agents=[researcher, auditor, counter_researcher, scriptwriter, personality],
    tasks=[research_task, gap_analysis_task, adversarial_task, audit_task, script_task, natural_language_task],
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
    create_pdf("Final Meta-Audit Verdict", audit_task.output.raw, "final_audit_report.pdf")
except Exception as e:
    print(f"Warning: PDF generation failed, but research is complete: {e}")

# --- AUDIO GENERATION ---
def generate_audio(crew_output):
    filename = output_dir / "podcast_final_audio.mp3"
    clean_text = re.sub(r'<think>.*?</think>', '', str(crew_output), flags=re.DOTALL)
    clean_text = re.sub(r'[*#_]', '', clean_text)
    
    print(f"Synthesizing speech to {filename}...")
    try:
        tts = gTTS(text=clean_text, lang='en')
        tts.save(str(filename))
        
        if platform.system() == "Darwin": os.system(f"open '{filename}'")
        elif platform.system() == "Windows": os.startfile(str(filename))
        else: os.system(f"xdg-open '{filename}' &")
    except Exception as e:
        print(f"Audio failed: {e}")

generate_audio(result)