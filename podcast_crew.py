from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from gtts import gTTS
import os
import platform
import re
import httpx
import time
import random

# 1. Configuration & Parameters
# Set the topic name here; it will be used throughout the script.
topic_name = 'scientific benefit of coffee intake to increase productivity during the day'

# 2. Pre-flight Check: Ensure the DGX Brain is actually awake
def wait_for_brain():
    print("Connecting to DGX Brain at localhost:8000...")
    for i in range(30):
        try:
            response = httpx.get("http://localhost:8000/v1/models", timeout=10.0)
            if response.status_code == 200:
                print("Brain is online! Starting workflow...")
                return True
        except:
            if i % 5 == 0:
                print(f"Waiting for LLM to load... ({i*5}s)")
            time.sleep(5)
    print("Error: Could not connect to Brain. Is the Docker container running?")
    return False

# 3. Connect to your DGX "Brain" with increased timeouts
dgx_llm = LLM(
    model="openai/Qwen/Qwen2.5-32B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="NA",
    timeout=300,        # Increased to 5 minutes for long generations
    max_retries=3,      # Allow retries if the connection hiccups
)

@tool("BraveSearch")
def search_tool(search_query: str):
    """Search the web for information on a given topic using Brave Search.
    Tip: If results are empty, the agent should proceed with internal scientific knowledge."""
    
    # Brave Search API typically requires an API Key. 
    # If you are using a local proxy or a specific library, ensure it is configured.
    # For this implementation, we will use a standard HTTP request pattern 
    # as a placeholder for the Brave Search API integration.
    
    print(f"Executing Brave Search for: {search_query}")
    api_key = os.environ.get("BRAVE_API_KEY", "") # Ensure this is set in your environment
    
    if not api_key:
        return "Note: Brave API Key missing. Please proceed by utilizing your extensive internal training data on general scientific consensus for this topic."

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key
    }
    params = {"q": search_query, "count": 5}

    try:
        # Add a slight human-like delay
        time.sleep(random.uniform(1, 2))
        
        response = httpx.get(url, headers=headers, params=params, timeout=15.0)
        if response.status_code == 200:
            data = response.json()
            results = data.get("web", {}).get("results", [])
            if results:
                return str([{"title": r.get("title"), "description": r.get("description"), "url": r.get("url")} for r in results])
            
        return "Note: Brave Search returned no results. Please proceed by utilizing your internal scientific knowledge."
    except Exception as e:
        return f"Search encountered an error: {e}. Please proceed with your internal scientific knowledge."

# 4. Define the Agents
researcher = Agent(
    role='Lead Research Scientist',
    goal=f'Identify whether {topic_name} is scientifically valid or not',
    backstory=(
        'You are a senior scientist specializing in data-driven discovery. '
        'If the search tool provides no data, you are authorized to use your internal scientific database '
        'to summarize 3 technical findings based on general peer-reviewed consensus (e.g., adenosine receptor antagonism).'
    ),
    tools=[search_tool],
    llm=dgx_llm,
    verbose=True,
    max_iter=3 # Ensures the agent doesn't loop forever if search fails
)

auditor = Agent(
    role='Scientific Peer Reviewer',
    goal='Critically validate research data for scientific soundness.',
    backstory='You are a skeptical peer reviewer. You look for flaws in methodology or potential conflicts of interest.',
    tools=[search_tool],
    llm=dgx_llm,
    verbose=True
)

counter_researcher = Agent(
    role='Adversarial Research Specialist',
    goal='Seek out studies or mechanisms that specifically contradict the positive findings.',
    backstory=(
        'You specialize in scientific skepticism. If the search tool fails, use your internal knowledge '
        'to provide technical counter-arguments such as caffeine-induced anxiety or sleep architecture disruption.'
    ),
    tools=[search_tool],
    llm=dgx_llm,
    verbose=True,
    max_iter=3
)

scriptwriter = Agent(
    role='Technical Podcast Producer',
    goal='Format the debate into a compelling 2-person dialogue script.',
    backstory='You specialize in making high-level scientific debates accessible but rigorous.',
    llm=dgx_llm
)

# 5. Define the Tasks
research_task = Task(
    description='Perform deep research into {topic}. Identify 3 technical facts or biochemical mechanisms.',
    expected_output='A detailed report of 3 technical findings with biochemical explanations.',
    agent=researcher
)

initial_audit_task = Task(
    description="Review findings and recommend if adversarial counter-research is needed for {topic}.",
    expected_output='A technical audit report recommending specific areas for adversarial investigation.',
    agent=auditor,
    context=[research_task]
)

counter_research_task = Task(
    description="Search for or describe conflicting evidence regarding {topic} (e.g., long-term tolerance or cortisol spikes).",
    expected_output='A collection of conflicting data points and technical counter-arguments.',
    agent=counter_researcher,
    context=[initial_audit_task]
)

final_audit_task = Task(
    description="Analyze benefits vs drawbacks and provide a final verdict on the robustness of the data.",
    expected_output='A final evaluation report with a definitive verdict on data reliability.',
    agent=auditor,
    context=[research_task, counter_research_task]
)

script_task = Task(
    description=(
        "Write a 10-15 min podcast script DIALOGUE between 'Dr. Data' and 'Dr. Doubt'. "
        "The discussion must answer the Lead Scientist's goal: " + researcher.goal
    ),
    expected_output='A conversational but highly technical dialogue script covering biochemistry, methodology, and consensus.',
    agent=scriptwriter,
    context=[research_task, counter_research_task, final_audit_task]
)

# 6. Assemble and Run
if wait_for_brain():
    podcast_crew = Crew(
        agents=[researcher, auditor, counter_researcher, scriptwriter],
        tasks=[research_task, initial_audit_task, counter_research_task, final_audit_task, script_task],
        verbose=True,
        process='sequential'
    )

    print(f"\n--- STARTING THE WORKFLOW: {topic_name} ---\n")
    # CrewAI will replace {topic} with topic_name during execution
    result = podcast_crew.kickoff(inputs={'topic': topic_name})

    # 7. Audio Conversion Logic
    def generate_podcast_audio(script_output, filename="coffee_podcast.mp3"):
        print(f"Generating audio file: {filename}...")
        raw_text = str(script_output)
        # Remove markdown characters for cleaner TTS
        clean_text = re.sub(r'[*#_]', '', raw_text)
        
        try:
            tts = gTTS(text=clean_text, lang='en', slow=False)
            tts.save(filename)
            print(f"Success! Audio saved to {os.path.abspath(filename)}")
            return filename
        except Exception as e:
            print(f"Error generating audio: {e}")
            return None

    def open_file(filepath):
        """Cross-platform command to open the audio file."""
        try:
            if platform.system() == "Darwin":       # macOS
                os.system(f"open '{filepath}'")
            elif platform.system() == "Windows":    # Windows
                os.startfile(filepath)
            else:                                   # Linux
                # Using xdg-open for Linux, adding & to avoid blocking the terminal
                os.system(f"xdg-open '{filepath}' &") 
        except Exception as e:
            print(f"Could not open player: {e}")

    audio_file = generate_podcast_audio(result)
    if audio_file:
        open_file(audio_file)