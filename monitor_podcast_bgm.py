
import json
import time
import os
import subprocess
import signal
import sys
from pathlib import Path
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bgm_monitor.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_DIR = Path("/home/korety/Project/DR_2_Podcast")
TASKS_FILE = PROJECT_DIR / "podcast_tasks.json"
BGM_OUTPUT_DIR = PROJECT_DIR / "Podcast BGM"
PODCAST_PYTHON = "/home/korety/miniconda3/envs/podcast_flow/bin/python"

def get_latest_task_status():
    if not TASKS_FILE.exists():
        logger.warning(f"Tasks file not found: {TASKS_FILE}")
        return None
    
    try:
        with open(TASKS_FILE, 'r') as f:
            tasks = json.load(f)
        
        if not tasks:
            return None
            
        # Sort by creation time, descending
        latest_task = sorted(tasks.values(), key=lambda x: x.get('created_at', ''), reverse=True)[0]
        return latest_task.get('status')
    except Exception as e:
        logger.error(f"Error reading tasks file: {e}")
        return None

def stop_process_by_name(proc_name):
    logger.info(f"Attempting to stop process matching '{proc_name}'...")
    try:
        # First try pkill/pgrep strategy
        cmd = ["pgrep", "-f", proc_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if not pid: continue
                logger.info(f"Killing PID {pid} ({proc_name})")
                os.kill(int(pid), signal.SIGTERM)
        else:
            logger.info(f"No host process found for '{proc_name}'")
            
        # Also try docker stop if it looks like vLLM
        if "vllm" in proc_name:
            logger.info("Attempting to stop docker container 'vllm-final'...")
            subprocess.run(["docker", "stop", "vllm-final"], capture_output=True)
            
    except Exception as e:
        logger.error(f"Error stopping {proc_name}: {e}")

def generate_bgm_tracks():
    logger.info("Starting BGM Generation...")
    
    # Ensure output directory exists
    BGM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # List of prompts for diverse BGM
    prompts = [
        "lofi hip hop beat, relax, study, chill",
        "ambient cinematic soundscape, emotional, deep",
        "upbeat corporate background music, inspiring, tech",
        "mysterious investigative journalism background, tension",
        "soft piano and strings, sentimental, warm",
        "futuristic synthwave, cyber, driving beat",
        "acoustic guitar folk background, calm, nature",
        "minimalist electronic pulse, news, serious",
        "jazz trio background, coffee shop, smooth",
        "orchestral strings building tension, dramatic",
        "light playful melody, curiosity, science",
        "dark ambient drone, suspense, mystery",
        "energetic rock beat, intro, excitement",
        "lofi beats with rain sounds, cozy",
        "abstract electronic texture, complex, data",
        "slow motion cinematic strings, epic",
        "retro 80s synth pop, nostalgia",
        "modern trap beat, heavy bass, dark",
        "gentle harp and flute, ethereal, magic",
        "driving techno beat, focus, momentum"
    ]
    
    # Build python script to run generation in the correct environment
    # We create a temporary script to run inside the conda env
    gen_script_content = f"""
import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.append("{PROJECT_DIR}")

from music_engine import MusicGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

prompts = {json.dumps(prompts)}
output_dir = "{BGM_OUTPUT_DIR}"

def main():
    gen = MusicGenerator()
    
    for i, prompt in enumerate(prompts):
        filename = os.path.join(output_dir, f"bgm_track_{{i+1:02d}}.wav")
        if os.path.exists(filename):
            print(f"Skipping {{filename}}, already exists")
            continue
            
        print(f"Generating track {{i+1}}/{{len(prompts)}}: {{prompt}}")
        # Generate 30s clips
        result = gen.generate_music(prompt, duration=30, output_filename=filename)
        
        if result:
            print(f"✓ Generated: {{filename}}")
        else:
            print(f"✗ Failed to generate: {{prompt}}")

if __name__ == "__main__":
    main()
"""
    
    temp_script = PROJECT_DIR / "temp_bgm_gen.py"
    with open(temp_script, 'w') as f:
        f.write(gen_script_content)
        
    logger.info(f"Running generation script with {PODCAST_PYTHON}")
    
    try:
        proc = subprocess.Popen(
            [PODCAST_PYTHON, str(temp_script)],
            cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        for line in proc.stdout:
            print(line, end='')
            
        proc.wait()
        
        if proc.returncode == 0:
            logger.info("BGM Generation Complete!")
        else:
            logger.error("BGM Generation script failed.")
            
    except Exception as e:
        logger.error(f"Failed to execute generation script: {e}")
    finally:
        if temp_script.exists():
            os.remove(temp_script)

def main():
    logger.info("Monitor script started.")
    
    # 1. Monitoring Loop
    while True:
        status = get_latest_task_status()
        logger.info(f"Current Task Status: {status}")
        
        if status in ['completed', 'failed']:
            logger.info("Workflow finished. Proceeding to cleanup.")
            break
        elif status is None:
            logger.info("No active tasks found or unable to read file. Waiting...")
        
        # Wait 10 minutes (600 seconds)
        # For testing purposes, check if the user wants a faster loop, but they asked for 10 mins.
        # I'll implement 10 mins but check frequently for interrupt? No, simple sleep is fine.
        logger.info("Waiting 10 minutes...")
        time.sleep(600)

    # 2. Cleanup Resources
    logger.info("Stopping vLLM and Ollama...")
    stop_process_by_name("vllm.entrypoints.openai.api_server")
    stop_process_by_name("ollama serve")
    
    # Wait for memory to free
    time.sleep(10)
    
    # 3. Generate BGM
    generate_bgm_tracks()

if __name__ == "__main__":
    main()
