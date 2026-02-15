
import sys
import os
import logging
import json
from pathlib import Path

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bgm_generation.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_DIR = Path("/home/korety/Project/DR_2_Podcast")
BGM_OUTPUT_DIR = PROJECT_DIR / "Podcast BGM"

# Add project root to path so we can import music_engine
sys.path.append(str(PROJECT_DIR))

try:
    from music_engine import MusicGenerator
except ImportError:
    logger.error("Could not import music_engine. Make sure you are running in the correct environment.")
    sys.exit(1)

def generate_library():
    logger.info("Starting BGM Library Generation...")
    
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
    
    gen = MusicGenerator()
    
    total = len(prompts)
    successful = 0
    
    for i, prompt in enumerate(prompts):
        filename = BGM_OUTPUT_DIR / f"bgm_track_{i+1:02d}.wav"
        
        if filename.exists():
            logger.info(f"Skipping {filename.name}, already exists")
            successful += 1
            continue
            
        logger.info(f"Generating track {i+1}/{total}: '{prompt}'")
        
        # Generate 30s clips
        # Using the generate_music method which handles fallback logic internally
        result = gen.generate_music(prompt, duration=30, output_filename=str(filename))
        
        if result:
            logger.info(f"✓ Generated: {filename.name}")
            successful += 1
        else:
            logger.error(f"✗ Failed to generate: '{prompt}'")
            
    logger.info(f"Generation Complete! {successful}/{total} tracks ready in '{BGM_OUTPUT_DIR}'")

if __name__ == "__main__":
    generate_library()
