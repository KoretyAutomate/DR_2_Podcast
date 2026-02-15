
import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from music_engine import MusicGenerator
from audio_mixer import AudioMixer
from audio_engine import post_process_audio
import soundfile as sf
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_voice(filename):
    """Create a 5-second sine wave to simulate voice"""
    sr = 24000
    t = np.linspace(0, 5, 5 * sr, endpoint=False)
    # 440Hz sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(filename, audio, sr)
    logger.info(f"Created dummy voice file: {filename}")

def test_music_generation():
    print("\n--- Testing Music Generation ---")
    gen = MusicGenerator()
    # Use very short duration for test
    output = gen.generate_music("simple drum beat", duration=5, output_filename="test_gen.wav")
    if output and os.path.exists(output):
        print("✓ Music generation successful")
        return output
    else:
        print("✗ Music generation failed")
        return None

def test_mixing(voice_path, music_path):
    print("\n--- Testing Mixing ---")
    mixer = AudioMixer()
    output = "test_mixed.wav"
    success = mixer.mix_podcast(voice_path, music_path, output)
    if success and os.path.exists(output):
        print("✓ Audio mixing successful")
    else:
        print("✗ Audio mixing failed")

def test_pipeline_integration():
    print("\n--- Testing Pipeline Integration (BGM Selection from Library) ---")
    
    # Create dummy parsed audio
    dummy_voice = "test_pipeline_voice.wav"
    create_dummy_voice(dummy_voice)
    
    # 3.1 Test Specific Library File (Assuming "Interesting BGM.wav" exists)
    print("Test 3.1: Library Lookup (Interesting BGM.wav)...")
    result_lib = post_process_audio(dummy_voice, bgm_target="Interesting BGM.wav")
    
    if result_lib and "_mixed" in result_lib and os.path.exists(result_lib):
        print(f"✓ Library integration successful. Result: {result_lib}")
    else:
        print(f"✗ Library integration failed. Result: {result_lib}")
    
    # 3.2 Test Random
    print("Test 3.2: Random Selection...")
    result_rand = post_process_audio(dummy_voice, bgm_target="random")
    
    if result_rand and "_mixed" in result_lib and os.path.exists(result_rand):
        print("✓ Random integration successful.")
    else:
        print("✗ Random integration failed.")

if __name__ == "__main__":
    # Ensure asset dir exists for temp files
    # os.makedirs("tests/temp", exist_ok=True)
    
    # 1. Test Gen
    music_file = test_music_generation()
    
    # 2. Test Mix
    if music_file:
        dummy_voice = "test_voice.wav"
        create_dummy_voice(dummy_voice)
        test_mixing(dummy_voice, music_file)
        
    # 3. Test Integrated Function
    test_pipeline_integration()
