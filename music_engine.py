import logging
import torch
import os
from pathlib import Path
from typing import Optional

try:
    from audiocraft.models.musicgen import MusicGen
    from audiocraft.data.audio import audio_write
except ImportError:
    MusicGen = None

logger = logging.getLogger(__name__)

class MusicGenerator:
    """
    Generates background music using Meta's AudioCraft (MusicGen).
    """
    def __init__(self, model_name: str = 'facebook/musicgen-small'):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None

    def _load_model(self):
        if not MusicGen:
            logger.error("AudioCraft not installed. Cannot generate music.")
            return False
            
        if self.model:
            return True
            
        try:
            # 1. Try loading on GPU first
            logger.info(f"Loading MusicGen model: {self.model_name} on {self.device}")
            self.model = MusicGen.get_pretrained(self.model_name, device=self.device)
            return True
        except (RuntimeError, Exception) as e:
            if self.device == 'cuda':
                # Catch OOM or Capability mismatch (CUDA error: no kernel image...)
                logger.warning(f"CUDA validation failed: {e}")
                logger.warning("Falling back to CPU for music generation.")
                self.device = 'cpu'
                try:
                    self.model = MusicGen.get_pretrained(self.model_name, device=self.device)
                    return True
                except Exception as ex:
                    logger.error(f"Failed to load MusicGen on CPU: {ex}")
                    return False
            else:
                logger.error(f"Failed to load MusicGen model: {e}")
                return False

    def generate_music(self, prompt: str, duration: int = 10, output_filename: str = "bgm.wav") -> Optional[str]:
        """
        Generates music based on a text prompt.
        
        Args:
            prompt: Text description of the music (e.g. "lofi hip hop beat")
            duration: Duration in seconds
            output_filename: Output filename (relative or absolute)
            
        Returns:
            Path to the generated audio file, or None if failed.
        """
        if not self._load_model():
            return None
            
        logger.info(f"Generating {duration}s of music for prompt: '{prompt}' on {self.device}")
        
        try:
            self.model.set_generation_params(duration=duration)
            wav = self.model.generate([prompt], progress=True)
            
            # Save the audio
            # audio_write adds the extension automatically, so we strip it from the filename
            output_path = Path(output_filename)
            stem = str(output_path.parent / output_path.stem)
            
            audio_write(stem, wav[0].cpu(), self.model.sample_rate, strategy="loudness", loudness_compressor=True)
            
            # audio_write saves as wav by default
            final_path = f"{stem}.wav"
            logger.info(f"Music generated and saved to: {final_path}")
            return final_path
            
        except (RuntimeError, Exception) as e:
            if "out of memory" in str(e).lower() and self.device == 'cuda':
                logger.warning(f"GPU OOM during generation: {e}")
                logger.warning("Attempting fallback to CPU...")
                # Clear GPU memory if possible? Hard in Python without process restart
                # But we can try to reload on CPU
                self.model = None
                self.device = 'cpu'
                if self._load_model():
                     return self.generate_music(prompt, duration, output_filename) # Retry
                
            logger.error(f"Music generation failed: {e}")
            return None

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    gen = MusicGenerator()
    gen.generate_music("lofi hip hop beat, chill, study", duration=5, output_filename="test_bgm.wav")
