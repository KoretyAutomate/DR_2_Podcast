import logging
import os
from pydub import AudioSegment, effects

logger = logging.getLogger(__name__)

class AudioMixer:
    """
    Mixes voice and background music with ducking capabilities.
    """
    def __init__(self):
        pass

    def mix_podcast(self, voice_path: str, music_path: str, output_path: str) -> bool:
        """
        Mixes voice and background music. Loop music and duck it.
        """
        try:
            logger.info(f"Mixing voice: {voice_path} with music: {music_path}")
            
            voice = AudioSegment.from_wav(voice_path)
            music = AudioSegment.from_wav(music_path)

            # 1. Loop music to match or exceed voice duration
            if len(music) < len(voice):
                repeats = (len(voice) // len(music)) + 1
                music = music * repeats
            
            # Trim music to exact length of voice (plus maybe a small fade out tail)
            music = music[:len(voice) + 2000] # + 2s tail
            
            # 2. Lower volume of music ("ducking")
            # Simple volume reduction. True sidechain compression is complex in pydub.
            # We'll set music to -20dB relative to voice peak.
            
            # Normalize voice first
            voice = effects.normalize(voice)
            music = effects.normalize(music)
            
            # Reduce music volume significantly
            music = music - 18 # Reduce by 18dB
            
            # 3. Overlay
            # position=0 means start at beginning
            final_mix = music.overlay(voice, position=0)
            
            # 4. Fade in/out music
            final_mix = final_mix.fade_in(2000).fade_out(3000)
            
            # Export
            final_mix.export(output_path, format="wav")
            logger.info(f"Mixed audio saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mix audio: {e}")
            return False

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    # create dummy files if they don't exist for testing
    # ...
    pass
