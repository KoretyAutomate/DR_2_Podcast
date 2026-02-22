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

    def mix_podcast_pro(self, voice_path: str, music_path: str, output_path: str,
                        pre_roll_ms: int = 4000, post_roll_ms: int = 6000,
                        transition_positions_ms: list = None,
                        voice_ducking_db: int = -18, transition_bump_db: int = -10,
                        transition_duration_ms: int = 1500) -> bool:
        """
        Pro-grade podcast mixing with BGM-only intro/outro and transition bumps.

        Sections:
        1. PRE-ROLL: BGM only at full volume, fading down to ducked level
        2. VOICE+BGM: Main content with BGM ducked
        3. TRANSITIONS: Brief BGM volume bumps at marked positions
        4. POST-ROLL: BGM fading up from ducked to full, then fading out
        """
        try:
            logger.info(f"Pro mixing: {voice_path} with {music_path}")
            logger.info(f"  Pre-roll: {pre_roll_ms}ms, Post-roll: {post_roll_ms}ms")
            if transition_positions_ms:
                logger.info(f"  Transition bumps at: {transition_positions_ms}")

            voice = AudioSegment.from_wav(voice_path)
            music = AudioSegment.from_wav(music_path)

            total_duration = len(voice) + pre_roll_ms + post_roll_ms

            # Loop and trim music to cover total duration
            if len(music) < total_duration + 2000:
                repeats = ((total_duration + 2000) // len(music)) + 1
                music = music * repeats
            music = music[:total_duration + 2000]

            # Normalize both tracks
            voice = effects.normalize(voice)
            music = effects.normalize(music)

            # --- 1. PRE-ROLL: BGM only, fading from full volume down to ducked ---
            pre_roll = music[:pre_roll_ms].fade(
                to_gain=voice_ducking_db, start=0, duration=pre_roll_ms
            )

            # --- 2. MAIN SECTION: ducked BGM under voice ---
            main_music = (music[pre_roll_ms:pre_roll_ms + len(voice)] + voice_ducking_db)

            # --- 3. TRANSITION BUMPS: brief volume increases at marked positions ---
            if transition_positions_ms:
                for pos_ms in transition_positions_ms:
                    bump_start = pos_ms
                    bump_end = min(bump_start + transition_duration_ms, len(main_music))
                    if 0 <= bump_start < len(main_music):
                        bump_gain = abs(voice_ducking_db - transition_bump_db)
                        bump_section = main_music[bump_start:bump_end] + bump_gain
                        bump_section = bump_section.fade_in(300).fade_out(300)
                        main_music = main_music[:bump_start] + bump_section + main_music[bump_end:]

            # --- 4. POST-ROLL: BGM fading up from ducked to full, then fading out ---
            post_start = pre_roll_ms + len(voice)
            post_music = music[post_start:post_start + post_roll_ms] + voice_ducking_db
            # Fade up from ducked to full volume over first half
            fade_up_duration = min(post_roll_ms // 2, 3000)
            post_music = post_music.fade(
                from_gain=0, to_gain=abs(voice_ducking_db),
                start=0, duration=fade_up_duration
            )
            # Fade out over second half
            post_music = post_music.fade_out(post_roll_ms - fade_up_duration)

            # --- ASSEMBLE: pre-roll + (main BGM overlaid with voice) + post-roll ---
            # Create voice track with silence padding
            silence_pre = AudioSegment.silent(duration=pre_roll_ms, frame_rate=voice.frame_rate)
            silence_post = AudioSegment.silent(duration=post_roll_ms, frame_rate=voice.frame_rate)
            full_voice = silence_pre + voice + silence_post

            # Create full BGM track
            full_bgm = pre_roll + main_music + post_music

            # Ensure same length (trim to shorter)
            min_len = min(len(full_bgm), len(full_voice))
            full_bgm = full_bgm[:min_len]
            full_voice = full_voice[:min_len]

            # Overlay voice on top of BGM
            final_mix = full_bgm.overlay(full_voice)

            # Gentle global fade in/out
            final_mix = final_mix.fade_in(1500).fade_out(2000)

            final_mix.export(output_path, format="wav")
            logger.info(f"Pro-mixed audio saved to: {output_path}")
            logger.info(f"  Total duration: {len(final_mix) / 1000:.1f}s "
                        f"(pre-roll {pre_roll_ms/1000:.1f}s + voice {len(voice)/1000:.1f}s + "
                        f"post-roll {post_roll_ms/1000:.1f}s)")
            return True

        except Exception as e:
            logger.error(f"Pro mixing failed: {e}, falling back to basic mix")
            return self.mix_podcast(voice_path, music_path, output_path)


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    pass
