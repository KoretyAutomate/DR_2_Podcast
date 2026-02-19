# Reference Audio for Voice Cloning

Fish Speech V1.5 supports voice cloning via reference audio files. This is optional — if no reference files are present, the default Fish Speech voice is used (still high quality for Japanese).

## Setup

To give the two podcast speakers distinct voices, place two WAV files here:

```
reference_audio/
├── kaz.wav      → Speaker 1 (カズ, the presenter/expert)
└── erika.wav    → Speaker 2 (エリカ, the questioner/skeptic)
```

## Requirements

- **Format**: WAV, mono or stereo
- **Length**: 5-15 seconds (longer is better for voice capture)
- **Content**: Japanese speech from the target voice persona
- **Quality**: Clean audio, no background noise

## Example

Record or obtain a clean Japanese voice sample:

```bash
# Example: use a professional voice actor recording, or synthesize a seed voice
ffmpeg -i input.mp3 -ar 44100 -ac 1 kaz.wav
ffmpeg -i input2.mp3 -ar 44100 -ac 1 erika.wav
```

Once these files are present, `audio_engine.py` will automatically use them for voice cloning on all Japanese podcasts.

## Testing

After adding reference audio, generate a test podcast:

```bash
cd /home/korety/Project/DR_2_Podcast
conda activate podcast_flow
python podcast_crew.py --topic "test Japanese voices" --language ja
```

Listen to the output — the two speakers should now have the distinct voices from your reference files.
