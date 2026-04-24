# content-creator

CLI-first Python tool for generating videos for YouTube workflows using Hugging Face inference APIs.

## What it does

The tool is designed around two primary operating modes:

1. `from-text`: takes narration text and either a video prompt or `--generate-video-prompt`, generates speech with Hugging Face TTS, plans visuals with an LLM, generates scene images with Stable Diffusion, and assembles a video locally with `ffmpeg`.
2. `from-audio`: takes an existing audio file and either a video prompt or `--generate-video-prompt`, transcribes the audio with Whisper, uses the transcript plus your prompt to plan scenes, generates images, and assembles a final video using the supplied audio track.

There is also a focused utility command:

1. `transcribe`: transcribes an audio file only, with optional ffmpeg chunking for better STT quality on long files.

This gives you a single CLI that uses:

- LLM for storyboard planning
- speech-to-text for transcript extraction
- text-to-speech for narration synthesis
- stable diffusion for image generation
- optional speaker diarization for speaker-labeled transcripts

## Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe` installed and on `PATH`
- Hugging Face access token in `HF_TOKEN`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp examples/envrc.example .envrc
export $(grep -v '^#' .env | xargs)

```

## Commands

Detailed command documentation with flowcharts is available in [docs/README.md](docs/README.md):

- [Global CLI options](docs/commands/global-options.md)
- [from-text command](docs/commands/from-text.md)
- [from-audio command](docs/commands/from-audio.md)
- [transcribe command](docs/commands/transcribe.md)
- [doctor command](docs/commands/doctor.md)

Generate a video from text and a visual prompt:

```bash
content-creator from-text \
  --text-transcription "Welcome to the channel. Today we are exploring how coral reefs recover after storms." \
  --video-prompt "documentary b-roll of coral reefs recovering after a storm, cinematic nature photography, vivid oceans, natural lighting" \
  --output ./renders/coral-reef.mp4
```

`--text-transcription` and `--video-prompt` can also point to UTF-8 text files using `file://` URIs.
Examples: `file://output/voicecall.txt` (relative) or `file:///Users/me/prompts/video.txt` (absolute).

Generate a video from text and let the LLM create the visual direction from the narration:

```bash
content-creator from-text \
  --text-transcription file://prompts/test-transcription.txt \
  --generate-video-prompt \
  --output ./renders/generated-direction.mp4
```

Generate a video from an existing audio file:

```bash
content-creator from-audio \
  --audio-file ./assets/voiceover.mp3 \
  --video-prompt "clean futuristic data center visuals with slow cinematic camera motion, blue accents, realistic lighting" \
  --preserve-speaker \
  --chunk-seconds 45 \
  --output ./renders/data-center.mp4
```

`--generate-video-prompt` is also available on `from-audio`, which makes `--video-prompt` optional and asks the LLM to derive a reusable visual direction from the transcript.

Transcribe audio only:

```bash
content-creator transcribe \
  --audio-file ./assets/voiceover.mp3 \
  --preserve-speaker \
  --chunk-seconds 45 \
  --output ./renders/voiceover.txt
```

`--preserve-speaker` enables diarization + STT, producing speaker-labeled transcript lines like `SPEAKER_00: ...`.

Requirements for `--preserve-speaker`:

- Install pyannote locally: `pip install pyannote.audio`
- Accept model terms on Hugging Face for `pyannote/speaker-diarization-3.1` (and any dependent pyannote model pages)
- Ensure `HF_TOKEN` has access to those gated models
- Optional: set `HF_DIARIZATION_MODEL` to override the diarization model

Enable debug mode (emoji status + verbose chunk progress + full tracebacks):

```bash
content-creator --debug transcribe \
  --audio-file ./assets/voiceover.mp3 \
  --chunk-seconds 30
```

Override models from the CLI (takes precedence over environment variables):

```bash
content-creator \
  --llm-model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --stt-model openai/whisper-large-v3 \
  --tts-model espnet/kan-bayashi_ljspeech_vits \
  --image-model stabilityai/stable-diffusion-xl-base-1.0 \
  from-text \
  --text-transcription "A short narration" \
  --video-prompt "cinematic skyline at dawn" \
  --output ./renders/example.mp4
```

Short aliases are also available: `-L`, `-S`, `-T`, `-I`.

Compact short-flag example:

```bash
content-creator -L mistralai/Mixtral-8x7B-Instruct-v0.1 -S openai/whisper-large-v3 -T espnet/kan-bayashi_ljspeech_vits -I stabilityai/stable-diffusion-xl-base-1.0 from-audio --audio-file ./assets/voiceover.mp3 --video-prompt "cinematic skyline at dawn" --chunk-seconds 45 --output ./renders/example-shortflags.mp4
```

Validate configuration:

```bash
content-creator doctor
```

Run tests:

```bash
pip install -e '.[test]'
pytest -q
```

## Output structure

Each run writes intermediate assets to the configured work directory:

- generated narration or the referenced audio track
- per-scene images
- temporary scene clips
- `manifest.json` with prompts, transcript text, and scene durations

## Notes

- The final video is assembled locally using static AI images with gentle camera motion. This keeps the pipeline dependable and inexpensive while still using Stable Diffusion for visual generation.
- STT chunking uses `ffmpeg` segmenting and transcribes each chunk sequentially, then joins chunk transcripts in order. Set `--chunk-seconds 0` to disable chunking.
- When `--preserve-speaker` is enabled, diarization segmentation is used instead of chunk-based STT.
- Model availability depends on your Hugging Face account permissions and inference quota.
- If you want to adapt this for direct YouTube upload later, the natural next step is adding metadata generation and YouTube Data API integration.
