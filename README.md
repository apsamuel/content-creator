# content-creator

CLI-first Python tool for generating videos for YouTube workflows using Hugging Face inference APIs.

## What it does

The tool is designed around two primary operating modes:

1. `from-text`: takes narration text and either a video prompt or `--generate-video-prompt`, generates speech with Hugging Face TTS, plans visuals with an LLM, generates scene images with Stable Diffusion, and assembles a video locally with `ffmpeg`.
2. `from-audio`: takes an existing audio file and either a video prompt or `--generate-video-prompt`, transcribes the audio with Whisper, uses the transcript plus your prompt to plan scenes, generates images, and assembles a final video using the supplied audio track.

There is also a focused utility command:

1. `transcribe`: transcribes an audio file only, with optional ffmpeg chunking for better STT quality on long files.
2. `lexicon-doctor`: audits profanity lexicon files for duplicate and near-duplicate entries.
3. `profanity-debug`: renders a debug audio artifact that announces and previews each profanity replacement event.

This gives you a single CLI that uses:

- LLM for storyboard planning
- speech-to-text for transcript extraction
- text-to-speech for narration synthesis
- stable diffusion for image generation
- optional speaker diarization for speaker-labeled transcripts
- optional content safety labeling/filtering on transcribed audio or chunks
- optional profanity replacement with cute sound effects using word timestamps

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
set -a
source .envrc
set +a

```

`HF_IMAGE_NEGATIVE_PROMPT` in `.envrc` controls the default negative prompt passed to Hugging Face image generation. Use it to suppress recurring artifacts such as blur, anatomy errors, watermarks, text overlays, flat lighting, muddy colors, or unwanted photorealism.

`HF_IMAGE_COMPOSITION_MODE` sets the default still-image framing strategy used when the planner prepares scene prompts for image generation. Supported values are `balanced`, `dynamic`, `portrait`, and `establishing`.

## Commands

Detailed command documentation with flowcharts is available in [docs/README.md](docs/README.md):

- [Global CLI options](docs/commands/global-options.md)
- [from-text command](docs/commands/from-text.md)
- [from-audio command](docs/commands/from-audio.md)
- [transcribe command](docs/commands/transcribe.md)
- [doctor command](docs/commands/doctor.md)
- [lexicon-doctor command](docs/commands/lexicon-doctor.md)
- [profanity-debug command](docs/commands/profanity-debug.md)
- [calibrate command](docs/commands/calibrate.md)

### Modern Cinematic Features

**Automatic Cinematic Transitions** - The system applies modern, professional-grade transitions between scenes:

- **Dissolve** (soft cross-fade) — Emotional continuity, mood changes
- **Match Cut** (thematic connection) — Narrative bridges via visual rhyme
- **Whip Pan** (dynamic pan) — Action, urgency, high-energy moments
- **Focus Shift** (depth transition) — Intimate moments, attention shifts
- **Color Match** (palette gradient) — Mood maintenance, visual harmony
- **Light Leak** (cinematic flare) — Hopeful moments, dramatic reveals
- **Tracking** (camera motion) — Spatial storytelling, connected spaces
- **Bokeh** (foreground blur) — Romantic/dreamy sequences, elegance

Transitions are **automatically selected** based on narrative pacing, content intensity, and scene position. Full documentation: [Cinematic Transitions Guide](docs/CINEMATIC_TRANSITIONS.md)

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

Content safety options (available on `from-audio` and `transcribe`):

- `--content-safety` enables moderation labeling on transcript text (full-file or per chunk).
- `--content-safety-filter` drops flagged transcript segments.
- `--content-safety-threshold` controls flagging sensitivity (0.0 to 1.0, default `0.7`).
- `--content-safety-model` selects the moderation model (default `unitary/unbiased-toxic-roberta`).

Example with filtering enabled:

```bash
content-creator transcribe \
  --audio-file ./assets/voiceover.mp3 \
  --chunk-seconds 45 \
  --content-safety \
  --content-safety-filter \
  --content-safety-threshold 0.8 \
  --content-safety-model unitary/toxic-bert \
  --output ./renders/voiceover-safe.txt
```

Profanity SFX replacement options:

- `--profanity-sfx` enables timestamped profanity detection and replacement.
- `--profanity-sound-pack-dir` points to a folder of effect files (`wav`, `mp3`, `m4a`, `flac`, `ogg`).
- `--profanity-words-file` overrides the bundled `data/profanity_words.txt` lexicon. Each line can be a single word or a multi-word phrase.
- `--profanity-pad-ms` adds timing padding around each detected word.
- `--profanity-duck-db` controls source-audio ducking while the SFX plays.

Example for `from-audio` (replacement is embedded into final video audio):

```bash
content-creator from-audio \
  --audio-file ./assets/voiceover.mp3 \
  --generate-video-prompt \
  --profanity-sfx \
  --profanity-sound-pack-dir ./src/content_creator/sound \
  --output ./renders/voiceover-clean.mp4
```

Example for `transcribe` (requires explicit audio output path):

```bash
content-creator transcribe \
  --audio-file ./assets/voiceover.mp3 \
  --profanity-sfx \
  --profanity-sfx-output ./renders/voiceover-clean.m4a \
  --output ./renders/voiceover.txt
```

Notes for timestamped profanity replacement:

- Word-level timestamps depend on STT model capability. `openai/whisper-large-v3` supports this flow.
- No additional model is required by default if you keep the current STT default.
- `ffmpeg` and `ffprobe` remain required for audio analysis and rendering.

Hugging Face model families you can evaluate for safety labeling/filtering:

- Toxicity / abuse classifiers: `unitary/unbiased-toxic-roberta`, `unitary/toxic-bert`
- Moderation-focused text classifiers: `KoalaAI/Text-Moderation`, `NemoraAi/roberta-chat-moderation-X`
- Guardrail-style classifiers: `allenai/wildguard`, `meta-llama/Llama-Guard-3-8B`

Pick one model and tune the threshold with your own audio corpus before enforcing hard filtering in production.

Requirements for `--preserve-speaker`:

- Install pyannote locally: `pip install pyannote.audio`
- Accept model terms on Hugging Face for `pyannote/speaker-diarization-3.1` (and any dependent pyannote model pages)
- Ensure `HF_TOKEN` has access to those gated models
- Optional: set `HF_DIARIZATION_MODEL` to override the diarization model

Hugging Face inference reliability controls (all optional, with safe defaults):

- `HF_INFERENCE_MAX_RETRIES` (default: `5`) controls retry attempts after the first request.
- `HF_INFERENCE_BASE_DELAY_SECONDS` (default: `1.0`) sets the initial exponential backoff delay.
- `HF_INFERENCE_MAX_DELAY_SECONDS` (default: `30.0`) caps per-retry wait time.
- `HF_INFERENCE_JITTER_SECONDS` (default: `0.35`) adds random jitter to reduce synchronized retries.
- `HF_INFERENCE_MIN_INTERVAL_SECONDS` (default: `0.25`) enforces a minimum gap between outbound HF requests.

These apply to chat completion, STT, TTS, moderation, and image generation calls.

Additional environment controls:

- `HF_CONTENT_SAFETY_MODEL` selects the default moderation model used when `--content-safety-model` is not passed.
- `HF_DIARIZATION_MODEL` selects the default pyannote diarization model used with `--preserve-speaker`.
- `HF_DIARIZATION_MIN_SEGMENT_SECONDS` drops diarization segments shorter than this duration before chunk transcription (default `0.5`).
- `HF_IMAGE_WORKERS` sets the default worker count for scene image generation when `--image-workers` is omitted.
- `HF_TRANSCRIBE_WORKERS` sets the default worker count for STT chunk processing when `--transcribe-workers` is omitted.
- `HF_SPEAKER_DOMINANCE_THRESHOLD` sets the default threshold for auto-collapsing sparse secondary speakers when explicit speaker bounds are not provided.

Image generation environment variables:

- `HF_IMAGE_MODEL` selects the Hugging Face image model.
- `HF_IMAGE_NEGATIVE_PROMPT` sets the default negative prompt applied to every generated image request.
- `HF_IMAGE_COMPOSITION_MODE` controls the default composition rotation used while preparing scene prompts. Use `balanced` for mixed coverage, `dynamic` for more aggressive angles, `portrait` for character-forward framing, or `establishing` for wider scene coverage.
- `CONTENT_CREATOR_WORK_DIR` controls where intermediate assets and manifests are written.

You can use [examples/envrc.example](examples/envrc.example) as the authoritative template for the full current environment variable set.

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
- `manifest.json` with prompts, transcript text, scene durations, and video-prompt preclassification metadata

Example `video_prompt_preclassification` block written to `manifest.json`:

```json
{
  "video_prompt_preclassification": {
    "mood": "informative",
    "has_foul_language": false,
    "word_count": 312,
    "sentence_count": 18,
    "truthfulness_assessment": {
      "label": "MostlyFactual",
      "confidence_score": 0.82,
      "reason": "Content contains verifiable technical claims with no apparent exaggeration."
    },
    "interaction_style_assessment": {
      "formality": {
        "label": "Mixed",
        "confidence_score": 0.75,
        "reason": "Speaker alternates between structured explanations and conversational asides."
      },
      "certainty_hedging": {
        "label": "Balanced",
        "confidence_score": 0.7,
        "reason": "Some hedging phrases like 'I think' and 'probably' appear, but most claims are direct."
      },
      "persuasion_intent": {
        "label": "Moderate",
        "confidence_score": 0.65,
        "reason": "Occasional calls to action and framing of benefits suggest light persuasion."
      },
      "claim_density": {
        "label": "Medium",
        "confidence_score": 0.78,
        "reason": "Several factual assertions per paragraph without overwhelming the listener."
      },
      "speaker_sentiment": [
        {
          "speaker": "SPEAKER_00",
          "sentiment": "Neutral",
          "confidence_score": 0.85,
          "reason": "Measured, even tone throughout with no strong emotional peaks."
        },
        {
          "speaker": "SPEAKER_01",
          "sentiment": "Positive",
          "confidence_score": 0.79,
          "reason": "Enthusiastic phrasing and positive framing detected in most segments."
        }
      ]
    }
  }
}
```

Use `--view-preclassification` on `from-text` or `from-audio` to print this block to the terminal immediately after the LLM analysis completes, without waiting for the full video render.

## Notes

- The final video is assembled locally using static AI images with gentle camera motion. This keeps the pipeline dependable and inexpensive while still using Stable Diffusion for visual generation.
- STT chunking uses `ffmpeg` segmenting and transcribes each chunk sequentially, then joins chunk transcripts in order. Set `--chunk-seconds 0` to disable chunking.
- When `--preserve-speaker` is enabled, diarization segmentation is used instead of chunk-based STT.
- Model availability depends on your Hugging Face account permissions and inference quota.
- If you want to adapt this for direct YouTube upload later, the natural next step is adding metadata generation and YouTube Data API integration.
