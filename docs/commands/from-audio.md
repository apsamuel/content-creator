# from-audio Command

`from-audio` generates a complete video starting from an existing audio track.

## What this command does

This command performs an end-to-end workflow:

1. Accepts an input audio file.
2. Transcribes the audio into text (chunked STT by default).
3. Resolves visual direction from either:
   - `--video-prompt`, or
   - `--generate-video-prompt` (LLM-generated from transcript).
4. Plans scenes from transcript content and visual direction.
5. Generates scene images.
6. Assembles final video using the original audio.

## When to use it

Use `from-audio` when you already have narration audio and want matching visuals.

## Required and Optional Inputs

- Required:
  - `--audio-file FILE`
  - `--output FILE`
- One visual-direction choice is required:
  - `--video-prompt TEXT`, or
  - `--generate-video-prompt`
- Optional:
  - `--chunk-seconds FLOAT` (default `45.0`; set to `0` to disable chunking)
  - `--preserve-speaker / --no-preserve-speaker` (default `--no-preserve-speaker`)
  - `--speaker-count INTEGER` (force diarization to an exact speaker count)
  - `--min-speakers INTEGER` (minimum speaker count bound for diarization)
  - `--max-speakers INTEGER` (maximum speaker count bound for diarization)
  - `--speaker-dominance-threshold FLOAT` (default `HF_SPEAKER_DOMINANCE_THRESHOLD` or `0.9`; only used when auto-collapse is active)
  - `--content-safety / --no-content-safety` (default `--no-content-safety`)
  - `--content-safety-filter / --no-content-safety-filter` (default `--no-content-safety-filter`)
  - `--content-safety-threshold FLOAT` (default `0.7`)
  - `--content-safety-model TEXT` (default `unitary/unbiased-toxic-roberta`)
  - `--profanity-sfx / --no-profanity-sfx` (default `--no-profanity-sfx`)
  - `--profanity-sound-pack-dir DIR` (default bundled `src/content_creator/sound`)
  - `--profanity-words-file FILE` (optional custom lexicon; default is bundled `data/profanity_words.txt`, with one word or phrase per line)
  - `--profanity-pad-ms INTEGER` (default `80`)
  - `--profanity-duck-db FLOAT` (default `-42.0`)
  - `--work-dir TEXT`

## STT and diarization behavior

- Default mode uses STT with optional chunking (`--chunk-seconds`).
- If `--preserve-speaker` is enabled, diarization is used so transcript lines are speaker-labeled.
- In speaker mode, if one speaker dominates ~90%+ of diarized duration, sparse secondary labels are automatically collapsed into the primary speaker.
- Speaker-preserving mode requires diarization dependencies and model access.
- Explicit speaker constraints (`--speaker-count` or `--min-speakers/--max-speakers`) disable automatic dominant-speaker collapse.
- `--speaker-dominance-threshold` (or `HF_SPEAKER_DOMINANCE_THRESHOLD`) controls when automatic collapse triggers.
- If `--content-safety` is enabled, transcript text is labeled for unsafe content.
- If `--content-safety-filter` is also enabled, flagged chunks are dropped before scene planning.
- If `--profanity-sfx` is enabled, word-level timestamps are used to replace profane words in final audio with effects from the selected sound pack.

## Mechanism Flow

```mermaid
flowchart TD
    A[User runs from-audio] --> B[Validate audio path]
    B --> C{Video prompt provided?}
    C -->|Yes| D[Use provided visual direction]
    C -->|No + generate flag| E[Mark prompt for LLM generation]
    C -->|No + no generate flag| F[Fail with CLI error]
    D --> G[Build pipeline]
    E --> G
    G --> H{Preserve speaker enabled?}
    H -->|No| I[Transcribe audio with STT chunking rules]
    H -->|Yes| J[Run diarization and produce speaker-labeled transcript]
    I --> K[Resolve final video prompt]
    J --> K
    K --> L[Calculate scene count from audio duration]
    L --> M[Split narration into per-scene chunks]
    M --> N[Send style template + per-scene chunks to LLM]
    N --> O[LLM returns story_anchor + scenes JSON]
    O --> P[Extract story_anchor, summary, continuity, prompt per scene]
    P --> Q[Compose final prompt per scene:\nstory_anchor + previous beat + carry-forward continuity + scene prompt]
    Q --> R[Spread total duration evenly across scenes]
    R --> S[Generate images per scene]
    S --> T[Assemble final video with original audio]
    T --> U[Write output MP4]
```

## Practical Examples

Prompt provided directly:

```bash
content-creator from-audio \
  --audio-file ./assets/voiceover.mp3 \
  --video-prompt "clean futuristic datacenter, cinematic camera movement, realistic light" \
  --chunk-seconds 45 \
  --output ./output/datacenter.mp4
```

Prompt generated from transcript:

```bash
content-creator from-audio \
  --audio-file ./assets/voiceover.mp3 \
  --generate-video-prompt \
  --output ./output/generated-style-from-audio.mp4
```

Speaker-preserving transcript mode:

```bash
content-creator from-audio \
  --audio-file ./assets/interview.wav \
  --generate-video-prompt \
  --preserve-speaker \
  --speaker-count 1 \
  --chunk-seconds 0 \
  --output ./output/interview.mp4
```

Moderate and filter transcript chunks before planning scenes:

```bash
content-creator from-audio \
  --audio-file ./assets/voiceover.mp3 \
  --generate-video-prompt \
  --chunk-seconds 45 \
  --content-safety \
  --content-safety-filter \
  --content-safety-threshold 0.8 \
  --content-safety-model unitary/toxic-bert \
  --output ./output/filtered.mp4
```

Apply profanity replacement with a custom sound pack:

```bash
content-creator from-audio \
  --audio-file ./assets/voiceover.mp3 \
  --generate-video-prompt \
  --profanity-sfx \
  --profanity-sound-pack-dir ./src/content_creator/sound \
  --profanity-pad-ms 100 \
  --profanity-duck-db -42 \
  --output ./output/voiceover-clean.mp4
```

## Failure Modes to Expect

- Missing both `--video-prompt` and `--generate-video-prompt`: command fails early.
- Invalid audio path: command fails before pipeline execution.
- Diarization requirements not satisfied with `--preserve-speaker`: runtime error with setup details.
- Missing token, model permissions, or ffmpeg tools: startup or pipeline failure.
