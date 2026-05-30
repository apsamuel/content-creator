# `rebuild-video` command

Rebuild the final MP4 from an existing run directory without re-running transcription, LLM scene planning, or image generation.

This is designed for fast, low-cost retries when ffmpeg assembly fails late in the pipeline.

## What it reuses

- `manifest.json` in the run directory
- existing scene images (usually `images/scene_XX_frame_YY.png`)
- existing audio from manifest (or optional override)

## Usage

```bash
content-creator rebuild-video \
  --run-dir ./output/toxic-masculinity \
  --output ./output/toxic-masculinity-retry.mp4
```

## Common retry variants

Retry with the same settings from the manifest:

```bash
content-creator rebuild-video \
  --run-dir ./output/toxic-masculinity \
  --output ./output/toxic-masculinity-retry.mp4
```

Retry while disabling TV overlay effects:

```bash
content-creator rebuild-video \
  --run-dir ./output/toxic-masculinity \
  --output ./output/toxic-masculinity-retry.mp4 \
  --television-overlay-effects off
```

Retry with replacement audio:

```bash
content-creator rebuild-video \
  --run-dir ./output/toxic-masculinity \
  --output ./output/toxic-masculinity-retry.mp4 \
  --audio-file ./output/toxic-masculinity/audio_censored.m4a
```

## Options

- `--run-dir` (required): existing run directory that contains `manifest.json`
- `--output` (required): output MP4 path for rebuilt video
- `--audio-file`: optional audio override (default comes from manifest)
- `--cinematic-intro`: `auto|on|off` (default `auto`)
- `--cinematic-intro-duration`: optional intro duration override
- `--cinematic-transitions`: `auto|on|off` (default `auto`)
- `--television-overlay-effects`: `auto|on|off` (default `auto`)
- `--work-dir`: optional pipeline work dir override
