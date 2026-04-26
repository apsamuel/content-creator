# Global CLI Options

Global options are specified before the command name. They apply to all commands.

Example:

```bash
content-creator --debug --llm-model mistralai/Mixtral-8x7B-Instruct-v0.1 from-text ...
```

## Option Reference

### `--debug / --no-debug`

- Default: `--no-debug`
- Enables verbose runtime behavior and full tracebacks on failure.
- Practical effect:
  - In normal mode, unexpected exceptions are converted into a concise CLI error.
  - In debug mode, the original exception is raised, which is useful when diagnosing stack traces.

### `-L, --llm-model TEXT`

- Overrides the LLM model used for scene planning and optional video prompt generation.
- Takes precedence over environment configuration.

### `-S, --stt-model TEXT`

- Overrides the speech-to-text model used for transcription.
- Relevant for `from-audio` and `transcribe`.

### `-T, --tts-model TEXT`

- Overrides the text-to-speech model used to synthesize narration.
- Relevant for `from-text`.

### `-I, --image-model TEXT`

- Overrides the image generation model used for scene visuals.
- Relevant for `from-text` and `from-audio`.

### `--progress / --no-progress`

- Default: `--progress`
- Controls whether status lines containing progress updates are printed.
- Useful when piping command output or keeping logs quieter during long runs.

## Core Environment Variables

- `HF_TOKEN` is required for Hugging Face inference API access.
- `CONTENT_CREATOR_WORK_DIR` sets the default directory used for manifests, clips, audio intermediates, and generated scene images (default `./output`).

## Model Selection Environment Variables

- `HF_LLM_MODEL` selects the default LLM for scene planning and `--generate-video-prompt`.
- `HF_STT_MODEL` selects the default speech-to-text model.
- `HF_TTS_MODEL` selects the default narration model.
- `HF_IMAGE_MODEL` selects the default image model when `-I/--image-model` is not passed.
- `HF_CONTENT_SAFETY_MODEL` selects the default moderation model used when `--content-safety-model` is omitted.
- `HF_DIARIZATION_MODEL` selects the default pyannote diarization model used with `--preserve-speaker`.

CLI model overrides still win over these environment variables.

## Tuning Profile Environment Variables

- `HF_TUNING_PROFILE` picks a bundled inference profile. Supported values: `balanced`, `cinematic`, `consistent`, `fast`.
- `HF_LLM_MAX_TOKENS` overrides the profile/default LLM token budget.
- `HF_LLM_TEMPERATURE` overrides the profile/default LLM temperature.
- `HF_LLM_TOP_P` overrides the profile/default LLM top-p.
- `HF_IMAGE_NUM_INFERENCE_STEPS` overrides the profile/default image inference steps.
- `HF_IMAGE_GUIDANCE_SCALE` overrides the profile/default image guidance scale.
- `HF_IMAGE_SEED` sets a deterministic seed for image generation.
- `HF_SAFETY_TOP_K` overrides the moderation classifier top-k setting.

## Hugging Face Resilience Environment Variables

These environment variables apply to all Hugging Face inference calls (LLM, STT, TTS, moderation, image generation):

- `HF_INFERENCE_MAX_RETRIES` (default `5`)
- `HF_INFERENCE_BASE_DELAY_SECONDS` (default `1.0`)
- `HF_INFERENCE_MAX_DELAY_SECONDS` (default `30.0`)
- `HF_INFERENCE_JITTER_SECONDS` (default `0.35`)
- `HF_INFERENCE_MIN_INTERVAL_SECONDS` (default `0.25`)

The gateway retries on rate limits (`429`) and transient server/network errors using jittered exponential backoff, and it also spaces outbound requests to reduce burst throttling.

## Image Generation Environment Variables

- `HF_IMAGE_NEGATIVE_PROMPT` sets the default negative prompt appended to every Hugging Face text-to-image request.
- `HF_IMAGE_COMPOSITION_MODE` sets the planner's default composition rotation. Supported values: `balanced`, `dynamic`, `portrait`, `establishing`.
- `HF_IMAGE_WORKERS` sets the default image worker count used by `from-text` and `from-audio` when `--image-workers` is omitted.

Use `HF_IMAGE_NEGATIVE_PROMPT` to suppress recurring artifacts globally instead of repeating negative terms in every scene prompt. The default preset targets blur, anatomy mistakes, duplicate limbs, text overlays, watermarks, borders, photorealism, flat lighting, and muddy colors.

Use `HF_IMAGE_COMPOSITION_MODE` to bias framing globally without rewriting individual prompts. `balanced` rotates between wide, hero, and portrait-friendly compositions; the other modes bias more heavily toward their named framing style.

## Audio Transcription and Diarization Environment Variables

- `HF_TRANSCRIBE_WORKERS` sets the default chunk transcription worker count used by `from-audio` and `transcribe` when `--transcribe-workers` is omitted.
- `HF_DIARIZATION_MIN_SEGMENT_SECONDS` drops diarization segments shorter than this duration before per-speaker chunk transcription (default `0.5`).
- `HF_SPEAKER_DOMINANCE_THRESHOLD` sets the default threshold used to auto-collapse sparse secondary speakers when explicit speaker constraints are not provided (default `0.9`).

## Image Provider Billing Environment Variables

By default, image generation is **routed through Hugging Face** and charged to your HF account credits. To bypass HF credits and bill a provider directly, set both of the following:

- `HF_INFERENCE_PROVIDER` — the inference provider to use for image generation. Supported values for `black-forest-labs/FLUX.1-dev`: `fal-ai`, `replicate`, `nebius`, `wavespeed`.
- `HF_PROVIDER_KEY` — the provider's own API key (**not** your HF token).

When both variables are set, a dedicated `InferenceClient` is constructed for image calls using the provider's key. All other inference calls (LLM, STT, TTS, moderation) continue to use the HF-routed client.

| Provider | Key format | Key dashboard |
|---|---|---|
| `fal-ai` | `key_…` | https://fal.ai/dashboard/keys |
| `replicate` | `r8_…` | https://replicate.com/account/api-tokens |
| `nebius` | — | https://studio.nebius.ai/settings/api-keys |
| `wavespeed` | — | https://wavespeed.ai/api-keys |

Example `.envrc` entry:

```bash
HF_INFERENCE_PROVIDER=replicate
HF_PROVIDER_KEY=r8_your_replicate_key
```

Leave both variables unset to use the default HF-routed path.

## Startup Behavior

For commands that build a pipeline (`from-text`, `from-audio`, `transcribe`), the CLI performs startup checks and prints active model and work directory information.

```mermaid
flowchart TD
    A[User runs content-creator with global options] --> B[Resolve environment and CLI model overrides]
    B --> C[Build AppConfig]
    C --> D[Print startup check output]
    D --> E[Run selected subcommand]
    E --> F{Error occurred?}
    F -->|No| G[Complete successfully]
    F -->|Yes + no debug| H[Show concise CLI error]
    F -->|Yes + debug| I[Raise original exception with traceback]
```

## Common Patterns

Use global overrides to test models without editing environment files:

```bash
content-creator \
  -L mistralai/Mixtral-8x7B-Instruct-v0.1 \
  -S openai/whisper-large-v3 \
  -T espnet/kan-bayashi_ljspeech_vits \
  -I stabilityai/stable-diffusion-xl-base-1.0 \
  doctor
```
