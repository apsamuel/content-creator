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

## Hugging Face Resilience Environment Variables

These environment variables apply to all Hugging Face inference calls (LLM, STT, TTS, moderation, image generation):

- `HF_INFERENCE_MAX_RETRIES` (default `5`)
- `HF_INFERENCE_BASE_DELAY_SECONDS` (default `1.0`)
- `HF_INFERENCE_MAX_DELAY_SECONDS` (default `30.0`)
- `HF_INFERENCE_JITTER_SECONDS` (default `0.35`)
- `HF_INFERENCE_MIN_INTERVAL_SECONDS` (default `0.25`)

The gateway retries on rate limits (`429`) and transient server/network errors using jittered exponential backoff, and it also spaces outbound requests to reduce burst throttling.

## Image Generation Environment Variables

- `HF_IMAGE_MODEL` selects the default image model when `-I/--image-model` is not passed.
- `HF_IMAGE_NEGATIVE_PROMPT` sets the default negative prompt appended to every Hugging Face text-to-image request.
- `CONTENT_CREATOR_WORK_DIR` controls where manifests, audio intermediates, and generated scene images are written.

Use `HF_IMAGE_NEGATIVE_PROMPT` to suppress recurring artifacts globally instead of repeating negative terms in every scene prompt. The default preset targets blur, anatomy mistakes, duplicate limbs, text overlays, watermarks, borders, photorealism, flat lighting, and muddy colors.

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
