from __future__ import annotations

import os
from pathlib import Path
from typing import Callable
from urllib.parse import unquote

import click

from content_creator.config import AppConfig
from content_creator.pipeline import VideoGenerationPipeline
from content_creator.profanity_sfx import analyze_profanity_lexicon


def _print_startup_check(config: AppConfig) -> None:
    click.echo("🔎 Startup check")
    click.echo(f"📁 Work directory: {config.work_dir}")
    click.echo(f"🎛️ Tuning profile: {config.tuning_profile}")
    click.echo(f"🧠 LLM model: {config.models.llm_model}")
    click.echo(f"🎧 STT model: {config.models.stt_model}")
    click.echo(f"🔊 TTS model: {config.models.tts_model}")
    click.echo(f"🛡️ Content safety model: {config.models.safety_model}")
    click.echo(f"🎙️ Diarization model: {config.models.diarization_model}")
    click.echo(f"🖼️ Image model: {config.models.image_model}")
    click.echo(f"🎬 Image composition mode: {config.image_composition_mode}")
    click.echo(
        "🧠 LLM inference: "
        f"max_tokens={config.llm_inference.max_tokens}, "
        f"temperature={config.llm_inference.temperature}, "
        f"top_p={config.llm_inference.top_p}"
    )
    click.echo(
        "🖼️ Image inference: "
        f"steps={config.image_inference.num_inference_steps}, "
        f"guidance_scale={config.image_inference.guidance_scale}, "
        f"seed={config.image_inference.seed}"
    )
    click.echo(f"🛡️ Safety inference top_k: {config.safety_inference.top_k}")


def _build_pipeline(
    *,
    work_dir: str | None,
    debug: bool,
    status_callback: Callable[[str], None] | None,
    llm_model: str | None,
    stt_model: str | None,
    tts_model: str | None,
    image_model: str | None,
) -> VideoGenerationPipeline:
    try:
        config = AppConfig.from_env(
            work_dir=work_dir,
            llm_model=llm_model,
            stt_model=stt_model,
            tts_model=tts_model,
            image_model=image_model,
        )
        _print_startup_check(config)
        return VideoGenerationPipeline(
            config, debug=debug, status_callback=status_callback
        )
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


def _status(message: str) -> None:
    click.echo(message)


def _make_status_callback(*, progress_enabled: bool) -> Callable[[str], None]:
    def _callback(message: str) -> None:
        if not progress_enabled and "progress:" in message:
            return
        click.echo(message)

    return _callback


def _resolve_text_option(value: str, *, option_name: str) -> str:
    if not value.startswith("file://"):
        return value

    raw_path = unquote(value[len("file://") :]).strip()
    if not raw_path:
        raise click.ClickException(f"{option_name} file reference is empty")

    file_path = Path(raw_path).expanduser()
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path

    try:
        resolved_text = file_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise click.ClickException(
            f"Unable to read {option_name} from {file_path}: {exc}"
        ) from exc

    if not resolved_text:
        raise click.ClickException(f"{option_name} file is empty: {file_path}")
    return resolved_text


def _run_with_debug(ctx: click.Context, operation: Callable[[], None]) -> None:
    try:
        operation()
    except click.ClickException:
        raise
    except Exception as exc:
        if bool(ctx.obj.get("debug", False)):
            raise
        message = str(exc).strip() or repr(exc)
        raise click.ClickException(message) from exc


def _resolve_video_prompt_request(
    *,
    video_prompt: str | None,
    generate_video_prompt: bool,
    option_name: str = "--video-prompt",
) -> str | None:
    if video_prompt is not None:
        return _resolve_text_option(video_prompt, option_name=option_name)
    if generate_video_prompt:
        return None
    raise click.ClickException(
        f"{option_name} is required unless --generate-video-prompt is enabled"
    )


def _resolve_worker_count(
    value: int | None, *, env_var: str, option_name: str, default: int = 1
) -> int:
    if value is not None:
        return value

    raw_value = os.getenv(env_var, "").strip()
    if not raw_value:
        return default

    try:
        resolved = int(raw_value)
    except ValueError as exc:
        raise click.ClickException(
            f"{option_name} must be an integer >= 1 (from {env_var})"
        ) from exc

    if resolved < 1:
        raise click.ClickException(
            f"{option_name} must be an integer >= 1 (from {env_var})"
        )
    return resolved


def _resolve_diarization_speaker_options(
    *, speaker_count: int | None, min_speakers: int | None, max_speakers: int | None
) -> tuple[int | None, int | None, int | None]:
    if speaker_count is not None and (
        min_speakers is not None or max_speakers is not None
    ):
        raise click.ClickException(
            "--speaker-count cannot be used with --min-speakers/--max-speakers"
        )
    if (
        min_speakers is not None
        and max_speakers is not None
        and min_speakers > max_speakers
    ):
        raise click.ClickException(
            "--min-speakers cannot be greater than --max-speakers"
        )
    return speaker_count, min_speakers, max_speakers


def _resolve_speaker_dominance_threshold(value: float | None) -> float:
    if value is not None:
        return value

    raw_value = os.getenv("HF_SPEAKER_DOMINANCE_THRESHOLD", "").strip()
    if not raw_value:
        return 0.9

    try:
        resolved = float(raw_value)
    except ValueError as exc:
        raise click.ClickException(
            "--speaker-dominance-threshold must be between 0.0 and 1.0 "
            "(from HF_SPEAKER_DOMINANCE_THRESHOLD)"
        ) from exc

    if resolved < 0.0 or resolved > 1.0:
        raise click.ClickException(
            "--speaker-dominance-threshold must be between 0.0 and 1.0 "
            "(from HF_SPEAKER_DOMINANCE_THRESHOLD)"
        )
    return resolved


@click.group()
@click.option(
    "--debug/--no-debug",
    default=False,
    show_default=True,
    help="Enable verbose debug mode with full tracebacks.",
)
@click.option(
    "--llm-model", "-L", default=None, help="Override the LLM model for scene planning."
)
@click.option(
    "--stt-model",
    "-S",
    default=None,
    help="Override the speech-to-text model for transcription.",
)
@click.option(
    "--tts-model",
    "-T",
    default=None,
    help="Override the text-to-speech model for narration.",
)
@click.option(
    "--image-model",
    "-I",
    default=None,
    help="Override the image generation model for visuals.",
)
@click.option(
    "--progress/--no-progress",
    default=True,
    show_default=True,
    help="Show progress bars for long-running operations.",
)
@click.pass_context
def cli(
    ctx: click.Context,
    debug: bool,
    llm_model: str | None,
    stt_model: str | None,
    tts_model: str | None,
    image_model: str | None,
    progress: bool,
) -> None:
    """CLI-first AI video generation built on Hugging Face inference APIs."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["llm_model"] = llm_model
    ctx.obj["stt_model"] = stt_model
    ctx.obj["tts_model"] = tts_model
    ctx.obj["image_model"] = image_model
    ctx.obj["progress"] = progress
    if debug:
        click.echo("🐛 Debug mode enabled")


@cli.command("from-text")
@click.option(
    "--text-transcription",
    required=True,
    help="Narration text that will be synthesized into audio.",
)
@click.option(
    "--video-prompt",
    required=False,
    help="Creative direction for the generated visuals.",
)
@click.option(
    "--generate-video-prompt/--no-generate-video-prompt",
    default=False,
    show_default=True,
    help="Generate the visual direction from the narration with the configured LLM.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--cinematic-intro/--no-cinematic-intro",
    default=False,
    show_default=True,
    help=(
        "Generate an LLM-crafted cinematic title card and short description at the "
        "beginning of the rendered video."
    ),
)
@click.option(
    "--cinematic-intro-duration",
    default=5.8,
    show_default=True,
    type=click.FloatRange(2.0, 20.0),
    help=(
        "Duration in seconds for the cinematic intro title card when "
        "--cinematic-intro is enabled."
    ),
)
@click.option(
    "--image-workers",
    default=None,
    show_default=False,
    type=click.IntRange(1, None),
    help="Number of worker threads for scene image generation (default: HF_IMAGE_WORKERS or 1).",
)
@click.option(
    "--images-per-scene",
    default=None,
    show_default=False,
    type=click.IntRange(1, None),
    help="Number of images generated per scene clip (default: HF_IMAGES_PER_SCENE or 1).",
)
@click.option("--work-dir", default=None, help="Directory for intermediate assets.")
@click.option(
    "--view-preclassification/--no-view-preclassification",
    default=False,
    show_default=True,
    help="Print the pre-classification analysis to the terminal after LLM analysis.",
)
@click.pass_context
def from_text(
    ctx: click.Context,
    text_transcription: str,
    video_prompt: str | None,
    generate_video_prompt: bool,
    output_path: Path,
    cinematic_intro: bool,
    cinematic_intro_duration: float,
    image_workers: int | None,
    images_per_scene: int | None,
    work_dir: str | None,
    view_preclassification: bool,
) -> None:
    """Generate narration with TTS and create a matching AI video."""

    def _operation() -> None:
        resolved_text_transcription = _resolve_text_option(
            text_transcription, option_name="--text-transcription"
        )
        resolved_video_prompt = _resolve_video_prompt_request(
            video_prompt=video_prompt,
            generate_video_prompt=generate_video_prompt,
            option_name="--video-prompt",
        )
        resolved_image_workers = _resolve_worker_count(
            image_workers, env_var="HF_IMAGE_WORKERS", option_name="--image-workers"
        )
        resolved_images_per_scene = _resolve_worker_count(
            images_per_scene,
            env_var="HF_IMAGES_PER_SCENE",
            option_name="--images-per-scene",
        )
        pipeline = _build_pipeline(
            work_dir=work_dir,
            debug=bool(ctx.obj.get("debug", False)),
            status_callback=_make_status_callback(
                progress_enabled=bool(ctx.obj.get("progress", True))
            ),
            llm_model=ctx.obj.get("llm_model"),
            stt_model=ctx.obj.get("stt_model"),
            tts_model=ctx.obj.get("tts_model"),
            image_model=ctx.obj.get("image_model"),
        )
        result = pipeline.generate_from_text(
            narration_text=resolved_text_transcription,
            video_prompt=resolved_video_prompt,
            generate_video_prompt=generate_video_prompt,
            output_path=output_path,
            cinematic_intro=cinematic_intro,
            cinematic_intro_duration=cinematic_intro_duration,
            image_workers=resolved_image_workers,
            images_per_scene=resolved_images_per_scene,
            view_preclassification=view_preclassification,
        )
        click.echo(f"✅ Video written to {result}")

    _run_with_debug(ctx, _operation)


@cli.command("from-audio")
@click.option(
    "--audio-file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--video-prompt",
    required=False,
    help="Creative direction for the generated visuals.",
)
@click.option(
    "--generate-video-prompt/--no-generate-video-prompt",
    default=False,
    show_default=True,
    help="Generate the visual direction from the transcript with the configured LLM.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--cinematic-intro/--no-cinematic-intro",
    default=False,
    show_default=True,
    help=(
        "Generate an LLM-crafted cinematic title card and short description at the "
        "beginning of the rendered video."
    ),
)
@click.option(
    "--cinematic-intro-duration",
    default=5.8,
    show_default=True,
    type=click.FloatRange(2.0, 20.0),
    help=(
        "Duration in seconds for the cinematic intro title card when "
        "--cinematic-intro is enabled."
    ),
)
@click.option(
    "--image-workers",
    default=None,
    show_default=False,
    type=click.IntRange(1, None),
    help="Number of worker threads for scene image generation (default: HF_IMAGE_WORKERS or 1).",
)
@click.option(
    "--images-per-scene",
    default=None,
    show_default=False,
    type=click.IntRange(1, None),
    help="Number of images generated per scene clip (default: HF_IMAGES_PER_SCENE or 1).",
)
@click.option(
    "--chunk-seconds",
    default=45.0,
    show_default=True,
    type=float,
    help="Chunk duration for STT requests. Set to 0 to disable chunking.",
)
@click.option(
    "--transcribe-workers",
    default=None,
    show_default=False,
    type=click.IntRange(1, None),
    help="Number of worker threads for chunk transcription (default: HF_TRANSCRIBE_WORKERS or 1).",
)
@click.option(
    "--preserve-speaker/--no-preserve-speaker",
    default=False,
    show_default=True,
    help="Use speaker diarization so transcript text is labeled by speaker.",
)
@click.option(
    "--speaker-count",
    default=None,
    show_default=False,
    type=click.IntRange(1, None),
    help="Force diarization to a fixed number of speakers (mutually exclusive with min/max).",
)
@click.option(
    "--min-speakers",
    default=None,
    show_default=False,
    type=click.IntRange(1, None),
    help="Set minimum speaker count bound for diarization.",
)
@click.option(
    "--max-speakers",
    default=None,
    show_default=False,
    type=click.IntRange(1, None),
    help="Set maximum speaker count bound for diarization.",
)
@click.option(
    "--speaker-dominance-threshold",
    default=None,
    show_default=False,
    type=click.FloatRange(0.0, 1.0),
    help=(
        "Dominance threshold for auto-collapsing sparse secondary speakers "
        "(default: HF_SPEAKER_DOMINANCE_THRESHOLD or 0.9)."
    ),
)
@click.option(
    "--content-safety/--no-content-safety",
    default=False,
    show_default=True,
    help="Label transcript content safety for full audio/chunks using a Hugging Face classifier.",
)
@click.option(
    "--content-safety-filter/--no-content-safety-filter",
    default=False,
    show_default=True,
    help="Drop transcript segments flagged unsafe when content safety is enabled.",
)
@click.option(
    "--content-safety-threshold",
    default=0.7,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Unsafe score threshold used to flag/filter transcript segments.",
)
@click.option(
    "--content-safety-model",
    default=None,
    help="Override Hugging Face moderation model (default: cardiffnlp/twitter-roberta-base-offensive).",
)
@click.option(
    "--profanity-sfx/--no-profanity-sfx",
    default=False,
    show_default=True,
    help="Replace profane words in output audio using timestamped STT and sound effects.",
)
@click.option(
    "--profanity-sound-pack-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing sound effect files (wav/mp3/m4a/flac/ogg).",
)
@click.option(
    "--profanity-words-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Optional profanity lexicon file; defaults to bundled data/profanity_words.txt. "
        "Each line can be a single word or multi-word phrase."
    ),
)
@click.option(
    "--profanity-pad-ms",
    default=80,
    show_default=True,
    type=click.IntRange(0, None),
    help="Padding around each profane word when applying replacement SFX.",
)
@click.option(
    "--profanity-duck-db",
    default=-42.0,
    show_default=True,
    type=float,
    help=(
        "Volume reduction (dB) applied to source audio during profanity windows; "
        "defaults to near-muting under the replacement SFX."
    ),
)
@click.option("--work-dir", default=None, help="Directory for intermediate assets.")
@click.option(
    "--view-preclassification/--no-view-preclassification",
    default=False,
    show_default=True,
    help="Print the pre-classification analysis to the terminal after LLM analysis.",
)
@click.pass_context
def from_audio(
    ctx: click.Context,
    audio_file: Path,
    video_prompt: str | None,
    generate_video_prompt: bool,
    output_path: Path,
    cinematic_intro: bool,
    cinematic_intro_duration: float,
    image_workers: int | None,
    images_per_scene: int | None,
    chunk_seconds: float,
    transcribe_workers: int | None,
    preserve_speaker: bool,
    speaker_count: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    speaker_dominance_threshold: float | None,
    content_safety: bool,
    content_safety_filter: bool,
    content_safety_threshold: float,
    content_safety_model: str | None,
    profanity_sfx: bool,
    profanity_sound_pack_dir: Path | None,
    profanity_words_file: Path | None,
    profanity_pad_ms: int,
    profanity_duck_db: float,
    work_dir: str | None,
    view_preclassification: bool,
) -> None:
    """Transcribe supplied audio and generate a matching AI video track."""

    def _operation() -> None:
        resolved_video_prompt = _resolve_video_prompt_request(
            video_prompt=video_prompt,
            generate_video_prompt=generate_video_prompt,
            option_name="--video-prompt",
        )
        resolved_image_workers = _resolve_worker_count(
            image_workers, env_var="HF_IMAGE_WORKERS", option_name="--image-workers"
        )
        resolved_images_per_scene = _resolve_worker_count(
            images_per_scene,
            env_var="HF_IMAGES_PER_SCENE",
            option_name="--images-per-scene",
        )
        resolved_transcribe_workers = _resolve_worker_count(
            transcribe_workers,
            env_var="HF_TRANSCRIBE_WORKERS",
            option_name="--transcribe-workers",
        )
        (resolved_speaker_count, resolved_min_speakers, resolved_max_speakers) = (
            _resolve_diarization_speaker_options(
                speaker_count=speaker_count,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        )
        resolved_speaker_dominance_threshold = _resolve_speaker_dominance_threshold(
            speaker_dominance_threshold
        )
        pipeline = _build_pipeline(
            work_dir=work_dir,
            debug=bool(ctx.obj.get("debug", False)),
            status_callback=_make_status_callback(
                progress_enabled=bool(ctx.obj.get("progress", True))
            ),
            llm_model=ctx.obj.get("llm_model"),
            stt_model=ctx.obj.get("stt_model"),
            tts_model=ctx.obj.get("tts_model"),
            image_model=ctx.obj.get("image_model"),
        )
        result = pipeline.generate_from_audio(
            audio_path=audio_file,
            video_prompt=resolved_video_prompt,
            generate_video_prompt=generate_video_prompt,
            output_path=output_path,
            cinematic_intro=cinematic_intro,
            cinematic_intro_duration=cinematic_intro_duration,
            image_workers=resolved_image_workers,
            images_per_scene=resolved_images_per_scene,
            chunk_seconds=chunk_seconds,
            transcribe_workers=resolved_transcribe_workers,
            preserve_speaker=preserve_speaker,
            diarization_speaker_count=resolved_speaker_count,
            diarization_min_speakers=resolved_min_speakers,
            diarization_max_speakers=resolved_max_speakers,
            speaker_dominance_threshold=resolved_speaker_dominance_threshold,
            content_safety_enabled=content_safety,
            content_safety_filter=content_safety_filter,
            content_safety_threshold=content_safety_threshold,
            content_safety_model=content_safety_model,
            profanity_sfx_enabled=profanity_sfx,
            profanity_sound_pack_dir=profanity_sound_pack_dir,
            profanity_words_file=profanity_words_file,
            profanity_pad_seconds=(profanity_pad_ms / 1000.0),
            profanity_duck_db=profanity_duck_db,
            view_preclassification=view_preclassification,
        )
        click.echo(f"✅ Video written to {result}")

    _run_with_debug(ctx, _operation)


@cli.command("transcribe")
@click.option(
    "--audio-file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output",
    "output_path",
    required=False,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Optional .txt output file for transcript text.",
)
@click.option(
    "--chunk-seconds",
    default=45.0,
    show_default=True,
    type=float,
    help="Chunk duration for STT requests. Set to 0 to disable chunking.",
)
@click.option(
    "--transcribe-workers",
    default=None,
    show_default=False,
    type=click.IntRange(1, None),
    help="Number of worker threads for chunk transcription (default: HF_TRANSCRIBE_WORKERS or 1).",
)
@click.option(
    "--preserve-speaker/--no-preserve-speaker",
    default=False,
    show_default=True,
    help="Use speaker diarization so transcript text is labeled by speaker.",
)
@click.option(
    "--speaker-count",
    default=None,
    show_default=False,
    type=click.IntRange(1, None),
    help="Force diarization to a fixed number of speakers (mutually exclusive with min/max).",
)
@click.option(
    "--min-speakers",
    default=None,
    show_default=False,
    type=click.IntRange(1, None),
    help="Set minimum speaker count bound for diarization.",
)
@click.option(
    "--max-speakers",
    default=None,
    show_default=False,
    type=click.IntRange(1, None),
    help="Set maximum speaker count bound for diarization.",
)
@click.option(
    "--speaker-dominance-threshold",
    default=None,
    show_default=False,
    type=click.FloatRange(0.0, 1.0),
    help=(
        "Dominance threshold for auto-collapsing sparse secondary speakers "
        "(default: HF_SPEAKER_DOMINANCE_THRESHOLD or 0.9)."
    ),
)
@click.option(
    "--content-safety/--no-content-safety",
    default=False,
    show_default=True,
    help="Label transcript content safety for full audio/chunks using a Hugging Face classifier.",
)
@click.option(
    "--content-safety-filter/--no-content-safety-filter",
    default=False,
    show_default=True,
    help="Drop transcript segments flagged unsafe when content safety is enabled.",
)
@click.option(
    "--content-safety-threshold",
    default=0.7,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Unsafe score threshold used to flag/filter transcript segments.",
)
@click.option(
    "--content-safety-model",
    default=None,
    help="Override Hugging Face moderation model (default: cardiffnlp/twitter-roberta-base-offensive).",
)
@click.option(
    "--profanity-sfx/--no-profanity-sfx",
    default=False,
    show_default=True,
    help="Replace profane words in a rendered audio file using timestamped STT and sound effects.",
)
@click.option(
    "--profanity-sfx-output",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output audio path for profanity-SFX rendering (required when --profanity-sfx is enabled).",
)
@click.option(
    "--profanity-sound-pack-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing sound effect files (wav/mp3/m4a/flac/ogg).",
)
@click.option(
    "--profanity-words-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Optional profanity lexicon file; defaults to bundled data/profanity_words.txt. "
        "Each line can be a single word or multi-word phrase."
    ),
)
@click.option(
    "--profanity-pad-ms",
    default=80,
    show_default=True,
    type=click.IntRange(0, None),
    help="Padding around each profane word when applying replacement SFX.",
)
@click.option(
    "--profanity-duck-db",
    default=-42.0,
    show_default=True,
    type=float,
    help=(
        "Volume reduction (dB) applied to source audio during profanity windows; "
        "defaults to near-muting under the replacement SFX."
    ),
)
@click.option("--work-dir", default=None, help="Directory for intermediate assets.")
@click.pass_context
def transcribe(
    ctx: click.Context,
    audio_file: Path,
    output_path: Path | None,
    chunk_seconds: float,
    transcribe_workers: int | None,
    preserve_speaker: bool,
    speaker_count: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    speaker_dominance_threshold: float | None,
    content_safety: bool,
    content_safety_filter: bool,
    content_safety_threshold: float,
    content_safety_model: str | None,
    profanity_sfx: bool,
    profanity_sfx_output: Path | None,
    profanity_sound_pack_dir: Path | None,
    profanity_words_file: Path | None,
    profanity_pad_ms: int,
    profanity_duck_db: float,
    work_dir: str | None,
) -> None:
    """Generate a transcript for an audio file using the configured AI STT model."""

    def _operation() -> None:
        if profanity_sfx and profanity_sfx_output is None:
            raise click.ClickException(
                "--profanity-sfx-output is required when --profanity-sfx is enabled"
            )
        resolved_transcribe_workers = _resolve_worker_count(
            transcribe_workers,
            env_var="HF_TRANSCRIBE_WORKERS",
            option_name="--transcribe-workers",
        )
        (resolved_speaker_count, resolved_min_speakers, resolved_max_speakers) = (
            _resolve_diarization_speaker_options(
                speaker_count=speaker_count,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        )
        resolved_speaker_dominance_threshold = _resolve_speaker_dominance_threshold(
            speaker_dominance_threshold
        )
        pipeline = _build_pipeline(
            work_dir=work_dir,
            debug=bool(ctx.obj.get("debug", False)),
            status_callback=_make_status_callback(
                progress_enabled=bool(ctx.obj.get("progress", True))
            ),
            llm_model=ctx.obj.get("llm_model"),
            stt_model=ctx.obj.get("stt_model"),
            tts_model=ctx.obj.get("tts_model"),
            image_model=ctx.obj.get("image_model"),
        )
        transcript = pipeline.transcribe_audio_file(
            audio_path=audio_file,
            output_path=output_path,
            chunk_seconds=chunk_seconds,
            transcribe_workers=resolved_transcribe_workers,
            preserve_speaker=preserve_speaker,
            diarization_speaker_count=resolved_speaker_count,
            diarization_min_speakers=resolved_min_speakers,
            diarization_max_speakers=resolved_max_speakers,
            speaker_dominance_threshold=resolved_speaker_dominance_threshold,
            content_safety_enabled=content_safety,
            content_safety_filter=content_safety_filter,
            content_safety_threshold=content_safety_threshold,
            content_safety_model=content_safety_model,
            profanity_sfx_enabled=profanity_sfx,
            profanity_sfx_output_path=profanity_sfx_output,
            profanity_sound_pack_dir=profanity_sound_pack_dir,
            profanity_words_file=profanity_words_file,
            profanity_pad_seconds=(profanity_pad_ms / 1000.0),
            profanity_duck_db=profanity_duck_db,
        )
        if output_path is not None:
            click.echo(f"✅ Transcript written to {output_path}")
        else:
            click.echo(transcript)

    _run_with_debug(ctx, _operation)


@cli.command("doctor")
@click.option("--work-dir", default=None, help="Directory for intermediate assets.")
@click.pass_context
def doctor(ctx: click.Context, work_dir: str | None) -> None:
    """Validate local prerequisites and show active model configuration."""

    def _operation() -> None:
        try:
            config = AppConfig.from_env(
                work_dir=work_dir,
                llm_model=ctx.obj.get("llm_model"),
                stt_model=ctx.obj.get("stt_model"),
                tts_model=ctx.obj.get("tts_model"),
                image_model=ctx.obj.get("image_model"),
            )
            VideoGenerationPipeline(
                config,
                debug=bool(ctx.obj.get("debug", False)),
                status_callback=_make_status_callback(
                    progress_enabled=bool(ctx.obj.get("progress", True))
                ),
            )
        except (RuntimeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo("✅ Environment looks ready.")
        click.echo(f"📁 Work directory: {config.work_dir}")
        click.echo(f"🧠 LLM model: {config.models.llm_model}")
        click.echo(f"🎧 STT model: {config.models.stt_model}")
        click.echo(f"🔊 TTS model: {config.models.tts_model}")
        click.echo(f"🖼️ Image model: {config.models.image_model}")

    _run_with_debug(ctx, _operation)


@cli.command("lexicon-doctor")
@click.option(
    "--profanity-words-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Optional profanity lexicon file; defaults to bundled data/profanity_words.txt. "
        "Each line can be a single word or multi-word phrase."
    ),
)
@click.option(
    "--max-groups",
    default=20,
    show_default=True,
    type=click.IntRange(1, None),
    help="Maximum near-duplicate groups to print.",
)
@click.pass_context
def lexicon_doctor(
    ctx: click.Context, profanity_words_file: Path | None, max_groups: int
) -> None:
    """Inspect profanity lexicon quality for duplicates and normalization collisions."""

    def _operation() -> None:
        report = analyze_profanity_lexicon(profanity_words_file)
        click.echo(f"📄 Lexicon file: {report.path}")
        click.echo(f"🧾 Total lines: {report.total_lines}")
        click.echo(f"🔎 Active entries: {report.active_lines}")
        click.echo(f"🧠 Unique normalized entries: {report.unique_normalized_entries}")

        exact_count = len(report.exact_duplicates)
        near_count = len(report.near_duplicates)
        click.echo(f"♻️ Exact duplicate entries: {exact_count}")
        click.echo(f"🧬 Near-duplicate groups: {near_count}")

        if report.exact_duplicates:
            click.echo("\nExact duplicate entries:")
            for entry, count in report.exact_duplicates.items():
                click.echo(f"- {entry} (x{count})")

        if report.near_duplicates:
            click.echo("\nNear-duplicate groups (same normalized phrase):")
            for index, (normalized, forms) in enumerate(report.near_duplicates.items()):
                if index >= max_groups:
                    remaining = near_count - max_groups
                    if remaining > 0:
                        click.echo(f"- ... and {remaining} more groups")
                    break
                click.echo(f"- {normalized}: {', '.join(forms)}")

        if not report.exact_duplicates and not report.near_duplicates:
            click.echo("✅ No duplicates detected.")

    _run_with_debug(ctx, _operation)


@cli.command("profanity-debug")
@click.option(
    "--audio-file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Source audio file containing the speech to inspect.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path for the generated debug audio file (e.g. debug.m4a).",
)
@click.option(
    "--manifest",
    "manifest_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Path to an existing manifest.json whose profanity_sfx.events are used "
        "instead of re-running STT transcription."
    ),
)
@click.option(
    "--profanity-words-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Optional profanity lexicon file; defaults to bundled data/profanity_words.txt. "
        "Ignored when --manifest is provided."
    ),
)
@click.option(
    "--profanity-sound-pack-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help=(
        "Directory containing sound effect files for bleep generation. "
        "Ignored when --manifest is provided."
    ),
)
@click.option(
    "--profanity-pad-ms",
    default=80,
    show_default=True,
    type=click.IntRange(0, None),
    help="Padding around each profane word in milliseconds. Ignored when --manifest is provided.",
)
@click.option(
    "--context-seconds",
    default=0.5,
    show_default=True,
    type=click.FloatRange(0.0, None),
    help="Seconds of original audio captured before and after each event snippet.",
)
@click.option(
    "--gap-seconds",
    default=0.3,
    show_default=True,
    type=click.FloatRange(0.0, None),
    help="Duration of silence inserted between audio sections within each event.",
)
@click.option("--work-dir", default=None, help="Directory for intermediate assets.")
@click.pass_context
def profanity_debug(
    ctx: click.Context,
    audio_file: Path,
    output_path: Path,
    manifest_path: Path | None,
    profanity_words_file: Path | None,
    profanity_sound_pack_dir: Path | None,
    profanity_pad_ms: int,
    context_seconds: float,
    gap_seconds: float,
    work_dir: str | None,
) -> None:
    """Generate a debug audio file illustrating each detected profanity event.

    For every event the output contains:
    a synthesized voice announcing the detected word with start/end/duration;
    the raw audio snippet from the source; a synthesized voice saying
    "Profanity filter implemented"; and the exact bleep production would overlay.

    Use --manifest to skip re-transcription and load events from an existing
    manifest.json produced by the transcribe command.
    """

    def _operation() -> None:
        import json as _json

        manifest_events: list[dict[str, object]] | None = None
        preclassification_data: dict[str, object] | None = None
        transcript_text: str | None = None
        if manifest_path is not None:
            payload = _json.loads(manifest_path.read_text(encoding="utf-8"))
            sfx_section = payload.get("profanity_sfx", {})
            if not isinstance(sfx_section, dict):
                raise click.ClickException(
                    "Manifest does not contain a 'profanity_sfx' section."
                )
            raw_events = sfx_section.get("events", [])
            if not isinstance(raw_events, list):
                raise click.ClickException(
                    "Manifest 'profanity_sfx.events' is not a list."
                )
            manifest_events = [e for e in raw_events if isinstance(e, dict)]
            raw_preclassification = payload.get("video_prompt_preclassification")
            if isinstance(raw_preclassification, dict):
                preclassification_data = raw_preclassification
            raw_transcript_text = payload.get("narration_text")
            if isinstance(raw_transcript_text, str) and raw_transcript_text.strip():
                transcript_text = raw_transcript_text.strip()
            click.echo(
                f"📋 Loaded {len(manifest_events)} event(s) from manifest: {manifest_path}"
            )

        pipeline = _build_pipeline(
            work_dir=work_dir,
            debug=bool(ctx.obj.get("debug", False)),
            status_callback=_make_status_callback(
                progress_enabled=bool(ctx.obj.get("progress", True))
            ),
            llm_model=ctx.obj.get("llm_model"),
            stt_model=ctx.obj.get("stt_model"),
            tts_model=ctx.obj.get("tts_model"),
            image_model=ctx.obj.get("image_model"),
        )
        event_count = pipeline.build_profanity_debug_audio(
            audio_path=audio_file,
            output_path=output_path,
            manifest_events=manifest_events,
            preclassification_data=preclassification_data,
            transcript_text=transcript_text,
            sound_pack_dir=profanity_sound_pack_dir,
            profanity_words_file=profanity_words_file,
            pad_seconds=profanity_pad_ms / 1000.0,
            context_seconds=context_seconds,
            gap_seconds=gap_seconds,
        )
        if event_count == 0:
            click.echo("ℹ️ No profanity events found — no debug audio generated.")
        else:
            click.echo(
                f"✅ Debug audio with {event_count} event(s) written to: {output_path}"
            )

    _run_with_debug(ctx, _operation)


@cli.command("calibrate")
@click.option(
    "--manifest-old",
    "manifest_old_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the old manifest.json (baseline/LLM-only preclassification).",
)
@click.option(
    "--manifest-new",
    "manifest_new_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the new manifest.json (ensemble-enhanced preclassification).",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Optional path to write calibration report JSON; prints to stdout if omitted.",
)
@click.pass_context
def calibrate(
    ctx: click.Context,
    manifest_old_path: Path,
    manifest_new_path: Path,
    output_path: Path | None,
) -> None:
    """Compare old (LLM-only) vs new (ensemble) preclassification for calibration."""
    import json

    def _operation() -> None:
        try:
            old_manifest = json.loads(manifest_old_path.read_text(encoding="utf-8"))
            new_manifest = json.loads(manifest_new_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise click.ClickException(f"Failed to load manifest: {exc}") from exc

        old_preclass = old_manifest.get("video_prompt_preclassification", {})
        new_preclass = new_manifest.get("video_prompt_preclassification", {})

        # Extract key metrics
        old_mood = old_preclass.get("mood", "N/A")
        new_mood = new_preclass.get("mood", "N/A")

        old_truthfulness = old_preclass.get("truthfulness_assessment", {})
        new_truthfulness = new_preclass.get("truthfulness_assessment", {})

        old_ensemble = old_preclass.get("ensemble_scorecard", {})
        new_ensemble = new_preclass.get("ensemble_scorecard", {})

        old_risk_score = old_ensemble.get("weighted_risk_score", "N/A")
        new_risk_score = new_ensemble.get("weighted_risk_score", "N/A")

        old_risk_level = old_ensemble.get("risk_level", "N/A")
        new_risk_level = new_ensemble.get("risk_level", "N/A")

        old_intensity = old_ensemble.get("recommended_visual_intensity", "N/A")
        new_intensity = new_ensemble.get("recommended_visual_intensity", "N/A")

        # Build comparison report
        report = {
            "calibration_report": True,
            "old_manifest": str(manifest_old_path),
            "new_manifest": str(manifest_new_path),
            "comparison": {
                "mood": {
                    "old": old_mood,
                    "new": new_mood,
                    "changed": old_mood != new_mood,
                },
                "truthfulness": {
                    "old": old_truthfulness.get("label", "N/A"),
                    "new": new_truthfulness.get("label", "N/A"),
                    "old_confidence": old_truthfulness.get("confidence_score", "N/A"),
                    "new_confidence": new_truthfulness.get("confidence_score", "N/A"),
                },
                "ensemble_scoring": {
                    "old_risk_score": old_risk_score,
                    "new_risk_score": new_risk_score,
                    "risk_score_delta": (
                        round(float(new_risk_score) - float(old_risk_score), 3)
                        if isinstance(new_risk_score, (int, float))
                        and isinstance(old_risk_score, (int, float))
                        else "N/A"
                    ),
                    "old_risk_level": old_risk_level,
                    "new_risk_level": new_risk_level,
                    "risk_level_changed": old_risk_level != new_risk_level,
                    "old_visual_intensity": old_intensity,
                    "new_visual_intensity": new_intensity,
                    "visual_intensity_changed": old_intensity != new_intensity,
                },
                "signal_count": {
                    "old_signals": len(old_ensemble.get("signals", [])),
                    "new_signals": len(new_ensemble.get("signals", [])),
                },
            },
        }

        # Add warnings if present
        if old_ensemble.get("warnings"):
            report["old_warnings"] = old_ensemble.get("warnings")
        if new_ensemble.get("warnings"):
            report["new_warnings"] = new_ensemble.get("warnings")

        # Output report
        report_json = json.dumps(report, indent=2)
        if output_path:
            output_path.write_text(report_json, encoding="utf-8")
            click.echo(f"✅ Calibration report written to: {output_path}")
        else:
            click.echo(report_json)

        # Print summary
        click.echo("\n📊 Calibration Summary:")
        click.echo(f"  Mood: {old_mood} → {new_mood}")
        if isinstance(new_risk_score, (int, float)) and isinstance(
            old_risk_score, (int, float)
        ):
            delta = float(new_risk_score) - float(old_risk_score)
            direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            click.echo(
                f"  Risk Score: {old_risk_score:.2f} {direction} {new_risk_score:.2f} (Δ={delta:+.3f})"
            )
        click.echo(f"  Risk Level: {old_risk_level} → {new_risk_level}")
        click.echo(f"  Visual Intensity: {old_intensity} → {new_intensity}")
        click.echo(
            f"  Signals: {report['comparison']['signal_count']['old_signals']} → {report['comparison']['signal_count']['new_signals']}"
        )

    _run_with_debug(ctx, _operation)
