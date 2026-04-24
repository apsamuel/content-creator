from __future__ import annotations

from pathlib import Path
from typing import Callable
from urllib.parse import unquote

import click

from content_creator.config import AppConfig
from content_creator.pipeline import VideoGenerationPipeline


def _print_startup_check(config: AppConfig) -> None:
    click.echo("🔎 Startup check")
    click.echo(f"📁 Work directory: {config.work_dir}")
    click.echo(f"🧠 LLM model: {config.models.llm_model}")
    click.echo(f"🎧 STT model: {config.models.stt_model}")
    click.echo(f"🔊 TTS model: {config.models.tts_model}")
    click.echo(f"🖼️ Image model: {config.models.image_model}")


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
@click.pass_context
def cli(
    ctx: click.Context,
    debug: bool,
    llm_model: str | None,
    stt_model: str | None,
    tts_model: str | None,
    image_model: str | None,
) -> None:
    """CLI-first AI video generation built on Hugging Face inference APIs."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["llm_model"] = llm_model
    ctx.obj["stt_model"] = stt_model
    ctx.obj["tts_model"] = tts_model
    ctx.obj["image_model"] = image_model
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
@click.option("--work-dir", default=None, help="Directory for intermediate assets.")
@click.pass_context
def from_text(
    ctx: click.Context,
    text_transcription: str,
    video_prompt: str | None,
    generate_video_prompt: bool,
    output_path: Path,
    work_dir: str | None,
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
        pipeline = _build_pipeline(
            work_dir=work_dir,
            debug=bool(ctx.obj.get("debug", False)),
            status_callback=_status,
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
    "--chunk-seconds",
    default=45.0,
    show_default=True,
    type=float,
    help="Chunk duration for STT requests. Set to 0 to disable chunking.",
)
@click.option(
    "--preserve-speaker/--no-preserve-speaker",
    default=False,
    show_default=True,
    help="Use speaker diarization so transcript text is labeled by speaker.",
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
    help="Override Hugging Face moderation model (default: unitary/unbiased-toxic-roberta).",
)
@click.option("--work-dir", default=None, help="Directory for intermediate assets.")
@click.pass_context
def from_audio(
    ctx: click.Context,
    audio_file: Path,
    video_prompt: str | None,
    generate_video_prompt: bool,
    output_path: Path,
    chunk_seconds: float,
    preserve_speaker: bool,
    content_safety: bool,
    content_safety_filter: bool,
    content_safety_threshold: float,
    content_safety_model: str | None,
    work_dir: str | None,
) -> None:
    """Transcribe supplied audio and generate a matching AI video track."""

    def _operation() -> None:
        resolved_video_prompt = _resolve_video_prompt_request(
            video_prompt=video_prompt,
            generate_video_prompt=generate_video_prompt,
            option_name="--video-prompt",
        )
        pipeline = _build_pipeline(
            work_dir=work_dir,
            debug=bool(ctx.obj.get("debug", False)),
            status_callback=_status,
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
            chunk_seconds=chunk_seconds,
            preserve_speaker=preserve_speaker,
            content_safety_enabled=content_safety,
            content_safety_filter=content_safety_filter,
            content_safety_threshold=content_safety_threshold,
            content_safety_model=content_safety_model,
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
    "--preserve-speaker/--no-preserve-speaker",
    default=False,
    show_default=True,
    help="Use speaker diarization so transcript text is labeled by speaker.",
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
    help="Override Hugging Face moderation model (default: unitary/unbiased-toxic-roberta).",
)
@click.option("--work-dir", default=None, help="Directory for intermediate assets.")
@click.pass_context
def transcribe(
    ctx: click.Context,
    audio_file: Path,
    output_path: Path | None,
    chunk_seconds: float,
    preserve_speaker: bool,
    content_safety: bool,
    content_safety_filter: bool,
    content_safety_threshold: float,
    content_safety_model: str | None,
    work_dir: str | None,
) -> None:
    """Generate a transcript for an audio file using the configured AI STT model."""

    def _operation() -> None:
        pipeline = _build_pipeline(
            work_dir=work_dir,
            debug=bool(ctx.obj.get("debug", False)),
            status_callback=_status,
            llm_model=ctx.obj.get("llm_model"),
            stt_model=ctx.obj.get("stt_model"),
            tts_model=ctx.obj.get("tts_model"),
            image_model=ctx.obj.get("image_model"),
        )
        transcript = pipeline.transcribe_audio_file(
            audio_path=audio_file,
            output_path=output_path,
            chunk_seconds=chunk_seconds,
            preserve_speaker=preserve_speaker,
            content_safety_enabled=content_safety,
            content_safety_filter=content_safety_filter,
            content_safety_threshold=content_safety_threshold,
            content_safety_model=content_safety_model,
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
                config, debug=bool(ctx.obj.get("debug", False)), status_callback=_status
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
