from __future__ import annotations

import json
import shutil
import textwrap
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable
from uuid import uuid4

from video_generator.config import AppConfig
from video_generator.hf_client import HuggingFaceGateway
from video_generator.media import MediaAssembler
from video_generator.planner import ScenePlanner, VideoPromptPlan


def wrap_transcription(text: str, *, width: int = 100) -> str:
    if width <= 0:
        return text

    wrapped_paragraphs: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            wrapped_paragraphs.append("")
            continue
        wrapped_paragraphs.append(
            textwrap.fill(
                paragraph, width=width, break_long_words=False, break_on_hyphens=False
            )
        )
    return "\n".join(wrapped_paragraphs)


class VideoGenerationPipeline:
    def __init__(
        self,
        config: AppConfig,
        *,
        debug: bool = False,
        status_callback: Callable[[str], None] | None = None,
    ):
        self._config = config
        self._debug = debug
        self._status_callback = status_callback
        self._gateway = HuggingFaceGateway(config)
        self._planner = ScenePlanner(self._gateway)
        self._media = MediaAssembler(
            width=config.width, height=config.height, fps=config.fps
        )

    def generate_from_text(
        self,
        *,
        narration_text: str,
        video_prompt: str | None,
        output_path: Path,
        generate_video_prompt: bool = False,
    ) -> Path:
        self._ensure_video_dependencies()
        run_dir = self._prepare_run_dir(output_path)
        manifest: dict[str, object] = {
            "pipeline": "from-text",
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "output": str(output_path),
            "run_dir": str(run_dir),
            "narration_text": narration_text,
            "video_prompt": video_prompt,
            "generate_video_prompt": generate_video_prompt,
        }
        self._write_manifest(run_dir, manifest)
        audio_path = run_dir / "narration.wav"
        self._status("🎙️ Synthesizing narration audio")
        self._gateway.synthesize_speech(narration_text, audio_path)
        manifest["audio"] = str(audio_path)
        manifest["status"] = "narration_synthesized"
        self._write_manifest(run_dir, manifest)
        self._status("⏱️ Measuring audio duration")
        duration = self._media.get_audio_duration(audio_path)
        manifest["duration_seconds"] = duration
        manifest["status"] = "duration_measured"
        self._write_manifest(run_dir, manifest)
        return self._render_project(
            narration_text=narration_text,
            video_prompt=video_prompt,
            generate_video_prompt=generate_video_prompt,
            audio_path=audio_path,
            duration_seconds=duration,
            output_path=output_path,
            run_dir=run_dir,
            manifest=manifest,
        )

    def generate_from_audio(
        self,
        *,
        audio_path: Path,
        video_prompt: str | None,
        output_path: Path,
        chunk_seconds: float = 45.0,
        generate_video_prompt: bool = False,
        preserve_speaker: bool = False,
    ) -> Path:
        self._ensure_video_dependencies()
        run_dir = self._prepare_run_dir(output_path)
        manifest: dict[str, object] = {
            "pipeline": "from-audio",
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "output": str(output_path),
            "run_dir": str(run_dir),
            "audio": str(audio_path),
            "chunk_seconds": chunk_seconds,
            "preserve_speaker": preserve_speaker,
            "generate_video_prompt": generate_video_prompt,
            "video_prompt": video_prompt,
        }
        self._write_manifest(run_dir, manifest)
        transcript = self._transcribe_with_optional_chunking(
            audio_path=audio_path,
            chunk_seconds=chunk_seconds,
            chunk_dir_root=run_dir / "stt_chunks",
            preserve_speaker=preserve_speaker,
        )
        manifest["narration_text"] = transcript
        manifest["status"] = "transcribed"
        self._write_manifest(run_dir, manifest)
        self._status("⏱️ Measuring audio duration")
        duration = self._media.get_audio_duration(audio_path)
        manifest["duration_seconds"] = duration
        manifest["status"] = "duration_measured"
        self._write_manifest(run_dir, manifest)
        return self._render_project(
            narration_text=transcript,
            video_prompt=video_prompt,
            generate_video_prompt=generate_video_prompt,
            audio_path=audio_path,
            duration_seconds=duration,
            output_path=output_path,
            run_dir=run_dir,
            manifest=manifest,
        )

    def transcribe_audio_file(
        self,
        *,
        audio_path: Path,
        output_path: Path | None = None,
        chunk_seconds: float = 45.0,
        preserve_speaker: bool = False,
    ) -> str:
        transcript = self._transcribe_with_optional_chunking(
            audio_path=audio_path,
            chunk_seconds=chunk_seconds,
            chunk_dir_root=self._config.work_dir / "transcribe_chunks",
            preserve_speaker=preserve_speaker,
        )
        if output_path is not None:
            self._status("💾 Writing transcript to disk")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(wrap_transcription(transcript), encoding="utf-8")
        self._status("✅ Transcription complete")
        return transcript

    def _render_project(
        self,
        *,
        narration_text: str,
        video_prompt: str | None,
        generate_video_prompt: bool,
        audio_path: Path,
        duration_seconds: float,
        output_path: Path,
        run_dir: Path,
        manifest: dict[str, object] | None = None,
    ) -> Path:
        if manifest is None:
            manifest = {
                "pipeline": "render",
                "status": "started",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "output": str(output_path),
                "audio": str(audio_path),
                "duration_seconds": duration_seconds,
                "narration_text": narration_text,
                "video_prompt": video_prompt,
            }
        manifest["status"] = "resolving_video_prompt"
        self._write_manifest(run_dir, manifest)
        video_prompt_plan = self._resolve_video_prompt_plan(
            narration_text=narration_text,
            video_prompt=video_prompt,
            generate_video_prompt=generate_video_prompt,
        )
        resolved_video_prompt = video_prompt_plan.video_prompt
        manifest["video_prompt"] = resolved_video_prompt
        manifest["video_prompt_preclassification"] = (
            {
                "mood": video_prompt_plan.preclassification.mood,
                "has_foul_language": video_prompt_plan.preclassification.has_foul_language,
                "word_count": video_prompt_plan.preclassification.word_count,
                "sentence_count": video_prompt_plan.preclassification.sentence_count,
            }
            if video_prompt_plan.preclassification is not None
            else None
        )
        manifest["status"] = "planning_scenes"
        self._write_manifest(run_dir, manifest)
        self._status("🧠 Planning scenes from narration")
        scenes = self._planner.build_scenes(
            narration_text=narration_text,
            video_prompt=resolved_video_prompt,
            total_duration_seconds=duration_seconds,
        )
        manifest["scenes"] = [asdict(scene) for scene in scenes]
        manifest["status"] = "scenes_planned"
        self._write_manifest(run_dir, manifest)
        self._status(f"🧩 Planned {len(scenes)} scenes")
        images_dir = run_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        image_paths = []
        manifest["status"] = "generating_images"
        manifest["images"] = []
        self._write_manifest(run_dir, manifest)
        self._status("🖼️ Generating images for scenes")
        for scene in scenes:
            if self._debug:
                self._status(
                    f"🐛 Rendering image for scene {scene.index}/{len(scenes)}"
                )
            image_path = images_dir / f"scene_{scene.index:02d}.png"
            self._gateway.generate_image(scene.prompt, image_path)
            image_paths.append(image_path)
            images = manifest.get("images")
            if isinstance(images, list):
                images.append(str(image_path))
                self._write_manifest(run_dir, manifest)

        manifest["status"] = "assembling_video"
        self._write_manifest(run_dir, manifest)
        self._status("🎬 Assembling video with ffmpeg")
        final_path = self._media.render_video(
            images=image_paths,
            scenes=scenes,
            audio_path=audio_path,
            output_path=output_path,
            work_dir=run_dir,
        )
        manifest["output"] = str(final_path)
        manifest["audio"] = str(audio_path)
        manifest["duration_seconds"] = duration_seconds
        manifest["narration_text"] = narration_text
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._write_manifest(run_dir, manifest)
        self._status("✅ Video generation complete")
        return final_path

    def _resolve_video_prompt_plan(
        self,
        *,
        narration_text: str,
        video_prompt: str | None,
        generate_video_prompt: bool,
    ) -> VideoPromptPlan:
        if video_prompt:
            return VideoPromptPlan(video_prompt=video_prompt, preclassification=None)
        if generate_video_prompt:
            self._status("🧪 Preclassifying transcript for visual planning")
            self._status("🪄 Generating video prompt from narration")
            return self._planner.generate_video_prompt_plan(
                narration_text=narration_text
            )
        raise ValueError(
            "video_prompt is required unless generate_video_prompt is enabled"
        )

    def _ensure_video_dependencies(self) -> None:
        missing = [
            binary for binary in ("ffmpeg", "ffprobe") if shutil.which(binary) is None
        ]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"Missing required system dependencies: {joined}")

    def _prepare_run_dir(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        run_dir = self._config.work_dir / output_path.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        if self._debug:
            self._status(f"🐛 Using run directory: {run_dir}")
        return run_dir

    def _transcribe_with_optional_chunking(
        self,
        *,
        audio_path: Path,
        chunk_seconds: float,
        chunk_dir_root: Path,
        preserve_speaker: bool,
    ) -> str:
        if chunk_seconds <= 0:
            if preserve_speaker:
                self._status(
                    "🧩 Transcribing audio with speaker diarization (full file)"
                )
                return self._gateway.transcribe_audio_with_speakers(audio_path)
            self._status("📝 Transcribing audio with speech-to-text model")
            return self._gateway.transcribe_audio(audio_path)

        if preserve_speaker:
            if shutil.which("ffmpeg") is None:
                self._status(
                    "⚠️ ffmpeg not found; falling back to full-file diarization"
                )
                self._status(
                    "🧩 Transcribing audio with speaker diarization (full file)"
                )
                return self._gateway.transcribe_audio_with_speakers(audio_path)
            chunk_dir = chunk_dir_root / f"{audio_path.stem}_{uuid4().hex[:8]}"
            self._status(f"✂️ Chunking audio into ~{int(chunk_seconds)}s segments")
            chunks = self._media.chunk_audio(
                audio_path=audio_path, output_dir=chunk_dir, chunk_seconds=chunk_seconds
            )
            self._status(f"🧩 Processing {len(chunks)} chunks with speaker diarization")
            chunk_texts: list[str] = []
            total_start = perf_counter()
            for chunk_idx, chunk_path in enumerate(chunks, start=1):
                self._status(
                    f"  Chunk {chunk_idx}/{len(chunks)}: diarizing and transcribing"
                )
                chunk_start = perf_counter()
                text = self._gateway.transcribe_audio_with_speakers(chunk_path)
                elapsed = perf_counter() - chunk_start
                self._status(
                    f"  Chunk {chunk_idx}/{len(chunks)} complete in {elapsed:.1f}s"
                )
                chunk_texts.append(text)
            total_elapsed = perf_counter() - total_start
            self._status(f"⏱️ Chunk processing completed in {total_elapsed:.1f}s")
            shutil.rmtree(chunk_dir, ignore_errors=True)
            return "\n".join(chunk_texts)

        if shutil.which("ffmpeg") is None:
            self._status(
                "⚠️ ffmpeg not found; falling back to single-pass transcription"
            )
            self._status("📝 Transcribing audio with speech-to-text model")
            return self._gateway.transcribe_audio(audio_path)

        chunk_dir = chunk_dir_root / f"{audio_path.stem}_{uuid4().hex[:8]}"
        self._status(f"✂️ Chunking audio into ~{int(chunk_seconds)}s segments")
        chunks = self._media.chunk_audio(
            audio_path=audio_path, output_dir=chunk_dir, chunk_seconds=chunk_seconds
        )
        self._status(f"📝 Transcribing {len(chunks)} audio chunks")
        chunk_texts: list[str] = []
        total_start = perf_counter()
        for index, chunk in enumerate(chunks, start=1):
            if self._debug:
                self._status(
                    f"🐛 Transcribing chunk {index}/{len(chunks)}: {chunk.name}"
                )
            chunk_start = perf_counter()
            text = self._gateway.transcribe_audio(chunk).strip()
            elapsed = perf_counter() - chunk_start
            self._status(f"  Chunk {index}/{len(chunks)} complete in {elapsed:.1f}s")
            if text:
                chunk_texts.append(text)
        total_elapsed = perf_counter() - total_start
        self._status(f"⏱️ Chunk processing completed in {total_elapsed:.1f}s")
        return " ".join(chunk_texts).strip()

    def _write_manifest(self, run_dir: Path, manifest: dict[str, object]) -> None:
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def _status(self, message: str) -> None:
        if self._status_callback is not None:
            self._status_callback(message)


# a 45 second clip takes about 1.5 minutes to transcribe with speaker diarization, so chunk into ~45s segments by default to balance speed and context retention.
# 8:07:28 (start)
# 8:08:41 (end)
