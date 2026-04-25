from __future__ import annotations

import json
import shutil
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable
from uuid import uuid4

from content_creator.config import AppConfig
from content_creator.hf_client import HuggingFaceGateway
from content_creator.media import MediaAssembler
from content_creator.planner import ScenePlanner, VideoPromptPlan


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
        self._last_content_safety_report: dict[str, object] | None = None

    def generate_from_text(
        self,
        *,
        narration_text: str,
        video_prompt: str | None,
        output_path: Path,
        generate_video_prompt: bool = False,
        image_workers: int = 1,
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
            image_workers=image_workers,
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
        content_safety_enabled: bool = False,
        content_safety_filter: bool = False,
        content_safety_threshold: float = 0.7,
        content_safety_model: str | None = None,
        transcribe_workers: int = 1,
        image_workers: int = 1,
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
            "content_safety_enabled": content_safety_enabled,
            "content_safety_filter": content_safety_filter,
            "content_safety_threshold": content_safety_threshold,
            "content_safety_model": content_safety_model,
            "transcribe_workers": transcribe_workers,
            "image_workers": image_workers,
            "generate_video_prompt": generate_video_prompt,
            "video_prompt": video_prompt,
        }
        self._write_manifest(run_dir, manifest)
        transcript = self._transcribe_with_optional_chunking(
            audio_path=audio_path,
            chunk_seconds=chunk_seconds,
            chunk_dir_root=run_dir / "stt_chunks",
            preserve_speaker=preserve_speaker,
            content_safety_enabled=content_safety_enabled,
            content_safety_filter=content_safety_filter,
            content_safety_threshold=content_safety_threshold,
            content_safety_model=content_safety_model,
            transcribe_workers=transcribe_workers,
        )
        if not transcript.strip() and content_safety_enabled and content_safety_filter:
            raise ValueError(
                "Transcription produced no allowed content after content safety filtering"
            )
        manifest["narration_text"] = transcript
        if self._last_content_safety_report is not None:
            manifest["content_safety"] = self._last_content_safety_report
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
            image_workers=image_workers,
        )

    def transcribe_audio_file(
        self,
        *,
        audio_path: Path,
        output_path: Path | None = None,
        chunk_seconds: float = 45.0,
        preserve_speaker: bool = False,
        content_safety_enabled: bool = False,
        content_safety_filter: bool = False,
        content_safety_threshold: float = 0.7,
        content_safety_model: str | None = None,
        transcribe_workers: int = 1,
    ) -> str:
        transcript = self._transcribe_with_optional_chunking(
            audio_path=audio_path,
            chunk_seconds=chunk_seconds,
            chunk_dir_root=self._config.work_dir / "transcribe_chunks",
            preserve_speaker=preserve_speaker,
            content_safety_enabled=content_safety_enabled,
            content_safety_filter=content_safety_filter,
            content_safety_threshold=content_safety_threshold,
            content_safety_model=content_safety_model,
            transcribe_workers=transcribe_workers,
        )
        self._emit_content_safety_summary()
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
        image_workers: int = 1,
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
        total_scenes = len(scenes)
        worker_count = max(1, image_workers)

        def _render_scene_image(
            scene_index: int, scene_prompt: str
        ) -> tuple[int, Path, float]:
            start = perf_counter()
            destination = images_dir / f"scene_{scene_index:02d}.png"
            self._gateway.generate_image(scene_prompt, destination)
            return scene_index, destination, perf_counter() - start

        rendered_paths: dict[int, Path] = {}
        completed = 0

        if worker_count == 1:
            for scene in scenes:
                if self._debug:
                    self._status(
                        f"🐛 Rendering image for scene {scene.index}/{len(scenes)}"
                    )
                scene_index, image_path, elapsed = _render_scene_image(
                    scene.index, scene.prompt
                )
                rendered_paths[scene_index] = image_path
                completed += 1
                self._emit_progress(
                    "📷 Image generation progress",
                    current=completed,
                    total=total_scenes,
                    elapsed_seconds=elapsed,
                )
        else:
            self._status(f"🧵 Using {worker_count} workers for image generation")
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        _render_scene_image, scene.index, scene.prompt
                    ): scene
                    for scene in scenes
                }
                for future in as_completed(futures):
                    scene_index, image_path, elapsed = future.result()
                    rendered_paths[scene_index] = image_path
                    completed += 1
                    self._emit_progress(
                        "📷 Image generation progress",
                        current=completed,
                        total=total_scenes,
                        elapsed_seconds=elapsed,
                    )

        image_paths = [rendered_paths[index] for index in sorted(rendered_paths)]
        images = manifest.get("images")
        if isinstance(images, list):
            images.clear()
            images.extend(str(path) for path in image_paths)
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
        content_safety_enabled: bool,
        content_safety_filter: bool,
        content_safety_threshold: float,
        content_safety_model: str | None,
        transcribe_workers: int,
    ) -> str:
        self._last_content_safety_report = None
        report: dict[str, object] | None = None
        if content_safety_enabled:
            report = {
                "enabled": True,
                "filter_enabled": content_safety_filter,
                "threshold": content_safety_threshold,
                "model": (
                    content_safety_model or "unitary/unbiased-toxic-roberta"
                ).strip(),
                "chunks": [],
                "dropped_chunks": 0,
                "kept_chunks": 0,
            }

        if chunk_seconds <= 0:
            if preserve_speaker:
                self._status(
                    "🧩 Transcribing audio with speaker diarization (full file)"
                )
                transcript = self._gateway.transcribe_audio_with_speakers(audio_path)
            else:
                self._status("📝 Transcribing audio with speech-to-text model")
                transcript = self._gateway.transcribe_audio(audio_path)

            if content_safety_enabled:
                if report is not None:
                    report["full_audio"] = self._evaluate_content_safety(
                        text=transcript,
                        segment_name=audio_path.name,
                        content_safety_threshold=content_safety_threshold,
                        content_safety_model=content_safety_model,
                    )
                self._last_content_safety_report = report
                full_audio = report.get("full_audio") if report is not None else None
                if (
                    content_safety_filter
                    and isinstance(full_audio, dict)
                    and bool(full_audio.get("flagged", False))
                ):
                    self._status(
                        "🚫 Filtered full transcript due to content safety policy"
                    )
                    return ""
            return transcript

        if preserve_speaker:
            if shutil.which("ffmpeg") is None:
                self._status(
                    "⚠️ ffmpeg not found; falling back to full-file diarization"
                )
                self._status(
                    "🧩 Transcribing audio with speaker diarization (full file)"
                )
                transcript = self._gateway.transcribe_audio_with_speakers(audio_path)
                if content_safety_enabled:
                    if report is not None:
                        report["full_audio"] = self._evaluate_content_safety(
                            text=transcript,
                            segment_name=audio_path.name,
                            content_safety_threshold=content_safety_threshold,
                            content_safety_model=content_safety_model,
                        )
                    self._last_content_safety_report = report
                    full_audio = (
                        report.get("full_audio") if report is not None else None
                    )
                    if (
                        content_safety_filter
                        and isinstance(full_audio, dict)
                        and bool(full_audio.get("flagged", False))
                    ):
                        self._status(
                            "🚫 Filtered full transcript due to content safety policy"
                        )
                        return ""
                return transcript
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
                self._emit_progress(
                    "Diarization chunk progress",
                    current=chunk_idx,
                    total=len(chunks),
                    elapsed_seconds=elapsed,
                )
                if content_safety_enabled:
                    evaluation = self._evaluate_content_safety(
                        text=text,
                        segment_name=chunk_path.name,
                        content_safety_threshold=content_safety_threshold,
                        content_safety_model=content_safety_model,
                    )
                    evaluation["chunk_index"] = chunk_idx
                    if report is not None:
                        chunks_report = report.get("chunks")
                        if isinstance(chunks_report, list):
                            chunks_report.append(evaluation)
                    if content_safety_filter and bool(evaluation.get("flagged", False)):
                        if report is not None:
                            report["dropped_chunks"] = (
                                self._as_int(report.get("dropped_chunks")) + 1
                            )
                        self._status(
                            f"🚫 Filtered chunk {chunk_idx}/{len(chunks)} ({chunk_path.name})"
                        )
                        continue

                if report is not None:
                    report["kept_chunks"] = self._as_int(report.get("kept_chunks")) + 1
                chunk_texts.append(text)
            total_elapsed = perf_counter() - total_start
            self._status(f"⏱️ Chunk processing completed in {total_elapsed:.1f}s")
            shutil.rmtree(chunk_dir, ignore_errors=True)
            self._last_content_safety_report = report
            return "\n".join(chunk_texts)

        if shutil.which("ffmpeg") is None:
            self._status(
                "⚠️ ffmpeg not found; falling back to single-pass transcription"
            )
            self._status("📝 Transcribing audio with speech-to-text model")
            transcript = self._gateway.transcribe_audio(audio_path)
            if content_safety_enabled:
                if report is not None:
                    report["full_audio"] = self._evaluate_content_safety(
                        text=transcript,
                        segment_name=audio_path.name,
                        content_safety_threshold=content_safety_threshold,
                        content_safety_model=content_safety_model,
                    )
                self._last_content_safety_report = report
                full_audio = report.get("full_audio") if report is not None else None
                if (
                    content_safety_filter
                    and isinstance(full_audio, dict)
                    and bool(full_audio.get("flagged", False))
                ):
                    self._status(
                        "🚫 Filtered full transcript due to content safety policy"
                    )
                    return ""
            return transcript

        chunk_dir = chunk_dir_root / f"{audio_path.stem}_{uuid4().hex[:8]}"
        self._status(f"✂️ Chunking audio into ~{int(chunk_seconds)}s segments")
        chunks = self._media.chunk_audio(
            audio_path=audio_path, output_dir=chunk_dir, chunk_seconds=chunk_seconds
        )
        self._status(f"📝 Transcribing {len(chunks)} audio chunks")
        chunk_texts: list[str] = []
        total_start = perf_counter()
        worker_count = max(1, transcribe_workers)

        def _transcribe_chunk(
            chunk_index: int, chunk_path: Path
        ) -> tuple[int, str, float]:
            chunk_start = perf_counter()
            chunk_text = self._gateway.transcribe_audio(chunk_path).strip()
            elapsed_seconds = perf_counter() - chunk_start
            return chunk_index, chunk_text, elapsed_seconds

        chunk_results: dict[int, str] = {}
        completed = 0

        if worker_count == 1:
            for index, chunk in enumerate(chunks, start=1):
                if self._debug:
                    self._status(
                        f"🐛 Transcribing chunk {index}/{len(chunks)}: {chunk.name}"
                    )
                _, text, elapsed = _transcribe_chunk(index, chunk)
                completed += 1
                self._emit_progress(
                    "Transcription chunk progress",
                    current=completed,
                    total=len(chunks),
                    elapsed_seconds=elapsed,
                )
                if text:
                    chunk_results[index] = text
        else:
            self._status(f"🧵 Using {worker_count} workers for chunk transcription")
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(_transcribe_chunk, index, chunk): chunk
                    for index, chunk in enumerate(chunks, start=1)
                }
                for future in as_completed(futures):
                    index, text, elapsed = future.result()
                    completed += 1
                    self._emit_progress(
                        "Transcription chunk progress",
                        current=completed,
                        total=len(chunks),
                        elapsed_seconds=elapsed,
                    )
                    if text:
                        chunk_results[index] = text

        for index, chunk in enumerate(chunks, start=1):
            text = chunk_results.get(index, "")
            if not text:
                continue

            if content_safety_enabled:
                evaluation = self._evaluate_content_safety(
                    text=text,
                    segment_name=chunk.name,
                    content_safety_threshold=content_safety_threshold,
                    content_safety_model=content_safety_model,
                )
                evaluation["chunk_index"] = index
                if report is not None:
                    chunks_report = report.get("chunks")
                    if isinstance(chunks_report, list):
                        chunks_report.append(evaluation)

                if content_safety_filter and bool(evaluation.get("flagged", False)):
                    if report is not None:
                        report["dropped_chunks"] = (
                            self._as_int(report.get("dropped_chunks")) + 1
                        )
                    self._status(
                        f"🚫 Filtered chunk {index}/{len(chunks)} ({chunk.name})"
                    )
                    continue

            if report is not None:
                report["kept_chunks"] = self._as_int(report.get("kept_chunks")) + 1
            chunk_texts.append(text)
        total_elapsed = perf_counter() - total_start
        self._status(f"⏱️ Chunk processing completed in {total_elapsed:.1f}s")
        self._last_content_safety_report = report
        return " ".join(chunk_texts).strip()

    def _evaluate_content_safety(
        self,
        *,
        text: str,
        segment_name: str,
        content_safety_threshold: float,
        content_safety_model: str | None,
    ) -> dict[str, object]:
        if not text.strip():
            return {
                "segment": segment_name,
                "text_length": 0,
                "flagged": False,
                "unsafe_score": 0.0,
                "top_label": "",
                "top_score": 0.0,
                "labels": [],
            }

        moderation = self._gateway.classify_content_safety(
            text, model=content_safety_model
        )
        unsafe_score = float(moderation.get("unsafe_score", 0.0) or 0.0)
        return {
            "segment": segment_name,
            "text_length": len(text),
            "flagged": unsafe_score >= content_safety_threshold,
            "unsafe_score": unsafe_score,
            "top_label": str(moderation.get("top_label", "")),
            "top_score": float(moderation.get("top_score", 0.0) or 0.0),
            "labels": moderation.get("labels", []),
        }

    def _emit_content_safety_summary(self) -> None:
        if self._last_content_safety_report is None:
            return

        dropped = self._as_int(self._last_content_safety_report.get("dropped_chunks"))
        kept = self._as_int(self._last_content_safety_report.get("kept_chunks"))
        threshold = self._last_content_safety_report.get("threshold")
        model = self._last_content_safety_report.get("model")
        self._status(
            "🛡️ Content safety summary: "
            f"model={model}, threshold={threshold}, kept={kept}, dropped={dropped}"
        )

    @staticmethod
    def _as_int(value: object, *, default: int = 0) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return default
        return default

    def _emit_progress(
        self,
        label: str,
        *,
        current: int,
        total: int,
        elapsed_seconds: float | None = None,
    ) -> None:
        if total <= 0:
            return
        bounded_current = max(0, min(current, total))
        remaining = total - bounded_current
        ratio = bounded_current / total
        width = 24
        filled = int(ratio * width)
        bar = "#" * filled + "-" * (width - filled)
        percent = int(ratio * 100)
        details = f"chunk {bounded_current}/{total}, {remaining} remaining"
        if elapsed_seconds is not None:
            details = f"{details}, {elapsed_seconds:.1f}s"
        self._status(f"{label}: [{bar}] {percent}% ({details})")

    def _write_manifest(self, run_dir: Path, manifest: dict[str, object]) -> None:
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def _status(self, message: str) -> None:
        if self._status_callback is not None:
            normalized = message.strip()
            if normalized and not self._starts_with_emoji(normalized):
                message = f"ℹ️ {message}"
            self._status_callback(message)

    @staticmethod
    def _starts_with_emoji(value: str) -> bool:
        return value.startswith(
            (
                "🔎",
                "📁",
                "🧠",
                "🎧",
                "🔊",
                "🛡",
                "🎙",
                "🖼",
                "⏱",
                "🧪",
                "🪄",
                "🧩",
                "🐛",
                "✂",
                "📝",
                "⚠",
                "🚫",
                "🧵",
                "✅",
                "📷",
                "🎬",
                "💾",
                "ℹ",
                "❌",
            )
        )
