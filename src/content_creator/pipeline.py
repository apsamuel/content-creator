from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
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
from content_creator.media import AudioOverlayEvent, CinematicIntroCard, MediaAssembler
from content_creator.planner import (
    CinematicTransition,
    Scene,
    ScenePlanner,
    ScenePlan,
    VideoPromptPlan,
    VideoPromptPreclassification,
)
from content_creator.profanity_sfx import (
    build_profanity_sfx_plan,
    load_profanity_words,
    load_sound_pack,
    scan_text_for_profanity,
)


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
    _DEFAULT_CINEMATIC_INTRO_DURATION_SECONDS = 5.8

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
        self._planner = ScenePlanner(
            self._gateway,
            image_composition_mode=config.image_composition_mode,
            preclassification_ensemble_enabled=config.preclassification_ensemble_enabled,
            preclass_emotion_model=config.models.preclass_emotion_model,
            preclass_intent_model=config.models.preclass_intent_model,
            safety_primary_model=config.models.safety_model,
            safety_secondary_model=config.models.safety_secondary_model,
        )
        self._media = MediaAssembler(
            width=config.width, height=config.height, fps=config.fps
        )
        self._last_content_safety_report: dict[str, object] | None = None
        self._chunk_ensemble_scores: list[dict[str, object]] = []

    def generate_from_text(
        self,
        *,
        narration_text: str,
        video_prompt: str | None,
        output_path: Path,
        generate_video_prompt: bool = False,
        cinematic_intro: bool = False,
        cinematic_intro_duration: float = _DEFAULT_CINEMATIC_INTRO_DURATION_SECONDS,
        cinematic_transitions: bool = False,
        television_overlay_effects: bool = False,
        image_workers: int = 1,
        images_per_scene: int = 1,
        view_preclassification: bool = False,
        feedback_tier: str = "standard",
        enhanced_rationale: bool = False,
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
            "cinematic_intro": cinematic_intro,
            "cinematic_intro_duration": cinematic_intro_duration,
            "cinematic_transitions": cinematic_transitions,
            "television_overlay_effects": television_overlay_effects,
            "images_per_scene": images_per_scene,
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
            cinematic_intro=cinematic_intro,
            cinematic_intro_duration=cinematic_intro_duration,
            cinematic_transitions=cinematic_transitions,
            television_overlay_effects=television_overlay_effects,
            image_workers=image_workers,
            images_per_scene=images_per_scene,
            view_preclassification=view_preclassification,
            feedback_tier=feedback_tier,
            enhanced_rationale=enhanced_rationale,
        )

    def generate_from_audio(
        self,
        *,
        audio_path: Path,
        video_prompt: str | None,
        output_path: Path,
        chunk_seconds: float = 45.0,
        generate_video_prompt: bool = False,
        cinematic_intro: bool = False,
        cinematic_intro_duration: float = _DEFAULT_CINEMATIC_INTRO_DURATION_SECONDS,
        cinematic_transitions: bool = False,
        television_overlay_effects: bool = False,
        preserve_speaker: bool = False,
        diarization_speaker_count: int | None = None,
        diarization_min_speakers: int | None = None,
        diarization_max_speakers: int | None = None,
        speaker_dominance_threshold: float = 0.9,
        content_safety_enabled: bool = False,
        content_safety_filter: bool = False,
        content_safety_threshold: float = 0.7,
        content_safety_model: str | None = None,
        profanity_sfx_enabled: bool = False,
        profanity_sound_pack_dir: Path | None = None,
        profanity_words_file: Path | None = None,
        profanity_pad_seconds: float = 0.08,
        profanity_duck_db: float = -42.0,
        transcribe_workers: int = 1,
        image_workers: int = 1,
        images_per_scene: int = 1,
        view_preclassification: bool = False,
        feedback_tier: str = "standard",
        enhanced_rationale: bool = False,
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
            "diarization_speaker_count": diarization_speaker_count,
            "diarization_min_speakers": diarization_min_speakers,
            "diarization_max_speakers": diarization_max_speakers,
            "speaker_dominance_threshold": speaker_dominance_threshold,
            "content_safety_enabled": content_safety_enabled,
            "content_safety_filter": content_safety_filter,
            "content_safety_threshold": content_safety_threshold,
            "content_safety_model": content_safety_model,
            "profanity_sfx_enabled": profanity_sfx_enabled,
            "profanity_sound_pack_dir": (
                str(profanity_sound_pack_dir) if profanity_sound_pack_dir else None
            ),
            "profanity_words_file": (
                str(profanity_words_file) if profanity_words_file else None
            ),
            "profanity_pad_seconds": profanity_pad_seconds,
            "profanity_duck_db": profanity_duck_db,
            "transcribe_workers": transcribe_workers,
            "image_workers": image_workers,
            "images_per_scene": images_per_scene,
            "generate_video_prompt": generate_video_prompt,
            "cinematic_intro": cinematic_intro,
            "cinematic_intro_duration": cinematic_intro_duration,
            "cinematic_transitions": cinematic_transitions,
            "television_overlay_effects": television_overlay_effects,
            "video_prompt": video_prompt,
        }
        self._write_manifest(run_dir, manifest)
        transcript = self._transcribe_with_optional_chunking(
            audio_path=audio_path,
            chunk_seconds=chunk_seconds,
            chunk_dir_root=run_dir / "stt_chunks",
            preserve_speaker=preserve_speaker,
            diarization_speaker_count=diarization_speaker_count,
            diarization_min_speakers=diarization_min_speakers,
            diarization_max_speakers=diarization_max_speakers,
            speaker_dominance_threshold=speaker_dominance_threshold,
            content_safety_enabled=content_safety_enabled,
            content_safety_filter=content_safety_filter,
            content_safety_threshold=content_safety_threshold,
            content_safety_model=content_safety_model,
            transcribe_workers=transcribe_workers,
            profanity_words=(
                load_profanity_words(profanity_words_file)
                if content_safety_enabled
                else None
            ),
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

        audio_for_render = audio_path
        if profanity_sfx_enabled:
            self._status("🤖 Building profanity replacement plan from word timestamps")
            censored_audio_path = run_dir / "audio_censored.m4a"
            censorship_report = self._apply_profanity_sound_effects(
                source_audio=audio_path,
                output_audio=censored_audio_path,
                sound_pack_dir=profanity_sound_pack_dir,
                profanity_words_file=profanity_words_file,
                pad_seconds=profanity_pad_seconds,
                duck_db=profanity_duck_db,
            )
            manifest["profanity_sfx"] = censorship_report
            if bool(censorship_report.get("events_applied", 0)):
                audio_for_render = censored_audio_path

        self._write_manifest(run_dir, manifest)
        return self._render_project(
            narration_text=transcript,
            video_prompt=video_prompt,
            generate_video_prompt=generate_video_prompt,
            audio_path=audio_for_render,
            duration_seconds=duration,
            output_path=output_path,
            run_dir=run_dir,
            manifest=manifest,
            cinematic_intro=cinematic_intro,
            cinematic_intro_duration=cinematic_intro_duration,
            cinematic_transitions=cinematic_transitions,
            television_overlay_effects=television_overlay_effects,
            image_workers=image_workers,
            images_per_scene=images_per_scene,
            view_preclassification=view_preclassification,
            feedback_tier=feedback_tier,
            enhanced_rationale=enhanced_rationale,
        )

    def rebuild_video_from_run(
        self,
        *,
        run_dir: Path,
        output_path: Path | None = None,
        audio_path: Path | None = None,
        cinematic_intro_enabled: bool | None = None,
        cinematic_intro_duration: float | None = None,
        cinematic_transitions: bool | None = None,
        television_overlay_effects: bool | None = None,
        reuse_visual_assets: bool = True,
    ) -> Path:
        """Rebuild the final video using an existing run directory.

        This skips transcription, planning, and image generation by reusing
        `manifest.json`, generated images, and existing audio artifacts.
        """

        self._ensure_video_dependencies()
        resolved_run_dir = run_dir.expanduser().resolve()
        if not resolved_run_dir.exists() or not resolved_run_dir.is_dir():
            raise ValueError(f"Run directory does not exist: {resolved_run_dir}")

        manifest_path = resolved_run_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Run directory is missing manifest.json: {manifest_path}")

        manifest_raw = manifest_path.read_text(encoding="utf-8")
        try:
            manifest: dict[str, object] = json.loads(manifest_raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid manifest JSON at {manifest_path}: {exc}"
            ) from exc

        resolved_audio_path = (
            audio_path.expanduser().resolve()
            if audio_path is not None
            else self._resolve_manifest_audio_path(
                run_dir=resolved_run_dir, manifest=manifest
            )
        )
        if not resolved_audio_path.exists():
            raise ValueError(f"Audio file does not exist: {resolved_audio_path}")

        resolved_output_path = self._resolve_manifest_output_path(
            run_dir=resolved_run_dir, manifest=manifest, output_path=output_path
        )

        manifest_intro = manifest.get("cinematic_intro")
        intro_enabled_default = False
        intro_title = ""
        intro_description = ""
        intro_duration_default = self._DEFAULT_CINEMATIC_INTRO_DURATION_SECONDS
        if isinstance(manifest_intro, dict):
            intro_enabled_default = bool(manifest_intro.get("enabled", False))
            intro_title = str(manifest_intro.get("title", "")).strip()
            intro_description = str(manifest_intro.get("description", "")).strip()
            intro_duration_default = max(
                2.0,
                self._coerce_float(
                    manifest_intro.get("duration_seconds"),
                    self._DEFAULT_CINEMATIC_INTRO_DURATION_SECONDS,
                ),
            )

        resolved_intro_enabled = (
            intro_enabled_default
            if cinematic_intro_enabled is None
            else cinematic_intro_enabled
        )
        resolved_intro_duration = (
            intro_duration_default
            if cinematic_intro_duration is None
            else max(2.0, float(cinematic_intro_duration))
        )
        resolved_transitions = (
            bool(manifest.get("cinematic_transitions", False))
            if cinematic_transitions is None
            else cinematic_transitions
        )
        resolved_tv_effects = (
            bool(manifest.get("television_overlay_effects", False))
            if television_overlay_effects is None
            else television_overlay_effects
        )
        default_transitions = bool(manifest.get("cinematic_transitions", False))
        default_tv_effects = bool(manifest.get("television_overlay_effects", False))

        intro_card: CinematicIntroCard | None = None
        if resolved_intro_enabled:
            title = intro_title or "Recovered Intro"
            description = (
                intro_description
                or "Recovered cinematic intro from existing run metadata."
            )
            intro_card = CinematicIntroCard(
                title=title,
                description=description,
                duration_seconds=resolved_intro_duration,
            )

        manifest["status"] = "rebuilding_video"
        manifest["run_dir"] = str(resolved_run_dir)
        manifest["audio"] = str(resolved_audio_path)
        manifest["output"] = str(resolved_output_path)
        manifest["cinematic_transitions"] = resolved_transitions
        manifest["television_overlay_effects"] = resolved_tv_effects
        manifest["cinematic_intro"] = (
            {
                "enabled": True,
                "title": intro_card.title,
                "description": intro_card.description,
                "duration_seconds": intro_card.duration_seconds,
            }
            if intro_card is not None
            else {"enabled": False}
        )
        self._write_manifest(resolved_run_dir, manifest)

        visual_asset_reuse_allowed = (
            reuse_visual_assets
            and resolved_intro_enabled == intro_enabled_default
            and resolved_transitions == default_transitions
            and resolved_tv_effects == default_tv_effects
        )
        if resolved_intro_enabled:
            visual_asset_reuse_allowed = visual_asset_reuse_allowed and (
                abs(resolved_intro_duration - intro_duration_default) < 0.01
            )

        reusable_visual_path = (
            self._resolve_reusable_visual_path(
                run_dir=resolved_run_dir,
                intro_enabled=resolved_intro_enabled,
                television_overlay_effects=resolved_tv_effects,
            )
            if visual_asset_reuse_allowed
            else None
        )

        if reusable_visual_path is not None:
            if self._media.probe_media_file(reusable_visual_path):
                self._status(
                    "♻️ Reusing existing stitched visuals to skip scene re-render"
                )
                final_path = self._media.mux_visual_with_audio(
                    visual_path=reusable_visual_path,
                    audio_path=resolved_audio_path,
                    output_path=resolved_output_path,
                    intro_delay_seconds=(
                        intro_card.duration_seconds if intro_card is not None else 0.0
                    ),
                )
            else:
                self._status(
                    "⚠️ Existing stitched visual is unreadable; falling back to full rebuild"
                )
                final_path = self._render_rebuild_from_images(
                    run_dir=resolved_run_dir,
                    manifest=manifest,
                    audio_path=resolved_audio_path,
                    output_path=resolved_output_path,
                    intro_card=intro_card,
                    cinematic_transitions=resolved_transitions,
                    television_overlay_effects=resolved_tv_effects,
                )
        else:
            final_path = self._render_rebuild_from_images(
                run_dir=resolved_run_dir,
                manifest=manifest,
                audio_path=resolved_audio_path,
                output_path=resolved_output_path,
                intro_card=intro_card,
                cinematic_transitions=resolved_transitions,
                television_overlay_effects=resolved_tv_effects,
            )

        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._write_manifest(resolved_run_dir, manifest)
        self._status("✅ Video rebuild complete")
        return final_path

    def _resolve_reusable_visual_path(
        self, *, run_dir: Path, intro_enabled: bool, television_overlay_effects: bool
    ) -> Path | None:
        if television_overlay_effects:
            candidate = run_dir / "stitched_tv_effects.mp4"
            if not candidate.exists():
                return None
            if intro_enabled and not (run_dir / "stitched_with_intro.mp4").exists():
                return None
            return candidate

        if intro_enabled:
            candidate = run_dir / "stitched_with_intro.mp4"
            return candidate if candidate.exists() else None

        candidate = run_dir / "stitched.mp4"
        return candidate if candidate.exists() else None

    def _render_rebuild_from_images(
        self,
        *,
        run_dir: Path,
        manifest: dict[str, object],
        audio_path: Path,
        output_path: Path,
        intro_card: CinematicIntroCard | None,
        cinematic_transitions: bool,
        television_overlay_effects: bool,
    ) -> Path:
        self._status("🎬 Rebuilding final video from existing run assets")
        scenes = self._load_scenes_from_manifest(manifest)
        scene_image_sequences = self._load_scene_images_from_manifest(
            run_dir=run_dir, manifest=manifest, scenes=scenes
        )
        return self._media.render_video(
            images=scene_image_sequences,
            scenes=scenes,
            audio_path=audio_path,
            output_path=output_path,
            work_dir=run_dir,
            cinematic_intro=intro_card,
            cinematic_transitions=cinematic_transitions,
            television_overlay_effects=television_overlay_effects,
        )

    def transcribe_audio_file(
        self,
        *,
        audio_path: Path,
        output_path: Path | None = None,
        chunk_seconds: float = 45.0,
        preserve_speaker: bool = False,
        diarization_speaker_count: int | None = None,
        diarization_min_speakers: int | None = None,
        diarization_max_speakers: int | None = None,
        speaker_dominance_threshold: float = 0.9,
        content_safety_enabled: bool = False,
        content_safety_filter: bool = False,
        content_safety_threshold: float = 0.7,
        content_safety_model: str | None = None,
        profanity_sfx_enabled: bool = False,
        profanity_sfx_output_path: Path | None = None,
        profanity_sound_pack_dir: Path | None = None,
        profanity_words_file: Path | None = None,
        profanity_pad_seconds: float = 0.08,
        profanity_duck_db: float = -42.0,
        transcribe_workers: int = 1,
    ) -> str:
        transcript = self._transcribe_with_optional_chunking(
            audio_path=audio_path,
            chunk_seconds=chunk_seconds,
            chunk_dir_root=self._config.work_dir / "transcribe_chunks",
            preserve_speaker=preserve_speaker,
            diarization_speaker_count=diarization_speaker_count,
            diarization_min_speakers=diarization_min_speakers,
            diarization_max_speakers=diarization_max_speakers,
            speaker_dominance_threshold=speaker_dominance_threshold,
            content_safety_enabled=content_safety_enabled,
            content_safety_filter=content_safety_filter,
            content_safety_threshold=content_safety_threshold,
            content_safety_model=content_safety_model,
            transcribe_workers=transcribe_workers,
            profanity_words=(
                load_profanity_words(profanity_words_file)
                if content_safety_enabled
                else None
            ),
        )
        self._emit_content_safety_summary()
        if profanity_sfx_enabled:
            if profanity_sfx_output_path is None:
                raise ValueError(
                    "profanity_sfx_output_path is required when profanity_sfx_enabled is true"
                )
            self._status("🤖 Building profanity replacement plan from word timestamps")
            report = self._apply_profanity_sound_effects(
                source_audio=audio_path,
                output_audio=profanity_sfx_output_path,
                sound_pack_dir=profanity_sound_pack_dir,
                profanity_words_file=profanity_words_file,
                pad_seconds=profanity_pad_seconds,
                duck_db=profanity_duck_db,
            )
            self._status(
                "✅ Profanity SFX output written "
                f"({report.get('events_applied', 0)} replacements): {profanity_sfx_output_path}"
            )
        if output_path is not None:
            self._status("💾 Writing transcript to disk")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(wrap_transcription(transcript), encoding="utf-8")
        self._status("✅ Transcription complete")
        return transcript

    def build_profanity_debug_audio(
        self,
        *,
        audio_path: Path,
        output_path: Path,
        manifest_events: list[dict[str, object]] | None = None,
        preclassification_data: dict[str, object] | None = None,
        transcript_text: str | None = None,
        sound_pack_dir: Path | None = None,
        profanity_words_file: Path | None = None,
        pad_seconds: float = 0.08,
        context_seconds: float = 0.5,
        gap_seconds: float = 0.3,
        preclassification_position: str = "prepend",
        feedback_tier: str = "standard",
        enhanced_rationale: bool = False,
    ) -> int:
        """Build a debug audio file illustrating each profanity detection event.

        For every event the output contains: a synthesized voice announcing the
        detected word, start/end/duration; the raw audio snippet; a synthesized
        voice saying "Profanity filter implemented"; and the exact bleep that
        production would overlay.

        Returns the number of events processed (0 if none found).
        """
        normalized_preclass_position = preclassification_position.strip().lower()
        if normalized_preclass_position not in {"prepend", "append", "off"}:
            raise ValueError(
                "preclassification_position must be one of: prepend, append, off"
            )

        if manifest_events is not None:
            events = manifest_events
        else:
            self._status(
                "🎤 Transcribing audio with word timestamps for profanity detection…"
            )
            default_sound_dir = Path(__file__).resolve().parent / "sound"
            resolved_sound_dir = (
                sound_pack_dir.expanduser().resolve()
                if sound_pack_dir
                else default_sound_dir
            )
            generated_transcript_text, timed_words = (
                self._gateway.transcribe_audio_with_word_timestamps(audio_path)
            )
            if transcript_text is None:
                transcript_text = generated_transcript_text
            sound_pack = load_sound_pack(sound_pack_dir=resolved_sound_dir)
            profanity_words = load_profanity_words(profanity_words_file)
            plan = build_profanity_sfx_plan(
                timed_words=timed_words,
                sound_pack=sound_pack,
                profanity_words=profanity_words,
                pad_seconds=pad_seconds,
            )
            events = [
                {
                    "word": ev.word,
                    "start_seconds": ev.start_seconds,
                    "end_seconds": ev.end_seconds,
                    "sfx": str(ev.sfx_path),
                    "sfx_duration_seconds": ev.sfx_duration_seconds,
                    "sfx_gain_db": ev.sfx_gain_db,
                }
                for ev in plan.events
            ]

        if not events:
            self._status("ℹ️ No profanity events found — no debug audio to generate.")
            return 0

        source_duration_seconds: float | None = None
        try:
            source_duration_seconds = self._media.get_audio_duration(audio_path)
        except Exception:
            source_duration_seconds = None

        if (
            preclassification_data is None
            and transcript_text
            and transcript_text.strip()
        ):
            self._status("🧪 Generating live pre-classification for debug narration")
            try:
                preclass_plan = self._resolve_video_prompt_plan(
                    narration_text=transcript_text,
                    video_prompt=None,
                    generate_video_prompt=True,
                    duration_seconds=source_duration_seconds,
                )
                preclassification_data = self._serialize_preclassification(
                    preclass_plan.preclassification
                )
                if preclassification_data is not None:
                    preclassification_data = self._attach_feedback_annotations(
                        narration_text=transcript_text,
                        preclassification_data=preclassification_data,
                        feedback_tier=feedback_tier,
                        enhanced_rationale=enhanced_rationale,
                    )
            except Exception as exc:
                self._status(
                    "⚠️ Unable to generate pre-classification for debug narration: "
                    f"{exc}"
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="profanity_debug_") as tmp_str:
            tmp = Path(tmp_str)
            segment_paths: list[Path] = []

            silence_path = tmp / "silence.wav"
            self._ffmpeg_generate_silence(silence_path, duration_seconds=gap_seconds)

            input_summary_text = self._build_debug_input_summary(
                audio_path=audio_path,
                output_path=output_path,
                event_count=len(events),
                source_duration_seconds=source_duration_seconds,
                pad_seconds=pad_seconds,
                context_seconds=context_seconds,
                gap_seconds=gap_seconds,
                using_manifest_events=manifest_events is not None,
                preclassification_data=(
                    preclassification_data
                    if normalized_preclass_position == "prepend"
                    else None
                ),
                feedback_tier=feedback_tier,
            )
            if input_summary_text:
                self._status("🎤 Prepending synthesized input summary")
                intro_raw = tmp / "intro_raw.wav"
                intro_path = tmp / "intro.wav"
                self._synthesize_long_speech(input_summary_text, intro_raw, tmp)
                self._ffmpeg_normalize_audio(intro_raw, intro_path)
                segment_paths.append(intro_path)
                segment_paths.append(silence_path)

            for idx, event in enumerate(events):
                word = str(event.get("word", ""))
                start = self._coerce_float(event.get("start_seconds"))
                end = self._coerce_float(event.get("end_seconds"))
                sfx_path = Path(str(event.get("sfx", "")))
                sfx_duration = self._coerce_float(event.get("sfx_duration_seconds"))
                elapsed = max(0.0, end - start)

                self._status(
                    f"🎤 Building debug segment {idx + 1}/{len(events)}: '{word}'"
                )

                # 1. TTS announcement
                announce_text = (
                    f"Detected profanity: {word}. "
                    f"Start time: {start:.2f} seconds. "
                    f"End time: {end:.2f} seconds. "
                    f"Duration: {elapsed:.2f} seconds."
                )
                raw_announce = tmp / f"event_{idx:03d}_announce_raw.wav"
                announce_path = tmp / f"event_{idx:03d}_announce.wav"
                self._gateway.synthesize_speech(announce_text, raw_announce)
                self._ffmpeg_normalize_audio(raw_announce, announce_path)

                # 2. Raw audio snippet with context window
                snippet_start = max(0.0, start - context_seconds)
                snippet_end = end + context_seconds
                snippet_path = tmp / f"event_{idx:03d}_snippet.wav"
                self._ffmpeg_extract_audio_segment(
                    audio_path,
                    snippet_path,
                    start_seconds=snippet_start,
                    end_seconds=snippet_end,
                )

                # 3. TTS: "Profanity filter implemented."
                raw_filter = tmp / f"event_{idx:03d}_filter_raw.wav"
                filter_path = tmp / f"event_{idx:03d}_filter.wav"
                self._gateway.synthesize_speech(
                    "Profanity filter implemented.", raw_filter
                )
                self._ffmpeg_normalize_audio(raw_filter, filter_path)

                # 4. Bleep trimmed to its production duration
                bleep_path = tmp / f"event_{idx:03d}_bleep.wav"
                self._ffmpeg_extract_bleep(
                    sfx_path, bleep_path, duration_seconds=sfx_duration
                )

                # Prepend inter-event silence for every event after the first
                if segment_paths:
                    segment_paths.append(silence_path)
                segment_paths.extend(
                    [
                        announce_path,
                        silence_path,
                        snippet_path,
                        silence_path,
                        filter_path,
                        silence_path,
                        bleep_path,
                    ]
                )

            summary_text = self._build_debug_preclassification_summary(
                events=events,
                transcript_text=transcript_text,
                preclassification_data=(
                    preclassification_data
                    if normalized_preclass_position == "append"
                    else None
                ),
                feedback_tier=feedback_tier,
            )
            if summary_text:
                if normalized_preclass_position == "append" and preclassification_data:
                    self._status("🎤 Appending synthesized pre-classification summary")
                else:
                    self._status("🎤 Appending synthesized diagnostic summary")
                summary_raw = tmp / "summary_raw.wav"
                summary_path = tmp / "summary.wav"
                self._synthesize_long_speech(summary_text, summary_raw, tmp)
                self._ffmpeg_normalize_audio(summary_raw, summary_path)
                if segment_paths:
                    segment_paths.append(silence_path)
                segment_paths.append(summary_path)

            concat_list = tmp / "concat.txt"
            concat_list.write_text(
                "\n".join(f"file '{p.as_posix()}'" for p in segment_paths),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_list),
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        return len(events)

    def _build_debug_input_summary(
        self,
        *,
        audio_path: Path,
        output_path: Path,
        event_count: int,
        source_duration_seconds: float | None,
        pad_seconds: float,
        context_seconds: float,
        gap_seconds: float,
        using_manifest_events: bool,
        preclassification_data: dict[str, object] | None = None,
        feedback_tier: str = "standard",
    ) -> str:
        summary_parts = [
            "Debug input summary.",
            f"Input file: {audio_path.name}.",
            f"Output file: {output_path.name}.",
        ]
        if source_duration_seconds is not None:
            summary_parts.append(
                f"Source elapsed time: {source_duration_seconds:.2f} seconds."
            )
        summary_parts.extend(
            [
                f"Events to process: {event_count}.",
                (
                    "Event source: manifest input."
                    if using_manifest_events
                    else "Event source: live transcription."
                ),
                f"Timing settings: pad {pad_seconds:.2f} seconds, context {context_seconds:.2f} seconds, gap {gap_seconds:.2f} seconds.",
            ]
        )
        if preclassification_data:
            mood = preclassification_data.get("mood")
            has_foul = preclassification_data.get("has_foul_language", False)
            word_count = preclassification_data.get("word_count")
            sentence_count = preclassification_data.get("sentence_count")
            truthfulness = preclassification_data.get("truthfulness_assessment")
            style = preclassification_data.get("interaction_style_assessment")

            summary_parts.append("Pre-classification report.")

            # Overview line
            overview: list[str] = []
            if isinstance(mood, str) and mood.strip():
                overview.append(f"mood: {mood.strip()}")
            foul_str = (
                "foul language detected" if has_foul else "no foul language detected"
            )
            overview.append(foul_str)
            if isinstance(word_count, int):
                overview.append(f"{word_count} words")
            if isinstance(sentence_count, int):
                overview.append(f"{sentence_count} sentences")
            if overview:
                summary_parts.append("Overview: " + "; ".join(overview) + ".")

            # Truthfulness
            if isinstance(truthfulness, dict):
                truth_label = truthfulness.get("label", "")
                truth_confidence = truthfulness.get("confidence_score")
                truth_reason = truthfulness.get("reason", "")
                if truth_label:
                    truth_entry = f"Truthfulness assessment: {truth_label}"
                    if isinstance(truth_confidence, (int, float)):
                        truth_entry += f", confidence {float(truth_confidence):.0%}"
                    if truth_reason:
                        truth_entry += f". {truth_reason}"
                    summary_parts.append(truth_entry + ".")

            fact_check = preclassification_data.get("fact_check_assessment")
            if isinstance(fact_check, dict):
                label = fact_check.get("label", "")
                confidence = fact_check.get("confidence_score")
                reason = fact_check.get("reason", "")
                if label:
                    line = f"Fact-check estimate: {label}"
                    if isinstance(confidence, (int, float)):
                        line += f", confidence {float(confidence):.0%}"
                    if reason:
                        line += f". {reason}"
                    summary_parts.append(line + ".")

            aggression = preclassification_data.get("aggression_assessment")
            if isinstance(aggression, dict):
                label = aggression.get("label", "")
                confidence = aggression.get("confidence_score")
                reason = aggression.get("reason", "")
                if label:
                    line = f"Aggression estimate: {label}"
                    if isinstance(confidence, (int, float)):
                        line += f", confidence {float(confidence):.0%}"
                    if reason:
                        line += f". {reason}"
                    summary_parts.append(line + ".")

            contemporary = preclassification_data.get(
                "contemporary_alignment_assessment"
            )
            if isinstance(contemporary, dict):
                label = contemporary.get("label", "")
                confidence = contemporary.get("confidence_score")
                reason = contemporary.get("reason", "")
                if label:
                    line = f"Contemporary alignment: {label}"
                    if isinstance(confidence, (int, float)):
                        line += f", confidence {float(confidence):.0%}"
                    if reason:
                        line += f". {reason}"
                    summary_parts.append(line + ".")

            propaganda = preclassification_data.get("propaganda_assessment")
            if isinstance(propaganda, dict):
                label = propaganda.get("label", "")
                confidence = propaganda.get("confidence_score")
                reason = propaganda.get("reason", "")
                if label:
                    line = f"Propaganda alignment: {label}"
                    if isinstance(confidence, (int, float)):
                        line += f", confidence {float(confidence):.0%}"
                    if reason:
                        line += f". {reason}"
                    summary_parts.append(line + ".")

            social = preclassification_data.get("social_score_assessment")
            if isinstance(social, dict):
                label = social.get("composite_label", "")
                score = social.get("composite_social_score")
                reason = social.get("reason", "")
                if label:
                    line = f"Social score: {label}"
                    if isinstance(score, (int, float)):
                        line += f" at {float(score):.2f}"
                    if reason:
                        line += f". {reason}"
                    summary_parts.append(line + ".")

            # Interaction style sub-dimensions
            if isinstance(style, dict):
                style_dim_labels = {
                    "formality": "Formality",
                    "certainty_hedging": "Certainty hedging",
                    "persuasion_intent": "Persuasion intent",
                    "claim_density": "Claim density",
                }
                for key, heading in style_dim_labels.items():
                    item = style.get(key)
                    if isinstance(item, dict):
                        lbl = item.get("label")
                        conf = item.get("confidence_score")
                        reason = item.get("reason", "")
                        if isinstance(lbl, str) and lbl.strip():
                            dim_entry = f"{heading}: {lbl.strip()}"
                            if isinstance(conf, (int, float)):
                                dim_entry += f", confidence {float(conf):.0%}"
                            if reason:
                                dim_entry += f". {reason}"
                            summary_parts.append(dim_entry + ".")

                # Speaker sentiment
                speaker_sentiment = style.get("speaker_sentiment")
                if isinstance(speaker_sentiment, list) and speaker_sentiment:
                    first = speaker_sentiment[0]
                    if isinstance(first, dict):
                        sentiment = first.get("sentiment")
                        speaker = first.get("speaker")
                        conf = first.get("confidence_score")
                        reason = first.get("reason", "")
                        if isinstance(sentiment, str) and sentiment.strip():
                            sent_entry = "Speaker sentiment"
                            if (
                                isinstance(speaker, str)
                                and speaker.strip()
                                and speaker.strip().lower() != "unknown"
                            ):
                                sent_entry += f" ({speaker.strip()})"
                            sent_entry += f": {sentiment.strip()}"
                            if isinstance(conf, (int, float)):
                                sent_entry += f", confidence {float(conf):.0%}"
                            if reason:
                                sent_entry += f". {reason}"
                            summary_parts.append(sent_entry + ".")

            communication_metrics = preclassification_data.get("communication_metrics")
            if isinstance(communication_metrics, dict):
                profanity_per_sentence_ratio = communication_metrics.get(
                    "profanity_per_sentence_ratio"
                )
                if isinstance(profanity_per_sentence_ratio, (int, float)):
                    summary_parts.append(
                        "Profanity per sentence ratio: "
                        f"{float(profanity_per_sentence_ratio):.4f}."
                    )

                profanity_to_non_profanity_ratio = communication_metrics.get(
                    "profanity_to_non_profanity_ratio"
                )
                if isinstance(profanity_to_non_profanity_ratio, (int, float)):
                    summary_parts.append(
                        "Profanity to non-profanity ratio: "
                        f"{float(profanity_to_non_profanity_ratio):.4f}."
                    )

                sentence_complexity_score = communication_metrics.get(
                    "sentence_complexity_score"
                )
                sentence_complexity_label = communication_metrics.get(
                    "sentence_complexity_label"
                )
                if isinstance(sentence_complexity_score, (int, float)):
                    if isinstance(sentence_complexity_label, str) and sentence_complexity_label.strip():
                        summary_parts.append(
                            "Sentence complexity: "
                            f"{sentence_complexity_label.strip()} at "
                            f"{float(sentence_complexity_score):.4f}."
                        )
                    else:
                        summary_parts.append(
                            "Sentence complexity score: "
                            f"{float(sentence_complexity_score):.4f}."
                        )

            self._append_feedback_summary_lines(
                summary_parts=summary_parts,
                preclassification_data=preclassification_data,
                feedback_tier=feedback_tier,
            )

        summary_parts.append("Begin event diagnostics.")
        return " ".join(summary_parts)

    def _build_debug_preclassification_summary(
        self,
        *,
        events: list[dict[str, object]],
        transcript_text: str | None,
        preclassification_data: dict[str, object] | None,
        feedback_tier: str = "standard",
    ) -> str:
        unique_words = {
            str(event.get("word", "")).strip().lower()
            for event in events
            if str(event.get("word", "")).strip()
        }
        avg_duration = 0.0
        if events:
            total_duration = 0.0
            for event in events:
                start = self._coerce_float(event.get("start_seconds"))
                end = self._coerce_float(event.get("end_seconds"))
                total_duration += max(0.0, end - start)
            avg_duration = total_duration / len(events)

        summary_parts = [
            "Diagnostic summary.",
            f"Event count: {len(events)}.",
            f"Unique matches: {len(unique_words)}.",
            f"Average event duration: {avg_duration:.2f} seconds.",
        ]

        if transcript_text:
            transcript_word_count = len(transcript_text.split())
            summary_parts.append(
                f"Transcript words before classification: {transcript_word_count}."
            )

        if preclassification_data:
            mood = preclassification_data.get("mood")
            if isinstance(mood, str) and mood.strip():
                summary_parts.append(f"Mood: {mood.strip()}.")

            has_foul_language = preclassification_data.get("has_foul_language")
            if isinstance(has_foul_language, bool):
                summary_parts.append(
                    "Foul language signal: " + ("yes." if has_foul_language else "no.")
                )

            truthfulness = preclassification_data.get("truthfulness_assessment")
            if isinstance(truthfulness, dict):
                label = truthfulness.get("label")
                confidence = truthfulness.get("confidence_score")
                if isinstance(label, str) and label.strip():
                    if isinstance(confidence, (int, float)):
                        summary_parts.append(
                            f"Truthfulness: {label.strip()} at {float(confidence):.2f} confidence."
                        )
                    else:
                        summary_parts.append(f"Truthfulness: {label.strip()}.")

            fact_check = preclassification_data.get("fact_check_assessment")
            if isinstance(fact_check, dict):
                label = fact_check.get("label")
                confidence = fact_check.get("confidence_score")
                if isinstance(label, str) and label.strip():
                    if isinstance(confidence, (int, float)):
                        summary_parts.append(
                            f"Fact-check estimate: {label.strip()} at {float(confidence):.2f} confidence."
                        )
                    else:
                        summary_parts.append(f"Fact-check estimate: {label.strip()}.")

            aggression = preclassification_data.get("aggression_assessment")
            if isinstance(aggression, dict):
                label = aggression.get("label")
                confidence = aggression.get("confidence_score")
                if isinstance(label, str) and label.strip():
                    if isinstance(confidence, (int, float)):
                        summary_parts.append(
                            f"Aggression estimate: {label.strip()} at {float(confidence):.2f} confidence."
                        )
                    else:
                        summary_parts.append(f"Aggression estimate: {label.strip()}.")

            social = preclassification_data.get("social_score_assessment")
            if isinstance(social, dict):
                label = social.get("composite_label")
                score = social.get("composite_social_score")
                if isinstance(label, str) and label.strip():
                    if isinstance(score, (int, float)):
                        summary_parts.append(
                            f"Composite social score: {label.strip()} at {float(score):.2f}."
                        )
                    else:
                        summary_parts.append(
                            f"Composite social score: {label.strip()}."
                        )

            contemporary = preclassification_data.get(
                "contemporary_alignment_assessment"
            )
            if isinstance(contemporary, dict):
                label = contemporary.get("label")
                confidence = contemporary.get("confidence_score")
                if isinstance(label, str) and label.strip():
                    if isinstance(confidence, (int, float)):
                        summary_parts.append(
                            f"Contemporary alignment: {label.strip()} at {float(confidence):.2f} confidence."
                        )
                    else:
                        summary_parts.append(
                            f"Contemporary alignment: {label.strip()}."
                        )

            propaganda = preclassification_data.get("propaganda_assessment")
            if isinstance(propaganda, dict):
                label = propaganda.get("label")
                confidence = propaganda.get("confidence_score")
                if isinstance(label, str) and label.strip():
                    if isinstance(confidence, (int, float)):
                        summary_parts.append(
                            f"Propaganda alignment: {label.strip()} at {float(confidence):.2f} confidence."
                        )
                    else:
                        summary_parts.append(f"Propaganda alignment: {label.strip()}.")

            style = preclassification_data.get("interaction_style_assessment")
            if isinstance(style, dict):
                style_labels: list[str] = []
                for key in (
                    "formality",
                    "certainty_hedging",
                    "persuasion_intent",
                    "claim_density",
                ):
                    item = style.get(key)
                    if isinstance(item, dict):
                        label = item.get("label")
                        if isinstance(label, str) and label.strip():
                            style_labels.append(
                                f"{key.replace('_', ' ')}: {label.strip()}"
                            )
                if style_labels:
                    summary_parts.append("Style: " + "; ".join(style_labels) + ".")

                speaker_sentiment = style.get("speaker_sentiment")
                if isinstance(speaker_sentiment, list) and speaker_sentiment:
                    first = speaker_sentiment[0]
                    if isinstance(first, dict):
                        sentiment = first.get("sentiment")
                        speaker = first.get("speaker")
                        if isinstance(sentiment, str) and sentiment.strip():
                            if isinstance(speaker, str) and speaker.strip():
                                summary_parts.append(
                                    f"Primary sentiment: {speaker.strip()}, {sentiment.strip()}."
                                )
                            else:
                                summary_parts.append(
                                    f"Primary sentiment: {sentiment.strip()}."
                                )

            communication_metrics = preclassification_data.get("communication_metrics")
            if isinstance(communication_metrics, dict):
                profanity_per_sentence_ratio = communication_metrics.get(
                    "profanity_per_sentence_ratio"
                )
                if isinstance(profanity_per_sentence_ratio, (int, float)):
                    summary_parts.append(
                        "Profanity per sentence ratio: "
                        f"{float(profanity_per_sentence_ratio):.4f}."
                    )

                profanity_to_non_profanity_ratio = communication_metrics.get(
                    "profanity_to_non_profanity_ratio"
                )
                if isinstance(profanity_to_non_profanity_ratio, (int, float)):
                    summary_parts.append(
                        "Profanity to non-profanity ratio: "
                        f"{float(profanity_to_non_profanity_ratio):.4f}."
                    )

                sentence_complexity_score = communication_metrics.get(
                    "sentence_complexity_score"
                )
                sentence_complexity_label = communication_metrics.get(
                    "sentence_complexity_label"
                )
                if isinstance(sentence_complexity_score, (int, float)):
                    if isinstance(sentence_complexity_label, str) and sentence_complexity_label.strip():
                        summary_parts.append(
                            "Sentence complexity: "
                            f"{sentence_complexity_label.strip()} at "
                            f"{float(sentence_complexity_score):.4f}."
                        )
                    else:
                        summary_parts.append(
                            "Sentence complexity score: "
                            f"{float(sentence_complexity_score):.4f}."
                        )

            self._append_feedback_summary_lines(
                summary_parts=summary_parts,
                preclassification_data=preclassification_data,
                feedback_tier=feedback_tier,
            )

        summary_parts.append("End diagnostic summary.")
        return " ".join(summary_parts)

    def _apply_profanity_sound_effects(
        self,
        *,
        source_audio: Path,
        output_audio: Path,
        sound_pack_dir: Path | None,
        profanity_words_file: Path | None,
        pad_seconds: float,
        duck_db: float,
    ) -> dict[str, object]:
        default_sound_dir = Path(__file__).resolve().parent / "sound"
        resolved_sound_dir = (
            sound_pack_dir.expanduser().resolve()
            if sound_pack_dir
            else default_sound_dir
        )

        transcript_text, timed_words = (
            self._gateway.transcribe_audio_with_word_timestamps(source_audio)
        )
        sound_pack = load_sound_pack(sound_pack_dir=resolved_sound_dir)
        profanity_words = load_profanity_words(profanity_words_file)
        plan = build_profanity_sfx_plan(
            timed_words=timed_words,
            sound_pack=sound_pack,
            profanity_words=profanity_words,
            pad_seconds=pad_seconds,
        )

        events = [
            AudioOverlayEvent(
                start_seconds=event.start_seconds,
                end_seconds=event.end_seconds,
                sfx_path=event.sfx_path,
                sfx_duration_seconds=event.sfx_duration_seconds,
                sfx_gain_db=event.sfx_gain_db,
            )
            for event in plan.events
        ]
        self._media.overlay_sound_effects(
            audio_path=source_audio,
            output_path=output_audio,
            events=events,
            duck_db=duck_db,
        )
        return {
            "enabled": True,
            "sound_pack": plan.sound_pack_name,
            "sound_pack_dir": str(plan.sound_pack_dir),
            "transcript_text_length": len(transcript_text),
            "total_words": plan.total_words,
            "matches_found": plan.matches_found,
            "events_applied": len(plan.events),
            "output_audio": str(output_audio),
            "events": [
                {
                    "word": event.word,
                    "start_seconds": event.start_seconds,
                    "end_seconds": event.end_seconds,
                    "sfx": str(event.sfx_path),
                    "sfx_duration_seconds": event.sfx_duration_seconds,
                    "sfx_gain_db": event.sfx_gain_db,
                }
                for event in plan.events
            ],
        }

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
        cinematic_intro: bool = False,
        cinematic_intro_duration: float = _DEFAULT_CINEMATIC_INTRO_DURATION_SECONDS,
        cinematic_transitions: bool = False,
        television_overlay_effects: bool = False,
        image_workers: int = 1,
        images_per_scene: int = 1,
        view_preclassification: bool = False,
        feedback_tier: str = "standard",
        enhanced_rationale: bool = False,
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
            duration_seconds=duration_seconds,
        )
        resolved_video_prompt = video_prompt_plan.video_prompt
        manifest["video_prompt"] = resolved_video_prompt
        serialized_preclassification = self._serialize_preclassification(
            video_prompt_plan.preclassification
        )
        if serialized_preclassification is not None:
            serialized_preclassification = self._attach_feedback_annotations(
                narration_text=narration_text,
                preclassification_data=serialized_preclassification,
                feedback_tier=feedback_tier,
                enhanced_rationale=enhanced_rationale,
            )
        manifest["video_prompt_preclassification"] = serialized_preclassification
        analysis_summary_path = self._write_analysis_summary_artifact(
            run_dir=run_dir,
            narration_text=narration_text,
            video_prompt=resolved_video_prompt,
            preclassification=manifest.get("video_prompt_preclassification"),
        )
        if analysis_summary_path is not None:
            manifest["analysis_summary"] = {
                "path": str(analysis_summary_path),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        if (
            view_preclassification
            and manifest.get("video_prompt_preclassification") is not None
        ):
            self._status(
                f"🔬 High-fidelity analysis summary: {analysis_summary_path}\n"
                + self._format_preclassification_rollup(
                    manifest.get("video_prompt_preclassification")
                )
                + "\n"
                + "🔍 Pre-classification:\n"
                + json.dumps(manifest["video_prompt_preclassification"], indent=2)
            )
        manifest["status"] = "planning_scenes"
        self._write_manifest(run_dir, manifest)
        self._status("🧠 Planning scenes from narration")
        scene_plan = self._planner.build_scenes(
            narration_text=narration_text,
            video_prompt=resolved_video_prompt,
            total_duration_seconds=duration_seconds,
            cinematic_transitions=cinematic_transitions,
        )
        scenes = scene_plan.scenes
        if video_prompt_plan.prompts is not None:
            manifest["llm_prompts"] = {
                **video_prompt_plan.prompts,
                "scene_planning": scene_plan.scene_prompt,
            }
        else:
            manifest["llm_prompts"] = {"scene_planning": scene_plan.scene_prompt}

        # Serialize scenes with transition data
        manifest["scenes"] = [
            {
                **asdict(scene),
                "transition_to_next": (
                    asdict(scene.transition_to_next)
                    if scene.transition_to_next is not None
                    else None
                ),
            }
            for scene in scenes
        ]
        manifest["status"] = "scenes_planned"
        self._write_manifest(run_dir, manifest)
        if cinematic_transitions:
            self._status(f"🧩 Planned {len(scenes)} scenes with cinematic transitions")
        else:
            self._status(f"🧩 Planned {len(scenes)} scenes")
        images_dir = run_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        scene_images_per_scene = max(1, images_per_scene)
        manifest["status"] = "generating_images"
        manifest["images"] = []
        manifest["images_per_scene"] = scene_images_per_scene
        self._write_manifest(run_dir, manifest)
        self._status("🖼️ Generating images for scenes")
        total_scenes = len(scenes)
        worker_count = max(1, image_workers)
        total_images = total_scenes * scene_images_per_scene

        # Extract visual intensity from preclassification if available
        visual_intensity = None
        if (
            video_prompt_plan.preclassification
            and video_prompt_plan.preclassification.ensemble_scorecard
        ):
            visual_intensity = (
                video_prompt_plan.preclassification.ensemble_scorecard.recommended_visual_intensity
            )

        def _render_scene_image(
            scene_index: int, scene_prompt: str, frame_index: int
        ) -> tuple[int, int, Path, float, str]:
            start = perf_counter()
            destination = (
                images_dir / f"scene_{scene_index:02d}_frame_{frame_index + 1:02d}.png"
            )
            prepared_prompt = self._build_scene_frame_prompt(
                scene_prompt=scene_prompt,
                scene_index=scene_index,
                total_scenes=total_scenes,
                frame_index=frame_index,
                frames_per_scene=scene_images_per_scene,
                visual_intensity=visual_intensity,
            )
            self._gateway.generate_image(prepared_prompt, destination)
            return (
                scene_index,
                frame_index,
                destination,
                perf_counter() - start,
                prepared_prompt,
            )

        rendered_paths: dict[int, dict[int, Path]] = {}
        rendered_prepared_prompts: dict[int, dict[int, str]] = {}
        completed = 0

        if worker_count == 1:
            for scene in scenes:
                for frame_index in range(scene_images_per_scene):
                    if self._debug:
                        self._status(
                            "🐛 Rendering image for "
                            f"scene {scene.index}/{len(scenes)} "
                            f"frame {frame_index + 1}/{scene_images_per_scene}"
                        )
                    (
                        scene_index,
                        rendered_frame_index,
                        image_path,
                        elapsed,
                        prepared,
                    ) = _render_scene_image(scene.index, scene.prompt, frame_index)
                    rendered_paths.setdefault(scene_index, {})[
                        rendered_frame_index
                    ] = image_path
                    rendered_prepared_prompts.setdefault(scene_index, {})[
                        rendered_frame_index
                    ] = prepared
                    completed += 1
                    self._emit_progress(
                        "📷 Image generation progress",
                        current=completed,
                        total=total_images,
                        elapsed_seconds=elapsed,
                    )
        else:
            self._status(f"🧵 Using {worker_count} workers for image generation")
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        _render_scene_image, scene.index, scene.prompt, frame_index
                    ): (scene.index, frame_index)
                    for scene in scenes
                    for frame_index in range(scene_images_per_scene)
                }
                for future in as_completed(futures):
                    (
                        scene_index,
                        rendered_frame_index,
                        image_path,
                        elapsed,
                        prepared,
                    ) = future.result()
                    rendered_paths.setdefault(scene_index, {})[
                        rendered_frame_index
                    ] = image_path
                    rendered_prepared_prompts.setdefault(scene_index, {})[
                        rendered_frame_index
                    ] = prepared
                    completed += 1
                    self._emit_progress(
                        "📷 Image generation progress",
                        current=completed,
                        total=total_images,
                        elapsed_seconds=elapsed,
                    )

        scene_image_sequences = [
            [frame_paths[frame_index] for frame_index in sorted(frame_paths)]
            for _, frame_paths in sorted(rendered_paths.items())
        ]
        image_paths = [
            image_path
            for scene_sequence in scene_image_sequences
            for image_path in scene_sequence
        ]
        if rendered_prepared_prompts:
            manifest_scenes = manifest.get("scenes")
            if isinstance(manifest_scenes, list):
                for scene_dict in manifest_scenes:
                    if not isinstance(scene_dict, dict):
                        continue
                    idx = scene_dict.get("index")
                    if not isinstance(idx, int):
                        continue
                    prompts_for_scene = rendered_prepared_prompts.get(idx)
                    if not prompts_for_scene:
                        continue
                    ordered_prompts = [
                        prompts_for_scene[frame_index]
                        for frame_index in sorted(prompts_for_scene)
                    ]
                    scene_dict["prepared_prompts"] = ordered_prompts
                    scene_dict["prepared_prompt"] = ordered_prompts[0]
        images = manifest.get("images")
        if isinstance(images, list):
            images.clear()
            images.extend(str(path) for path in image_paths)
            self._write_manifest(run_dir, manifest)

        manifest["status"] = "assembling_video"
        intro_card: CinematicIntroCard | None = None
        if cinematic_intro:
            intro_card = self._build_cinematic_intro_card(
                narration_text=narration_text, duration_seconds=cinematic_intro_duration
            )
            manifest["cinematic_intro"] = {
                "enabled": True,
                "title": intro_card.title,
                "description": intro_card.description,
                "duration_seconds": intro_card.duration_seconds,
            }
        else:
            manifest["cinematic_intro"] = {"enabled": False}
        manifest["television_overlay_effects"] = television_overlay_effects
        self._write_manifest(run_dir, manifest)
        self._status("🎬 Assembling video with ffmpeg")
        final_path = self._media.render_video(
            images=scene_image_sequences,
            scenes=scenes,
            audio_path=audio_path,
            output_path=output_path,
            work_dir=run_dir,
            cinematic_intro=intro_card,
            cinematic_transitions=cinematic_transitions,
            television_overlay_effects=television_overlay_effects,
        )
        manifest["output"] = str(final_path)
        manifest["audio"] = str(audio_path)
        manifest["duration_seconds"] = duration_seconds
        manifest["narration_text"] = narration_text
        if self._chunk_ensemble_scores:
            manifest["chunk_ensemble_scores"] = self._chunk_ensemble_scores
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._write_manifest(run_dir, manifest)
        self._status("✅ Video generation complete")
        return final_path

    def _build_cinematic_intro_card(
        self, *, narration_text: str, duration_seconds: float
    ) -> CinematicIntroCard:
        self._status("🎞️ Generating cinematic intro title and description")
        resolved_duration = max(2.0, float(duration_seconds))
        fallback = CinematicIntroCard(
            title="Tonight on Accidental Genius",
            description=(
                "A deeply serious brief delivered with enough chaos to keep everyone awake."
            ),
            duration_seconds=resolved_duration,
        )

        try:
            prompt = self._build_cinematic_intro_prompt(narration_text=narration_text)
            raw = self._gateway.generate_text(prompt)
            payload = self._extract_json_payload(raw)
            if payload is None:
                return fallback

            title = self._normalize_cinematic_line(str(payload.get("title", "")))
            description = self._normalize_cinematic_line(
                str(payload.get("description", ""))
            )
            if not title:
                title = fallback.title
            if not description:
                description = fallback.description

            return CinematicIntroCard(
                title=self._truncate_words(title, max_words=12, max_chars=86),
                description=self._truncate_words(
                    description, max_words=24, max_chars=170
                ),
                duration_seconds=resolved_duration,
            )
        except Exception as exc:
            self._status(
                f"⚠️ Cinematic intro copy generation failed; using fallback: {exc}"
            )
            return fallback

    def _build_cinematic_intro_prompt(self, *, narration_text: str) -> str:
        snippet = narration_text.strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200].rstrip() + "..."
        return (
            "You are writing a cinematic title card for a short-form video. "
            "Return valid JSON only with this exact schema: "
            '{"title": "...", "description": "..."}. '
            "Requirements: title must be ironic, witty, and comedic while staying clean for broad audiences; "
            "4 to 10 words; no hashtags; no emoji; no profanity. "
            "Description must be one sentence, ironic/witty/comedic, and under 150 characters. "
            "Do not include quotes around title or description text. "
            "Narration/transcript context:\n"
            f"{snippet}"
        )

    def _extract_json_payload(self, text: str) -> dict[str, object] | None:
        candidate = text.strip()
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            return payload
        return None

    def _normalize_cinematic_line(self, text: str) -> str:
        return " ".join(text.strip().strip('"').strip("'").split())

    def _truncate_words(self, text: str, *, max_words: int, max_chars: int) -> str:
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words])
        if len(text) > max_chars:
            text = text[: max_chars - 1].rstrip() + "..."
        return text

    def _build_scene_frame_prompt(
        self,
        *,
        scene_prompt: str,
        scene_index: int,
        total_scenes: int,
        frame_index: int,
        frames_per_scene: int,
        visual_intensity: str | None = None,
    ) -> str:
        prepared = self._planner.prepare_image_prompt(
            scene_prompt, scene_index=scene_index - 1, total_scenes=total_scenes
        )

        # Apply visual intensity style guidance if provided
        intensity_guidance = ""
        if visual_intensity:
            intensity_guidance_map = {
                "restrained": "Use muted, desaturated color palette, subtle lighting contrasts, minimal motion energy, serene composition.",
                "balanced": "Maintain balanced composition with moderate color saturation and normal lighting contrasts.",
                "expressive": "Use dynamic camera angles, expressive character gestures, varied composition depth, engaging emotional intensity.",
                "vivid": "Use saturated, vivid colors, dramatic lighting with strong contrasts, dynamic framing, high visual energy throughout.",
            }
            intensity_guidance = intensity_guidance_map.get(
                visual_intensity.lower(), ""
            )
            if intensity_guidance:
                intensity_guidance = (
                    f" Style intensity ({visual_intensity}): {intensity_guidance}"
                )

        if frames_per_scene <= 1:
            return prepared + intensity_guidance

        variation_cues = (
            "slight camera angle shift",
            "small expression change",
            "subtle gesture progression",
            "gentle lighting variation",
            "minor background parallax",
        )
        cue = variation_cues[frame_index % len(variation_cues)]
        return (
            f"{prepared}. Keep exact same scene continuity, character identity, wardrobe, "
            f"location, and visual style. Frame {frame_index + 1}/{frames_per_scene} with {cue}."
            f"{intensity_guidance}"
        )

    def _resolve_video_prompt_plan(
        self,
        *,
        narration_text: str,
        video_prompt: str | None,
        generate_video_prompt: bool,
        duration_seconds: float | None = None,
    ) -> VideoPromptPlan:
        if video_prompt:
            self._status("🧪 Preclassifying transcript for visual planning")
            preclass_plan = self._planner.generate_video_prompt_plan(
                narration_text=narration_text, duration_seconds=duration_seconds
            )
            return VideoPromptPlan(
                video_prompt=video_prompt,
                preclassification=preclass_plan.preclassification,
            )
        if generate_video_prompt:
            self._status("🧪 Preclassifying transcript for visual planning")
            self._status("🪄 Generating video prompt from narration")
            return self._planner.generate_video_prompt_plan(
                narration_text=narration_text, duration_seconds=duration_seconds
            )
        raise ValueError(
            "video_prompt is required unless generate_video_prompt is enabled"
        )

    def _serialize_preclassification(
        self, preclassification: VideoPromptPreclassification | None
    ) -> dict[str, object] | None:
        if preclassification is None:
            return None

        payload: dict[str, object] = {
            "mood": preclassification.mood,
            "has_foul_language": preclassification.has_foul_language,
            "word_count": preclassification.word_count,
            "sentence_count": preclassification.sentence_count,
            "truthfulness_assessment": {
                "label": preclassification.truthfulness_assessment.label,
                "confidence_score": preclassification.truthfulness_assessment.confidence_score,
                "reason": preclassification.truthfulness_assessment.reason,
            },
            "interaction_style_assessment": {
                "formality": {
                    "label": preclassification.interaction_style_assessment.formality.label,
                    "confidence_score": preclassification.interaction_style_assessment.formality.confidence_score,
                    "reason": preclassification.interaction_style_assessment.formality.reason,
                },
                "certainty_hedging": {
                    "label": preclassification.interaction_style_assessment.certainty_hedging.label,
                    "confidence_score": preclassification.interaction_style_assessment.certainty_hedging.confidence_score,
                    "reason": preclassification.interaction_style_assessment.certainty_hedging.reason,
                },
                "persuasion_intent": {
                    "label": preclassification.interaction_style_assessment.persuasion_intent.label,
                    "confidence_score": preclassification.interaction_style_assessment.persuasion_intent.confidence_score,
                    "reason": preclassification.interaction_style_assessment.persuasion_intent.reason,
                },
                "claim_density": {
                    "label": preclassification.interaction_style_assessment.claim_density.label,
                    "confidence_score": preclassification.interaction_style_assessment.claim_density.confidence_score,
                    "reason": preclassification.interaction_style_assessment.claim_density.reason,
                },
                "speaker_sentiment": [
                    {
                        "speaker": item.speaker,
                        "sentiment": item.sentiment,
                        "confidence_score": item.confidence_score,
                        "reason": item.reason,
                    }
                    for item in preclassification.interaction_style_assessment.speaker_sentiment
                ],
            },
        }

        if preclassification.fact_check_assessment is not None:
            payload["fact_check_assessment"] = {
                "label": preclassification.fact_check_assessment.label,
                "confidence_score": preclassification.fact_check_assessment.confidence_score,
                "reason": preclassification.fact_check_assessment.reason,
            }
        if preclassification.aggression_assessment is not None:
            payload["aggression_assessment"] = {
                "label": preclassification.aggression_assessment.label,
                "confidence_score": preclassification.aggression_assessment.confidence_score,
                "reason": preclassification.aggression_assessment.reason,
            }
        if preclassification.social_score_assessment is not None:
            payload["social_score_assessment"] = {
                "prosocial_antisocial": {
                    "label": preclassification.social_score_assessment.prosocial_antisocial.label,
                    "confidence_score": preclassification.social_score_assessment.prosocial_antisocial.confidence_score,
                    "reason": preclassification.social_score_assessment.prosocial_antisocial.reason,
                },
                "cohesion_divisiveness": {
                    "label": preclassification.social_score_assessment.cohesion_divisiveness.label,
                    "confidence_score": preclassification.social_score_assessment.cohesion_divisiveness.confidence_score,
                    "reason": preclassification.social_score_assessment.cohesion_divisiveness.reason,
                },
                "norm_alignment": {
                    "label": preclassification.social_score_assessment.norm_alignment.label,
                    "confidence_score": preclassification.social_score_assessment.norm_alignment.confidence_score,
                    "reason": preclassification.social_score_assessment.norm_alignment.reason,
                },
                "composite_social_score": preclassification.social_score_assessment.composite_social_score,
                "composite_label": preclassification.social_score_assessment.composite_label,
                "reason": preclassification.social_score_assessment.reason,
            }
        if preclassification.contemporary_alignment_assessment is not None:
            payload["contemporary_alignment_assessment"] = {
                "label": preclassification.contemporary_alignment_assessment.label,
                "confidence_score": preclassification.contemporary_alignment_assessment.confidence_score,
                "reason": preclassification.contemporary_alignment_assessment.reason,
            }
        if preclassification.propaganda_assessment is not None:
            payload["propaganda_assessment"] = {
                "label": preclassification.propaganda_assessment.label,
                "confidence_score": preclassification.propaganda_assessment.confidence_score,
                "reason": preclassification.propaganda_assessment.reason,
            }
        if preclassification.conversation_insights is not None:
            payload["conversation_insights"] = {
                "conversation_type": {
                    "label": preclassification.conversation_insights.conversation_type.label,
                    "confidence_score": preclassification.conversation_insights.conversation_type.confidence_score,
                    "reason": preclassification.conversation_insights.conversation_type.reason,
                },
                "primary_goal": {
                    "label": preclassification.conversation_insights.primary_goal.label,
                    "confidence_score": preclassification.conversation_insights.primary_goal.confidence_score,
                    "reason": preclassification.conversation_insights.primary_goal.reason,
                },
                "participant_dynamic": {
                    "label": preclassification.conversation_insights.participant_dynamic.label,
                    "confidence_score": preclassification.conversation_insights.participant_dynamic.confidence_score,
                    "reason": preclassification.conversation_insights.participant_dynamic.reason,
                },
                "decision_signal": {
                    "label": preclassification.conversation_insights.decision_signal.label,
                    "confidence_score": preclassification.conversation_insights.decision_signal.confidence_score,
                    "reason": preclassification.conversation_insights.decision_signal.reason,
                },
                "conflict_level": {
                    "label": preclassification.conversation_insights.conflict_level.label,
                    "confidence_score": preclassification.conversation_insights.conflict_level.confidence_score,
                    "reason": preclassification.conversation_insights.conflict_level.reason,
                },
                "concise_summary": preclassification.conversation_insights.concise_summary,
            }
        if preclassification.communication_metrics is not None:
            payload["communication_metrics"] = {
                "profanity_word_count": preclassification.communication_metrics.profanity_word_count,
                "profanity_rate": preclassification.communication_metrics.profanity_rate,
                "profanity_per_sentence_ratio": preclassification.communication_metrics.profanity_per_sentence_ratio,
                "profanity_to_non_profanity_ratio": preclassification.communication_metrics.profanity_to_non_profanity_ratio,
                "words_per_minute": preclassification.communication_metrics.words_per_minute,
                "average_words_per_sentence": preclassification.communication_metrics.average_words_per_sentence,
                "sentence_complexity_score": preclassification.communication_metrics.sentence_complexity_score,
                "sentence_complexity_label": preclassification.communication_metrics.sentence_complexity_label,
                "communication_capability_score": preclassification.communication_metrics.communication_capability_score,
                "communication_capability_label": preclassification.communication_metrics.communication_capability_label,
                "communication_notes": preclassification.communication_metrics.communication_notes,
            }
        if preclassification.ensemble_scorecard is not None:
            payload["ensemble_scorecard"] = {
                "weighted_risk_score": preclassification.ensemble_scorecard.weighted_risk_score,
                "risk_level": preclassification.ensemble_scorecard.risk_level,
                "recommended_visual_intensity": preclassification.ensemble_scorecard.recommended_visual_intensity,
                "signals": [
                    {
                        "source": signal.source,
                        "model": signal.model,
                        "label": signal.label,
                        "confidence_score": signal.confidence_score,
                        "normalized_risk": signal.normalized_risk,
                        "weight": signal.weight,
                        "reason": signal.reason,
                    }
                    for signal in preclassification.ensemble_scorecard.signals
                ],
                "warnings": preclassification.ensemble_scorecard.warnings,
            }

        return payload

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
        diarization_speaker_count: int | None,
        diarization_min_speakers: int | None,
        diarization_max_speakers: int | None,
        speaker_dominance_threshold: float,
        content_safety_enabled: bool,
        content_safety_filter: bool,
        content_safety_threshold: float,
        content_safety_model: str | None,
        transcribe_workers: int,
        profanity_words: set[str] | None = None,
    ) -> str:
        self._last_content_safety_report = None
        report: dict[str, object] | None = None
        if content_safety_enabled:
            report = {
                "enabled": True,
                "filter_enabled": content_safety_filter,
                "threshold": content_safety_threshold,
                "model": (
                    content_safety_model or "cardiffnlp/twitter-roberta-base-offensive"
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
                transcript = self._gateway.transcribe_audio_with_speakers(
                    audio_path,
                    speaker_count=diarization_speaker_count,
                    min_speakers=diarization_min_speakers,
                    max_speakers=diarization_max_speakers,
                    speaker_dominance_threshold=speaker_dominance_threshold,
                )
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
                        profanity_words=profanity_words,
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
                transcript = self._gateway.transcribe_audio_with_speakers(
                    audio_path,
                    speaker_count=diarization_speaker_count,
                    min_speakers=diarization_min_speakers,
                    max_speakers=diarization_max_speakers,
                    speaker_dominance_threshold=speaker_dominance_threshold,
                )
                if content_safety_enabled:
                    if report is not None:
                        report["full_audio"] = self._evaluate_content_safety(
                            text=transcript,
                            segment_name=audio_path.name,
                            content_safety_threshold=content_safety_threshold,
                            content_safety_model=content_safety_model,
                            profanity_words=profanity_words,
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
            self._chunk_ensemble_scores = []
            total_start = perf_counter()
            for chunk_idx, chunk_path in enumerate(chunks, start=1):
                self._status(
                    f"  Chunk {chunk_idx}/{len(chunks)}: diarizing and transcribing"
                )
                chunk_start = perf_counter()
                text = self._gateway.transcribe_audio_with_speakers(
                    chunk_path,
                    speaker_count=diarization_speaker_count,
                    min_speakers=diarization_min_speakers,
                    max_speakers=diarization_max_speakers,
                    speaker_dominance_threshold=speaker_dominance_threshold,
                )
                elapsed = perf_counter() - chunk_start
                self._emit_progress(
                    "Diarization chunk progress",
                    current=chunk_idx,
                    total=len(chunks),
                    elapsed_seconds=elapsed,
                )

                # Compute chunk-level ensemble scorecard
                chunk_scorecard = self._planner.compute_chunk_ensemble_scorecard(text)
                self._chunk_ensemble_scores.append(
                    {
                        "chunk_index": chunk_idx,
                        "text_length": len(text),
                        "weighted_risk_score": chunk_scorecard.weighted_risk_score,
                        "risk_level": chunk_scorecard.risk_level,
                        "recommended_visual_intensity": chunk_scorecard.recommended_visual_intensity,
                    }
                )

                if content_safety_enabled:
                    evaluation = self._evaluate_content_safety(
                        text=text,
                        segment_name=chunk_path.name,
                        content_safety_threshold=content_safety_threshold,
                        content_safety_model=content_safety_model,
                        profanity_words=profanity_words,
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
                        profanity_words=profanity_words,
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
                    profanity_words=profanity_words,
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
        profanity_words: set[str] | None = None,
    ) -> dict[str, object]:
        if not text.strip():
            return {
                "segment": segment_name,
                "text_length": 0,
                "flagged": False,
                "ml_flagged": False,
                "lexicon_flagged": False,
                "lexicon_matched": [],
                "unsafe_score": 0.0,
                "top_label": "",
                "top_score": 0.0,
                "labels": [],
            }

        moderation = self._gateway.classify_content_safety(
            text, model=content_safety_model
        )
        unsafe_score = float(moderation.get("unsafe_score", 0.0) or 0.0)
        ml_flagged = unsafe_score >= content_safety_threshold

        lexicon_matched = scan_text_for_profanity(text, profanity_words)
        lexicon_flagged = bool(lexicon_matched)

        return {
            "segment": segment_name,
            "text_length": len(text),
            "flagged": ml_flagged or lexicon_flagged,
            "ml_flagged": ml_flagged,
            "lexicon_flagged": lexicon_flagged,
            "lexicon_matched": lexicon_matched,
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

    def _synthesize_long_speech(
        self, text: str, destination: Path, tmp_dir: Path
    ) -> None:
        """Synthesize *text* to *destination*, chunking at sentence boundaries.

        Many TTS APIs enforce a ~500-character limit per request.  This helper
        splits the text into chunks of at most 490 characters (breaking only at
        sentence endings) and concatenates the resulting audio segments so that
        the full text is always synthesised regardless of length.
        """
        import re as _re

        max_chars = 490
        sentences = _re.split(r"(?<=\.)\s+", text)
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = (current + " " + sentence).strip() if current else sentence
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If a single sentence is itself > max_chars, split it hard
                if len(sentence) > max_chars:
                    for i in range(0, len(sentence), max_chars):
                        chunks.append(sentence[i : i + max_chars])
                    current = ""
                else:
                    current = sentence
        if current:
            chunks.append(current)

        if len(chunks) <= 1:
            self._gateway.synthesize_speech(text, destination)
            return

        chunk_paths: list[Path] = []
        for i, chunk in enumerate(chunks):
            chunk_path = tmp_dir / f"_tts_chunk_{destination.stem}_{i:03d}.wav"
            self._gateway.synthesize_speech(chunk, chunk_path)
            chunk_paths.append(chunk_path)

        concat_txt = tmp_dir / f"_tts_concat_{destination.stem}.txt"
        concat_txt.write_text(
            "\n".join(f"file '{p.as_posix()}'" for p in chunk_paths), encoding="utf-8"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_txt),
                "-c:a",
                "pcm_s16le",
                str(destination),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def _ffmpeg_generate_silence(
        self, output: Path, *, duration_seconds: float
    ) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=24000:cl=mono",
                "-t",
                str(duration_seconds),
                "-c:a",
                "pcm_s16le",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def _ffmpeg_normalize_audio(self, input_path: Path, output_path: Path) -> None:
        """Re-encode to a common PCM WAV format so concat works without gaps."""
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-c:a",
                "pcm_s16le",
                "-ar",
                "24000",
                "-ac",
                "1",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def _ffmpeg_extract_audio_segment(
        self,
        audio_path: Path,
        output_path: Path,
        *,
        start_seconds: float,
        end_seconds: float,
    ) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-ss",
                f"{start_seconds:.3f}",
                "-to",
                f"{end_seconds:.3f}",
                "-c:a",
                "pcm_s16le",
                "-ar",
                "24000",
                "-ac",
                "1",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def _ffmpeg_extract_bleep(
        self, sfx_path: Path, output_path: Path, *, duration_seconds: float
    ) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(sfx_path),
                "-t",
                f"{max(0.1, duration_seconds):.3f}",
                "-c:a",
                "pcm_s16le",
                "-ar",
                "24000",
                "-ac",
                "1",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def _write_manifest(self, run_dir: Path, manifest: dict[str, object]) -> None:
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def _write_analysis_summary_artifact(
        self,
        *,
        run_dir: Path,
        narration_text: str,
        video_prompt: str,
        preclassification: object,
    ) -> Path | None:
        if not isinstance(preclassification, dict):
            return None

        summary_path = run_dir / "analysis_summary.json"
        sentence_count = 0
        if narration_text.strip():
            sentence_count = len(
                [
                    part
                    for part in re.split(r"(?<=[.!?])\s+", narration_text.strip())
                    if part.strip()
                ]
            )

        summary_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "narration": {
                "word_count": len(narration_text.split()),
                "sentence_count": sentence_count,
                "excerpt": narration_text[:280].strip(),
            },
            "video_prompt": video_prompt,
            "preclassification": preclassification,
            "dimensions": {
                "truthfulness": preclassification.get("truthfulness_assessment"),
                "fact_check": preclassification.get("fact_check_assessment"),
                "aggression": preclassification.get("aggression_assessment"),
                "social_score": preclassification.get("social_score_assessment"),
                "contemporary_alignment": preclassification.get(
                    "contemporary_alignment_assessment"
                ),
                "propaganda_alignment": preclassification.get("propaganda_assessment"),
                "interaction_style": preclassification.get(
                    "interaction_style_assessment"
                ),
                "conversation_insights": preclassification.get("conversation_insights"),
                "communication_metrics": preclassification.get("communication_metrics"),
                "ensemble_scorecard": preclassification.get("ensemble_scorecard"),
            },
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        return summary_path

    def _attach_feedback_annotations(
        self,
        *,
        narration_text: str,
        preclassification_data: dict[str, object],
        feedback_tier: str,
        enhanced_rationale: bool,
    ) -> dict[str, object]:
        payload = dict(preclassification_data)
        if enhanced_rationale:
            self._status(
                "🧠 Running optional enhanced pre-classification rationale pass"
            )
        if hasattr(self._planner, "generate_feedback_annotations"):
            feedback = self._planner.generate_feedback_annotations(
                narration_text=narration_text,
                preclassification_data=payload,
                feedback_tier=feedback_tier,
                enhanced_rationale=enhanced_rationale,
            )
            if isinstance(feedback, dict):
                payload["feedback"] = feedback
        return payload

    def _append_feedback_summary_lines(
        self,
        *,
        summary_parts: list[str],
        preclassification_data: dict[str, object],
        feedback_tier: str,
    ) -> None:
        feedback = preclassification_data.get("feedback")
        if not isinstance(feedback, dict):
            return

        tier = str(feedback.get("tier", feedback_tier)).strip().lower() or "standard"
        if tier == "minimal":
            return

        confidence_flags = feedback.get("confidence_flags")
        if isinstance(confidence_flags, list) and confidence_flags:
            summary_parts.append(
                "Confidence flags: "
                + "; ".join(str(item).strip() for item in confidence_flags if str(item).strip())
                + "."
            )

        if tier != "expert":
            return

        contradiction_flags = feedback.get("contradiction_flags")
        if isinstance(contradiction_flags, list) and contradiction_flags:
            summary_parts.append(
                "Contradiction flags: "
                + "; ".join(
                    str(item).strip() for item in contradiction_flags if str(item).strip()
                )
                + "."
            )

        recommendations = feedback.get("recommendations")
        if isinstance(recommendations, list) and recommendations:
            summary_parts.append(
                "Recommendations: "
                + "; ".join(str(item).strip() for item in recommendations if str(item).strip())
                + "."
            )

        enhanced_pass = feedback.get("enhanced_pass")
        if isinstance(enhanced_pass, dict):
            observations = enhanced_pass.get("global_observations")
            if isinstance(observations, list) and observations:
                summary_parts.append(
                    "Enhanced observations: "
                    + "; ".join(str(item).strip() for item in observations if str(item).strip())
                    + "."
                )

    def _format_preclassification_rollup(self, preclassification: object) -> str:
        if not isinstance(preclassification, dict):
            return ""

        lines = ["High-fidelity rollup:"]
        for key, title in (
            ("truthfulness_assessment", "Truthfulness"),
            ("fact_check_assessment", "Fact-check"),
            ("aggression_assessment", "Aggression"),
            ("contemporary_alignment_assessment", "Contemporary alignment"),
            ("propaganda_assessment", "Propaganda alignment"),
        ):
            item = preclassification.get(key)
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            confidence = item.get("confidence_score")
            if not label:
                continue
            line = f"- {title}: {label}"
            if isinstance(confidence, (int, float)):
                line += f" ({float(confidence):.2f})"
            lines.append(line)

        social = preclassification.get("social_score_assessment")
        if isinstance(social, dict):
            label = str(social.get("composite_label", "")).strip()
            score = social.get("composite_social_score")
            if label:
                line = f"- Social score: {label}"
                if isinstance(score, (int, float)):
                    line += f" ({float(score):.2f})"
                lines.append(line)

        ensemble = preclassification.get("ensemble_scorecard")
        if isinstance(ensemble, dict):
            risk_level = str(ensemble.get("risk_level", "")).strip()
            weighted = ensemble.get("weighted_risk_score")
            if risk_level:
                line = f"- Ensemble risk: {risk_level}"
                if isinstance(weighted, (int, float)):
                    line += f" ({float(weighted):.2f})"
                lines.append(line)

        feedback = preclassification.get("feedback")
        if isinstance(feedback, dict):
            tier = str(feedback.get("tier", "")).strip()
            if tier:
                lines.append(f"- Feedback tier: {tier}")
            confidence_flags = feedback.get("confidence_flags")
            if isinstance(confidence_flags, list) and confidence_flags:
                lines.append(
                    "- Confidence flags: "
                    + "; ".join(
                        str(item).strip()
                        for item in confidence_flags
                        if str(item).strip()
                    )
                )
            recommendations = feedback.get("recommendations")
            if isinstance(recommendations, list) and recommendations:
                lines.append(
                    "- Recommendations: "
                    + "; ".join(
                        str(item).strip()
                        for item in recommendations
                        if str(item).strip()
                    )
                )

        return "\n".join(lines)

    def _load_scenes_from_manifest(self, manifest: dict[str, object]) -> list[Scene]:
        raw_scenes = manifest.get("scenes")
        if not isinstance(raw_scenes, list) or not raw_scenes:
            raise ValueError("Manifest is missing scene metadata")

        scenes: list[Scene] = []
        for item in raw_scenes:
            if not isinstance(item, dict):
                raise ValueError("Manifest contains invalid scene entry")

            transition: CinematicTransition | None = None
            raw_transition = item.get("transition_to_next")
            if isinstance(raw_transition, dict):
                transition = CinematicTransition(
                    transition_type=str(
                        raw_transition.get("transition_type", "dissolve")
                    ),
                    duration_frames=max(
                        1,
                        int(
                            self._coerce_float(
                                raw_transition.get("duration_frames"), 12
                            )
                        ),
                    ),
                    intensity=str(raw_transition.get("intensity", "subtle")),
                    visual_cue=str(raw_transition.get("visual_cue", "")),
                    semantic_bridge=str(raw_transition.get("semantic_bridge", "")),
                )

            scenes.append(
                Scene(
                    index=int(self._coerce_float(item.get("index"), 0)),
                    prompt=str(item.get("prompt", "")).strip() or "Recovered scene",
                    duration_seconds=max(
                        1.0 / max(1, self._config.fps),
                        self._coerce_float(item.get("duration_seconds"), 1.0),
                    ),
                    transition_to_next=transition,
                )
            )

        scenes.sort(key=lambda scene: scene.index)
        if any(scene.index <= 0 for scene in scenes):
            raise ValueError("Manifest scene indexes must be positive integers")
        return scenes

    def _load_scene_images_from_manifest(
        self, *, run_dir: Path, manifest: dict[str, object], scenes: list[Scene]
    ) -> list[list[Path]]:
        scene_paths: dict[int, list[Path]] = {}

        raw_images = manifest.get("images")
        if isinstance(raw_images, list) and raw_images:
            for value in raw_images:
                if not isinstance(value, str):
                    continue
                candidate = self._resolve_manifest_path(run_dir=run_dir, raw_path=value)
                if not candidate.exists():
                    continue
                scene_index = self._extract_scene_index(candidate)
                if scene_index is None:
                    continue
                scene_paths.setdefault(scene_index, []).append(candidate)

        for scene in scenes:
            if scene.index in scene_paths:
                continue
            fallback = sorted(
                (run_dir / "images").glob(f"scene_{scene.index:02d}_frame_*.png")
            )
            if fallback:
                scene_paths[scene.index] = fallback

        scene_image_sequences: list[list[Path]] = []
        for scene in scenes:
            image_paths = scene_paths.get(scene.index)
            if not image_paths:
                raise ValueError(
                    f"No scene images found for scene {scene.index} in run directory"
                )
            ordered_paths = sorted(image_paths, key=self._extract_frame_index)
            scene_image_sequences.append(ordered_paths)
        return scene_image_sequences

    @staticmethod
    def _resolve_manifest_path(*, run_dir: Path, raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (run_dir / candidate).resolve()

    def _resolve_manifest_audio_path(
        self, *, run_dir: Path, manifest: dict[str, object]
    ) -> Path:
        audio_value = manifest.get("audio")
        if not isinstance(audio_value, str) or not audio_value.strip():
            raise ValueError("Manifest is missing audio path")
        return self._resolve_manifest_path(run_dir=run_dir, raw_path=audio_value)

    def _resolve_manifest_output_path(
        self, *, run_dir: Path, manifest: dict[str, object], output_path: Path | None
    ) -> Path:
        if output_path is not None:
            return output_path.expanduser().resolve()

        output_value = manifest.get("output")
        if isinstance(output_value, str) and output_value.strip():
            return self._resolve_manifest_path(run_dir=run_dir, raw_path=output_value)
        raise ValueError("Output path is required when manifest has no output field")

    @staticmethod
    def _extract_scene_index(path: Path) -> int | None:
        match = re.search(r"scene_(\d+)", path.name)
        if match is None:
            return None
        return int(match.group(1))

    @staticmethod
    def _extract_frame_index(path: Path) -> int:
        match = re.search(r"frame_(\d+)", path.name)
        if match is None:
            return 0
        return int(match.group(1))

    def _status(self, message: str) -> None:
        if self._status_callback is not None:
            normalized = message.strip()
            if normalized and not self._starts_with_emoji(normalized):
                message = f"ℹ️ {message}"
            self._status_callback(message)

    @staticmethod
    def _coerce_float(value: object, default: float = 0.0) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return default
            try:
                return float(stripped)
            except ValueError:
                return default
        return default

    @staticmethod
    def _starts_with_emoji(value: str) -> bool:
        return value.startswith(
            (
                "🎤",
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
