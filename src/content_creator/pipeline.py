from __future__ import annotations

import json
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
from content_creator.media import AudioOverlayEvent, MediaAssembler
from content_creator.planner import ScenePlanner, ScenePlan, VideoPromptPlan
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
            self._gateway, image_composition_mode=config.image_composition_mode
        )
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
        images_per_scene: int = 1,
        view_preclassification: bool = False,
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
            image_workers=image_workers,
            images_per_scene=images_per_scene,
            view_preclassification=view_preclassification,
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
            image_workers=image_workers,
            images_per_scene=images_per_scene,
            view_preclassification=view_preclassification,
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
    ) -> int:
        """Build a debug audio file illustrating each profanity detection event.

        For every event the output contains: a synthesized voice announcing the
        detected word, start/end/duration; the raw audio snippet; a synthesized
        voice saying "Profanity filter implemented"; and the exact bleep that
        production would overlay.

        Returns the number of events processed (0 if none found).
        """
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
                preclassification_data=preclassification_data,
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
                preclassification_data=preclassification_data,
            )
            if summary_text:
                self._status("🎤 Appending synthesized pre-classification summary")
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

        summary_parts.append("Begin event diagnostics.")
        return " ".join(summary_parts)

    def _build_debug_preclassification_summary(
        self,
        *,
        events: list[dict[str, object]],
        transcript_text: str | None,
        preclassification_data: dict[str, object] | None,
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
        image_workers: int = 1,
        images_per_scene: int = 1,
        view_preclassification: bool = False,
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
                "truthfulness_assessment": {
                    "label": video_prompt_plan.preclassification.truthfulness_assessment.label,
                    "confidence_score": video_prompt_plan.preclassification.truthfulness_assessment.confidence_score,
                    "reason": video_prompt_plan.preclassification.truthfulness_assessment.reason,
                },
                "interaction_style_assessment": {
                    "formality": {
                        "label": video_prompt_plan.preclassification.interaction_style_assessment.formality.label,
                        "confidence_score": video_prompt_plan.preclassification.interaction_style_assessment.formality.confidence_score,
                        "reason": video_prompt_plan.preclassification.interaction_style_assessment.formality.reason,
                    },
                    "certainty_hedging": {
                        "label": video_prompt_plan.preclassification.interaction_style_assessment.certainty_hedging.label,
                        "confidence_score": video_prompt_plan.preclassification.interaction_style_assessment.certainty_hedging.confidence_score,
                        "reason": video_prompt_plan.preclassification.interaction_style_assessment.certainty_hedging.reason,
                    },
                    "persuasion_intent": {
                        "label": video_prompt_plan.preclassification.interaction_style_assessment.persuasion_intent.label,
                        "confidence_score": video_prompt_plan.preclassification.interaction_style_assessment.persuasion_intent.confidence_score,
                        "reason": video_prompt_plan.preclassification.interaction_style_assessment.persuasion_intent.reason,
                    },
                    "claim_density": {
                        "label": video_prompt_plan.preclassification.interaction_style_assessment.claim_density.label,
                        "confidence_score": video_prompt_plan.preclassification.interaction_style_assessment.claim_density.confidence_score,
                        "reason": video_prompt_plan.preclassification.interaction_style_assessment.claim_density.reason,
                    },
                    "speaker_sentiment": [
                        {
                            "speaker": item.speaker,
                            "sentiment": item.sentiment,
                            "confidence_score": item.confidence_score,
                            "reason": item.reason,
                        }
                        for item in video_prompt_plan.preclassification.interaction_style_assessment.speaker_sentiment
                    ],
                },
            }
            if video_prompt_plan.preclassification is not None
            else None
        )
        if (
            view_preclassification
            and manifest.get("video_prompt_preclassification") is not None
        ):
            self._status(
                "🔍 Pre-classification:\n"
                + json.dumps(manifest["video_prompt_preclassification"], indent=2)
            )
        manifest["status"] = "planning_scenes"
        self._write_manifest(run_dir, manifest)
        self._status("🧠 Planning scenes from narration")
        scene_plan = self._planner.build_scenes(
            narration_text=narration_text,
            video_prompt=resolved_video_prompt,
            total_duration_seconds=duration_seconds,
        )
        scenes = scene_plan.scenes
        if video_prompt_plan.prompts is not None:
            manifest["llm_prompts"] = {
                **video_prompt_plan.prompts,
                "scene_planning": scene_plan.scene_prompt,
            }
        else:
            manifest["llm_prompts"] = {"scene_planning": scene_plan.scene_prompt}
        manifest["scenes"] = [asdict(scene) for scene in scenes]
        manifest["status"] = "scenes_planned"
        self._write_manifest(run_dir, manifest)
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
        self._write_manifest(run_dir, manifest)
        self._status("🎬 Assembling video with ffmpeg")
        final_path = self._media.render_video(
            images=scene_image_sequences,
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

    def _build_scene_frame_prompt(
        self,
        *,
        scene_prompt: str,
        scene_index: int,
        total_scenes: int,
        frame_index: int,
        frames_per_scene: int,
    ) -> str:
        prepared = self._planner.prepare_image_prompt(
            scene_prompt, scene_index=scene_index - 1, total_scenes=total_scenes
        )
        if frames_per_scene <= 1:
            return prepared

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
        )

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
