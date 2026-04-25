from __future__ import annotations

import os
import random
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import monotonic, sleep
from typing import Any

from huggingface_hub import InferenceClient
from PIL import Image

from content_creator.config import AppConfig


@dataclass(slots=True)
class AudioAsset:
    path: Path
    transcript: str
    duration_seconds: float


@dataclass(slots=True)
class TimedWord:
    word: str
    start_seconds: float
    end_seconds: float


@dataclass(slots=True)
class InferenceRetryConfig:
    max_retries: int
    base_delay_seconds: float
    max_delay_seconds: float
    jitter_seconds: float
    min_request_interval_seconds: float


class HuggingFaceGateway:
    def __init__(self, config: AppConfig):
        self._config = config
        self._client = InferenceClient(token=config.hf_token)
        self._retry = self._load_retry_config_from_env()
        self._request_lock = Lock()
        self._next_request_time = 0.0

    def generate_text(self, prompt: str) -> str:
        response = self._call_with_retries(
            operation_name="chat completion",
            call=lambda: self._client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self._config.models.llm_model,
                max_tokens=900,
                temperature=0.6,
            ),
        )
        return response.choices[0].message.content or ""

    def synthesize_speech(self, text: str, destination: Path) -> Path:
        audio_bytes = self._call_with_retries(
            operation_name="text to speech",
            call=lambda: self._client.text_to_speech(
                text, model=self._config.models.tts_model
            ),
        )
        destination.write_bytes(audio_bytes)
        return destination

    def transcribe_audio(self, audio_path: Path) -> str:
        audio_bytes = audio_path.read_bytes()
        result = self._call_with_retries(
            operation_name="speech recognition",
            call=lambda: self._client.automatic_speech_recognition(
                audio_bytes, model=self._config.models.stt_model
            ),
        )
        if isinstance(result, dict):
            return str(result.get("text", "")).strip()
        return str(result).strip()

    def transcribe_audio_with_word_timestamps(
        self, audio_path: Path
    ) -> tuple[str, list[TimedWord]]:
        audio_bytes = audio_path.read_bytes()
        result = self._call_with_retries(
            operation_name="speech recognition with word timestamps",
            call=lambda: self._client.automatic_speech_recognition(
                audio_bytes,
                model=self._config.models.stt_model,
                extra_body={"return_timestamps": "word"},
            ),
        )

        if isinstance(result, dict):
            text = str(result.get("text", "")).strip()
            chunks = result.get("chunks")
        else:
            text = str(getattr(result, "text", "") or "").strip()
            chunks = getattr(result, "chunks", None)

        words = self._parse_timed_words(chunks)
        if not text and words:
            text = " ".join(word.word for word in words).strip()
        if not words:
            raise RuntimeError(
                "Word-level timestamps are unavailable for the configured STT model. "
                "Use a Whisper model that supports return_timestamps=word, such as openai/whisper-large-v3."
            )
        return text, words

    def classify_content_safety(
        self, text: str, *, model: str | None = None
    ) -> dict[str, Any]:
        model_id = (model or "unitary/unbiased-toxic-roberta").strip()
        result = self._call_with_retries(
            operation_name="text classification",
            call=lambda: self._client.text_classification(text, model=model_id),
        )

        normalized: list[dict[str, Any]] = []
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    label = str(item.get("label", "")).strip()
                    score = float(item.get("score", 0.0) or 0.0)
                else:
                    label = str(getattr(item, "label", "")).strip()
                    score = float(getattr(item, "score", 0.0) or 0.0)
                if label:
                    normalized.append({"label": label, "score": score})
        elif isinstance(result, dict):
            label = str(result.get("label", "")).strip()
            score = float(result.get("score", 0.0) or 0.0)
            if label:
                normalized.append({"label": label, "score": score})
        else:
            label = str(getattr(result, "label", "")).strip()
            score = float(getattr(result, "score", 0.0) or 0.0)
            if label:
                normalized.append({"label": label, "score": score})

        normalized.sort(key=lambda entry: float(entry.get("score", 0.0)), reverse=True)
        top = normalized[0] if normalized else {"label": "", "score": 0.0}

        unsafe_tokens = (
            "toxic",
            "hate",
            "offensive",
            "obscene",
            "sexual",
            "nsfw",
            "profan",
            "insult",
            "threat",
            "violence",
            "self-harm",
            "harmful",
        )

        unsafe_score = 0.0
        for entry in normalized:
            label = str(entry.get("label", "")).lower()
            score = float(entry.get("score", 0.0) or 0.0)
            if any(token in label for token in unsafe_tokens):
                unsafe_score = max(unsafe_score, score)

        if unsafe_score == 0.0:
            top_label = str(top.get("label", "")).lower()
            top_score = float(top.get("score", 0.0) or 0.0)
            if any(token in top_label for token in unsafe_tokens):
                unsafe_score = top_score

        return {
            "model": model_id,
            "unsafe_score": unsafe_score,
            "top_label": str(top.get("label", "")),
            "top_score": float(top.get("score", 0.0) or 0.0),
            "labels": normalized,
        }

    def transcribe_audio_with_speakers(
        self,
        audio_path: Path,
        *,
        speaker_count: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        speaker_dominance_threshold: float | None = None,
    ) -> str:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "--preserve-speaker requires ffmpeg to extract diarized audio segments"
            )

        try:
            from pyannote.audio import Pipeline as PyannotePipeline
        except ImportError as exc:
            raise RuntimeError(
                "--preserve-speaker requires pyannote.audio. Install with: pip install pyannote.audio"
            ) from exc

        diarization_model = (
            os.getenv(
                "HF_DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1"
            ).strip()
            or "pyannote/speaker-diarization-3.1"
        )
        with tempfile.TemporaryDirectory(prefix="content_creator_diarization_") as tmp:
            temp_dir = Path(tmp)
            prepared_audio_path = temp_dir / "prepared_audio.wav"
            print("🔄 Converting source audio for diarization (mono 16kHz WAV)...")
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        str(audio_path),
                        "-vn",
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        str(prepared_audio_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print("✅ Audio conversion complete")
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                detail = f" ffmpeg error: {stderr}" if stderr else ""
                raise RuntimeError(
                    "--preserve-speaker could not decode audio input for diarization. "
                    "Please provide a valid audio file (e.g. wav, mp3, flac, m4a)."
                    f"{detail}"
                ) from exc

            try:
                print("🧠 Running speaker diarization...")
                diarizer = PyannotePipeline.from_pretrained(
                    diarization_model, use_auth_token=self._config.hf_token
                )
                if speaker_count is not None and speaker_count < 1:
                    raise ValueError("speaker_count must be >= 1")
                if min_speakers is not None and min_speakers < 1:
                    raise ValueError("min_speakers must be >= 1")
                if max_speakers is not None and max_speakers < 1:
                    raise ValueError("max_speakers must be >= 1")
                if (
                    min_speakers is not None
                    and max_speakers is not None
                    and min_speakers > max_speakers
                ):
                    raise ValueError("min_speakers cannot be greater than max_speakers")

                diarization_kwargs: dict[str, int] = {}
                if speaker_count is not None:
                    diarization_kwargs["num_speakers"] = speaker_count
                if min_speakers is not None:
                    diarization_kwargs["min_speakers"] = min_speakers
                if max_speakers is not None:
                    diarization_kwargs["max_speakers"] = max_speakers

                diarization = diarizer(str(prepared_audio_path), **diarization_kwargs)
                print("✅ Speaker diarization complete")
            except Exception as exc:
                message = str(exc)
                if "NoneType" in message and "eval" in message:
                    raise RuntimeError(
                        "--preserve-speaker could not access required pyannote models. "
                        "Ensure HF_TOKEN is set and has access to gated models: "
                        "https://huggingface.co/pyannote/speaker-diarization-3.1 and "
                        "https://huggingface.co/pyannote/segmentation-3.0"
                    ) from exc
                raise

            utterances: list[tuple[str, str, float]] = []
            for index, (segment, _, speaker) in enumerate(
                diarization.itertracks(yield_label=True), start=1
            ):
                start_time = float(getattr(segment, "start", 0.0))
                end_time = float(getattr(segment, "end", 0.0))
                if end_time <= start_time:
                    continue

                chunk_path = temp_dir / f"speaker_{index:04d}.wav"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-v",
                        "error",
                        "-ss",
                        f"{start_time:.3f}",
                        "-to",
                        f"{end_time:.3f}",
                        "-i",
                        str(prepared_audio_path),
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        str(chunk_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )

                text = self.transcribe_audio(chunk_path).strip()
                if text:
                    utterances.append((str(speaker), text, end_time - start_time))

        resolved_speaker_dominance_threshold = (
            self._resolve_speaker_dominance_threshold(speaker_dominance_threshold)
        )

        should_auto_collapse_primary = (
            speaker_count is None and min_speakers is None and max_speakers is None
        )
        if should_auto_collapse_primary:
            merge_inputs = self._collapse_to_primary_speaker(
                utterances, dominance_threshold=resolved_speaker_dominance_threshold
            )
        else:
            merge_inputs = [(speaker, text) for speaker, text, _ in utterances]

        lines = self._merge_speaker_utterances(merge_inputs)
        if not lines:
            return self.transcribe_audio(audio_path)
        return "\n".join(lines).strip()

    def _collapse_to_primary_speaker(
        self, utterances: list[tuple[str, str, float]], *, dominance_threshold: float
    ) -> list[tuple[str, str]]:
        if not utterances:
            return []

        speaker_durations: dict[str, float] = {}
        total_duration = 0.0
        for speaker, _, duration in utterances:
            safe_duration = max(0.0, float(duration))
            speaker_durations[speaker] = (
                speaker_durations.get(speaker, 0.0) + safe_duration
            )
            total_duration += safe_duration

        if total_duration <= 0.0:
            return [(speaker, text) for speaker, text, _ in utterances]

        primary_speaker, primary_duration = max(
            speaker_durations.items(), key=lambda item: item[1]
        )
        dominance = primary_duration / total_duration

        if len(speaker_durations) == 1 or dominance < dominance_threshold:
            return [(speaker, text) for speaker, text, _ in utterances]
        return [(primary_speaker, text) for _, text, _ in utterances]

    def _resolve_speaker_dominance_threshold(self, value: float | None) -> float:
        if value is not None:
            if value < 0.0 or value > 1.0:
                raise ValueError(
                    "speaker_dominance_threshold must be between 0.0 and 1.0"
                )
            return value

        raw_value = os.getenv("HF_SPEAKER_DOMINANCE_THRESHOLD", "").strip()
        if not raw_value:
            return 0.9

        try:
            resolved = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                "HF_SPEAKER_DOMINANCE_THRESHOLD must be a float between 0.0 and 1.0"
            ) from exc

        if resolved < 0.0 or resolved > 1.0:
            raise ValueError(
                "HF_SPEAKER_DOMINANCE_THRESHOLD must be a float between 0.0 and 1.0"
            )
        return resolved

    def _merge_speaker_utterances(self, utterances: list[tuple[str, str]]) -> list[str]:
        merged: list[list[str]] = []
        for speaker, text in utterances:
            clean_speaker = speaker.strip()
            clean_text = text.strip()
            if not clean_speaker or not clean_text:
                continue
            if merged and merged[-1][0] == clean_speaker:
                merged[-1][1] = f"{merged[-1][1]} {clean_text}".strip()
                continue
            merged.append([clean_speaker, clean_text])
        return [f"{speaker}: {text}" for speaker, text in merged]

    def _parse_timed_words(self, chunks: Any) -> list[TimedWord]:
        if not isinstance(chunks, list):
            return []

        timed_words: list[TimedWord] = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                chunk_text = str(chunk.get("text", "")).strip()
                timestamp = chunk.get("timestamp") or chunk.get("timestamps")
            else:
                chunk_text = str(getattr(chunk, "text", "") or "").strip()
                timestamp = getattr(chunk, "timestamp", None) or getattr(
                    chunk, "timestamps", None
                )

            if not chunk_text or not isinstance(timestamp, (list, tuple)):
                continue
            if len(timestamp) != 2:
                continue

            start, end = timestamp
            if start is None or end is None:
                continue

            try:
                start_value = float(start)
                end_value = float(end)
            except (TypeError, ValueError):
                continue

            if end_value <= start_value:
                continue

            split_words = [token for token in chunk_text.split() if token.strip()]
            if not split_words:
                continue

            duration = end_value - start_value
            unit_duration = duration / len(split_words)
            for index, word in enumerate(split_words):
                token_start = start_value + (index * unit_duration)
                token_end = start_value + ((index + 1) * unit_duration)
                timed_words.append(
                    TimedWord(
                        word=word, start_seconds=token_start, end_seconds=token_end
                    )
                )

        return timed_words

    def generate_image(self, prompt: str, destination: Path) -> Path:
        extra_kwargs: dict[str, Any] = {}
        if self._config.image_negative_prompt.strip():
            extra_kwargs["negative_prompt"] = self._config.image_negative_prompt
        image = self._call_with_retries(
            operation_name="text to image",
            call=lambda: self._client.text_to_image(
                prompt,
                model=self._config.models.image_model,
                width=self._config.width,
                height=self._config.height,
                **extra_kwargs,
            ),
        )
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Expected a PIL image from Hugging Face, received {type(image)!r}"
            )
        image.save(destination)
        return destination

    def _call_with_retries(self, *, operation_name: str, call: Any) -> Any:
        last_error: Exception | None = None
        max_attempts = self._retry.max_retries + 1

        for attempt in range(1, max_attempts + 1):
            self._wait_for_request_slot()
            try:
                return call()
            except Exception as exc:
                last_error = exc
                if attempt >= max_attempts or not self._is_retryable_error(exc):
                    break
                delay_seconds = self._compute_retry_delay_seconds(exc, attempt)
                sleep(delay_seconds)

        if last_error is None:
            raise RuntimeError(f"Unexpected retry state for {operation_name}")

        raise RuntimeError(
            "Hugging Face inference request failed after "
            f"{max_attempts} attempts ({operation_name}): {last_error}"
        ) from last_error

    def _wait_for_request_slot(self) -> None:
        wait_seconds = 0.0
        with self._request_lock:
            now = monotonic()
            if self._next_request_time > now:
                wait_seconds = self._next_request_time - now

            interval = self._retry.min_request_interval_seconds
            next_allowed = max(now, self._next_request_time) + interval
            self._next_request_time = next_allowed

        if wait_seconds > 0.0:
            sleep(wait_seconds)

    def _compute_retry_delay_seconds(self, exc: Exception, attempt: int) -> float:
        header_delay = self._extract_server_delay_seconds(exc)
        if header_delay is not None:
            jitter = random.uniform(0.0, self._retry.jitter_seconds)
            return min(self._retry.max_delay_seconds, header_delay + jitter)

        backoff = self._retry.base_delay_seconds * (2 ** max(0, attempt - 1))
        jitter = random.uniform(0.0, self._retry.jitter_seconds)
        return min(self._retry.max_delay_seconds, backoff + jitter)

    def _extract_server_delay_seconds(self, exc: Exception) -> float | None:
        headers = self._extract_response_headers(exc)
        if not headers:
            return None

        retry_after_raw = headers.get("retry-after")
        if retry_after_raw:
            try:
                return max(0.0, float(retry_after_raw))
            except ValueError:
                pass

        rate_limit_raw = headers.get("ratelimit")
        if not rate_limit_raw:
            return None

        # Example format: "api|pages|resolvers";r=123;t=42
        match = re.search(r"(?:^|;)\s*t=(\d+(?:\.\d+)?)\b", rate_limit_raw)
        if not match:
            return None

        try:
            return max(0.0, float(match.group(1)))
        except ValueError:
            return None

    def _is_retryable_error(self, exc: Exception) -> bool:
        status_code = self._extract_status_code(exc)
        if status_code is not None:
            if status_code == 429:
                return True
            if status_code in {408, 500, 502, 503, 504}:
                return True
            return False

        error_type = type(exc).__name__.lower()
        if any(
            token in error_type
            for token in ("timeout", "connection", "tempor", "network")
        ):
            return True

        message = str(exc).lower()
        transient_markers = (
            "too many requests",
            "rate limit",
            "429",
            "timed out",
            "timeout",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
        )
        return any(marker in message for marker in transient_markers)

    def _extract_status_code(self, exc: Exception) -> int | None:
        direct_status = getattr(exc, "status_code", None)
        if isinstance(direct_status, int):
            return direct_status

        response = getattr(exc, "response", None) or getattr(
            exc, "server_response", None
        )
        if response is None:
            return None

        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        return None

    def _extract_response_headers(self, exc: Exception) -> dict[str, str]:
        response = getattr(exc, "response", None) or getattr(
            exc, "server_response", None
        )
        if response is None:
            return {}

        raw_headers = getattr(response, "headers", None)
        if raw_headers is None:
            return {}

        try:
            return {str(key).lower(): str(value) for key, value in raw_headers.items()}
        except Exception:
            return {}

    def _load_retry_config_from_env(self) -> InferenceRetryConfig:
        return InferenceRetryConfig(
            max_retries=self._read_env_int("HF_INFERENCE_MAX_RETRIES", default=5),
            base_delay_seconds=self._read_env_float(
                "HF_INFERENCE_BASE_DELAY_SECONDS", default=1.0
            ),
            max_delay_seconds=self._read_env_float(
                "HF_INFERENCE_MAX_DELAY_SECONDS", default=30.0
            ),
            jitter_seconds=self._read_env_float(
                "HF_INFERENCE_JITTER_SECONDS", default=0.35
            ),
            min_request_interval_seconds=self._read_env_float(
                "HF_INFERENCE_MIN_INTERVAL_SECONDS", default=0.25
            ),
        )

    def _read_env_int(self, var_name: str, *, default: int) -> int:
        raw = os.getenv(var_name, "").strip()
        if not raw:
            return default

        try:
            value = int(raw)
        except ValueError:
            return default
        return max(0, value)

    def _read_env_float(self, var_name: str, *, default: float) -> float:
        raw = os.getenv(var_name, "").strip()
        if not raw:
            return default

        try:
            value = float(raw)
        except ValueError:
            return default
        return max(0.0, value)
