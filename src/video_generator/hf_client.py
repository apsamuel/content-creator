from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import InferenceClient
from PIL import Image

from video_generator.config import AppConfig


@dataclass(slots=True)
class AudioAsset:
    path: Path
    transcript: str
    duration_seconds: float


class HuggingFaceGateway:
    def __init__(self, config: AppConfig):
        self._config = config
        self._client = InferenceClient(token=config.hf_token)

    def generate_text(self, prompt: str) -> str:
        response = self._client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self._config.models.llm_model,
            max_tokens=900,
            temperature=0.6,
        )
        return response.choices[0].message.content or ""

    def synthesize_speech(self, text: str, destination: Path) -> Path:
        audio_bytes = self._client.text_to_speech(
            text, model=self._config.models.tts_model
        )
        destination.write_bytes(audio_bytes)
        return destination

    def transcribe_audio(self, audio_path: Path) -> str:
        audio_bytes = audio_path.read_bytes()
        result = self._client.automatic_speech_recognition(
            audio_bytes, model=self._config.models.stt_model
        )
        if isinstance(result, dict):
            return str(result.get("text", "")).strip()
        return str(result).strip()

    def transcribe_audio_with_speakers(self, audio_path: Path) -> str:
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
        with tempfile.TemporaryDirectory(prefix="video_generator_diarization_") as tmp:
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
                diarization = diarizer(str(prepared_audio_path))
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

            utterances: list[tuple[str, str]] = []
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
                    utterances.append((str(speaker), text))

        lines = self._merge_speaker_utterances(utterances)
        if not lines:
            return self.transcribe_audio(audio_path)
        return "\n".join(lines).strip()

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

    def generate_image(self, prompt: str, destination: Path) -> Path:
        image = self._client.text_to_image(
            prompt,
            model=self._config.models.image_model,
            width=self._config.width,
            height=self._config.height,
        )
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Expected a PIL image from Hugging Face, received {type(image)!r}"
            )
        image.save(destination)
        return destination
