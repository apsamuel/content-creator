from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from video_generator.config import AppConfig, ModelConfig
from video_generator.hf_client import HuggingFaceGateway


class FakeInferenceClient:
    def __init__(self, token: str):
        self.token = token
        self.calls: dict[str, tuple] = {}

    def chat_completion(self, messages, **kwargs):
        self.calls["chat_completion"] = (messages, kwargs)

        class _Message:
            content = "generated text"

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        return _Response()

    def text_to_speech(self, text: str, **kwargs):
        self.calls["text_to_speech"] = (text, kwargs)
        return b"audio-bytes"

    def automatic_speech_recognition(self, inputs, **kwargs):
        self.calls["automatic_speech_recognition"] = (inputs, kwargs)
        return {"text": "  transcript text  "}

    def text_to_image(self, prompt: str, **kwargs):
        self.calls["text_to_image"] = (prompt, kwargs)
        return Image.new("RGB", (64, 64), color="black")


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        hf_token="token",
        work_dir=tmp_path,
        models=ModelConfig(
            llm_model="llm/model",
            stt_model="stt/model",
            tts_model="tts/model",
            image_model="image/model",
        ),
    )


def test_generate_text_uses_configured_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import video_generator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    gateway = HuggingFaceGateway(_config(tmp_path))

    output = gateway.generate_text("hello")

    assert output == "generated text"
    messages, kwargs = gateway._client.calls["chat_completion"]
    assert messages == [{"role": "user", "content": "hello"}]
    assert kwargs["model"] == "llm/model"


def test_synthesize_speech_writes_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import video_generator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    gateway = HuggingFaceGateway(_config(tmp_path))
    destination = tmp_path / "speech.wav"

    result = gateway.synthesize_speech("say this", destination)

    assert result == destination
    assert destination.read_bytes() == b"audio-bytes"


def test_transcribe_audio_passes_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import video_generator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    gateway = HuggingFaceGateway(_config(tmp_path))
    audio_path = tmp_path / "sample.m4a"
    audio_path.write_bytes(b"abc123")

    text = gateway.transcribe_audio(audio_path)

    assert text == "transcript text"
    inputs, kwargs = gateway._client.calls["automatic_speech_recognition"]
    assert inputs == b"abc123"
    assert kwargs["model"] == "stt/model"


def test_generate_image_raises_for_non_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import video_generator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    gateway = HuggingFaceGateway(_config(tmp_path))

    def _bad_text_to_image(prompt: str, **kwargs):
        return "not-an-image"

    gateway._client.text_to_image = _bad_text_to_image

    with pytest.raises(TypeError, match="Expected a PIL image"):
        gateway.generate_image("prompt", tmp_path / "x.png")


def test_transcribe_audio_with_speakers_uses_diarization_segments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import video_generator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    monkeypatch.setattr(hf_module.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    class _Segment:
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end

    class _Diarization:
        def itertracks(self, yield_label: bool = False):
            assert yield_label is True
            yield (_Segment(0.0, 1.2), None, "SPEAKER_00")
            yield (_Segment(1.2, 2.4), None, "SPEAKER_01")

    class _PyannotePipeline:
        @classmethod
        def from_pretrained(cls, model_id: str, use_auth_token: str):
            assert model_id.startswith("pyannote/speaker-diarization")
            assert use_auth_token == "token"
            return cls()

        def __call__(self, audio_path: str):
            assert audio_path.endswith(".wav")
            return _Diarization()

    fake_pkg = types.ModuleType("pyannote")
    fake_audio = types.ModuleType("pyannote.audio")
    fake_audio.Pipeline = _PyannotePipeline
    monkeypatch.setitem(sys.modules, "pyannote", fake_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)

    calls: list[list[str]] = []

    def _run(command, check, capture_output, text):
        calls.append(command)
        Path(command[-1]).write_bytes(b"chunk")
        return types.SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(hf_module.subprocess, "run", _run)

    gateway = HuggingFaceGateway(_config(tmp_path))
    source_audio = tmp_path / "sample.wav"
    source_audio.write_bytes(b"audio")

    transcripts = iter(["hello there", "general kenobi"])
    monkeypatch.setattr(
        gateway, "transcribe_audio", lambda _path: next(transcripts), raising=False
    )

    text = gateway.transcribe_audio_with_speakers(source_audio)

    assert text == "SPEAKER_00: hello there\nSPEAKER_01: general kenobi"
    assert len(calls) == 3
    assert all(command[0] == "ffmpeg" for command in calls)
    assert calls[0][calls[0].index("-i") + 1].endswith("sample.wav")
    assert calls[0][-1].endswith("prepared_audio.wav")


def test_transcribe_audio_with_speakers_merges_consecutive_segments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import video_generator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    monkeypatch.setattr(hf_module.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    class _Segment:
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end

    class _Diarization:
        def itertracks(self, yield_label: bool = False):
            assert yield_label is True
            yield (_Segment(0.0, 0.9), None, "SPEAKER_00")
            yield (_Segment(0.9, 1.8), None, "SPEAKER_00")
            yield (_Segment(1.8, 2.7), None, "SPEAKER_01")

    class _PyannotePipeline:
        @classmethod
        def from_pretrained(cls, model_id: str, use_auth_token: str):
            assert model_id.startswith("pyannote/speaker-diarization")
            assert use_auth_token == "token"
            return cls()

        def __call__(self, audio_path: str):
            assert audio_path.endswith(".wav")
            return _Diarization()

    fake_pkg = types.ModuleType("pyannote")
    fake_audio = types.ModuleType("pyannote.audio")
    fake_audio.Pipeline = _PyannotePipeline
    monkeypatch.setitem(sys.modules, "pyannote", fake_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)

    def _run(command, check, capture_output, text):
        Path(command[-1]).write_bytes(b"chunk")
        return types.SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(hf_module.subprocess, "run", _run)

    gateway = HuggingFaceGateway(_config(tmp_path))
    source_audio = tmp_path / "sample.wav"
    source_audio.write_bytes(b"audio")

    transcripts = iter(["hello", "there", "general kenobi"])
    monkeypatch.setattr(
        gateway, "transcribe_audio", lambda _path: next(transcripts), raising=False
    )

    text = gateway.transcribe_audio_with_speakers(source_audio)

    assert text == "SPEAKER_00: hello there\nSPEAKER_01: general kenobi"


def test_transcribe_audio_with_speakers_raises_helpful_error_for_gated_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import video_generator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    monkeypatch.setattr(hf_module.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    class _PyannotePipeline:
        @classmethod
        def from_pretrained(cls, model_id: str, use_auth_token: str):
            assert model_id.startswith("pyannote/speaker-diarization")
            assert use_auth_token == "token"
            return cls()

        def __call__(self, _audio_path: str):
            raise AttributeError("'NoneType' object has no attribute 'eval'")

    fake_pkg = types.ModuleType("pyannote")
    fake_audio = types.ModuleType("pyannote.audio")
    fake_audio.Pipeline = _PyannotePipeline
    monkeypatch.setitem(sys.modules, "pyannote", fake_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)

    def _run(command, check, capture_output, text):
        Path(command[-1]).write_bytes(b"audio")
        return types.SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(hf_module.subprocess, "run", _run)

    gateway = HuggingFaceGateway(_config(tmp_path))
    source_audio = tmp_path / "sample.wav"
    source_audio.write_bytes(b"audio")

    with pytest.raises(RuntimeError, match="could not access required pyannote models"):
        gateway.transcribe_audio_with_speakers(source_audio)
