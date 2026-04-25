from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from content_creator.config import AppConfig, ModelConfig
from content_creator.hf_client import HuggingFaceGateway


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

    def text_classification(self, text: str, **kwargs):
        self.calls["text_classification"] = (text, kwargs)
        return [
            {"label": "toxic", "score": 0.81},
            {"label": "non-toxic", "score": 0.19},
        ]


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
    import content_creator.hf_client as hf_module

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
    import content_creator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    gateway = HuggingFaceGateway(_config(tmp_path))
    destination = tmp_path / "speech.wav"

    result = gateway.synthesize_speech("say this", destination)

    assert result == destination
    assert destination.read_bytes() == b"audio-bytes"


def test_transcribe_audio_passes_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    gateway = HuggingFaceGateway(_config(tmp_path))
    audio_path = tmp_path / "sample.m4a"
    audio_path.write_bytes(b"abc123")

    text = gateway.transcribe_audio(audio_path)

    assert text == "transcript text"
    inputs, kwargs = gateway._client.calls["automatic_speech_recognition"]
    assert inputs == b"abc123"
    assert kwargs["model"] == "stt/model"


def test_transcribe_audio_with_word_timestamps_returns_words(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    gateway = HuggingFaceGateway(_config(tmp_path))
    audio_path = tmp_path / "sample.m4a"
    audio_path.write_bytes(b"abc123")

    def _asr(_inputs, **kwargs):
        assert kwargs["model"] == "stt/model"
        assert kwargs["extra_body"] == {"return_timestamps": "word"}
        return {
            "text": "hello world",
            "chunks": [
                {"text": "hello", "timestamp": [0.0, 0.4]},
                {"text": "world", "timestamp": [0.4, 0.8]},
            ],
        }

    gateway._client.automatic_speech_recognition = _asr

    text, words = gateway.transcribe_audio_with_word_timestamps(audio_path)

    assert text == "hello world"
    assert [word.word for word in words] == ["hello", "world"]
    assert words[0].start_seconds == pytest.approx(0.0)
    assert words[1].end_seconds == pytest.approx(0.8)


def test_transcribe_audio_with_word_timestamps_raises_when_missing_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    gateway = HuggingFaceGateway(_config(tmp_path))
    audio_path = tmp_path / "sample.m4a"
    audio_path.write_bytes(b"abc123")

    gateway._client.automatic_speech_recognition = lambda _inputs, **_kwargs: {
        "text": "hello world"
    }

    with pytest.raises(RuntimeError, match="Word-level timestamps are unavailable"):
        gateway.transcribe_audio_with_word_timestamps(audio_path)


def test_classify_content_safety_uses_requested_model_and_parses_scores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    gateway = HuggingFaceGateway(_config(tmp_path))

    result = gateway.classify_content_safety("bad words", model="unitary/toxic-bert")

    assert result["model"] == "unitary/toxic-bert"
    assert result["unsafe_score"] == pytest.approx(0.81)
    assert result["top_label"] == "toxic"
    text, kwargs = gateway._client.calls["text_classification"]
    assert text == "bad words"
    assert kwargs["model"] == "unitary/toxic-bert"


def test_generate_image_raises_for_non_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    gateway = HuggingFaceGateway(_config(tmp_path))

    def _bad_text_to_image(prompt: str, **kwargs):
        return "not-an-image"

    gateway._client.text_to_image = _bad_text_to_image

    with pytest.raises(TypeError, match="Expected a PIL image"):
        gateway.generate_image("prompt", tmp_path / "x.png")


def test_generate_image_passes_negative_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    gateway = HuggingFaceGateway(_config(tmp_path))
    destination = tmp_path / "image.png"

    result = gateway.generate_image("prompt", destination)

    assert result == destination
    prompt, kwargs = gateway._client.calls["text_to_image"]
    assert prompt == "prompt"
    assert kwargs["model"] == "image/model"
    assert kwargs["negative_prompt"] == gateway._config.image_negative_prompt


def test_transcribe_audio_with_speakers_uses_diarization_segments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.hf_client as hf_module

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
    import content_creator.hf_client as hf_module

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


def test_transcribe_audio_with_speakers_passes_diarization_speaker_constraints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    monkeypatch.setattr(hf_module.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    class _Segment:
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end

    class _Diarization:
        def itertracks(self, yield_label: bool = False):
            assert yield_label is True
            yield (_Segment(0.0, 1.0), None, "SPEAKER_00")

    class _PyannotePipeline:
        last_call_kwargs: dict[str, int] | None = None

        @classmethod
        def from_pretrained(cls, model_id: str, use_auth_token: str):
            assert model_id.startswith("pyannote/speaker-diarization")
            assert use_auth_token == "token"
            return cls()

        def __call__(self, _audio_path: str, **kwargs):
            _PyannotePipeline.last_call_kwargs = kwargs
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
    monkeypatch.setattr(
        gateway, "transcribe_audio", lambda _path: "hello", raising=False
    )

    text = gateway.transcribe_audio_with_speakers(source_audio, speaker_count=1)

    assert text == "SPEAKER_00: hello"
    assert _PyannotePipeline.last_call_kwargs == {"num_speakers": 1}


def test_transcribe_audio_with_speakers_auto_collapses_to_primary_speaker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    monkeypatch.setattr(hf_module.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    class _Segment:
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end

    class _Diarization:
        def itertracks(self, yield_label: bool = False):
            assert yield_label is True
            yield (_Segment(0.0, 9.0), None, "SPEAKER_00")
            yield (_Segment(9.0, 10.0), None, "SPEAKER_01")

    class _PyannotePipeline:
        @classmethod
        def from_pretrained(cls, model_id: str, use_auth_token: str):
            assert model_id.startswith("pyannote/speaker-diarization")
            assert use_auth_token == "token"
            return cls()

        def __call__(self, _audio_path: str, **_kwargs):
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
    transcripts = iter(["long section", "short section"])
    monkeypatch.setattr(
        gateway, "transcribe_audio", lambda _path: next(transcripts), raising=False
    )

    text = gateway.transcribe_audio_with_speakers(source_audio)

    assert text == "SPEAKER_00: long section short section"


def test_transcribe_audio_with_speakers_does_not_auto_collapse_with_explicit_speaker_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    monkeypatch.setattr(hf_module.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    class _Segment:
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end

    class _Diarization:
        def itertracks(self, yield_label: bool = False):
            assert yield_label is True
            yield (_Segment(0.0, 9.0), None, "SPEAKER_00")
            yield (_Segment(9.0, 10.0), None, "SPEAKER_01")

    class _PyannotePipeline:
        @classmethod
        def from_pretrained(cls, model_id: str, use_auth_token: str):
            assert model_id.startswith("pyannote/speaker-diarization")
            assert use_auth_token == "token"
            return cls()

        def __call__(self, _audio_path: str, **_kwargs):
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
    transcripts = iter(["long section", "short section"])
    monkeypatch.setattr(
        gateway, "transcribe_audio", lambda _path: next(transcripts), raising=False
    )

    text = gateway.transcribe_audio_with_speakers(source_audio, speaker_count=2)

    assert text == "SPEAKER_00: long section\nSPEAKER_01: short section"


def test_transcribe_audio_with_speakers_respects_custom_dominance_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    monkeypatch.setattr(hf_module.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    class _Segment:
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end

    class _Diarization:
        def itertracks(self, yield_label: bool = False):
            assert yield_label is True
            yield (_Segment(0.0, 9.0), None, "SPEAKER_00")
            yield (_Segment(9.0, 10.0), None, "SPEAKER_01")

    class _PyannotePipeline:
        @classmethod
        def from_pretrained(cls, model_id: str, use_auth_token: str):
            assert model_id.startswith("pyannote/speaker-diarization")
            assert use_auth_token == "token"
            return cls()

        def __call__(self, _audio_path: str, **_kwargs):
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
    transcripts = iter(["long section", "short section"])
    monkeypatch.setattr(
        gateway, "transcribe_audio", lambda _path: next(transcripts), raising=False
    )

    text = gateway.transcribe_audio_with_speakers(
        source_audio, speaker_dominance_threshold=0.95
    )

    assert text == "SPEAKER_00: long section\nSPEAKER_01: short section"


def test_transcribe_audio_with_speakers_uses_env_dominance_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    monkeypatch.setattr(hf_module.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setenv("HF_SPEAKER_DOMINANCE_THRESHOLD", "0.95")

    class _Segment:
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end

    class _Diarization:
        def itertracks(self, yield_label: bool = False):
            assert yield_label is True
            yield (_Segment(0.0, 9.0), None, "SPEAKER_00")
            yield (_Segment(9.0, 10.0), None, "SPEAKER_01")

    class _PyannotePipeline:
        @classmethod
        def from_pretrained(cls, model_id: str, use_auth_token: str):
            assert model_id.startswith("pyannote/speaker-diarization")
            assert use_auth_token == "token"
            return cls()

        def __call__(self, _audio_path: str, **_kwargs):
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
    transcripts = iter(["long section", "short section"])
    monkeypatch.setattr(
        gateway, "transcribe_audio", lambda _path: next(transcripts), raising=False
    )

    text = gateway.transcribe_audio_with_speakers(source_audio)

    assert text == "SPEAKER_00: long section\nSPEAKER_01: short section"


def test_transcribe_audio_with_speakers_raises_helpful_error_for_gated_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.hf_client as hf_module

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


def test_generate_text_retries_on_http_429_with_retry_after_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    monkeypatch.setenv("HF_INFERENCE_MAX_RETRIES", "2")

    sleep_calls: list[float] = []
    monkeypatch.setattr(hf_module, "sleep", lambda value: sleep_calls.append(value))

    gateway = HuggingFaceGateway(_config(tmp_path))

    class _Response:
        status_code = 429
        headers = {"Retry-After": "1.25"}

    class _RateLimitError(Exception):
        def __init__(self):
            super().__init__("429 Too Many Requests")
            self.response = _Response()

    attempts = {"count": 0}

    def _flaky_chat(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise _RateLimitError()

        class _Message:
            content = "eventual success"

        class _Choice:
            message = _Message()

        class _Output:
            choices = [_Choice()]

        return _Output()

    gateway._client.chat_completion = _flaky_chat

    text = gateway.generate_text("hello")

    assert text == "eventual success"
    assert attempts["count"] == 2
    assert sleep_calls
    assert sleep_calls[0] >= 1.25


def test_generate_text_raises_after_retry_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.hf_client as hf_module

    monkeypatch.setattr(hf_module, "InferenceClient", FakeInferenceClient)
    monkeypatch.setenv("HF_INFERENCE_MAX_RETRIES", "1")
    monkeypatch.setattr(hf_module, "sleep", lambda _value: None)

    gateway = HuggingFaceGateway(_config(tmp_path))

    class _ServerError(Exception):
        status_code = 503

    def _always_fails(*args, **kwargs):
        raise _ServerError("service unavailable")

    gateway._client.chat_completion = _always_fails

    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        gateway.generate_text("hello")
