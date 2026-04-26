from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
import pytest

import content_creator.cli as cli_module
from content_creator.config import AppConfig, ModelConfig
from content_creator.profanity_sfx import LexiconDoctorReport


class FakePipeline:
    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []

    def generate_from_text(self, **kwargs):
        self.calls.append(("from_text", (), kwargs))
        return kwargs["output_path"]

    def generate_from_audio(self, **kwargs):
        self.calls.append(("from_audio", (), kwargs))
        return kwargs["output_path"]

    def transcribe_audio_file(self, **kwargs):
        self.calls.append(("transcribe", (), kwargs))
        output_path = kwargs.get("output_path")
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("sample transcript", encoding="utf-8")
        return "sample transcript"

    def build_profanity_debug_audio(self, **kwargs):
        self.calls.append(("profanity_debug", (), kwargs))
        return 2


def test_transcribe_command_writes_output_file(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    def _build_pipeline(
        *,
        work_dir,
        debug,
        status_callback,
        llm_model,
        stt_model,
        tts_model,
        image_model,
    ):
        assert work_dir == str(tmp_path / "work")
        assert debug is False
        assert callable(status_callback)
        assert llm_model is None
        assert stt_model is None
        assert tts_model is None
        assert image_model is None
        return fake_pipeline

    monkeypatch.setattr(cli_module, "_build_pipeline", _build_pipeline)

    audio_file = tmp_path / "audio.m4a"
    audio_file.write_bytes(b"audio")
    output_file = tmp_path / "output" / "transcript.txt"

    result = runner.invoke(
        cli_module.cli,
        [
            "transcribe",
            "--audio-file",
            str(audio_file),
            "--output",
            str(output_file),
            "--work-dir",
            str(tmp_path / "work"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "✅ Transcript written to" in result.output
    assert output_file.read_text(encoding="utf-8") == "sample transcript"
    assert fake_pipeline.calls[0][0] == "transcribe"
    assert fake_pipeline.calls[0][2]["chunk_seconds"] == 45.0
    assert fake_pipeline.calls[0][2]["preserve_speaker"] is False


def test_transcribe_profanity_sfx_requires_output(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: FakePipeline())

    audio_file = tmp_path / "audio.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        ["transcribe", "--audio-file", str(audio_file), "--profanity-sfx"],
    )

    assert result.exit_code != 0
    assert "--profanity-sfx-output is required" in result.output


def test_transcribe_profanity_sfx_passes_options(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()
    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "audio.m4a"
    audio_file.write_bytes(b"audio")
    output_audio = tmp_path / "clean.m4a"

    result = runner.invoke(
        cli_module.cli,
        [
            "transcribe",
            "--audio-file",
            str(audio_file),
            "--profanity-sfx",
            "--profanity-sfx-output",
            str(output_audio),
            "--profanity-pad-ms",
            "120",
            "--profanity-duck-db",
            "-10",
        ],
    )

    assert result.exit_code == 0, result.output
    call_kwargs = fake_pipeline.calls[0][2]
    assert call_kwargs["profanity_sfx_enabled"] is True
    assert call_kwargs["profanity_sfx_output_path"] == output_audio
    assert call_kwargs["profanity_pad_seconds"] == pytest.approx(0.12)
    assert call_kwargs["profanity_duck_db"] == pytest.approx(-10.0)


def test_profanity_debug_passes_manifest_and_timing_options(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()
    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "audio.m4a"
    audio_file.write_bytes(b"audio")
    output_file = tmp_path / "debug.m4a"
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(
        '{"profanity_sfx":{"events":[{"word":"heck","start_seconds":1.2,"end_seconds":1.8,"sfx":"/tmp/beep.wav","sfx_duration_seconds":0.6}]},"video_prompt_preclassification":{"mood":"serious"},"narration_text":"example transcript"}',
        encoding="utf-8",
    )

    result = runner.invoke(
        cli_module.cli,
        [
            "profanity-debug",
            "--audio-file",
            str(audio_file),
            "--output",
            str(output_file),
            "--manifest",
            str(manifest_file),
            "--profanity-pad-ms",
            "125",
            "--context-seconds",
            "0.75",
            "--gap-seconds",
            "0.4",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Loaded 1 event(s) from manifest" in result.output
    assert "Debug audio with 2 event(s) written" in result.output

    call_kwargs = fake_pipeline.calls[0][2]
    assert fake_pipeline.calls[0][0] == "profanity_debug"
    assert call_kwargs["audio_path"] == audio_file
    assert call_kwargs["output_path"] == output_file
    assert call_kwargs["manifest_events"] == [
        {
            "word": "heck",
            "start_seconds": 1.2,
            "end_seconds": 1.8,
            "sfx": "/tmp/beep.wav",
            "sfx_duration_seconds": 0.6,
        }
    ]
    assert call_kwargs["preclassification_data"] == {"mood": "serious"}
    assert call_kwargs["transcript_text"] == "example transcript"
    assert call_kwargs["pad_seconds"] == pytest.approx(0.125)
    assert call_kwargs["context_seconds"] == pytest.approx(0.75)
    assert call_kwargs["gap_seconds"] == pytest.approx(0.4)


def test_global_debug_flag_prints_message(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    def _build_pipeline(
        *,
        work_dir,
        debug,
        status_callback,
        llm_model,
        stt_model,
        tts_model,
        image_model,
    ):
        assert debug is True
        return FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", _build_pipeline)

    audio_file = tmp_path / "audio.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        [
            "--debug",
            "transcribe",
            "--audio-file",
            str(audio_file),
            "--work-dir",
            str(tmp_path / "work"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "🐛 Debug mode enabled" in result.output
    assert "sample transcript" in result.output


def test_non_debug_wraps_unexpected_exceptions(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    class FailingPipeline(FakePipeline):
        def generate_from_text(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        cli_module, "_build_pipeline", lambda **kwargs: FailingPipeline()
    )

    result = runner.invoke(
        cli_module.cli,
        [
            "from-text",
            "--text-transcription",
            "hello",
            "--video-prompt",
            "style",
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code != 0
    assert "Error: boom" in result.output


def test_debug_mode_surfaces_traceback(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    class FailingPipeline(FakePipeline):
        def generate_from_text(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        cli_module, "_build_pipeline", lambda **kwargs: FailingPipeline()
    )

    result = runner.invoke(
        cli_module.cli,
        [
            "--debug",
            "from-text",
            "--text-transcription",
            "hello",
            "--video-prompt",
            "style",
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert str(result.exception) == "boom"


def test_global_model_flags_passed_to_pipeline(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, str | None] = {}

    def _build_pipeline(
        *,
        work_dir,
        debug,
        status_callback,
        llm_model,
        stt_model,
        tts_model,
        image_model,
    ):
        captured["work_dir"] = work_dir
        captured["llm_model"] = llm_model
        captured["stt_model"] = stt_model
        captured["tts_model"] = tts_model
        captured["image_model"] = image_model
        return FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", _build_pipeline)

    audio_file = tmp_path / "audio.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        [
            "--llm-model",
            "llm/cli",
            "--stt-model",
            "stt/cli",
            "--tts-model",
            "tts/cli",
            "--image-model",
            "img/cli",
            "transcribe",
            "--audio-file",
            str(audio_file),
            "--work-dir",
            str(tmp_path / "work"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["work_dir"] == str(tmp_path / "work")
    assert captured["llm_model"] == "llm/cli"
    assert captured["stt_model"] == "stt/cli"
    assert captured["tts_model"] == "tts/cli"
    assert captured["image_model"] == "img/cli"


def test_short_model_alias_flags_passed_to_pipeline(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    captured: dict[str, str | None] = {}

    def _build_pipeline(
        *,
        work_dir,
        debug,
        status_callback,
        llm_model,
        stt_model,
        tts_model,
        image_model,
    ):
        captured["llm_model"] = llm_model
        captured["stt_model"] = stt_model
        captured["tts_model"] = tts_model
        captured["image_model"] = image_model
        return FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", _build_pipeline)

    audio_file = tmp_path / "audio.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        [
            "-L",
            "llm/short",
            "-S",
            "stt/short",
            "-T",
            "tts/short",
            "-I",
            "img/short",
            "transcribe",
            "--audio-file",
            str(audio_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["llm_model"] == "llm/short"
    assert captured["stt_model"] == "stt/short"
    assert captured["tts_model"] == "tts/short"
    assert captured["image_model"] == "img/short"


def test_build_pipeline_prints_startup_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_config = AppConfig(
        hf_token="token",
        work_dir=tmp_path / "work",
        models=ModelConfig(
            llm_model="llm/test",
            stt_model="stt/test",
            tts_model="tts/test",
            image_model="img/test",
        ),
    )

    class DummyPipeline:
        def __init__(self, config, *, debug, status_callback):
            self.config = config
            self.debug = debug
            self.status_callback = status_callback

    monkeypatch.setattr(cli_module.AppConfig, "from_env", lambda **_kwargs: fake_config)
    monkeypatch.setattr(cli_module, "VideoGenerationPipeline", DummyPipeline)

    pipeline = cli_module._build_pipeline(
        work_dir=None,
        debug=False,
        status_callback=None,
        llm_model=None,
        stt_model=None,
        tts_model=None,
        image_model=None,
    )

    captured = capsys.readouterr()
    assert isinstance(pipeline, DummyPipeline)
    assert "🔎 Startup check" in captured.out
    assert "🧠 LLM model: llm/test" in captured.out
    assert "🎧 STT model: stt/test" in captured.out
    assert "🔊 TTS model: tts/test" in captured.out
    assert "🖼️ Image model: img/test" in captured.out


def test_from_text_resolves_file_uri_inputs(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    transcription_file = tmp_path / "transcription.txt"
    video_prompt_file = tmp_path / "video_prompt.txt"
    transcription_file.write_text("Narration from file", encoding="utf-8")
    video_prompt_file.write_text("Visual prompt from file", encoding="utf-8")

    result = runner.invoke(
        cli_module.cli,
        [
            "from-text",
            "--text-transcription",
            f"file://{transcription_file}",
            "--video-prompt",
            f"file://{video_prompt_file}",
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][0] == "from_text"
    assert fake_pipeline.calls[0][2]["narration_text"] == "Narration from file"
    assert fake_pipeline.calls[0][2]["video_prompt"] == "Visual prompt from file"


def test_from_audio_resolves_relative_file_uri_for_video_prompt(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("output").mkdir(parents=True, exist_ok=True)
        Path("output/voicecall.txt").write_text(
            "Prompt from relative URI", encoding="utf-8"
        )

        result = runner.invoke(
            cli_module.cli,
            [
                "from-audio",
                "--audio-file",
                str(audio_file),
                "--video-prompt",
                "file://output/voicecall.txt",
                "--output",
                str(tmp_path / "video.mp4"),
            ],
        )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][0] == "from_audio"
    assert fake_pipeline.calls[0][2]["video_prompt"] == "Prompt from relative URI"


def test_from_text_allows_generated_video_prompt(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    result = runner.invoke(
        cli_module.cli,
        [
            "from-text",
            "--text-transcription",
            "Narration from text",
            "--generate-video-prompt",
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["video_prompt"] is None
    assert fake_pipeline.calls[0][2]["generate_video_prompt"] is True


def test_from_audio_allows_generated_video_prompt(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        [
            "from-audio",
            "--audio-file",
            str(audio_file),
            "--generate-video-prompt",
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["video_prompt"] is None
    assert fake_pipeline.calls[0][2]["generate_video_prompt"] is True


def test_from_audio_passes_preserve_speaker(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        [
            "from-audio",
            "--audio-file",
            str(audio_file),
            "--video-prompt",
            "Style",
            "--preserve-speaker",
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["preserve_speaker"] is True


def test_transcribe_passes_preserve_speaker(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        ["transcribe", "--audio-file", str(audio_file), "--preserve-speaker"],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["preserve_speaker"] is True


def test_transcribe_passes_diarization_speaker_constraints(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        [
            "transcribe",
            "--audio-file",
            str(audio_file),
            "--preserve-speaker",
            "--min-speakers",
            "1",
            "--max-speakers",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["diarization_speaker_count"] is None
    assert fake_pipeline.calls[0][2]["diarization_min_speakers"] == 1
    assert fake_pipeline.calls[0][2]["diarization_max_speakers"] == 2


def test_transcribe_rejects_conflicting_diarization_speaker_options(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        [
            "transcribe",
            "--audio-file",
            str(audio_file),
            "--preserve-speaker",
            "--speaker-count",
            "1",
            "--min-speakers",
            "1",
        ],
    )

    assert result.exit_code != 0
    assert "--speaker-count cannot be used" in result.output


def test_transcribe_passes_speaker_dominance_threshold(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        [
            "transcribe",
            "--audio-file",
            str(audio_file),
            "--speaker-dominance-threshold",
            "0.95",
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["speaker_dominance_threshold"] == 0.95


def test_transcribe_uses_speaker_dominance_threshold_env_default(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setenv("HF_SPEAKER_DOMINANCE_THRESHOLD", "0.85")
    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli, ["transcribe", "--audio-file", str(audio_file)]
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["speaker_dominance_threshold"] == 0.85


def test_transcribe_rejects_invalid_speaker_dominance_threshold_env(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setenv("HF_SPEAKER_DOMINANCE_THRESHOLD", "1.5")
    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli, ["transcribe", "--audio-file", str(audio_file)]
    )

    assert result.exit_code != 0
    assert "--speaker-dominance-threshold must be between 0.0 and 1.0" in result.output


def test_transcribe_passes_content_safety_options(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        [
            "transcribe",
            "--audio-file",
            str(audio_file),
            "--content-safety",
            "--content-safety-filter",
            "--content-safety-threshold",
            "0.85",
            "--content-safety-model",
            "unitary/toxic-bert",
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["content_safety_enabled"] is True
    assert fake_pipeline.calls[0][2]["content_safety_filter"] is True
    assert fake_pipeline.calls[0][2]["content_safety_threshold"] == 0.85
    assert fake_pipeline.calls[0][2]["content_safety_model"] == "unitary/toxic-bert"


def test_transcribe_passes_transcribe_workers(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        ["transcribe", "--audio-file", str(audio_file), "--transcribe-workers", "3"],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["transcribe_workers"] == 3


def test_from_audio_passes_transcribe_workers(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        [
            "from-audio",
            "--audio-file",
            str(audio_file),
            "--video-prompt",
            "Style",
            "--transcribe-workers",
            "4",
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["transcribe_workers"] == 4


def test_from_audio_passes_diarization_speaker_count(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        [
            "from-audio",
            "--audio-file",
            str(audio_file),
            "--video-prompt",
            "Style",
            "--preserve-speaker",
            "--speaker-count",
            "1",
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["diarization_speaker_count"] == 1


def test_lexicon_doctor_reports_summary(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    lexicon_file = tmp_path / "words.txt"
    lexicon_file.write_text("foo\n", encoding="utf-8")

    def _analyze(path):
        assert path == lexicon_file
        return LexiconDoctorReport(
            path=lexicon_file,
            total_lines=5,
            active_lines=4,
            unique_normalized_entries=3,
            exact_duplicates={"foo": 2},
            near_duplicates={"wet back": ["wet back", "wet-back"]},
        )

    monkeypatch.setattr(cli_module, "analyze_profanity_lexicon", _analyze)

    result = runner.invoke(
        cli_module.cli,
        [
            "lexicon-doctor",
            "--profanity-words-file",
            str(lexicon_file),
            "--max-groups",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Lexicon file:" in result.output
    assert "Exact duplicate entries: 1" in result.output
    assert "Near-duplicate groups: 1" in result.output
    assert "foo (x2)" in result.output
    assert "wet back: wet back, wet-back" in result.output


def test_from_audio_passes_speaker_dominance_threshold(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        [
            "from-audio",
            "--audio-file",
            str(audio_file),
            "--video-prompt",
            "Style",
            "--speaker-dominance-threshold",
            "0.92",
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["speaker_dominance_threshold"] == 0.92


def test_transcribe_uses_hf_transcribe_workers_env_default(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setenv("HF_TRANSCRIBE_WORKERS", "3")
    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli, ["transcribe", "--audio-file", str(audio_file)]
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["transcribe_workers"] == 3


def test_from_audio_uses_hf_transcribe_workers_env_default(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setenv("HF_TRANSCRIBE_WORKERS", "2")
    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        [
            "from-audio",
            "--audio-file",
            str(audio_file),
            "--video-prompt",
            "Style",
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["transcribe_workers"] == 2


def test_from_text_passes_image_workers(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    result = runner.invoke(
        cli_module.cli,
        [
            "from-text",
            "--text-transcription",
            "Narration",
            "--video-prompt",
            "Style",
            "--image-workers",
            "3",
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["image_workers"] == 3


def test_from_audio_passes_image_workers(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        [
            "from-audio",
            "--audio-file",
            str(audio_file),
            "--video-prompt",
            "Style",
            "--image-workers",
            "4",
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["image_workers"] == 4


def test_from_text_passes_images_per_scene(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    result = runner.invoke(
        cli_module.cli,
        [
            "from-text",
            "--text-transcription",
            "Narration",
            "--video-prompt",
            "Style",
            "--images-per-scene",
            "3",
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["images_per_scene"] == 3


def test_from_audio_passes_images_per_scene(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        [
            "from-audio",
            "--audio-file",
            str(audio_file),
            "--video-prompt",
            "Style",
            "--images-per-scene",
            "4",
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["images_per_scene"] == 4


def test_from_text_uses_hf_image_workers_env_default(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setenv("HF_IMAGE_WORKERS", "3")
    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    result = runner.invoke(
        cli_module.cli,
        [
            "from-text",
            "--text-transcription",
            "Narration",
            "--video-prompt",
            "Style",
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["image_workers"] == 3


def test_from_audio_uses_hf_image_workers_env_default(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setenv("HF_IMAGE_WORKERS", "2")
    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        [
            "from-audio",
            "--audio-file",
            str(audio_file),
            "--video-prompt",
            "Style",
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["image_workers"] == 2


def test_from_text_uses_hf_images_per_scene_env_default(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setenv("HF_IMAGES_PER_SCENE", "2")
    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    result = runner.invoke(
        cli_module.cli,
        [
            "from-text",
            "--text-transcription",
            "Narration",
            "--video-prompt",
            "Style",
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["images_per_scene"] == 2


def test_from_audio_uses_hf_images_per_scene_env_default(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setenv("HF_IMAGES_PER_SCENE", "3")
    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        [
            "from-audio",
            "--audio-file",
            str(audio_file),
            "--video-prompt",
            "Style",
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["images_per_scene"] == 3


def test_from_audio_passes_content_safety_options(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: fake_pipeline)

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        [
            "from-audio",
            "--audio-file",
            str(audio_file),
            "--video-prompt",
            "Style",
            "--content-safety",
            "--content-safety-threshold",
            "0.65",
            "--content-safety-model",
            "unitary/unbiased-toxic-roberta",
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline.calls[0][2]["content_safety_enabled"] is True
    assert fake_pipeline.calls[0][2]["content_safety_filter"] is False
    assert fake_pipeline.calls[0][2]["content_safety_threshold"] == 0.65
    assert (
        fake_pipeline.calls[0][2]["content_safety_model"]
        == "unitary/unbiased-toxic-roberta"
    )


def test_from_audio_requires_video_prompt_without_generation(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()

    monkeypatch.setattr(cli_module, "_build_pipeline", lambda **_kwargs: FakePipeline())

    audio_file = tmp_path / "input.m4a"
    audio_file.write_bytes(b"audio")

    result = runner.invoke(
        cli_module.cli,
        [
            "from-audio",
            "--audio-file",
            str(audio_file),
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code != 0
    assert (
        "Error: --video-prompt is required unless --generate-video-prompt is enabled"
        in result.output
    )


def test_status_callback_hides_progress_when_disabled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    callback = cli_module._make_status_callback(progress_enabled=False)

    callback("Transcription chunk progress: [############------------] 1/2 (50%)")
    callback("✅ Transcription complete")

    captured = capsys.readouterr()
    assert "Transcription chunk progress:" not in captured.out
    assert "✅ Transcription complete" in captured.out
