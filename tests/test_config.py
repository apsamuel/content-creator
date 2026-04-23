from __future__ import annotations

from pathlib import Path

import pytest

from video_generator.config import AppConfig


def test_from_env_requires_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(ValueError, match="HF_TOKEN is required"):
        AppConfig.from_env()


def test_from_env_uses_work_dir_arg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_TOKEN", "token-123")
    work_dir = tmp_path / "work"

    config = AppConfig.from_env(work_dir=str(work_dir))

    assert config.hf_token == "token-123"
    assert config.work_dir == work_dir.resolve()
    assert config.work_dir.exists()


def test_from_env_reads_model_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_TOKEN", "token-xyz")
    monkeypatch.setenv("HF_LLM_MODEL", "llm/custom")
    monkeypatch.setenv("HF_STT_MODEL", "stt/custom")
    monkeypatch.setenv("HF_TTS_MODEL", "tts/custom")
    monkeypatch.setenv("HF_IMAGE_MODEL", "img/custom")

    config = AppConfig.from_env(work_dir=tmp_path)

    assert config.models.llm_model == "llm/custom"
    assert config.models.stt_model == "stt/custom"
    assert config.models.tts_model == "tts/custom"
    assert config.models.image_model == "img/custom"


def test_from_env_prefers_explicit_model_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_TOKEN", "token-xyz")
    monkeypatch.setenv("HF_LLM_MODEL", "llm/env")
    monkeypatch.setenv("HF_STT_MODEL", "stt/env")
    monkeypatch.setenv("HF_TTS_MODEL", "tts/env")
    monkeypatch.setenv("HF_IMAGE_MODEL", "img/env")

    config = AppConfig.from_env(
        work_dir=tmp_path,
        llm_model="llm/cli",
        stt_model="stt/cli",
        tts_model="tts/cli",
        image_model="img/cli",
    )

    assert config.models.llm_model == "llm/cli"
    assert config.models.stt_model == "stt/cli"
    assert config.models.tts_model == "tts/cli"
    assert config.models.image_model == "img/cli"


def test_from_env_uses_builtin_model_defaults_when_env_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_TOKEN", "token-xyz")
    monkeypatch.delenv("HF_LLM_MODEL", raising=False)
    monkeypatch.delenv("HF_STT_MODEL", raising=False)
    monkeypatch.delenv("HF_TTS_MODEL", raising=False)
    monkeypatch.delenv("HF_IMAGE_MODEL", raising=False)

    config = AppConfig.from_env(work_dir=tmp_path)

    assert isinstance(config.models.llm_model, str)
    assert isinstance(config.models.stt_model, str)
    assert isinstance(config.models.tts_model, str)
    assert isinstance(config.models.image_model, str)
    assert config.models.llm_model == "meta-llama/Llama-3.1-8B-Instruct"
    assert config.models.stt_model == "openai/whisper-large-v3"
    assert config.models.tts_model == "espnet/kan-bayashi_ljspeech_vits"
    assert config.models.image_model == "stabilityai/stable-diffusion-xl-base-1.0"
