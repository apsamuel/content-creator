from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_IMAGE_NEGATIVE_PROMPT = (
    "blurry, lowres, low detail, soft focus, bad anatomy, bad hands, extra fingers, "
    "missing fingers, extra limbs, duplicate subjects, deformed face, cross-eyed, "
    "text, subtitles, watermark, logo, border, frame, photorealistic, live action, "
    "flat lighting, muddy colors"
)

DEFAULT_IMAGE_COMPOSITION_MODE = "balanced"


@dataclass(slots=True)
class ModelConfig:
    llm_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    stt_model: str = "openai/whisper-large-v3"
    tts_model: str = "hexgrad/Kokoro-82M"
    safety_model: str = "unitary/unbiased-toxic-roberta"
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    image_model: str = "black-forest-labs/FLUX.1-dev"


@dataclass(slots=True)
class AppConfig:
    hf_token: str
    work_dir: Path
    models: ModelConfig
    width: int = 1280
    height: int = 720
    fps: int = 24
    image_negative_prompt: str = DEFAULT_IMAGE_NEGATIVE_PROMPT
    image_composition_mode: str = DEFAULT_IMAGE_COMPOSITION_MODE

    @classmethod
    def from_env(
        cls,
        work_dir: str | Path | None = None,
        *,
        llm_model: str | None = None,
        stt_model: str | None = None,
        tts_model: str | None = None,
        image_model: str | None = None,
    ) -> "AppConfig":
        token = os.getenv("HF_TOKEN", "").strip()
        if not token:
            raise ValueError(
                "HF_TOKEN is required to call Hugging Face inference APIs."
            )

        defaults = ModelConfig()

        def _resolve_model(
            cli_override: str | None, env_var: str, default_value: str
        ) -> str:
            if cli_override is not None and cli_override.strip():
                return cli_override.strip()
            env_value = os.getenv(env_var, "").strip()
            if env_value:
                return env_value
            return default_value

        base_dir = (
            Path(work_dir or os.getenv("CONTENT_CREATOR_WORK_DIR", "./output"))
            .expanduser()
            .resolve()
        )
        base_dir.mkdir(parents=True, exist_ok=True)

        models = ModelConfig(
            llm_model=_resolve_model(llm_model, "HF_LLM_MODEL", defaults.llm_model),
            stt_model=_resolve_model(stt_model, "HF_STT_MODEL", defaults.stt_model),
            tts_model=_resolve_model(tts_model, "HF_TTS_MODEL", defaults.tts_model),
            image_model=_resolve_model(
                image_model, "HF_IMAGE_MODEL", defaults.image_model
            ),
        )
        image_negative_prompt = (
            os.getenv("HF_IMAGE_NEGATIVE_PROMPT", "").strip()
            or DEFAULT_IMAGE_NEGATIVE_PROMPT
        )
        image_composition_mode = (
            os.getenv("HF_IMAGE_COMPOSITION_MODE", DEFAULT_IMAGE_COMPOSITION_MODE)
            .strip()
            .lower()
            or DEFAULT_IMAGE_COMPOSITION_MODE
        )
        if image_composition_mode not in {
            "balanced",
            "dynamic",
            "portrait",
            "establishing",
        }:
            raise ValueError(
                "HF_IMAGE_COMPOSITION_MODE must be one of: balanced, dynamic, portrait, establishing"
            )
        return cls(
            hf_token=token,
            work_dir=base_dir,
            models=models,
            image_negative_prompt=image_negative_prompt,
            image_composition_mode=image_composition_mode,
        )
