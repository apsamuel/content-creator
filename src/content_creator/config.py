from __future__ import annotations

import os
from dataclasses import dataclass, field
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
    safety_model: str = "cardiffnlp/twitter-roberta-base-offensive"
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    image_model: str = "black-forest-labs/FLUX.1-dev"


@dataclass(slots=True)
class LLMInferenceConfig:
    max_tokens: int = 900
    temperature: float = 0.6
    top_p: float = 1.0


@dataclass(slots=True)
class ImageInferenceConfig:
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    seed: int | None = None


@dataclass(slots=True)
class SafetyInferenceConfig:
    top_k: int | None = None


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
    tuning_profile: str = "balanced"
    llm_inference: LLMInferenceConfig = field(default_factory=LLMInferenceConfig)
    image_inference: ImageInferenceConfig = field(default_factory=ImageInferenceConfig)
    safety_inference: SafetyInferenceConfig = field(
        default_factory=SafetyInferenceConfig
    )
    # When set, image generation uses this provider and its API key directly,
    # bypassing HF-routed billing. Supported values: fal-ai, replicate, nebius,
    # wavespeed. HF_PROVIDER_KEY must be the provider's own API key (not an HF token).
    image_provider: str = ""
    image_provider_key: str = ""

    @classmethod
    def from_env(
        cls,
        work_dir: str | Path | None = None,
        *,
        llm_model: str | None = None,
        stt_model: str | None = None,
        tts_model: str | None = None,
        image_model: str | None = None,
        safety_model: str | None = None,
    ) -> "AppConfig":
        def _env_int(name: str, *, minimum: int | None = None) -> int | None:
            raw_value = os.getenv(name, "").strip()
            if not raw_value:
                return None
            try:
                resolved = int(raw_value)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer") from exc
            if minimum is not None and resolved < minimum:
                raise ValueError(f"{name} must be >= {minimum}")
            return resolved

        def _env_float(
            name: str,
            *,
            minimum: float | None = None,
            maximum: float | None = None,
            min_inclusive: bool = True,
            max_inclusive: bool = True,
        ) -> float | None:
            raw_value = os.getenv(name, "").strip()
            if not raw_value:
                return None
            try:
                resolved = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"{name} must be a float") from exc
            if minimum is not None:
                if min_inclusive and resolved < minimum:
                    raise ValueError(f"{name} must be >= {minimum}")
                if not min_inclusive and resolved <= minimum:
                    raise ValueError(f"{name} must be > {minimum}")
            if maximum is not None:
                if max_inclusive and resolved > maximum:
                    raise ValueError(f"{name} must be <= {maximum}")
                if not max_inclusive and resolved >= maximum:
                    raise ValueError(f"{name} must be < {maximum}")
            return resolved

        def _profile_defaults(
            profile_name: str,
        ) -> tuple[LLMInferenceConfig, ImageInferenceConfig, SafetyInferenceConfig]:
            profiles: dict[
                str,
                tuple[LLMInferenceConfig, ImageInferenceConfig, SafetyInferenceConfig],
            ] = {
                "balanced": (
                    LLMInferenceConfig(max_tokens=900, temperature=0.6, top_p=1.0),
                    ImageInferenceConfig(),
                    SafetyInferenceConfig(),
                ),
                "cinematic": (
                    LLMInferenceConfig(max_tokens=1100, temperature=0.45, top_p=0.9),
                    ImageInferenceConfig(num_inference_steps=40, guidance_scale=6.5),
                    SafetyInferenceConfig(top_k=5),
                ),
                "consistent": (
                    LLMInferenceConfig(max_tokens=950, temperature=0.35, top_p=0.85),
                    ImageInferenceConfig(
                        num_inference_steps=45, guidance_scale=7.0, seed=42
                    ),
                    SafetyInferenceConfig(top_k=5),
                ),
                "fast": (
                    LLMInferenceConfig(max_tokens=700, temperature=0.7, top_p=1.0),
                    ImageInferenceConfig(num_inference_steps=24, guidance_scale=5.5),
                    SafetyInferenceConfig(top_k=3),
                ),
            }
            resolved_profile = profile_name.strip().lower()
            if resolved_profile not in profiles:
                raise ValueError(
                    "HF_TUNING_PROFILE must be one of: balanced, cinematic, consistent, fast"
                )
            return profiles[resolved_profile]

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
            safety_model=_resolve_model(
                safety_model, "HF_CONTENT_SAFETY_MODEL", defaults.safety_model
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
        image_provider = os.getenv("HF_INFERENCE_PROVIDER", "").strip()
        image_provider_key = os.getenv("HF_PROVIDER_KEY", "").strip()
        tuning_profile = os.getenv("HF_TUNING_PROFILE", "balanced").strip().lower()
        if not tuning_profile:
            tuning_profile = "balanced"
        llm_defaults, image_defaults, safety_defaults = _profile_defaults(
            tuning_profile
        )
        llm_max_tokens = _env_int("HF_LLM_MAX_TOKENS", minimum=1)
        llm_temperature = _env_float("HF_LLM_TEMPERATURE", minimum=0.0, maximum=2.0)
        llm_top_p = _env_float(
            "HF_LLM_TOP_P", minimum=0.0, maximum=1.0, min_inclusive=False
        )
        llm_inference = LLMInferenceConfig(
            max_tokens=(
                llm_defaults.max_tokens if llm_max_tokens is None else llm_max_tokens
            ),
            temperature=(
                llm_defaults.temperature if llm_temperature is None else llm_temperature
            ),
            top_p=llm_defaults.top_p if llm_top_p is None else llm_top_p,
        )
        image_steps = _env_int("HF_IMAGE_NUM_INFERENCE_STEPS", minimum=1)
        image_guidance = _env_float(
            "HF_IMAGE_GUIDANCE_SCALE", minimum=0.0, min_inclusive=False
        )
        image_seed = _env_int("HF_IMAGE_SEED", minimum=0)
        image_inference = ImageInferenceConfig(
            num_inference_steps=(
                image_defaults.num_inference_steps
                if image_steps is None
                else image_steps
            ),
            guidance_scale=(
                image_defaults.guidance_scale
                if image_guidance is None
                else image_guidance
            ),
            seed=image_defaults.seed if image_seed is None else image_seed,
        )
        safety_top_k = _env_int("HF_SAFETY_TOP_K", minimum=1)
        safety_inference = SafetyInferenceConfig(
            top_k=safety_defaults.top_k if safety_top_k is None else safety_top_k
        )
        return cls(
            hf_token=token,
            work_dir=base_dir,
            models=models,
            image_negative_prompt=image_negative_prompt,
            image_composition_mode=image_composition_mode,
            tuning_profile=tuning_profile,
            llm_inference=llm_inference,
            image_inference=image_inference,
            safety_inference=safety_inference,
            image_provider=image_provider,
            image_provider_key=image_provider_key,
        )
