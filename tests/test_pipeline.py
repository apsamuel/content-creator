from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from content_creator.config import AppConfig, ModelConfig
from content_creator.planner import (
    CommunicationMetrics,
    ConversationInsights,
    InteractionStyleAssessment,
    Scene,
    ScenePlan,
    SocialScoreAssessment,
    SpeakerSentimentAssessment,
    TranscriptAssessment,
    VideoPromptPlan,
    VideoPromptPreclassification,
)
from content_creator.pipeline import VideoGenerationPipeline, wrap_transcription


class FakeGateway:
    def __init__(self, config: AppConfig):
        self.config = config
        self.generated_images: list[tuple[str, Path]] = []
        self.diarization_calls: list[
            tuple[int | None, int | None, int | None, float | None]
        ] = []

    def synthesize_speech(self, text: str, destination: Path) -> Path:
        destination.write_bytes(b"audio")
        return destination

    def transcribe_audio(self, audio_path: Path) -> str:
        return f"transcribed {audio_path.stem}"

    def transcribe_audio_with_word_timestamps(self, audio_path: Path):
        return (
            "hello damn world",
            [
                types.SimpleNamespace(word="hello", start_seconds=0.0, end_seconds=0.3),
                types.SimpleNamespace(word="damn", start_seconds=0.4, end_seconds=0.6),
                types.SimpleNamespace(word="world", start_seconds=0.7, end_seconds=1.0),
            ],
        )

    def transcribe_audio_with_speakers(
        self,
        audio_path: Path,
        *,
        speaker_count: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        speaker_dominance_threshold: float | None = None,
    ) -> str:
        self.diarization_calls.append(
            (speaker_count, min_speakers, max_speakers, speaker_dominance_threshold)
        )
        return f"SPEAKER_00: transcribed {audio_path.stem}"

    def classify_content_safety(
        self, text: str, *, model: str | None = None
    ) -> dict[str, object]:
        if "chunk_0001" in text:
            return {
                "model": model or "unitary/unbiased-toxic-roberta",
                "unsafe_score": 0.95,
                "top_label": "toxic",
                "top_score": 0.95,
                "labels": [
                    {"label": "toxic", "score": 0.95},
                    {"label": "non-toxic", "score": 0.05},
                ],
            }
        return {
            "model": model or "unitary/unbiased-toxic-roberta",
            "unsafe_score": 0.05,
            "top_label": "non-toxic",
            "top_score": 0.95,
            "labels": [
                {"label": "non-toxic", "score": 0.95},
                {"label": "toxic", "score": 0.05},
            ],
        }

    def generate_image(self, prompt: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"image")
        self.generated_images.append((prompt, destination))
        return destination

    def generate_text(self, prompt: str) -> str:
        if '"title"' in prompt and '"description"' in prompt:
            return json.dumps(
                {
                    "title": "Quarterly Chaos, Now in Widescreen",
                    "description": "A very official update that accidentally became a comedy special.",
                }
            )
        return "{}"


class FakePlanner:
    def __init__(
        self,
        _gateway: FakeGateway,
        *,
        image_composition_mode: str = "balanced",
        preclassification_ensemble_enabled: bool = True,
        preclass_emotion_model: str | None = None,
        preclass_intent_model: str | None = None,
        safety_primary_model: str | None = None,
        safety_secondary_model: str | None = None,
    ):
        self.calls: list[tuple[str, str, float]] = []
        self.generated_video_prompt_plan_inputs: list[str] = []
        self.prepared_image_prompts: list[tuple[str, int, int | None]] = []

    def generate_video_prompt_plan(
        self, *, narration_text: str, duration_seconds: float | None = None
    ) -> VideoPromptPlan:
        self.generated_video_prompt_plan_inputs.append(narration_text)
        return VideoPromptPlan(
            video_prompt="Generated visual direction",
            preclassification=VideoPromptPreclassification(
                mood="Hopeful",
                has_foul_language=False,
                word_count=12,
                sentence_count=2,
                truthfulness_assessment=TranscriptAssessment(
                    label="MixedOrUnverifiable",
                    confidence_score=0.58,
                    reason="The narration makes general claims without evidence that can be checked from the transcript alone.",
                ),
                fact_check_assessment=TranscriptAssessment(
                    label="MixedOrNeedsEvidence",
                    confidence_score=0.61,
                    reason="The transcript suggests plausible claims but lacks direct evidence for strict verification.",
                ),
                aggression_assessment=TranscriptAssessment(
                    label="Low",
                    confidence_score=0.83,
                    reason="The language is calm and not overtly confrontational.",
                ),
                social_score_assessment=SocialScoreAssessment(
                    prosocial_antisocial=TranscriptAssessment(
                        label="ProSocial",
                        confidence_score=0.72,
                        reason="The framing is constructive and cooperative.",
                    ),
                    cohesion_divisiveness=TranscriptAssessment(
                        label="Cohesive",
                        confidence_score=0.69,
                        reason="The tone encourages shared direction.",
                    ),
                    norm_alignment=TranscriptAssessment(
                        label="Aligned",
                        confidence_score=0.66,
                        reason="Most statements align with broad contemporary social framing.",
                    ),
                    composite_social_score=0.74,
                    composite_label="ProSocial",
                    reason="Composite social score indicates constructive social framing.",
                ),
                contemporary_alignment_assessment=TranscriptAssessment(
                    label="Aligned",
                    confidence_score=0.63,
                    reason="The transcript broadly aligns with contemporary mainstream framing.",
                ),
                propaganda_assessment=TranscriptAssessment(
                    label="Low",
                    confidence_score=0.79,
                    reason="Rhetorical pressure appears limited and not strongly propagandistic.",
                ),
                interaction_style_assessment=InteractionStyleAssessment(
                    formality=TranscriptAssessment(
                        label="Mixed",
                        confidence_score=0.72,
                        reason="The narration blends conversational language with some structured phrasing.",
                    ),
                    certainty_hedging=TranscriptAssessment(
                        label="Balanced",
                        confidence_score=0.63,
                        reason="The wording shows some confidence without sounding absolute.",
                    ),
                    persuasion_intent=TranscriptAssessment(
                        label="LowOrNone",
                        confidence_score=0.79,
                        reason="The transcript mainly informs rather than persuades.",
                    ),
                    claim_density=TranscriptAssessment(
                        label="Medium",
                        confidence_score=0.54,
                        reason="The narration contains some claims but also descriptive filler.",
                    ),
                    speaker_sentiment=[
                        SpeakerSentimentAssessment(
                            speaker="Unknown",
                            sentiment="Neutral",
                            confidence_score=0.46,
                            reason="The tone is mostly even and descriptive.",
                        )
                    ],
                ),
                conversation_insights=ConversationInsights(
                    conversation_type=TranscriptAssessment(
                        label="Meeting",
                        confidence_score=0.71,
                        reason="The exchange appears focused on status and coordination updates.",
                    ),
                    primary_goal=TranscriptAssessment(
                        label="Plan",
                        confidence_score=0.68,
                        reason="The participants discuss progress and next-step alignment.",
                    ),
                    participant_dynamic=TranscriptAssessment(
                        label="Collaborative",
                        confidence_score=0.74,
                        reason="The tone is constructive and mutually aligned.",
                    ),
                    decision_signal=TranscriptAssessment(
                        label="LeaningDecision",
                        confidence_score=0.57,
                        reason="There are directional cues but no explicit final commitment.",
                    ),
                    conflict_level=TranscriptAssessment(
                        label="Low",
                        confidence_score=0.86,
                        reason="No adversarial or confrontational language is present.",
                    ),
                    concise_summary="A collaborative planning conversation with low conflict and an emerging, but not finalized, decision.",
                ),
                communication_metrics=CommunicationMetrics(
                    profanity_word_count=0,
                    profanity_rate=0.0,
                    profanity_per_sentence_ratio=0.0,
                    profanity_to_non_profanity_ratio=0.0,
                    words_per_minute=(
                        round((12 / duration_seconds) * 60.0, 1)
                        if isinstance(duration_seconds, (int, float))
                        and duration_seconds > 0
                        else None
                    ),
                    average_words_per_sentence=6.0,
                    sentence_complexity_score=0.42,
                    sentence_complexity_label="Moderate",
                    communication_capability_score=0.81,
                    communication_capability_label="High",
                    communication_notes="Clear and effective communication with strong collaborative signals.",
                ),
            ),
        )

    def build_scenes(
        self,
        *,
        narration_text: str,
        video_prompt: str,
        total_duration_seconds: float,
        max_scenes: int = 8,
        cinematic_transitions: bool = False,
    ):
        self.calls.append((narration_text, video_prompt, total_duration_seconds))
        return ScenePlan(
            scenes=[
                Scene(index=1, prompt="Prompt A", duration_seconds=2.5),
                Scene(index=2, prompt="Prompt B", duration_seconds=2.5),
            ],
            scene_prompt="fake scene prompt",
        )

    def prepare_image_prompt(
        self, prompt_text: str, *, scene_index: int = 0, total_scenes: int | None = None
    ) -> str:
        self.prepared_image_prompts.append((prompt_text, scene_index, total_scenes))
        return f"{prompt_text} :: prepared {scene_index + 1}/{total_scenes}"

    def compute_chunk_ensemble_scorecard(
        self, chunk_text: str
    ) -> "PreclassificationEnsembleScorecard":
        from content_creator.planner import (
            PreclassificationEnsembleScorecard,
            EnsembleSignal,
        )

        return PreclassificationEnsembleScorecard(
            weighted_risk_score=0.35,
            risk_level="Low",
            recommended_visual_intensity="balanced",
            signals=[
                EnsembleSignal(
                    source="test",
                    model="test_model",
                    label="test_label",
                    confidence_score=0.7,
                    normalized_risk=0.35,
                    weight=1.0,
                    reason="Test signal",
                )
            ],
            warnings=[],
        )


class FakeMedia:
    def __init__(self, *, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps
        self.last_cinematic_intro = None
        self.last_cinematic_transitions = False
        self.last_television_overlay_effects = False
        self.render_calls = 0
        self.mux_calls = 0
        self.last_mux_visual_path: Path | None = None
        self.last_mux_intro_delay_seconds = 0.0

    def get_audio_duration(self, audio_path: Path) -> float:
        return 5.0

    def chunk_audio(
        self, *, audio_path: Path, output_dir: Path, chunk_seconds: float
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk1 = output_dir / "chunk_0000.wav"
        chunk2 = output_dir / "chunk_0001.wav"
        chunk1.write_bytes(b"a")
        chunk2.write_bytes(b"b")
        return [chunk1, chunk2]

    def render_video(
        self,
        *,
        images,
        scenes,
        audio_path: Path,
        output_path: Path,
        work_dir: Path,
        cinematic_intro=None,
        cinematic_transitions: bool = False,
        television_overlay_effects: bool = False,
    ) -> Path:
        self.render_calls += 1
        self.last_cinematic_intro = cinematic_intro
        self.last_cinematic_transitions = cinematic_transitions
        self.last_television_overlay_effects = television_overlay_effects
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"video")
        return output_path

    def mux_visual_with_audio(
        self,
        *,
        visual_path: Path,
        audio_path: Path,
        output_path: Path,
        intro_delay_seconds: float = 0.0,
    ) -> Path:
        self.mux_calls += 1
        self.last_mux_visual_path = visual_path
        self.last_mux_intro_delay_seconds = intro_delay_seconds
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"video")
        return output_path

    def probe_media_file(self, media_path: Path) -> bool:
        return media_path.exists()

    def overlay_sound_effects(
        self, *, audio_path: Path, output_path: Path, events, duck_db: float
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"censored")
        return output_path


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(hf_token="token", work_dir=tmp_path / "work", models=ModelConfig())


def test_transcribe_audio_file_writes_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    status_messages: list[str] = []
    pipeline = VideoGenerationPipeline(
        _config(tmp_path), status_callback=status_messages.append
    )
    audio_path = tmp_path / "input.m4a"
    audio_path.write_bytes(b"audio")
    output_path = tmp_path / "transcript" / "out.txt"

    text = pipeline.transcribe_audio_file(
        audio_path=audio_path, output_path=output_path
    )

    assert text == "transcribed chunk_0000 transcribed chunk_0001"
    assert output_path.read_text(encoding="utf-8") == wrap_transcription(text)
    assert "✂️ Chunking audio into ~45s segments" in status_messages
    assert any("Transcription chunk progress:" in msg for msg in status_messages)
    assert not any("Chunk 1/2 complete" in msg for msg in status_messages)
    assert "✅ Transcription complete" in status_messages


def test_wrap_transcription_splits_long_line() -> None:
    transcript = (
        "this is a long transcript line that should be wrapped for readability "
        "without changing word order"
    )

    wrapped = wrap_transcription(transcript, width=40)

    lines = wrapped.splitlines()
    assert len(lines) > 1
    assert all(len(line) <= 40 for line in lines)
    assert " ".join(lines) == transcript


def _stub_profanity_debug_media_ops(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: VideoGenerationPipeline,
    synthesized_long_speech: list[str],
) -> None:
    def _write_binary(path: Path, payload: bytes = b"audio") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    monkeypatch.setattr(
        pipeline,
        "_synthesize_long_speech",
        lambda text, destination, _tmp: (
            synthesized_long_speech.append(text),
            _write_binary(destination),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_ffmpeg_generate_silence",
        lambda output_path, duration_seconds: _write_binary(output_path),
    )
    monkeypatch.setattr(
        pipeline,
        "_ffmpeg_normalize_audio",
        lambda input_path, output_path: _write_binary(output_path),
    )
    monkeypatch.setattr(
        pipeline,
        "_ffmpeg_extract_audio_segment",
        lambda audio_path, output_path, start_seconds, end_seconds: _write_binary(
            output_path
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_ffmpeg_extract_bleep",
        lambda sfx_path, output_path, duration_seconds: _write_binary(output_path),
    )


def test_build_profanity_debug_audio_generates_live_preclassification_for_append_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    statuses: list[str] = []
    synthesized_long_speech: list[str] = []
    pipeline = VideoGenerationPipeline(
        _config(tmp_path), status_callback=statuses.append
    )
    _stub_profanity_debug_media_ops(monkeypatch, pipeline, synthesized_long_speech)

    def _fake_subprocess_run(args, check, capture_output, text):
        output_path = Path(str(args[-1]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"debug")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline_module.subprocess, "run", _fake_subprocess_run)

    output_path = tmp_path / "debug.m4a"
    event_count = pipeline.build_profanity_debug_audio(
        audio_path=tmp_path / "audio.m4a",
        output_path=output_path,
        manifest_events=[
            {
                "word": "damn",
                "start_seconds": 0.5,
                "end_seconds": 0.7,
                "sfx": str(tmp_path / "beep.wav"),
                "sfx_duration_seconds": 0.2,
            }
        ],
        transcript_text="this is transcript text for live preclassification",
        preclassification_data=None,
        preclassification_position="append",
    )

    assert event_count == 1
    assert output_path.exists()
    assert pipeline._planner.generated_video_prompt_plan_inputs == [
        "this is transcript text for live preclassification"
    ]
    assert any(
        "Generating live pre-classification for debug narration" in status
        for status in statuses
    )
    assert len(synthesized_long_speech) == 2
    assert "Pre-classification report." not in synthesized_long_speech[0]
    assert "Mood: Hopeful." in synthesized_long_speech[1]
    assert "Profanity per sentence ratio:" in synthesized_long_speech[1]
    assert "Profanity to non-profanity ratio:" in synthesized_long_speech[1]
    assert "Sentence complexity:" in synthesized_long_speech[1]


def test_build_profanity_debug_audio_prepend_mode_includes_preclassification_in_intro(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    synthesized_long_speech: list[str] = []
    pipeline = VideoGenerationPipeline(_config(tmp_path))
    _stub_profanity_debug_media_ops(monkeypatch, pipeline, synthesized_long_speech)
    monkeypatch.setattr(
        pipeline_module.subprocess,
        "run",
        lambda args, check, capture_output, text: types.SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )

    pipeline.build_profanity_debug_audio(
        audio_path=tmp_path / "audio.m4a",
        output_path=tmp_path / "debug.m4a",
        manifest_events=[
            {
                "word": "damn",
                "start_seconds": 0.5,
                "end_seconds": 0.7,
                "sfx": str(tmp_path / "beep.wav"),
                "sfx_duration_seconds": 0.2,
            }
        ],
        preclassification_data={
            "mood": "Serious",
            "has_foul_language": True,
            "communication_metrics": {
                "profanity_per_sentence_ratio": 0.5,
                "profanity_to_non_profanity_ratio": 0.125,
                "sentence_complexity_score": 0.62,
                "sentence_complexity_label": "Moderate",
            },
        },
        preclassification_position="prepend",
    )

    assert len(synthesized_long_speech) == 2
    assert "Pre-classification report." in synthesized_long_speech[0]
    assert "mood: Serious" in synthesized_long_speech[0]
    assert "Profanity per sentence ratio: 0.5000." in synthesized_long_speech[0]
    assert "Profanity to non-profanity ratio: 0.1250." in synthesized_long_speech[0]
    assert "Sentence complexity: Moderate at 0.6200." in synthesized_long_speech[0]
    assert "Mood: Serious." not in synthesized_long_speech[1]


def test_build_profanity_debug_audio_off_mode_excludes_preclassification_narration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    synthesized_long_speech: list[str] = []
    pipeline = VideoGenerationPipeline(_config(tmp_path))
    _stub_profanity_debug_media_ops(monkeypatch, pipeline, synthesized_long_speech)
    monkeypatch.setattr(
        pipeline_module.subprocess,
        "run",
        lambda args, check, capture_output, text: types.SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )

    pipeline.build_profanity_debug_audio(
        audio_path=tmp_path / "audio.m4a",
        output_path=tmp_path / "debug.m4a",
        manifest_events=[
            {
                "word": "damn",
                "start_seconds": 0.5,
                "end_seconds": 0.7,
                "sfx": str(tmp_path / "beep.wav"),
                "sfx_duration_seconds": 0.2,
            }
        ],
        preclassification_data={"mood": "Serious", "has_foul_language": True},
        preclassification_position="off",
    )

    assert len(synthesized_long_speech) == 2
    assert "Pre-classification report." not in synthesized_long_speech[0]
    assert "Mood: Serious." not in synthesized_long_speech[1]


def test_generate_from_text_writes_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    statuses: list[str] = []
    pipeline = VideoGenerationPipeline(
        _config(tmp_path), debug=True, status_callback=statuses.append
    )
    output_path = tmp_path / "out" / "result.mp4"

    result = pipeline.generate_from_text(
        narration_text="Narration",
        video_prompt="Video direction",
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()

    run_dir = _config(tmp_path).work_dir / output_path.stem
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["narration_text"] == "Narration"
    assert manifest["video_prompt"] == "Video direction"
    assert isinstance(manifest["video_prompt_preclassification"], dict)
    assert manifest["video_prompt_preclassification"]["mood"] == "Hopeful"
    assert len(manifest["scenes"]) == 2
    assert manifest["scenes"][0]["prepared_prompt"] == "Prompt A :: prepared 1/2"
    assert manifest["scenes"][1]["prepared_prompt"] == "Prompt B :: prepared 2/2"
    assert pipeline._planner.prepared_image_prompts == [
        ("Prompt A", 0, 2),
        ("Prompt B", 1, 2),
    ]
    assert pipeline._gateway.generated_images[0][0].endswith("prepared 1/2")
    assert any(msg.startswith("🐛 Rendering image for scene") for msg in statuses)


def test_generate_from_audio_applies_profanity_sfx_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    sound_dir = tmp_path / "sound"
    sound_dir.mkdir(parents=True, exist_ok=True)
    (sound_dir / "button.wav").write_bytes(b"sound")

    monkeypatch.setattr(
        pipeline_module,
        "load_sound_pack",
        lambda sound_pack_dir: types.SimpleNamespace(
            name="test-pack",
            root_dir=sound_pack_dir,
            target_mean_db=-18.0,
            assets=[
                types.SimpleNamespace(
                    path=sound_pack_dir / "button.wav",
                    duration_seconds=0.2,
                    mean_volume_db=-20.0,
                    max_volume_db=-5.0,
                )
            ],
        ),
    )
    monkeypatch.setattr(pipeline_module, "load_profanity_words", lambda _path: {"damn"})
    monkeypatch.setattr(
        pipeline_module,
        "build_profanity_sfx_plan",
        lambda **_kwargs: types.SimpleNamespace(
            sound_pack_name="test-pack",
            sound_pack_dir=sound_dir,
            total_words=3,
            matches_found=1,
            events=[
                types.SimpleNamespace(
                    word="damn",
                    start_seconds=0.32,
                    end_seconds=0.68,
                    sfx_path=sound_dir / "button.wav",
                    sfx_duration_seconds=0.2,
                    sfx_gain_db=2.0,
                )
            ],
        ),
    )

    pipeline = VideoGenerationPipeline(_config(tmp_path))
    audio_path = tmp_path / "input.m4a"
    audio_path.write_bytes(b"audio")
    output_path = tmp_path / "out" / "video.mp4"

    pipeline.generate_from_audio(
        audio_path=audio_path,
        video_prompt="style",
        output_path=output_path,
        profanity_sfx_enabled=True,
        profanity_sound_pack_dir=sound_dir,
    )

    manifest_path = _config(tmp_path).work_dir / output_path.stem / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["profanity_sfx"]["events_applied"] == 1
    assert "audio_censored.m4a" in manifest["profanity_sfx"]["output_audio"]


def test_generate_from_text_uses_threaded_image_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    statuses: list[str] = []
    pipeline = VideoGenerationPipeline(
        _config(tmp_path), status_callback=statuses.append
    )
    output_path = tmp_path / "out" / "threaded-images.mp4"

    result = pipeline.generate_from_text(
        narration_text="Narration",
        video_prompt="Video direction",
        output_path=output_path,
        image_workers=2,
    )

    assert result == output_path
    assert output_path.exists()
    assert any("Using 2 workers for image generation" in msg for msg in statuses)


def test_generate_from_text_generates_multiple_images_per_scene(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    pipeline = VideoGenerationPipeline(_config(tmp_path))
    output_path = tmp_path / "out" / "multi-images.mp4"

    result = pipeline.generate_from_text(
        narration_text="Narration",
        video_prompt="Video direction",
        output_path=output_path,
        images_per_scene=3,
    )

    assert result == output_path
    assert output_path.exists()
    assert len(pipeline._gateway.generated_images) == 6
    assert len(pipeline._planner.prepared_image_prompts) == 6
    assert "Frame 1/3" in pipeline._gateway.generated_images[0][0]

    manifest_path = _config(tmp_path).work_dir / output_path.stem / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["images_per_scene"] == 3
    assert len(manifest["images"]) == 6
    assert len(manifest["scenes"][0]["prepared_prompts"]) == 3
    assert len(manifest["scenes"][1]["prepared_prompts"]) == 3


def test_generate_from_text_can_generate_video_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    statuses: list[str] = []
    pipeline = VideoGenerationPipeline(
        _config(tmp_path), status_callback=statuses.append
    )
    output_path = tmp_path / "out" / "result.mp4"

    pipeline.generate_from_text(
        narration_text="Narration",
        video_prompt=None,
        generate_video_prompt=True,
        output_path=output_path,
    )

    assert pipeline._planner.generated_video_prompt_plan_inputs == ["Narration"]
    assert "🧪 Preclassifying transcript for visual planning" in statuses
    assert "🪄 Generating video prompt from narration" in statuses

    manifest_path = _config(tmp_path).work_dir / output_path.stem / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["video_prompt"] == "Generated visual direction"
    assert manifest["video_prompt_preclassification"] == {
        "mood": "Hopeful",
        "has_foul_language": False,
        "word_count": 12,
        "sentence_count": 2,
        "truthfulness_assessment": {
            "label": "MixedOrUnverifiable",
            "confidence_score": 0.58,
            "reason": "The narration makes general claims without evidence that can be checked from the transcript alone.",
        },
        "fact_check_assessment": {
            "label": "MixedOrNeedsEvidence",
            "confidence_score": 0.61,
            "reason": "The transcript suggests plausible claims but lacks direct evidence for strict verification.",
        },
        "aggression_assessment": {
            "label": "Low",
            "confidence_score": 0.83,
            "reason": "The language is calm and not overtly confrontational.",
        },
        "social_score_assessment": {
            "prosocial_antisocial": {
                "label": "ProSocial",
                "confidence_score": 0.72,
                "reason": "The framing is constructive and cooperative.",
            },
            "cohesion_divisiveness": {
                "label": "Cohesive",
                "confidence_score": 0.69,
                "reason": "The tone encourages shared direction.",
            },
            "norm_alignment": {
                "label": "Aligned",
                "confidence_score": 0.66,
                "reason": "Most statements align with broad contemporary social framing.",
            },
            "composite_social_score": 0.74,
            "composite_label": "ProSocial",
            "reason": "Composite social score indicates constructive social framing.",
        },
        "contemporary_alignment_assessment": {
            "label": "Aligned",
            "confidence_score": 0.63,
            "reason": "The transcript broadly aligns with contemporary mainstream framing.",
        },
        "propaganda_assessment": {
            "label": "Low",
            "confidence_score": 0.79,
            "reason": "Rhetorical pressure appears limited and not strongly propagandistic.",
        },
        "interaction_style_assessment": {
            "formality": {
                "label": "Mixed",
                "confidence_score": 0.72,
                "reason": "The narration blends conversational language with some structured phrasing.",
            },
            "certainty_hedging": {
                "label": "Balanced",
                "confidence_score": 0.63,
                "reason": "The wording shows some confidence without sounding absolute.",
            },
            "persuasion_intent": {
                "label": "LowOrNone",
                "confidence_score": 0.79,
                "reason": "The transcript mainly informs rather than persuades.",
            },
            "claim_density": {
                "label": "Medium",
                "confidence_score": 0.54,
                "reason": "The narration contains some claims but also descriptive filler.",
            },
            "speaker_sentiment": [
                {
                    "speaker": "Unknown",
                    "sentiment": "Neutral",
                    "confidence_score": 0.46,
                    "reason": "The tone is mostly even and descriptive.",
                }
            ],
        },
        "conversation_insights": {
            "conversation_type": {
                "label": "Meeting",
                "confidence_score": 0.71,
                "reason": "The exchange appears focused on status and coordination updates.",
            },
            "primary_goal": {
                "label": "Plan",
                "confidence_score": 0.68,
                "reason": "The participants discuss progress and next-step alignment.",
            },
            "participant_dynamic": {
                "label": "Collaborative",
                "confidence_score": 0.74,
                "reason": "The tone is constructive and mutually aligned.",
            },
            "decision_signal": {
                "label": "LeaningDecision",
                "confidence_score": 0.57,
                "reason": "There are directional cues but no explicit final commitment.",
            },
            "conflict_level": {
                "label": "Low",
                "confidence_score": 0.86,
                "reason": "No adversarial or confrontational language is present.",
            },
            "concise_summary": "A collaborative planning conversation with low conflict and an emerging, but not finalized, decision.",
        },
        "communication_metrics": {
            "profanity_word_count": 0,
            "profanity_rate": 0.0,
            "profanity_per_sentence_ratio": 0.0,
            "profanity_to_non_profanity_ratio": 0.0,
            "words_per_minute": 144.0,
            "average_words_per_sentence": 6.0,
            "sentence_complexity_score": 0.42,
            "sentence_complexity_label": "Moderate",
            "communication_capability_score": 0.81,
            "communication_capability_label": "High",
            "communication_notes": "Clear and effective communication with strong collaborative signals.",
        },
    }
    analysis_summary = manifest.get("analysis_summary")
    assert isinstance(analysis_summary, dict)
    summary_path = Path(str(analysis_summary["path"]))
    assert summary_path.exists()
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert (
        summary_payload["dimensions"]["fact_check"]["label"] == "MixedOrNeedsEvidence"
    )
    assert summary_payload["dimensions"]["aggression"]["label"] == "Low"
    assert (
        summary_payload["dimensions"]["social_score"]["composite_label"] == "ProSocial"
    )
    assert summary_payload["dimensions"]["contemporary_alignment"]["label"] == "Aligned"
    assert summary_payload["dimensions"]["propaganda_alignment"]["label"] == "Low"


def test_generate_from_text_includes_cinematic_intro_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    pipeline = VideoGenerationPipeline(_config(tmp_path))
    output_path = tmp_path / "out" / "cinematic-intro.mp4"

    pipeline.generate_from_text(
        narration_text="Narration text",
        video_prompt="Video direction",
        output_path=output_path,
        cinematic_intro=True,
    )

    manifest_path = _config(tmp_path).work_dir / output_path.stem / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["cinematic_intro"]["enabled"] is True
    assert manifest["cinematic_intro"]["title"] == "Quarterly Chaos, Now in Widescreen"
    assert "comedy special" in manifest["cinematic_intro"]["description"]

    intro_card = pipeline._media.last_cinematic_intro
    assert intro_card is not None
    assert intro_card.title == "Quarterly Chaos, Now in Widescreen"


def test_generate_from_text_uses_custom_cinematic_intro_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    pipeline = VideoGenerationPipeline(_config(tmp_path))
    output_path = tmp_path / "out" / "cinematic-intro-custom-duration.mp4"

    pipeline.generate_from_text(
        narration_text="Narration text",
        video_prompt="Video direction",
        output_path=output_path,
        cinematic_intro=True,
        cinematic_intro_duration=7.25,
    )

    manifest_path = _config(tmp_path).work_dir / output_path.stem / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["cinematic_intro"]["duration_seconds"] == pytest.approx(7.25)

    intro_card = pipeline._media.last_cinematic_intro
    assert intro_card is not None
    assert intro_card.duration_seconds == pytest.approx(7.25)


def test_generate_from_text_enables_cinematic_transitions_when_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    pipeline = VideoGenerationPipeline(_config(tmp_path))
    output_path = tmp_path / "out" / "cinematic-transitions.mp4"

    pipeline.generate_from_text(
        narration_text="Narration text",
        video_prompt="Video direction",
        output_path=output_path,
        cinematic_transitions=True,
    )

    manifest_path = _config(tmp_path).work_dir / output_path.stem / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["cinematic_transitions"] is True
    assert pipeline._media.last_cinematic_transitions is True


def test_generate_from_text_enables_television_overlay_effects_when_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    pipeline = VideoGenerationPipeline(_config(tmp_path))
    output_path = tmp_path / "out" / "television-overlay-effects.mp4"

    pipeline.generate_from_text(
        narration_text="Narration text",
        video_prompt="Video direction",
        output_path=output_path,
        television_overlay_effects=True,
    )

    manifest_path = _config(tmp_path).work_dir / output_path.stem / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["television_overlay_effects"] is True
    assert pipeline._media.last_television_overlay_effects is True


def test_rebuild_video_from_run_reuses_existing_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    pipeline = VideoGenerationPipeline(_config(tmp_path))
    run_dir = _config(tmp_path).work_dir / "existing-run"
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_1 = images_dir / "scene_01_frame_01.png"
    image_2 = images_dir / "scene_02_frame_01.png"
    image_1.write_bytes(b"img1")
    image_2.write_bytes(b"img2")

    audio_path = run_dir / "audio_censored.m4a"
    audio_path.write_bytes(b"audio")
    stitched_with_intro = run_dir / "stitched_with_intro.mp4"
    stitched_with_intro.write_bytes(b"video")
    output_path = tmp_path / "out" / "rebuilt.mp4"

    manifest = {
        "audio": str(audio_path),
        "output": str(output_path),
        "cinematic_transitions": True,
        "television_overlay_effects": False,
        "cinematic_intro": {
            "enabled": True,
            "title": "Recovered Title",
            "description": "Recovered Description",
            "duration_seconds": 6.1,
        },
        "scenes": [
            {"index": 1, "prompt": "Scene one", "duration_seconds": 2.5},
            {"index": 2, "prompt": "Scene two", "duration_seconds": 2.5},
        ],
        "images": [str(image_1), str(image_2)],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = pipeline.rebuild_video_from_run(run_dir=run_dir, output_path=output_path)

    assert result == output_path.resolve()
    assert output_path.exists()
    assert pipeline._media.render_calls == 0
    assert pipeline._media.mux_calls == 1
    assert pipeline._media.last_mux_visual_path == stitched_with_intro
    assert pipeline._media.last_mux_intro_delay_seconds == pytest.approx(6.1)

    updated_manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert updated_manifest["status"] == "complete"
    assert updated_manifest["output"] == str(output_path.resolve())


def test_rebuild_video_from_run_accepts_feature_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    pipeline = VideoGenerationPipeline(_config(tmp_path))
    run_dir = _config(tmp_path).work_dir / "existing-run-2"
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "scene_01_frame_01.png").write_bytes(b"img1")
    (images_dir / "scene_02_frame_01.png").write_bytes(b"img2")
    audio_path = run_dir / "audio_censored.m4a"
    audio_path.write_bytes(b"audio")
    (run_dir / "stitched_with_intro.mp4").write_bytes(b"video")
    (run_dir / "stitched_tv_effects.mp4").write_bytes(b"video")
    output_path = tmp_path / "out" / "rebuilt-overrides.mp4"

    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "audio": str(audio_path),
                "output": str(output_path),
                "cinematic_transitions": False,
                "television_overlay_effects": False,
                "cinematic_intro": {
                    "enabled": True,
                    "title": "Old",
                    "description": "Old description",
                    "duration_seconds": 5.8,
                },
                "scenes": [
                    {"index": 1, "prompt": "Scene one", "duration_seconds": 2.5},
                    {"index": 2, "prompt": "Scene two", "duration_seconds": 2.5},
                ],
            }
        ),
        encoding="utf-8",
    )

    pipeline.rebuild_video_from_run(
        run_dir=run_dir,
        output_path=output_path,
        cinematic_intro_enabled=False,
        cinematic_transitions=True,
        television_overlay_effects=True,
    )

    assert pipeline._media.last_cinematic_intro is None
    assert pipeline._media.last_cinematic_transitions is True
    assert pipeline._media.last_television_overlay_effects is True
    assert pipeline._media.render_calls == 1
    assert pipeline._media.mux_calls == 0


def test_rebuild_video_from_run_falls_back_when_reused_visual_is_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    pipeline = VideoGenerationPipeline(_config(tmp_path))
    run_dir = _config(tmp_path).work_dir / "existing-run-invalid-visual"
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_1 = images_dir / "scene_01_frame_01.png"
    image_2 = images_dir / "scene_02_frame_01.png"
    image_1.write_bytes(b"img1")
    image_2.write_bytes(b"img2")

    audio_path = run_dir / "audio_censored.m4a"
    audio_path.write_bytes(b"audio")
    (run_dir / "stitched_with_intro.mp4").write_bytes(b"video")
    output_path = tmp_path / "out" / "rebuilt-invalid-visual.mp4"

    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "audio": str(audio_path),
                "output": str(output_path),
                "cinematic_transitions": True,
                "television_overlay_effects": False,
                "cinematic_intro": {
                    "enabled": True,
                    "title": "Recovered Title",
                    "description": "Recovered Description",
                    "duration_seconds": 6.1,
                },
                "scenes": [
                    {"index": 1, "prompt": "Scene one", "duration_seconds": 2.5},
                    {"index": 2, "prompt": "Scene two", "duration_seconds": 2.5},
                ],
                "images": [str(image_1), str(image_2)],
            }
        ),
        encoding="utf-8",
    )

    pipeline._media.probe_media_file = lambda _path: False

    pipeline.rebuild_video_from_run(run_dir=run_dir, output_path=output_path)

    assert pipeline._media.render_calls == 1
    assert pipeline._media.mux_calls == 0


def test_rebuild_video_from_run_can_disable_visual_asset_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    pipeline = VideoGenerationPipeline(_config(tmp_path))
    run_dir = _config(tmp_path).work_dir / "existing-run-3"
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_1 = images_dir / "scene_01_frame_01.png"
    image_2 = images_dir / "scene_02_frame_01.png"
    image_1.write_bytes(b"img1")
    image_2.write_bytes(b"img2")

    audio_path = run_dir / "audio_censored.m4a"
    audio_path.write_bytes(b"audio")
    (run_dir / "stitched_with_intro.mp4").write_bytes(b"video")
    output_path = tmp_path / "out" / "rebuilt-no-reuse.mp4"

    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "audio": str(audio_path),
                "output": str(output_path),
                "cinematic_transitions": True,
                "television_overlay_effects": False,
                "cinematic_intro": {
                    "enabled": True,
                    "title": "Recovered Title",
                    "description": "Recovered Description",
                    "duration_seconds": 6.1,
                },
                "scenes": [
                    {"index": 1, "prompt": "Scene one", "duration_seconds": 2.5},
                    {"index": 2, "prompt": "Scene two", "duration_seconds": 2.5},
                ],
                "images": [str(image_1), str(image_2)],
            }
        ),
        encoding="utf-8",
    )

    pipeline.rebuild_video_from_run(
        run_dir=run_dir, output_path=output_path, reuse_visual_assets=False
    )

    assert pipeline._media.render_calls == 1
    assert pipeline._media.mux_calls == 0


def test_transcribe_audio_file_falls_back_without_ffmpeg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: None)

    statuses: list[str] = []
    pipeline = VideoGenerationPipeline(
        _config(tmp_path), status_callback=statuses.append
    )
    audio_path = tmp_path / "input.m4a"
    audio_path.write_bytes(b"audio")

    text = pipeline.transcribe_audio_file(audio_path=audio_path, chunk_seconds=30.0)

    assert text == "transcribed input"
    assert "⚠️ ffmpeg not found; falling back to single-pass transcription" in statuses


def test_transcribe_audio_file_preserves_speaker_labels_with_chunking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    statuses: list[str] = []
    pipeline = VideoGenerationPipeline(
        _config(tmp_path), status_callback=statuses.append
    )
    audio_path = tmp_path / "dialog.wav"
    audio_path.write_bytes(b"audio")

    text = pipeline.transcribe_audio_file(
        audio_path=audio_path, preserve_speaker=True, chunk_seconds=45.0
    )

    assert (
        text == "SPEAKER_00: transcribed chunk_0000\nSPEAKER_00: transcribed chunk_0001"
    )
    assert "✂️ Chunking audio into ~45s segments" in statuses
    assert "🧩 Processing 2 chunks with speaker diarization" in statuses


def test_transcribe_audio_file_preserves_speaker_labels_full_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)

    statuses: list[str] = []
    pipeline = VideoGenerationPipeline(
        _config(tmp_path), status_callback=statuses.append
    )
    audio_path = tmp_path / "dialog.wav"
    audio_path.write_bytes(b"audio")

    text = pipeline.transcribe_audio_file(
        audio_path=audio_path, preserve_speaker=True, chunk_seconds=0
    )

    assert text == "SPEAKER_00: transcribed dialog"
    assert "🧩 Transcribing audio with speaker diarization (full file)" in statuses


def test_transcribe_audio_file_forwards_diarization_speaker_constraints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)

    pipeline = VideoGenerationPipeline(_config(tmp_path))
    audio_path = tmp_path / "dialog.wav"
    audio_path.write_bytes(b"audio")

    pipeline.transcribe_audio_file(
        audio_path=audio_path,
        preserve_speaker=True,
        chunk_seconds=0,
        diarization_speaker_count=1,
    )

    assert pipeline._gateway.diarization_calls == [(1, None, None, 0.9)]


def test_transcribe_audio_file_filters_flagged_chunks_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    statuses: list[str] = []
    pipeline = VideoGenerationPipeline(
        _config(tmp_path), status_callback=statuses.append
    )
    audio_path = tmp_path / "input.m4a"
    audio_path.write_bytes(b"audio")

    text = pipeline.transcribe_audio_file(
        audio_path=audio_path,
        chunk_seconds=45.0,
        content_safety_enabled=True,
        content_safety_filter=True,
        content_safety_threshold=0.7,
    )

    assert text == "transcribed chunk_0000"
    assert any("Filtered chunk" in status for status in statuses)
    assert any("Content safety summary" in status for status in statuses)


def test_transcribe_audio_file_uses_threaded_workers_and_keeps_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    statuses: list[str] = []
    pipeline = VideoGenerationPipeline(
        _config(tmp_path), status_callback=statuses.append
    )
    audio_path = tmp_path / "threaded.m4a"
    audio_path.write_bytes(b"audio")

    text = pipeline.transcribe_audio_file(
        audio_path=audio_path, chunk_seconds=45.0, transcribe_workers=3
    )

    assert text == "transcribed chunk_0000 transcribed chunk_0001"
    assert any("Using 3 workers" in status for status in statuses)


def test_missing_video_dependencies_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)

    def _missing(_name: str) -> None:
        return None

    monkeypatch.setattr(pipeline_module.shutil, "which", _missing)

    pipeline = VideoGenerationPipeline(_config(tmp_path))

    with pytest.raises(RuntimeError, match="Missing required system dependencies"):
        pipeline.generate_from_text(
            narration_text="Narration",
            video_prompt="Visual direction",
            output_path=tmp_path / "out.mp4",
        )


# ---------------------------------------------------------------------------
# Combined lexicon + ML content safety tests
# ---------------------------------------------------------------------------


def _make_safety_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> "VideoGenerationPipeline":
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    return VideoGenerationPipeline(_config(tmp_path))


def _lexicon_file(tmp_path: Path, words: list[str]) -> Path:
    """Write a profanity words file and return its path."""
    path = tmp_path / "profanity_words.txt"
    path.write_text("\n".join(words), encoding="utf-8")
    return path


def test_content_safety_lexicon_only_flagging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Lexicon word in transcript with ML score below threshold → lexicon_flagged only."""
    pipeline = _make_safety_pipeline(monkeypatch, tmp_path)
    audio_path = tmp_path / "input.m4a"
    audio_path.write_bytes(b"audio")
    # FakeGateway.transcribe_audio returns "transcribed input"
    # FakeGateway.classify_content_safety returns unsafe_score=0.05 for this text
    words_file = _lexicon_file(tmp_path, ["input"])

    pipeline.transcribe_audio_file(
        audio_path=audio_path,
        chunk_seconds=0,
        content_safety_enabled=True,
        content_safety_filter=False,
        content_safety_threshold=0.5,  # 0.05 < 0.5 → ML not flagged
        profanity_words_file=words_file,
    )

    report = pipeline._last_content_safety_report
    assert report is not None
    full_audio = report["full_audio"]
    assert isinstance(full_audio, dict)
    assert full_audio["flagged"] is True
    assert full_audio["ml_flagged"] is False
    assert full_audio["lexicon_flagged"] is True
    assert "input" in full_audio["lexicon_matched"]


def test_content_safety_ml_only_flagging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ML score above threshold with no lexicon match → ml_flagged only."""
    pipeline = _make_safety_pipeline(monkeypatch, tmp_path)
    audio_path = tmp_path / "input.m4a"
    audio_path.write_bytes(b"audio")
    # FakeGateway.classify_content_safety returns unsafe_score=0.05
    # Set threshold below that to trigger ML flag
    words_file = _lexicon_file(tmp_path, ["xyznotaword"])

    pipeline.transcribe_audio_file(
        audio_path=audio_path,
        chunk_seconds=0,
        content_safety_enabled=True,
        content_safety_filter=False,
        content_safety_threshold=0.01,  # 0.05 >= 0.01 → ML flagged
        profanity_words_file=words_file,
    )

    report = pipeline._last_content_safety_report
    assert report is not None
    full_audio = report["full_audio"]
    assert isinstance(full_audio, dict)
    assert full_audio["flagged"] is True
    assert full_audio["ml_flagged"] is True
    assert full_audio["lexicon_flagged"] is False
    assert full_audio["lexicon_matched"] == []


def test_content_safety_both_ml_and_lexicon_flagging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both ML and lexicon signals trigger → flagged, ml_flagged, lexicon_flagged all True."""
    pipeline = _make_safety_pipeline(monkeypatch, tmp_path)
    audio_path = tmp_path / "input.m4a"
    audio_path.write_bytes(b"audio")
    words_file = _lexicon_file(tmp_path, ["input"])

    pipeline.transcribe_audio_file(
        audio_path=audio_path,
        chunk_seconds=0,
        content_safety_enabled=True,
        content_safety_filter=False,
        content_safety_threshold=0.01,  # ML triggers; lexicon "input" also matches
        profanity_words_file=words_file,
    )

    report = pipeline._last_content_safety_report
    assert report is not None
    full_audio = report["full_audio"]
    assert isinstance(full_audio, dict)
    assert full_audio["flagged"] is True
    assert full_audio["ml_flagged"] is True
    assert full_audio["lexicon_flagged"] is True
    assert "input" in full_audio["lexicon_matched"]


def test_content_safety_neither_signal_flagging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No ML trigger and no lexicon match → all flagging fields False."""
    pipeline = _make_safety_pipeline(monkeypatch, tmp_path)
    audio_path = tmp_path / "input.m4a"
    audio_path.write_bytes(b"audio")
    words_file = _lexicon_file(tmp_path, ["xyznotaword"])

    pipeline.transcribe_audio_file(
        audio_path=audio_path,
        chunk_seconds=0,
        content_safety_enabled=True,
        content_safety_filter=False,
        content_safety_threshold=0.5,
        profanity_words_file=words_file,
    )

    report = pipeline._last_content_safety_report
    assert report is not None
    full_audio = report["full_audio"]
    assert isinstance(full_audio, dict)
    assert full_audio["flagged"] is False
    assert full_audio["ml_flagged"] is False
    assert full_audio["lexicon_flagged"] is False
    assert full_audio["lexicon_matched"] == []


def test_content_safety_chunk_lexicon_flagging_filters_chunk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Lexicon word in a chunk causes that chunk to be filtered when filter is enabled."""
    import content_creator.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HuggingFaceGateway", FakeGateway)
    monkeypatch.setattr(pipeline_module, "ScenePlanner", FakePlanner)
    monkeypatch.setattr(pipeline_module, "MediaAssembler", FakeMedia)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "/usr/bin/fake")

    pipeline = VideoGenerationPipeline(_config(tmp_path))
    audio_path = tmp_path / "input.m4a"
    audio_path.write_bytes(b"audio")
    # FakeGateway chunks return "transcribed chunk_0000" and "transcribed chunk_0001"
    # ML: unsafe_score=0.95 for chunk_0001; 0.05 for chunk_0000
    # Set threshold high (0.99) so ML does NOT flag either chunk.
    # Lexicon contains "chunk_0001" so lexicon flags chunk_0001 only.
    words_file = _lexicon_file(tmp_path, ["chunk_0001"])

    statuses: list[str] = []
    pipeline._status_callback = statuses.append

    text = pipeline.transcribe_audio_file(
        audio_path=audio_path,
        chunk_seconds=45.0,
        content_safety_enabled=True,
        content_safety_filter=True,
        content_safety_threshold=0.99,
        profanity_words_file=words_file,
    )

    assert text == "transcribed chunk_0000"
    report = pipeline._last_content_safety_report
    assert report is not None
    assert report["dropped_chunks"] == 1
    chunks = report["chunks"]
    assert isinstance(chunks, list)
    flagged_chunks = [c for c in chunks if c["lexicon_flagged"]]
    assert len(flagged_chunks) == 1
    # Digits are leet-normalised during scanning so the matched entry is the
    # normalised form; just assert at least one lexicon entry was recorded.
    assert len(flagged_chunks[0]["lexicon_matched"]) > 0
