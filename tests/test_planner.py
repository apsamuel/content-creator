from __future__ import annotations

import json

from content_creator.planner import ScenePlanner


class StubLLM:
    def __init__(self, response: str | list[str]):
        if isinstance(response, list):
            self.responses = response
        else:
            self.responses = [response]
        self.prompts: list[str] = []

    def generate_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


def test_build_scenes_uses_llm_json_payload() -> None:
    llm = StubLLM(
        json.dumps(
            {
                "story_anchor": "Mira, a rabbit-eared traveler in a glowing plush forest, keeps the same blue cloak and watercolor anime style",
                "scenes": [
                    {
                        "summary": "Mira enters the clearing",
                        "prompt": "Mira steps into a sunrise clearing.",
                    },
                    {
                        "summary": "She follows the lanterns",
                        "continuity": "same cloak, same clearing, same lantern trail",
                        "prompt": "Mira walks deeper along glowing lanterns.",
                    },
                    {
                        "summary": "She finds the pond",
                        "continuity": "same traveler and lantern light reflected in water",
                        "prompt": "Mira reaches a still pond under night light.",
                    },
                ],
            }
        )
    )
    planner = ScenePlanner(llm)

    scenes = planner.build_scenes(
        narration_text="A short city montage.",
        video_prompt="Cinematic urban documentary style.",
        total_duration_seconds=12.0,
    )

    assert len(scenes) == 3
    assert scenes[0].prompt.startswith(
        "Mira, a rabbit-eared traveler in a glowing plush forest"
    )
    assert scenes[0].prompt.endswith("Mira steps into a sunrise clearing.")
    assert (
        "Carry forward: same cloak, same clearing, same lantern trail"
        in scenes[1].prompt
    )
    assert "Previous beat: Mira enters the clearing" in scenes[1].prompt
    assert scenes[1].index == 2
    assert round(sum(scene.duration_seconds for scene in scenes), 2) == 12.0
    assert '"story_anchor"' in llm.prompts[0]
    assert '"continuity"' in llm.prompts[0]
    assert "Style template" in llm.prompts[0]
    assert "Scene 1 narration chunk" in llm.prompts[0]


def test_build_scenes_falls_back_to_video_prompt_when_json_missing() -> None:
    llm = StubLLM("no json in this response")
    planner = ScenePlanner(llm)

    scenes = planner.build_scenes(
        narration_text="Some narration",
        video_prompt="Fallback visual direction",
        total_duration_seconds=9.0,
    )

    assert len(scenes) >= 3
    assert all(scene.prompt == "Fallback visual direction" for scene in scenes)


def test_generate_video_prompt_uses_llm_text() -> None:
    llm = StubLLM(
        [
            json.dumps(
                {
                    "mood": "Happy",
                    "has_foul_language": "No",
                    "video_prompt": "Stylized cartoon forest with a rabbit-eared heroine in a blue cloak, warm dawn glow, soft painted textures, consistent character design, expressive illustrated composition",
                }
            ),
            json.dumps(
                {
                    "truthfulness": {
                        "label": "MixedOrUnverifiable",
                        "confidence_score": 0.61,
                        "reason": "The transcript is descriptive but does not include enough evidence to verify its implied claims.",
                    },
                    "formality": {
                        "label": "Mixed",
                        "confidence_score": 0.73,
                        "reason": "The wording blends conversational phrasing with some descriptive structure.",
                    },
                    "certainty_hedging": {
                        "label": "Balanced",
                        "confidence_score": 0.66,
                        "reason": "The transcript makes observations without extreme certainty or heavy hedging.",
                    },
                    "persuasion_intent": {
                        "label": "LowOrNone",
                        "confidence_score": 0.81,
                        "reason": "The speaker mainly describes events rather than trying to convince the listener.",
                    },
                    "claim_density": {
                        "label": "Medium",
                        "confidence_score": 0.57,
                        "reason": "The narration contains some assertions but remains mostly atmospheric.",
                    },
                    "speaker_sentiment": [
                        {
                            "speaker": "Unknown",
                            "sentiment": "Positive",
                            "confidence_score": 0.62,
                            "reason": "The speaker uses curious and appreciative language.",
                        }
                    ],
                }
            ),
        ]
    )
    planner = ScenePlanner(llm)

    plan = planner.generate_video_prompt_plan(
        narration_text="Mira follows floating lanterns through a plush forest at dawn."
    )

    assert plan.video_prompt.startswith("Stylized cartoon forest")
    assert plan.preclassification is not None
    assert plan.preclassification.mood == "Happy"
    assert plan.preclassification.has_foul_language is False
    assert plan.preclassification.word_count == 10
    assert plan.preclassification.sentence_count == 1
    assert plan.preclassification.truthfulness_assessment.label == "MixedOrUnverifiable"
    assert plan.preclassification.truthfulness_assessment.confidence_score == 0.61
    assert (
        "verify its implied claims"
        in plan.preclassification.truthfulness_assessment.reason
    )
    assert (
        plan.preclassification.interaction_style_assessment.formality.label == "Mixed"
    )
    assert (
        plan.preclassification.interaction_style_assessment.certainty_hedging.label
        == "Balanced"
    )
    assert (
        plan.preclassification.interaction_style_assessment.persuasion_intent.label
        == "LowOrNone"
    )
    assert (
        plan.preclassification.interaction_style_assessment.claim_density.label
        == "Medium"
    )
    assert (
        plan.preclassification.interaction_style_assessment.speaker_sentiment[0].speaker
        == "Unknown"
    )
    assert (
        plan.preclassification.interaction_style_assessment.speaker_sentiment[
            0
        ].sentiment
        == "Positive"
    )
    assert '"has_foul_language"' in llm.prompts[0]
    assert '"truthfulness"' in llm.prompts[1]
    assert '"speaker_sentiment"' in llm.prompts[1]


def test_generate_video_prompt_falls_back_when_llm_returns_blank() -> None:
    llm = StubLLM("   ")
    planner = ScenePlanner(llm)

    video_prompt = planner.generate_video_prompt(
        narration_text="A traveler crosses a glowing valley at night."
    )

    assert video_prompt.startswith(
        "Cartoon style illustrated story sequence with a consistent protagonist"
    )
    assert "glowing valley at night" in video_prompt


def test_generate_video_prompt_enforces_cartoon_style_prefix_when_missing() -> None:
    llm = StubLLM(
        [
            json.dumps(
                {
                    "mood": "Tense",
                    "has_foul_language": "Yes",
                    "video_prompt": "dramatic stylized forest chase with recurring heroine and glowing embers",
                }
            ),
            json.dumps(
                {
                    "truthfulness": {
                        "label": "LikelyMisleading",
                        "confidence_score": 1.7,
                        "reason": "The speaker uses urgent certainty without providing supporting detail.",
                    },
                    "formality": {
                        "label": "Informal",
                        "confidence_score": 0.84,
                        "reason": "The phrasing is direct and conversational.",
                    },
                    "certainty_hedging": {
                        "label": "Confident",
                        "confidence_score": 1.3,
                        "reason": "The commands are delivered with little hesitation.",
                    },
                    "persuasion_intent": {
                        "label": "Strong",
                        "confidence_score": 0.76,
                        "reason": "The speaker is actively pushing the listener toward immediate action.",
                    },
                    "claim_density": {
                        "label": "Low",
                        "confidence_score": 0.7,
                        "reason": "The transcript is short and contains few explicit factual claims.",
                    },
                    "speaker_sentiment": [
                        {
                            "speaker": "Unknown",
                            "sentiment": "Negative",
                            "confidence_score": 1.2,
                            "reason": "The language carries urgency and pressure.",
                        }
                    ],
                }
            ),
        ]
    )
    planner = ScenePlanner(llm)

    plan = planner.generate_video_prompt_plan(narration_text="Run quickly! Move now!")

    assert plan.video_prompt.startswith("Cartoon style illustrated scene:")
    assert plan.preclassification is not None
    assert plan.preclassification.mood == "Tense"
    assert plan.preclassification.has_foul_language is True
    assert plan.preclassification.word_count == 4
    assert plan.preclassification.sentence_count == 2
    assert plan.preclassification.truthfulness_assessment.label == "LikelyMisleading"
    assert plan.preclassification.truthfulness_assessment.confidence_score == 1.0
    assert (
        plan.preclassification.interaction_style_assessment.formality.label
        == "Informal"
    )
    assert (
        plan.preclassification.interaction_style_assessment.certainty_hedging.confidence_score
        == 1.0
    )
    assert (
        plan.preclassification.interaction_style_assessment.speaker_sentiment[
            0
        ].sentiment
        == "Negative"
    )
    assert (
        plan.preclassification.interaction_style_assessment.speaker_sentiment[
            0
        ].confidence_score
        == 1.0
    )


def test_generate_video_prompt_defaults_truthfulness_when_json_missing() -> None:
    llm = StubLLM(
        [
            json.dumps(
                {
                    "mood": "Calm",
                    "has_foul_language": "No",
                    "video_prompt": "cartoon style village square with a storyteller near warm lanterns",
                }
            ),
            "not json",
        ]
    )
    planner = ScenePlanner(llm)

    plan = planner.generate_video_prompt_plan(narration_text="A gentle narration.")

    assert plan.preclassification is not None
    assert plan.preclassification.truthfulness_assessment.label == "MixedOrUnverifiable"
    assert plan.preclassification.truthfulness_assessment.confidence_score == 0.0
    assert (
        "limited to signals present in the transcript"
        in plan.preclassification.truthfulness_assessment.reason
    )
    assert (
        plan.preclassification.interaction_style_assessment.formality.label == "Mixed"
    )
    assert (
        plan.preclassification.interaction_style_assessment.certainty_hedging.label
        == "Balanced"
    )
    assert (
        plan.preclassification.interaction_style_assessment.persuasion_intent.label
        == "LowOrNone"
    )
    assert (
        plan.preclassification.interaction_style_assessment.claim_density.label
        == "Medium"
    )
    assert (
        plan.preclassification.interaction_style_assessment.speaker_sentiment[0].speaker
        == "Unknown"
    )


def test_generate_video_prompt_preserves_labeled_speaker_sentiment() -> None:
    llm = StubLLM(
        [
            json.dumps(
                {
                    "mood": "Neutral",
                    "has_foul_language": "No",
                    "video_prompt": "cartoon style meeting room with two coworkers talking across a table",
                }
            ),
            json.dumps(
                {
                    "truthfulness": {
                        "label": "LikelyTruthful",
                        "confidence_score": 0.52,
                        "reason": "The exchange is internally consistent and modest in scope.",
                    },
                    "formality": {
                        "label": "Formal",
                        "confidence_score": 0.64,
                        "reason": "The speakers use businesslike phrasing and explicit turn-taking.",
                    },
                    "certainty_hedging": {
                        "label": "HeavilyHedged",
                        "confidence_score": 0.55,
                        "reason": "The speakers qualify several points with uncertainty markers.",
                    },
                    "persuasion_intent": {
                        "label": "Moderate",
                        "confidence_score": 0.48,
                        "reason": "One speaker appears to be trying to steer the discussion without overt pressure.",
                    },
                    "claim_density": {
                        "label": "High",
                        "confidence_score": 0.69,
                        "reason": "Most lines contain explicit assertions or proposed conclusions.",
                    },
                    "speaker_sentiment": [
                        {
                            "speaker": "SPEAKER_00",
                            "sentiment": "Neutral",
                            "confidence_score": 0.51,
                            "reason": "The language is measured and procedural.",
                        },
                        {
                            "speaker": "SPEAKER_01",
                            "sentiment": "Mixed",
                            "confidence_score": 0.58,
                            "reason": "The speaker alternates between concern and agreement.",
                        },
                    ],
                }
            ),
        ]
    )
    planner = ScenePlanner(llm)

    plan = planner.generate_video_prompt_plan(
        narration_text="SPEAKER_00: I believe the rollout is on track. SPEAKER_01: It should be, although I still have some concerns."
    )

    assert plan.preclassification is not None
    assert (
        len(plan.preclassification.interaction_style_assessment.speaker_sentiment) == 2
    )
    assert (
        plan.preclassification.interaction_style_assessment.speaker_sentiment[0].speaker
        == "SPEAKER_00"
    )
    assert (
        plan.preclassification.interaction_style_assessment.speaker_sentiment[
            1
        ].sentiment
        == "Mixed"
    )


def test_split_narration_distributes_sentences_evenly() -> None:
    llm = StubLLM("{}")
    planner = ScenePlanner(llm)

    text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence. Sixth sentence."
    chunks = planner._split_narration(text, 3)

    assert len(chunks) == 3
    assert "First sentence" in chunks[0]
    assert "Third sentence" in chunks[1]
    assert "Fifth sentence" in chunks[2]


def test_split_narration_pads_when_fewer_sentences_than_chunks() -> None:
    llm = StubLLM("{}")
    planner = ScenePlanner(llm)

    chunks = planner._split_narration("Only one sentence here.", 3)

    assert len(chunks) == 3
    assert all("Only one sentence here" in c for c in chunks)


def test_extract_json_ignores_invalid_json() -> None:
    llm = StubLLM("{}")
    planner = ScenePlanner(llm)

    assert planner._extract_json("not-json") == {}
    assert planner._extract_json("{bad json}") == {}
