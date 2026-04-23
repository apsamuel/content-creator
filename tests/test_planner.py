from __future__ import annotations

import json

from video_generator.planner import ScenePlanner


class StubLLM:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def generate_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


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
        json.dumps(
            {
                "mood": "Happy",
                "has_foul_language": "No",
                "video_prompt": "Stylized cartoon forest with a rabbit-eared heroine in a blue cloak, warm dawn glow, soft painted textures, consistent character design, expressive illustrated composition",
            }
        )
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
    assert '"has_foul_language"' in llm.prompts[0]


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
        json.dumps(
            {
                "mood": "Tense",
                "has_foul_language": "Yes",
                "video_prompt": "dramatic stylized forest chase with recurring heroine and glowing embers",
            }
        )
    )
    planner = ScenePlanner(llm)

    plan = planner.generate_video_prompt_plan(narration_text="Run quickly! Move now!")

    assert plan.video_prompt.startswith("Cartoon style illustrated scene:")
    assert plan.preclassification is not None
    assert plan.preclassification.mood == "Tense"
    assert plan.preclassification.has_foul_language is True
    assert plan.preclassification.word_count == 4
    assert plan.preclassification.sentence_count == 2


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
