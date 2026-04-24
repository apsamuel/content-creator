from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Scene:
    index: int
    prompt: str
    duration_seconds: float


@dataclass(slots=True)
class VideoPromptPreclassification:
    mood: str
    has_foul_language: bool
    word_count: int
    sentence_count: int


@dataclass(slots=True)
class VideoPromptPlan:
    video_prompt: str
    preclassification: VideoPromptPreclassification | None


class ScenePlanner:
    def __init__(self, llm: Any):
        self._llm = llm

    def generate_video_prompt(self, *, narration_text: str) -> str:
        return self.generate_video_prompt_plan(
            narration_text=narration_text
        ).video_prompt

    def generate_video_prompt_plan(self, *, narration_text: str) -> VideoPromptPlan:
        prompt = self._build_video_prompt_prompt(narration_text=narration_text)
        raw = self._llm.generate_text(prompt)
        payload = self._extract_json(raw)

        mood = self._normalize_mood(str(payload.get("mood", "")))
        has_foul_language = self._parse_yes_no(payload.get("has_foul_language"))

        prompt_text = self._normalize_fragment(str(payload.get("video_prompt", "")))
        if not prompt_text:
            prompt_text = self._fallback_video_prompt(narration_text)
        prompt_text = self._enforce_cartoon_style(prompt_text)

        preclassification = VideoPromptPreclassification(
            mood=mood,
            has_foul_language=has_foul_language,
            word_count=self._count_words(narration_text),
            sentence_count=self._count_sentences(narration_text),
        )
        return VideoPromptPlan(
            video_prompt=prompt_text, preclassification=preclassification
        )

    def build_scenes(
        self,
        *,
        narration_text: str,
        video_prompt: str,
        total_duration_seconds: float,
        max_scenes: int = 8,
    ) -> list[Scene]:
        scene_count = max(3, min(max_scenes, math.ceil(total_duration_seconds / 4.5)))
        prompt = self._build_prompt(
            narration_text=narration_text,
            video_prompt=video_prompt,
            total_duration_seconds=total_duration_seconds,
            scene_count=scene_count,
        )
        raw = self._llm.generate_text(prompt)
        payload = self._extract_json(raw)
        scene_payloads = payload.get("scenes", [])
        if not scene_payloads:
            scene_payloads = [{"prompt": video_prompt}] * scene_count

        story_anchor = self._normalize_fragment(str(payload.get("story_anchor", "")))
        durations = self._spread_duration(total_duration_seconds, len(scene_payloads))
        scenes: list[Scene] = []
        previous_scene_summary = ""
        for index, item in enumerate(scene_payloads):
            prompt_text = self._extract_prompt_text(item, fallback=video_prompt)
            summary = self._normalize_fragment(str(item.get("summary", "")))
            continuity = self._normalize_fragment(str(item.get("continuity", "")))
            composed_prompt = self._compose_prompt(
                story_anchor=story_anchor,
                prompt_text=prompt_text,
                continuity=continuity,
                previous_scene_summary=previous_scene_summary,
            )
            scenes.append(
                Scene(
                    index=index + 1,
                    prompt=composed_prompt,
                    duration_seconds=durations[index],
                )
            )
            previous_scene_summary = summary or prompt_text
        return scenes

    def _split_narration(self, text: str, n: int) -> list[str]:
        """Split narration text into n roughly equal sentence-based chunks."""
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()
        ]
        if not sentences or n <= 1:
            return [text] * max(n, 1)
        per_chunk = math.ceil(len(sentences) / n)
        chunks: list[str] = []
        for i in range(n):
            start = i * per_chunk
            end = min(start + per_chunk, len(sentences))
            chunk = " ".join(sentences[start:end]).strip()
            chunks.append(chunk if chunk else text)
        while len(chunks) < n:
            chunks.append(text)
        return chunks[:n]

    def _build_prompt(
        self,
        *,
        narration_text: str,
        video_prompt: str,
        total_duration_seconds: float,
        scene_count: int,
    ) -> str:
        narration_chunks = self._split_narration(narration_text, scene_count)
        chunk_lines = "\n".join(
            f"Scene {i + 1} narration chunk:\n{chunk}"
            for i, chunk in enumerate(narration_chunks)
        )
        return f"""
You are a storyboard planner for short-form YouTube videos.
Create exactly {scene_count} consecutive scenes from one continuous story.
Return valid JSON only using this schema:
{{"story_anchor": "single sentence that must stay true in every shot", "scenes": [{{"summary": "what changes in this beat", "continuity": "what must carry forward from the previous scene", "prompt": "cinematic visual prompt"}}]}}

Constraints:
- The same protagonist, wardrobe, physical traits, location family, and visual style must persist across the full sequence.
- Each scene should feel like the next moment in the same story, not a reset to a new idea.
- story_anchor must define the recurring character and the stable visual language.
- continuity should be empty for scene 1 and specific for later scenes.
- Each scene prompt must be grounded in that scene's narration chunk and reflect its specific content.
- Each prompt must be a single sentence.
- Keep a consistent visual style across the video (follow the style template below).
- Avoid text overlays, subtitles, logos, watermarks, borders, and split screens.
- Use prompts that work well for text-to-image diffusion models.

Style template (visual direction, protagonist, and palette to apply consistently across all scenes):
{video_prompt}

Per-scene narration chunks:
{chunk_lines}

Approximate total duration in seconds: {total_duration_seconds}
""".strip()

    def _build_video_prompt_prompt(self, *, narration_text: str) -> str:
        return f"""
You are a visual development director for short-form AI videos.
You must classify the transcript and generate one robust visual prompt.

Return valid JSON only with this exact schema:
{{"mood": "Happy|Sad|Evil|Angry|Calm|Hopeful|Tense|Mysterious|Neutral", "has_foul_language": "Yes|No", "video_prompt": "single prompt string"}}

Constraints:
- Answer classification concretely from the transcript.
- video_prompt must be optimized for Stable Diffusion text-to-image generation.
- video_prompt must be cartoon style only. No photorealistic, live-action, or real camera language.
- video_prompt must describe recurring protagonist, environment, mood, lighting, palette, and stylized composition.
- video_prompt must be one sentence and under 110 words.
- Avoid text overlays, subtitles, logos, and watermarks.

Narration or transcript:
{narration_text}
""".strip()

    def _fallback_video_prompt(self, narration_text: str) -> str:
        snippet = self._normalize_fragment(narration_text)
        if len(snippet) > 220:
            snippet = snippet[:217].rstrip() + "..."
        return (
            "Cartoon style illustrated story sequence with a consistent protagonist, "
            f"expressive stylized lighting, and coherent environment continuity based on this narration: {snippet}"
        )

    def _extract_prompt_text(self, item: Any, *, fallback: str) -> str:
        if not isinstance(item, dict):
            return fallback
        prompt_text = str(item.get("prompt", "")).strip()
        return prompt_text or fallback

    def _compose_prompt(
        self,
        *,
        story_anchor: str,
        prompt_text: str,
        continuity: str,
        previous_scene_summary: str,
    ) -> str:
        fragments = [story_anchor]
        if previous_scene_summary and (story_anchor or continuity):
            fragments.append(f"Previous beat: {previous_scene_summary}")
        if continuity:
            fragments.append(f"Carry forward: {continuity}")
        fragments.append(prompt_text)
        return ". ".join(fragment for fragment in fragments if fragment).strip()

    def _normalize_fragment(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().rstrip("."))

    def _normalize_mood(self, value: str) -> str:
        normalized = self._normalize_fragment(value)
        return normalized or "Neutral"

    def _parse_yes_no(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        normalized = self._normalize_fragment(str(value)).lower()
        return normalized in {"yes", "true", "1"}

    def _count_words(self, text: str) -> int:
        return len(re.findall(r"\b\w+\b", text))

    def _count_sentences(self, text: str) -> int:
        matches = re.findall(r"[^.!?]+[.!?]", text)
        if matches:
            return len(matches)
        return 1 if self._normalize_fragment(text) else 0

    def _enforce_cartoon_style(self, prompt_text: str) -> str:
        normalized = self._normalize_fragment(prompt_text)
        if not normalized:
            return "Cartoon style illustrated fantasy scene with consistent character design"
        lowered = normalized.lower()
        if "cartoon" in lowered or "illustrated" in lowered or "anime" in lowered:
            return normalized
        return f"Cartoon style illustrated scene: {normalized}"

    def _extract_json(self, raw: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload

    def _spread_duration(
        self, total_duration_seconds: float, count: int
    ) -> list[float]:
        if count <= 0:
            return []
        base = total_duration_seconds / count
        durations = [round(base, 2) for _ in range(count)]
        delta = round(total_duration_seconds - sum(durations), 2)
        durations[-1] += delta
        return durations
