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
    truthfulness_assessment: "TranscriptAssessment"
    interaction_style_assessment: "InteractionStyleAssessment"


@dataclass(slots=True)
class TranscriptAssessment:
    label: str
    confidence_score: float
    reason: str


@dataclass(slots=True)
class SpeakerSentimentAssessment:
    speaker: str
    sentiment: str
    confidence_score: float
    reason: str


@dataclass(slots=True)
class InteractionStyleAssessment:
    formality: TranscriptAssessment
    certainty_hedging: TranscriptAssessment
    persuasion_intent: TranscriptAssessment
    claim_density: TranscriptAssessment
    speaker_sentiment: list[SpeakerSentimentAssessment]


@dataclass(slots=True)
class VideoPromptPlan:
    video_prompt: str
    preclassification: VideoPromptPreclassification | None
    prompts: dict[str, str] | None = None


@dataclass(slots=True)
class ScenePlan:
    scenes: list[Scene]
    scene_prompt: str


class ScenePlanner:
    _GLOBAL_VIDEO_STYLE = (
        "80s/90s retro anime aesthetic, vibrant colors, cel shading, detailed "
        "illustration, sharp linework, dramatic composition, expressive characters, "
        "subtle camera-shake energy, Studio Ghibli and Makoto Shinkai-inspired artistry"
    )
    _COMPOSITION_GUIDANCE = (
        "favor still-image-friendly composition cues such as low-angle hero framing, "
        "wide establishing shots, close emotional portraits, or dynamic diagonal layouts"
    )
    _MOTION_GUIDANCE = (
        "express motion as subtle motion energy or dynamic handheld framing rather than "
        "literal shake blur"
    )
    _COMPOSITION_SEQUENCES = {
        "balanced": (
            "wide establishing shot",
            "low-angle hero framing",
            "close emotional portrait",
            "dynamic diagonal layout",
        ),
        "dynamic": (
            "dynamic diagonal layout",
            "low-angle hero framing",
            "dynamic handheld framing",
            "wide establishing shot",
        ),
        "portrait": (
            "close emotional portrait",
            "low-angle hero framing",
            "dynamic diagonal layout",
            "close emotional portrait",
        ),
        "establishing": (
            "wide establishing shot",
            "dynamic diagonal layout",
            "low-angle hero framing",
            "wide establishing shot",
        ),
    }

    def __init__(self, llm: Any, *, image_composition_mode: str = "balanced"):
        self._llm = llm
        self._image_composition_mode = self._normalize_composition_mode(
            image_composition_mode
        )

    def generate_video_prompt(self, *, narration_text: str) -> str:
        return self.generate_video_prompt_plan(
            narration_text=narration_text
        ).video_prompt

    def generate_video_prompt_plan(self, *, narration_text: str) -> VideoPromptPlan:
        video_prompt_prompt = self._build_video_prompt_prompt(
            narration_text=narration_text
        )
        raw = self._llm.generate_text(video_prompt_prompt)
        payload = self._extract_json(raw)
        analysis_prompt = self._build_analysis_prompt(narration_text=narration_text)
        truthfulness_assessment, interaction_style_assessment = self._classify_analysis(
            narration_text=narration_text, prompt=analysis_prompt
        )

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
            truthfulness_assessment=truthfulness_assessment,
            interaction_style_assessment=interaction_style_assessment,
        )
        return VideoPromptPlan(
            video_prompt=prompt_text,
            preclassification=preclassification,
            prompts={
                "video_prompt_generation": video_prompt_prompt,
                "analysis": analysis_prompt,
            },
        )

    def _classify_analysis(
        self, *, narration_text: str, prompt: str | None = None
    ) -> tuple[TranscriptAssessment, InteractionStyleAssessment]:
        if prompt is None:
            prompt = self._build_analysis_prompt(narration_text=narration_text)
        raw = self._llm.generate_text(prompt)
        payload = self._extract_json(raw)

        truthfulness_assessment = self._parse_dimension_assessment(
            payload.get("truthfulness"),
            allowed_labels={
                "LikelyTruthful",
                "MixedOrUnverifiable",
                "LikelyMisleading",
            },
            fallback_reason=(
                "Assessment is limited to signals present in the transcript and does "
                "not verify external facts."
            ),
            default_label="MixedOrUnverifiable",
        )
        interaction_style_assessment = InteractionStyleAssessment(
            formality=self._parse_dimension_assessment(
                payload.get("formality"),
                allowed_labels={"Formal", "Mixed", "Informal"},
                fallback_reason=(
                    "Formality is estimated from wording and structure present in the transcript only."
                ),
                default_label="Mixed",
            ),
            certainty_hedging=self._parse_dimension_assessment(
                payload.get("certainty_hedging"),
                allowed_labels={"Confident", "Balanced", "HeavilyHedged"},
                fallback_reason=(
                    "Certainty and hedging are estimated from the transcript's phrasing only."
                ),
                default_label="Balanced",
            ),
            persuasion_intent=self._parse_dimension_assessment(
                payload.get("persuasion_intent"),
                allowed_labels={"Strong", "Moderate", "LowOrNone"},
                fallback_reason=(
                    "Persuasion intent is estimated from rhetorical cues in the transcript only."
                ),
                default_label="LowOrNone",
            ),
            claim_density=self._parse_dimension_assessment(
                payload.get("claim_density"),
                allowed_labels={"High", "Medium", "Low"},
                fallback_reason=(
                    "Claim density is estimated from the concentration of factual or assertive statements in the transcript only."
                ),
                default_label="Medium",
            ),
            speaker_sentiment=self._parse_speaker_sentiment_assessments(
                payload.get("speaker_sentiment")
            ),
        )
        return truthfulness_assessment, interaction_style_assessment

    def build_scenes(
        self,
        *,
        narration_text: str,
        video_prompt: str,
        total_duration_seconds: float,
        max_scenes: int = 8,
    ) -> ScenePlan:
        scene_count = max(3, min(max_scenes, math.ceil(total_duration_seconds / 4.5)))
        scene_prompt = self._build_prompt(
            narration_text=narration_text,
            video_prompt=video_prompt,
            total_duration_seconds=total_duration_seconds,
            scene_count=scene_count,
        )
        raw = self._llm.generate_text(scene_prompt)
        payload = self._extract_json(raw)
        scene_payloads = payload.get("scenes", [])
        if not scene_payloads:
            scene_payloads = [
                {"prompt": video_prompt, "_is_fallback": True}
            ] * scene_count

        story_anchor = self._normalize_fragment(str(payload.get("story_anchor", "")))
        durations = self._spread_duration(total_duration_seconds, len(scene_payloads))
        scenes: list[Scene] = []
        previous_scene_summary = ""
        for index, item in enumerate(scene_payloads):
            prompt_text = self._extract_prompt_text(item, fallback=video_prompt)
            if (
                isinstance(item, dict)
                and str(item.get("prompt", "")).strip()
                and not bool(item.get("_is_fallback"))
            ):
                prompt_text = self.prepare_image_prompt(
                    prompt_text, scene_index=index, total_scenes=len(scene_payloads)
                )
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
        return ScenePlan(scenes=scenes, scene_prompt=scene_prompt)

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
- Prefer composition language that diffusion models render well for still images: {self._COMPOSITION_GUIDANCE}.
- When the shot implies camera shake or kinetic energy, phrase it as {self._MOTION_GUIDANCE}.
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
- video_prompt must produce a anime cartoon style image only. No photorealistic, live-action, or real camera language.
- video_prompt must follow this fixed house style for every generated image: {self._GLOBAL_VIDEO_STYLE}.
- video_prompt must describe recurring protagonist, environment, mood, lighting, palette, and stylized composition.
- video_prompt should prefer still-image composition language: {self._COMPOSITION_GUIDANCE}.
- If motion or impact is important, phrase it as {self._MOTION_GUIDANCE}.
- video_prompt must be one sentence and under 110 words.
- DO NOT use text overlays, subtitles, logos, or watermarks.

Narration or transcript:
{narration_text}
""".strip()

    def _build_analysis_prompt(self, *, narration_text: str) -> str:
        return f"""
You analyze the truthfulness and interaction style of a transcript.

Return valid JSON only with this exact schema:
{{
    "truthfulness": {{"label": "LikelyTruthful|MixedOrUnverifiable|LikelyMisleading", "confidence_score": 0.0, "reason": "short explanation"}},
    "formality": {{"label": "Formal|Mixed|Informal", "confidence_score": 0.0, "reason": "short explanation"}},
    "certainty_hedging": {{"label": "Confident|Balanced|HeavilyHedged", "confidence_score": 0.0, "reason": "short explanation"}},
    "persuasion_intent": {{"label": "Strong|Moderate|LowOrNone", "confidence_score": 0.0, "reason": "short explanation"}},
    "claim_density": {{"label": "High|Medium|Low", "confidence_score": 0.0, "reason": "short explanation"}},
    "speaker_sentiment": [{{"speaker": "speaker identifier", "sentiment": "Positive|Negative|Neutral|Mixed", "confidence_score": 0.0, "reason": "short explanation"}}]
}}

Truthfulness constraints:
- Base the answer only on the transcript itself. Do not assume access to external facts.
- Use LikelyTruthful when the transcript is internally consistent, cautious, and avoids unsupported certainty.
- Use MixedOrUnverifiable when claims cannot be checked from the transcript or contain a mix of grounded and ungrounded statements.
- Use LikelyMisleading when the transcript contains strong unsupported certainty, internal contradictions, or obvious rhetorical manipulation.

Interaction style constraints:
- Base the answer only on the transcript itself.
- Keep each reason to one sentence.
- If speakers are explicitly labeled, preserve those labels.
- If speakers are not labeled, return one item with speaker set to Unknown.

General constraints:
- confidence_score values must be between 0 and 1.

Transcript:
{narration_text}
""".strip()

    def _fallback_video_prompt(self, narration_text: str) -> str:
        snippet = self._normalize_fragment(narration_text)
        if len(snippet) > 220:
            snippet = snippet[:217].rstrip() + "..."
        return (
            "Cartoon style illustrated story sequence in an 80s/90s retro anime style, "
            "with vibrant colors, cel shading, detailed illustration, sharp linework, "
            "dramatic composition, expressive characters, subtle camera-shake energy, "
            "and Studio Ghibli and Makoto Shinkai-inspired artistry, based on this narration: "
            f"{snippet}"
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

    def _normalize_assessment_label(self, value: str) -> str:
        normalized = self._normalize_fragment(value)
        allowed = {"LikelyTruthful", "MixedOrUnverifiable", "LikelyMisleading"}
        return normalized if normalized in allowed else "MixedOrUnverifiable"

    def _normalize_allowed_label(
        self, value: str, *, allowed_labels: set[str], default_label: str
    ) -> str:
        normalized = self._normalize_fragment(value)
        return normalized if normalized in allowed_labels else default_label

    def _parse_yes_no(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        normalized = self._normalize_fragment(str(value)).lower()
        return normalized in {"yes", "true", "1"}

    def _parse_confidence_score(self, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, parsed))

    def _parse_dimension_assessment(
        self,
        value: Any,
        *,
        allowed_labels: set[str],
        fallback_reason: str,
        default_label: str,
    ) -> TranscriptAssessment:
        if not isinstance(value, dict):
            return TranscriptAssessment(
                label=default_label, confidence_score=0.0, reason=fallback_reason
            )
        reason = self._normalize_fragment(str(value.get("reason", "")))
        if not reason:
            reason = fallback_reason
        return TranscriptAssessment(
            label=self._normalize_allowed_label(
                str(value.get("label", "")),
                allowed_labels=allowed_labels,
                default_label=default_label,
            ),
            confidence_score=self._parse_confidence_score(
                value.get("confidence_score")
            ),
            reason=reason,
        )

    def _parse_speaker_sentiment_assessments(
        self, value: Any
    ) -> list[SpeakerSentimentAssessment]:
        if not isinstance(value, list):
            value = []

        sentiments: list[SpeakerSentimentAssessment] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            speaker = self._normalize_fragment(str(item.get("speaker", "")))
            if not speaker:
                speaker = "Unknown"
            reason = self._normalize_fragment(str(item.get("reason", "")))
            if not reason:
                reason = "Sentiment is estimated from the speaker's wording in the transcript only."
            sentiments.append(
                SpeakerSentimentAssessment(
                    speaker=speaker,
                    sentiment=self._normalize_allowed_label(
                        str(item.get("sentiment", "")),
                        allowed_labels={"Positive", "Negative", "Neutral", "Mixed"},
                        default_label="Neutral",
                    ),
                    confidence_score=self._parse_confidence_score(
                        item.get("confidence_score")
                    ),
                    reason=reason,
                )
            )

        if sentiments:
            return sentiments

        return [
            SpeakerSentimentAssessment(
                speaker="Unknown",
                sentiment="Neutral",
                confidence_score=0.0,
                reason="Sentiment is estimated from the transcript only and no speaker-specific structure was available.",
            )
        ]

    def _count_words(self, text: str) -> int:
        return len(re.findall(r"\b\w+\b", text))

    def _count_sentences(self, text: str) -> int:
        matches = re.findall(r"[^.!?]+[.!?]", text)
        if matches:
            return len(matches)
        return 1 if self._normalize_fragment(text) else 0

    def prepare_image_prompt(
        self, prompt_text: str, *, scene_index: int = 0, total_scenes: int | None = None
    ) -> str:
        normalized = self._normalize_fragment(prompt_text)
        if not normalized:
            return normalized
        return self._normalize_still_image_language(
            normalized,
            preferred_composition=self._composition_cue_for_scene(
                scene_index=scene_index, total_scenes=total_scenes
            ),
        )

    def _normalize_composition_mode(self, value: str) -> str:
        normalized = self._normalize_fragment(value).lower().replace(" ", "-")
        if normalized not in self._COMPOSITION_SEQUENCES:
            return "balanced"
        return normalized

    def _composition_cue_for_scene(
        self, *, scene_index: int, total_scenes: int | None
    ) -> str:
        sequence = self._COMPOSITION_SEQUENCES[self._image_composition_mode]
        if self._image_composition_mode == "balanced" and total_scenes:
            if scene_index == 0:
                return "wide establishing shot"
            if total_scenes > 1 and scene_index == total_scenes - 1:
                return "close emotional portrait"
        return sequence[scene_index % len(sequence)]

    def _enforce_cartoon_style(self, prompt_text: str) -> str:
        normalized = self.prepare_image_prompt(prompt_text)
        if not normalized:
            return (
                "Cartoon style illustrated fantasy scene in an 80s/90s retro anime style, "
                "with vibrant colors, cel shading, detailed illustration, sharp linework, "
                "dramatic composition, expressive characters, low-angle hero framing, subtle motion energy, "
                "and Studio Ghibli and Makoto Shinkai-inspired artistry"
            )
        lowered = normalized.lower()
        needs_prefix = not (
            "cartoon" in lowered or "illustrated" in lowered or "anime" in lowered
        )
        needs_style = not all(
            phrase in lowered
            for phrase in (
                "retro anime",
                "vibrant",
                "cel shading",
                "sharp",
                "dramatic composition",
                "expressive characters",
                "camera-shake",
            )
        )
        if not needs_prefix and not needs_style:
            return normalized

        style_suffix = (
            "80s/90s retro anime style, vibrant colors, cel shading, detailed "
            "illustration, sharp linework, dramatic composition, expressive characters, "
            "low-angle hero framing, dynamic diagonal layout, subtle motion energy, "
            "Studio Ghibli and Makoto Shinkai-inspired artistry"
        )
        if needs_prefix:
            return f"Cartoon style illustrated scene in {style_suffix}: {normalized}"
        return f"{normalized}, {style_suffix}"

    def _normalize_still_image_language(
        self, prompt_text: str, *, preferred_composition: str | None = None
    ) -> str:
        if not prompt_text:
            return prompt_text

        normalized = re.sub(
            r"\bcamera shake\b|\bshaky camera\b|\bshake blur\b",
            "subtle motion energy",
            prompt_text,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"\bhandheld camera\b",
            "dynamic handheld framing",
            normalized,
            flags=re.IGNORECASE,
        )
        if not re.search(
            r"low-angle|wide establishing|close emotional portrait|dynamic diagonal|dynamic handheld framing",
            normalized,
            flags=re.IGNORECASE,
        ):
            normalized = (
                f"{normalized}, {preferred_composition or 'dynamic diagonal layout'}"
            )
        return normalized

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
