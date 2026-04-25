from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from content_creator.hf_client import TimedWord


@dataclass(slots=True)
class SoundEffectAsset:
    path: Path
    duration_seconds: float
    mean_volume_db: float | None
    max_volume_db: float | None


@dataclass(slots=True)
class SoundPack:
    name: str
    root_dir: Path
    target_mean_db: float
    assets: list[SoundEffectAsset]


@dataclass(slots=True)
class ProfanitySfxEvent:
    word: str
    start_seconds: float
    end_seconds: float
    sfx_path: Path
    sfx_duration_seconds: float
    sfx_gain_db: float


@dataclass(slots=True)
class ProfanitySfxPlan:
    sound_pack_name: str
    sound_pack_dir: Path
    total_words: int
    matches_found: int
    events: list[ProfanitySfxEvent]


@dataclass(slots=True)
class LexiconDoctorReport:
    path: Path
    total_lines: int
    active_lines: int
    unique_normalized_entries: int
    exact_duplicates: dict[str, int]
    near_duplicates: dict[str, list[str]]


def load_sound_pack(
    *, sound_pack_dir: Path, ffprobe_bin: str = "ffprobe", ffmpeg_bin: str = "ffmpeg"
) -> SoundPack:
    root = sound_pack_dir.expanduser().resolve()
    if not root.exists():
        raise ValueError(f"Sound pack directory not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Sound pack path is not a directory: {root}")

    manifest_path = root / "pack.json"
    manifest = _read_pack_manifest(manifest_path)
    target_mean_db = _coerce_float(
        manifest.get("target_mean_db"), default=-18.0, field_name="target_mean_db"
    )
    configured_files = manifest.get("sounds")

    candidate_files: list[Path]
    if isinstance(configured_files, list) and configured_files:
        candidate_files = []
        for entry in configured_files:
            file_name = ""
            if isinstance(entry, str):
                file_name = entry.strip()
            elif isinstance(entry, dict):
                file_name = str(entry.get("file", "")).strip()
            if not file_name:
                continue
            candidate_files.append((root / file_name).resolve())
    else:
        candidate_files = sorted(
            [
                *root.glob("*.wav"),
                *root.glob("*.mp3"),
                *root.glob("*.m4a"),
                *root.glob("*.flac"),
                *root.glob("*.ogg"),
            ]
        )

    assets: list[SoundEffectAsset] = []
    for file_path in candidate_files:
        if not file_path.exists() or not file_path.is_file():
            continue
        duration = _probe_duration_seconds(file_path, ffprobe_bin=ffprobe_bin)
        mean_db, max_db = _probe_volume_levels(file_path, ffmpeg_bin=ffmpeg_bin)
        assets.append(
            SoundEffectAsset(
                path=file_path,
                duration_seconds=duration,
                mean_volume_db=mean_db,
                max_volume_db=max_db,
            )
        )

    if not assets:
        raise ValueError(
            f"No sound effects found in sound pack: {root}. Supported files: wav, mp3, m4a, flac, ogg"
        )

    return SoundPack(
        name=str(manifest.get("name", root.name)).strip() or root.name,
        root_dir=root,
        target_mean_db=target_mean_db,
        assets=assets,
    )


def build_profanity_sfx_plan(
    *,
    timed_words: Iterable[TimedWord],
    sound_pack: SoundPack,
    profanity_words: set[str] | None = None,
    pad_seconds: float = 0.08,
) -> ProfanitySfxPlan:
    words = list(timed_words)
    lexicon = (
        profanity_words if profanity_words is not None else load_profanity_words(None)
    )
    effective_pad = max(0.0, float(pad_seconds))
    patterns = _compile_phrase_patterns(lexicon)

    if not patterns:
        return ProfanitySfxPlan(
            sound_pack_name=sound_pack.name,
            sound_pack_dir=sound_pack.root_dir,
            total_words=len(words),
            matches_found=0,
            events=[],
        )

    normalized_words = [_normalize_word(word.word) for word in words]
    onset_pad = _compute_onset_pad_seconds(effective_pad)
    merge_gap_seconds = _compute_merge_gap_seconds(effective_pad)

    matched_spans: list[tuple[str, float, float, int, int]] = []
    index = 0
    while index < len(words):
        if not normalized_words[index]:
            index += 1
            continue

        matched = False
        for pattern in patterns:
            pattern_length = len(pattern)
            upper_bound = index + pattern_length
            if upper_bound > len(words):
                continue

            if tuple(normalized_words[index:upper_bound]) != pattern:
                continue

            start = max(0.0, float(words[index].start_seconds) - onset_pad)
            end = max(start, float(words[upper_bound - 1].end_seconds) + effective_pad)
            phrase = " ".join(word.word for word in words[index:upper_bound]).strip()
            matched_spans.append((phrase, start, end, index, upper_bound))
            index = upper_bound
            matched = True
            break

        if not matched:
            index += 1

    merged_spans = _merge_spans(matched_spans, merge_gap_seconds=merge_gap_seconds)
    events: list[ProfanitySfxEvent] = []
    asset = _select_sound_effect_asset(sound_pack)
    for word, start, end in merged_spans:
        censored_window = max(0.08, end - start)
        sfx_duration = censored_window
        gain_db = _compute_sfx_gain_db(
            mean_volume_db=asset.mean_volume_db,
            target_mean_db=sound_pack.target_mean_db,
        )
        events.append(
            ProfanitySfxEvent(
                word=word,
                start_seconds=start,
                end_seconds=end,
                sfx_path=asset.path,
                sfx_duration_seconds=sfx_duration,
                sfx_gain_db=gain_db,
            )
        )

    return ProfanitySfxPlan(
        sound_pack_name=sound_pack.name,
        sound_pack_dir=sound_pack.root_dir,
        total_words=len(words),
        matches_found=len(merged_spans),
        events=events,
    )


def load_profanity_words(path: Path | None) -> set[str] | None:
    resolved = (
        path.expanduser().resolve()
        if path is not None
        else _default_profanity_words_path()
    )
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"Profanity word list not found: {resolved}")

    words: set[str] = set()
    for raw_line in resolved.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        normalized = _normalize_phrase_text(line)
        if normalized:
            words.add(normalized)

    if not words:
        raise ValueError(f"Profanity word list is empty: {resolved}")
    return words


def analyze_profanity_lexicon(path: Path | None) -> LexiconDoctorReport:
    resolved = (
        path.expanduser().resolve()
        if path is not None
        else _default_profanity_words_path()
    )
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"Profanity word list not found: {resolved}")

    raw_lines = resolved.read_text(encoding="utf-8").splitlines()
    normalized_entries: set[str] = set()
    exact_counts: dict[str, int] = {}
    exact_display: dict[str, str] = {}
    near_duplicate_groups: dict[str, set[str]] = {}
    active_lines = 0

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        normalized = _normalize_phrase_text(line)
        if not normalized:
            continue

        active_lines += 1
        normalized_entries.add(normalized)

        exact_key = line.casefold()
        exact_counts[exact_key] = exact_counts.get(exact_key, 0) + 1
        exact_display.setdefault(exact_key, line)

        group = near_duplicate_groups.setdefault(normalized, set())
        group.add(line)

    if not normalized_entries:
        raise ValueError(f"Profanity word list is empty: {resolved}")

    exact_duplicates = {
        exact_display[key]: count for key, count in exact_counts.items() if count > 1
    }
    near_duplicates = {
        normalized: sorted(forms, key=str.casefold)
        for normalized, forms in near_duplicate_groups.items()
        if len(forms) > 1
    }

    return LexiconDoctorReport(
        path=resolved,
        total_lines=len(raw_lines),
        active_lines=active_lines,
        unique_normalized_entries=len(normalized_entries),
        exact_duplicates=dict(
            sorted(exact_duplicates.items(), key=lambda item: item[0].casefold())
        ),
        near_duplicates=dict(sorted(near_duplicates.items(), key=lambda item: item[0])),
    )


def _default_profanity_words_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "profanity_words.txt"


def _read_pack_manifest(path: Path) -> dict[str, object]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in sound pack manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Sound pack manifest must be a JSON object: {path}")
    return payload


def _coerce_float(value: object, *, default: float, field_name: str) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return float(stripped)
        except ValueError as exc:
            raise ValueError(
                f"Sound pack field '{field_name}' must be a numeric value"
            ) from exc
    raise ValueError(f"Sound pack field '{field_name}' must be a numeric value")


def _probe_duration_seconds(path: Path, *, ffprobe_bin: str) -> float:
    result = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    if not output:
        raise RuntimeError(f"ffprobe returned empty duration for {path}")
    return max(0.0, float(output))


def _probe_volume_levels(
    path: Path, *, ffmpeg_bin: str
) -> tuple[float | None, float | None]:
    result = subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-i",
            str(path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    text = f"{result.stdout}\n{result.stderr}"
    mean_match = re.search(r"mean_volume:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*dB", text)
    max_match = re.search(r"max_volume:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*dB", text)
    mean_db = float(mean_match.group(1)) if mean_match else None
    max_db = float(max_match.group(1)) if max_match else None
    return mean_db, max_db


def _compute_sfx_gain_db(
    *, mean_volume_db: float | None, target_mean_db: float
) -> float:
    if mean_volume_db is None:
        return 0.0
    gain = target_mean_db - mean_volume_db
    return max(-12.0, min(12.0, gain))


def _compute_onset_pad_seconds(pad_seconds: float) -> float:
    # Whisper word timestamps often land slightly late on profanity onsets,
    # so bias the censor window earlier than later to avoid leaking consonants.
    return max(pad_seconds, min(0.18, pad_seconds + 0.08))


def _compute_merge_gap_seconds(pad_seconds: float) -> float:
    # Nearby profanity words should collapse into one censor window so short gaps
    # between STT tokens do not leak the second word.
    return max(0.20, min(0.50, pad_seconds + 0.42))


def _select_sound_effect_asset(sound_pack: SoundPack) -> SoundEffectAsset:
    for asset in sound_pack.assets:
        if asset.path.name.lower() == "bleep-1.wav":
            return asset
    return sound_pack.assets[0]


def _normalize_word(word: str) -> str:
    lowered = word.lower().strip()
    if not lowered:
        return ""
    substitutions = {
        "@": "a",
        "$": "s",
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "!": "i",
        "|": "i",
        "*": "",
        "_": "",
        "-": "",
        ".": "",
        "'": "",
    }
    for original, replacement in substitutions.items():
        lowered = lowered.replace(original, replacement)
    lowered = re.sub(r"[^a-z]", "", lowered)
    return lowered


def _normalize_phrase_text(phrase: str) -> str:
    normalized_words = [
        _normalize_word(token) for token in re.split(r"\s+", phrase.strip())
    ]
    normalized_words = [word for word in normalized_words if word]
    return " ".join(normalized_words)


def _compile_phrase_patterns(lexicon: set[str] | None) -> list[tuple[str, ...]]:
    if not lexicon:
        return []
    patterns = {
        tuple(normalized.split(" "))
        for entry in lexicon
        for normalized in [_normalize_phrase_text(entry)]
        if normalized
    }
    return sorted(patterns, key=lambda pattern: (-len(pattern), pattern))


def _merge_spans(
    spans: list[tuple[str, float, float, int, int]], *, merge_gap_seconds: float = 0.0
) -> list[tuple[str, float, float]]:
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda item: item[1])
    merged: list[tuple[str, float, float, int, int]] = [sorted_spans[0]]
    for word, start, end, start_index, end_index in sorted_spans[1:]:
        (
            previous_word,
            previous_start,
            previous_end,
            previous_start_index,
            previous_end_index,
        ) = merged[-1]
        if start_index == previous_end_index and start <= (
            previous_end + merge_gap_seconds
        ):
            merged[-1] = (
                f"{previous_word} {word}".strip(),
                previous_start,
                max(previous_end, end),
                previous_start_index,
                end_index,
            )
            continue
        merged.append((word, start, end, start_index, end_index))
    return [(word, start, end) for word, start, end, _start_index, _end_index in merged]
