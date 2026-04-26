from __future__ import annotations

import importlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from content_creator.hf_client import TimedWord

try:
    _jellyfish: Any = importlib.import_module("jellyfish")
    _HAS_JELLYFISH = True
except ImportError:  # pragma: no cover
    _jellyfish = None
    _HAS_JELLYFISH = False

# Ordered longest-first so the greediest suffix is stripped once.
_STEM_SUFFIXES: tuple[str, ...] = (
    "ings",
    "ers",
    "ing",
    "ied",
    "ies",
    "ed",
    "er",
    "es",
    "s",
)
_MIN_STEM_LENGTH: int = 3  # never reduce a word below 3 chars
_PHONETIC_MIN_NORM_LENGTH: int = 5  # skip phonetic encoding for very short words
_SPELLED_OUT_MAX_GAP_SECONDS: float = (
    0.8  # max gap between letters in a spelled-out run
)
_SPELLED_OUT_MIN_RUN: int = 3  # need at least 3 consecutive single-char tokens


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
    sensitivity: str = "stem",
) -> ProfanitySfxPlan:
    """Build a profanity SFX plan from *timed_words*.

    *sensitivity* controls how aggressively inflections and accent variants
    are matched:

    ``"exact"``
        Only the normalized exact form (leet-speak substituted).
    ``"stem"``  *(default)*
        Exact + morphological suffix stripping.  Catches inflected forms such
        as *fucking*, *fucked*, *fucker*, *motherfuckers*.
    ``"phonetic"``
        Stem + Double Metaphone phonetic encoding.  Also catches
        accent-induced transcription variants (e.g. *fok* for *fuck*).
        Requires the optional *jellyfish* package; falls back to ``"stem"``
        when the package is not installed.
    """
    words = list(timed_words)
    lexicon = (
        profanity_words if profanity_words is not None else load_profanity_words(None)
    )
    effective_pad = max(0.0, float(pad_seconds))
    onset_pad = _compute_onset_pad_seconds(effective_pad)
    merge_gap_seconds = _compute_merge_gap_seconds(effective_pad)

    use_stem = sensitivity in ("stem", "phonetic")
    use_phonetic = sensitivity == "phonetic"

    exact_patterns = _compile_phrase_patterns(lexicon)
    stem_patterns = _compile_stem_patterns(lexicon) if use_stem else []
    phonetic_patterns = _compile_phonetic_patterns(lexicon) if use_phonetic else []

    if not exact_patterns and not stem_patterns and not phonetic_patterns:
        return ProfanitySfxPlan(
            sound_pack_name=sound_pack.name,
            sound_pack_dir=sound_pack.root_dir,
            total_words=len(words),
            matches_found=0,
            events=[],
        )

    normalized_words = [_normalize_word(word.word) for word in words]
    stem_words = [_stem_word(nw) for nw in normalized_words] if use_stem else []
    phonetic_words = (
        [_phonetic_code(nw) or "" for nw in normalized_words] if use_phonetic else []
    )

    all_matched: list[tuple[str, float, float, int, int]] = []

    # Pre-pass: detect spelled-out profanity (e.g. "f u c k").
    all_matched.extend(
        _find_spelled_out_spans(
            words,
            normalized_words,
            exact_patterns,
            stem_patterns,
            onset_pad,
            effective_pad,
        )
    )
    skip: set[int] = {i for _, _, _, i0, i1 in all_matched for i in range(i0, i1)}

    # Pass 1: exact normalized match.
    exact_spans = _run_match_pass(
        words, normalized_words, exact_patterns, onset_pad, effective_pad, skip
    )
    all_matched.extend(exact_spans)
    skip |= {i for _, _, _, i0, i1 in exact_spans for i in range(i0, i1)}

    # Pass 2: morphological stem match — catches inflections like *fucking*.
    if use_stem and stem_patterns:
        stem_spans = _run_match_pass(
            words, stem_words, stem_patterns, onset_pad, effective_pad, skip
        )
        all_matched.extend(stem_spans)
        skip |= {i for _, _, _, i0, i1 in stem_spans for i in range(i0, i1)}

    # Pass 3: phonetic match — catches accent-variant transcriptions.
    if use_phonetic and phonetic_patterns:
        phonetic_spans = _run_match_pass(
            words, phonetic_words, phonetic_patterns, onset_pad, effective_pad, skip
        )
        all_matched.extend(phonetic_spans)

    merged_spans = _merge_spans(all_matched, merge_gap_seconds=merge_gap_seconds)
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


def _stem_word(word: str) -> str:
    """Strip the longest matching inflection suffix from *word*.

    Applied after :func:`_normalize_word` so the input is already lowercase
    alpha-only.  Never shortens a word below :data:`_MIN_STEM_LENGTH` chars.
    """
    for suffix in _STEM_SUFFIXES:
        end = len(word) - len(suffix)
        if word.endswith(suffix) and end >= _MIN_STEM_LENGTH:
            return word[:end]
    return word


def _compile_stem_patterns(lexicon: set[str] | None) -> list[tuple[str, ...]]:
    """Like :func:`_compile_phrase_patterns` but each token is additionally stemmed."""
    if not lexicon:
        return []
    patterns = {
        tuple(_stem_word(tok) for tok in normalized.split())
        for entry in lexicon
        for normalized in [_normalize_phrase_text(entry)]
        if normalized
    }
    return sorted(patterns, key=lambda p: (-len(p), p))


def _phonetic_code(word: str) -> str | None:
    """Return a Metaphone code for *word*, or ``None`` when unavailable.

    Returns ``None`` when *jellyfish* is not installed or the word is shorter
    than :data:`_PHONETIC_MIN_NORM_LENGTH` (very short words produce too many
    false-positive collisions).
    """
    if not _HAS_JELLYFISH or len(word) < _PHONETIC_MIN_NORM_LENGTH:
        return None
    try:
        code = _jellyfish.metaphone(word)  # type: ignore[union-attr]
        return code or None
    except Exception:
        return None


def _compile_phonetic_patterns(lexicon: set[str] | None) -> list[tuple[str, ...]]:
    """Patterns of Metaphone codes.

    Lexicon entries where any individual word encodes to ``None`` (too short or
    *jellyfish* unavailable) are excluded so every element in a returned tuple
    is a non-empty string.
    """
    if not lexicon or not _HAS_JELLYFISH:
        return []
    result: set[tuple[str, ...]] = set()
    for entry in lexicon:
        normalized = _normalize_phrase_text(entry)
        if not normalized:
            continue
        codes = tuple(_phonetic_code(tok) for tok in normalized.split())
        if None in codes:
            continue  # skip entries where any token is too short to encode
        result.add(codes)  # type: ignore[arg-type]
    return sorted(result, key=lambda p: (-len(p), p))


def _run_match_pass(
    words: list[TimedWord],
    token_seq: list[str],
    patterns: list[tuple[str, ...]],
    onset_pad: float,
    effective_pad: float,
    skip_indices: set[int],
) -> list[tuple[str, float, float, int, int]]:
    """Single left-to-right matching pass over *token_seq*.

    Returns ``(phrase, start, end, i0, i1)`` tuples for each match found.
    Positions in *skip_indices* are not used as match start points and pattern
    windows that overlap any skipped position are also rejected.
    """
    matched_spans: list[tuple[str, float, float, int, int]] = []
    index = 0
    while index < len(words):
        if index in skip_indices or not token_seq[index]:
            index += 1
            continue
        matched = False
        for pattern in patterns:
            plen = len(pattern)
            end_idx = index + plen
            if end_idx > len(words):
                continue
            # Reject window if any position is already covered by a prior pass.
            if skip_indices.intersection(range(index, end_idx)):
                continue
            if tuple(token_seq[index:end_idx]) != pattern:
                continue
            start = max(0.0, float(words[index].start_seconds) - onset_pad)
            end = max(start, float(words[end_idx - 1].end_seconds) + effective_pad)
            phrase = " ".join(w.word for w in words[index:end_idx]).strip()
            matched_spans.append((phrase, start, end, index, end_idx))
            index = end_idx
            matched = True
            break
        if not matched:
            index += 1
    return matched_spans


def _find_spelled_out_spans(
    words: list[TimedWord],
    normalized_words: list[str],
    exact_patterns: list[tuple[str, ...]],
    stem_patterns: list[tuple[str, ...]],
    onset_pad: float,
    effective_pad: float,
) -> list[tuple[str, float, float, int, int]]:
    """Detect profanity spelled out letter-by-letter (e.g. "f u c k").

    Scans for runs of consecutive single-character normalized tokens where the
    gap between adjacent tokens is at most :data:`_SPELLED_OUT_MAX_GAP_SECONDS`.
    Any sub-run of length ≥ :data:`_SPELLED_OUT_MIN_RUN` whose concatenated
    characters form a single-token pattern match is returned as a span.
    """
    # Only single-token patterns can be reconstructed from spelled-out letters.
    single_patterns = {p[0] for p in exact_patterns if len(p) == 1}
    single_patterns |= {p[0] for p in stem_patterns if len(p) == 1}
    if not single_patterns:
        return []

    n = len(words)
    spans: list[tuple[str, float, float, int, int]] = []
    i = 0
    while i < n:
        if len(normalized_words[i]) != 1:
            i += 1
            continue
        # Extend the run as long as each next token is a single char within the gap.
        run_start = i
        j = i + 1
        while j < n and len(normalized_words[j]) == 1:
            gap = float(words[j].start_seconds) - float(words[j - 1].end_seconds)
            if gap > _SPELLED_OUT_MAX_GAP_SECONDS:
                break
            j += 1
        run_end = j  # exclusive

        if run_end - run_start >= _SPELLED_OUT_MIN_RUN:
            # Check every contiguous sub-window within the run.
            for length in range(_SPELLED_OUT_MIN_RUN, run_end - run_start + 1):
                for s in range(run_start, run_end - length + 1):
                    concat = "".join(normalized_words[s : s + length])
                    if concat in single_patterns:
                        start_t = max(0.0, float(words[s].start_seconds) - onset_pad)
                        end_t = max(
                            start_t,
                            float(words[s + length - 1].end_seconds) + effective_pad,
                        )
                        phrase = " ".join(w.word for w in words[s : s + length]).strip()
                        spans.append((phrase, start_t, end_t, s, s + length))

        i = run_end if run_end > i + 1 else i + 1

    return spans


def scan_text_for_profanity(
    text: str, profanity_words: set[str] | None, sensitivity: str = "stem"
) -> list[str]:
    """Return the original lexicon entries found in *text* (no timestamps needed).

    Applies the same normalisation as the SFX planner.  Each matched entry is
    reported at most once.  Returns an empty list when *profanity_words* is
    ``None`` / empty or *text* is blank.

    *sensitivity* mirrors the same parameter on :func:`build_profanity_sfx_plan`
    (``"exact"``, ``"stem"``, or ``"phonetic"``).
    """
    if not profanity_words or not text.strip():
        return []

    raw_tokens = re.split(r"\s+", text.strip())
    norm_tokens = [_normalize_word(t) for t in raw_tokens if t]
    if not norm_tokens:
        return []

    use_stem = sensitivity in ("stem", "phonetic")
    use_phonetic = sensitivity == "phonetic"

    stem_tokens = [_stem_word(t) for t in norm_tokens] if use_stem else []
    phonetic_tokens = (
        [_phonetic_code(t) or "" for t in norm_tokens] if use_phonetic else []
    )

    # Build pattern → original-entry mappings.
    norm_to_entry: dict[tuple[str, ...], str] = {}
    stem_to_entry: dict[tuple[str, ...], str] = {}
    phonetic_to_entry: dict[tuple[str, ...], str] = {}
    for entry in profanity_words:
        norm = _normalize_phrase_text(entry)
        if not norm:
            continue
        toks = norm.split()
        exact_key: tuple[str, ...] = tuple(toks)
        norm_to_entry[exact_key] = entry
        if use_stem:
            stem_key: tuple[str, ...] = tuple(_stem_word(t) for t in toks)
            stem_to_entry[stem_key] = entry
        if use_phonetic:
            codes = tuple(_phonetic_code(t) for t in toks)
            if None not in codes:
                phonetic_to_entry[codes] = entry  # type: ignore[assignment]

    n = len(norm_tokens)
    seen: set[tuple[str, ...]] = set()
    matched: list[str] = []

    def _scan(
        token_seq: list[str],
        patterns: list[tuple[str, ...]],
        lookup: dict[tuple[str, ...], str],
    ) -> None:
        for pattern in patterns:
            plen = len(pattern)
            if plen > n:
                continue
            for i in range(n - plen + 1):
                if tuple(token_seq[i : i + plen]) == pattern:
                    if pattern not in seen:
                        seen.add(pattern)
                        matched.append(lookup.get(pattern, " ".join(pattern)))
                    break

    exact_patterns = _compile_phrase_patterns(profanity_words)
    _scan(norm_tokens, exact_patterns, norm_to_entry)

    if use_stem:
        stem_patterns = _compile_stem_patterns(profanity_words)
        _scan(stem_tokens, stem_patterns, stem_to_entry)

    if use_phonetic:
        phonetic_patterns = _compile_phonetic_patterns(profanity_words)
        _scan(phonetic_tokens, phonetic_patterns, phonetic_to_entry)

    return matched


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
