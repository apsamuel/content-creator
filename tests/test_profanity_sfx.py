from __future__ import annotations

from pathlib import Path

import pytest

from content_creator.hf_client import TimedWord
from content_creator.profanity_sfx import (
    analyze_profanity_lexicon,
    build_profanity_sfx_plan,
    load_profanity_words,
    load_sound_pack,
    scan_text_for_profanity,
)


class Completed:
    def __init__(self, stdout: str = "", stderr: str = ""):
        self.stdout = stdout
        self.stderr = stderr


def test_load_sound_pack_collects_duration_and_loudness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.profanity_sfx as module

    sound_dir = tmp_path / "sound"
    sound_dir.mkdir(parents=True, exist_ok=True)
    (sound_dir / "button.wav").write_bytes(b"sound")

    def _run(command, check, capture_output, text):
        if command[0] == "ffprobe":
            return Completed(stdout="0.42\n")
        return Completed(stderr="mean_volume: -22.1 dB\nmax_volume: -4.2 dB\n")

    monkeypatch.setattr(module.subprocess, "run", _run)

    pack = load_sound_pack(sound_pack_dir=sound_dir)

    assert pack.name == "sound"
    assert len(pack.assets) == 1
    assert pack.assets[0].duration_seconds == pytest.approx(0.42)
    assert pack.assets[0].mean_volume_db == pytest.approx(-22.1)


def test_build_profanity_sfx_plan_matches_and_merges_spans(tmp_path: Path) -> None:
    from content_creator.profanity_sfx import SoundEffectAsset, SoundPack

    sound_file = tmp_path / "button.wav"
    sound_file.write_bytes(b"sound")
    pack = SoundPack(
        name="pack",
        root_dir=tmp_path,
        target_mean_db=-18.0,
        assets=[
            SoundEffectAsset(
                path=sound_file,
                duration_seconds=0.25,
                mean_volume_db=-20.0,
                max_volume_db=-4.0,
            )
        ],
    )
    words = [
        TimedWord(word="hello", start_seconds=0.0, end_seconds=0.2),
        TimedWord(word="d4mn", start_seconds=0.3, end_seconds=0.5),
        TimedWord(word="shit", start_seconds=0.52, end_seconds=0.65),
    ]

    plan = build_profanity_sfx_plan(
        timed_words=words,
        sound_pack=pack,
        profanity_words={"damn", "shit"},
        pad_seconds=0.05,
    )

    assert plan.total_words == 3
    assert plan.matches_found == 1
    assert len(plan.events) == 1
    assert plan.events[0].word == "d4mn shit"
    assert plan.events[0].start_seconds == pytest.approx(0.17)
    assert plan.events[0].end_seconds == pytest.approx(0.70)
    assert plan.events[0].sfx_duration_seconds == pytest.approx(0.53)


def test_build_profanity_sfx_plan_prefers_bleep_1_and_covers_full_window(
    tmp_path: Path,
) -> None:
    from content_creator.profanity_sfx import SoundEffectAsset, SoundPack

    preferred = tmp_path / "bleep-1.wav"
    alternate = tmp_path / "bleep-2.wav"
    preferred.write_bytes(b"preferred")
    alternate.write_bytes(b"alternate")
    pack = SoundPack(
        name="pack",
        root_dir=tmp_path,
        target_mean_db=-18.0,
        assets=[
            SoundEffectAsset(
                path=alternate,
                duration_seconds=0.18,
                mean_volume_db=-21.0,
                max_volume_db=-4.0,
            ),
            SoundEffectAsset(
                path=preferred,
                duration_seconds=0.16,
                mean_volume_db=-20.0,
                max_volume_db=-4.0,
            ),
        ],
    )
    words = [
        TimedWord(word="damn", start_seconds=0.30, end_seconds=0.52),
        TimedWord(word="hello", start_seconds=0.60, end_seconds=0.72),
        TimedWord(word="fuck", start_seconds=0.90, end_seconds=1.24),
    ]

    plan = build_profanity_sfx_plan(
        timed_words=words,
        sound_pack=pack,
        profanity_words={"damn", "fuck"},
        pad_seconds=0.05,
    )

    assert len(plan.events) == 2
    assert all(event.sfx_path == preferred for event in plan.events)
    assert plan.events[0].start_seconds == pytest.approx(0.17)
    assert plan.events[0].sfx_duration_seconds == pytest.approx(0.40)
    assert plan.events[1].start_seconds == pytest.approx(0.77)
    assert plan.events[1].sfx_duration_seconds == pytest.approx(0.52)


def test_build_profanity_sfx_plan_biases_early_onset_coverage(tmp_path: Path) -> None:
    from content_creator.profanity_sfx import SoundEffectAsset, SoundPack

    sound_file = tmp_path / "bleep-1.wav"
    sound_file.write_bytes(b"sound")
    pack = SoundPack(
        name="pack",
        root_dir=tmp_path,
        target_mean_db=-18.0,
        assets=[
            SoundEffectAsset(
                path=sound_file,
                duration_seconds=0.20,
                mean_volume_db=-20.0,
                max_volume_db=-4.0,
            )
        ],
    )
    words = [
        TimedWord(word="damn", start_seconds=0.11, end_seconds=0.24),
        TimedWord(word="hello", start_seconds=0.40, end_seconds=0.52),
    ]

    plan = build_profanity_sfx_plan(
        timed_words=words, sound_pack=pack, profanity_words={"damn"}, pad_seconds=0.08
    )

    assert len(plan.events) == 1
    assert plan.events[0].start_seconds == pytest.approx(0.0)
    assert plan.events[0].end_seconds == pytest.approx(0.32)


def test_build_profanity_sfx_plan_merges_close_profanity_gaps(tmp_path: Path) -> None:
    from content_creator.profanity_sfx import SoundEffectAsset, SoundPack

    sound_file = tmp_path / "bleep-1.wav"
    sound_file.write_bytes(b"sound")
    pack = SoundPack(
        name="pack",
        root_dir=tmp_path,
        target_mean_db=-18.0,
        assets=[
            SoundEffectAsset(
                path=sound_file,
                duration_seconds=0.20,
                mean_volume_db=-20.0,
                max_volume_db=-4.0,
            )
        ],
    )
    words = [
        TimedWord(word="nigger", start_seconds=24.92, end_seconds=25.32),
        TimedWord(word="nigga", start_seconds=26.06, end_seconds=26.54),
    ]

    plan = build_profanity_sfx_plan(
        timed_words=words,
        sound_pack=pack,
        profanity_words={"nigger", "nigga"},
        pad_seconds=0.08,
    )

    assert plan.matches_found == 1
    assert len(plan.events) == 1
    assert plan.events[0].word == "nigger nigga"
    assert plan.events[0].start_seconds == pytest.approx(24.76)
    assert plan.events[0].end_seconds == pytest.approx(26.62)


def test_load_sound_pack_coerces_target_mean_db_from_manifest_string(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.profanity_sfx as module

    sound_dir = tmp_path / "sound"
    sound_dir.mkdir(parents=True, exist_ok=True)
    (sound_dir / "beep.wav").write_bytes(b"sound")
    (sound_dir / "pack.json").write_text(
        '{"target_mean_db": "-21.5", "sounds": ["beep.wav"]}', encoding="utf-8"
    )

    def _run(command, check, capture_output, text):
        if command[0] == "ffprobe":
            return Completed(stdout="0.50\n")
        return Completed(stderr="mean_volume: -19.0 dB\nmax_volume: -3.0 dB\n")

    monkeypatch.setattr(module.subprocess, "run", _run)

    pack = load_sound_pack(sound_pack_dir=sound_dir)

    assert pack.target_mean_db == pytest.approx(-21.5)


def test_load_sound_pack_rejects_invalid_target_mean_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import content_creator.profanity_sfx as module

    sound_dir = tmp_path / "sound"
    sound_dir.mkdir(parents=True, exist_ok=True)
    (sound_dir / "beep.wav").write_bytes(b"sound")
    (sound_dir / "pack.json").write_text(
        '{"target_mean_db": "loud", "sounds": ["beep.wav"]}', encoding="utf-8"
    )

    def _run(command, check, capture_output, text):
        if command[0] == "ffprobe":
            return Completed(stdout="0.50\n")
        return Completed(stderr="mean_volume: -19.0 dB\nmax_volume: -3.0 dB\n")

    monkeypatch.setattr(module.subprocess, "run", _run)

    with pytest.raises(ValueError, match="target_mean_db"):
        load_sound_pack(sound_pack_dir=sound_dir)


def test_load_profanity_words_defaults_to_bundled_file() -> None:
    words = load_profanity_words(None)

    assert words is not None
    assert "fuck" in words
    assert "sexual assault" in words


def test_build_profanity_sfx_plan_matches_phrase_entries(tmp_path: Path) -> None:
    from content_creator.profanity_sfx import SoundEffectAsset, SoundPack

    sound_file = tmp_path / "bleep-1.wav"
    sound_file.write_bytes(b"sound")
    pack = SoundPack(
        name="pack",
        root_dir=tmp_path,
        target_mean_db=-18.0,
        assets=[
            SoundEffectAsset(
                path=sound_file,
                duration_seconds=0.20,
                mean_volume_db=-20.0,
                max_volume_db=-4.0,
            )
        ],
    )
    words = [
        TimedWord(word="this", start_seconds=0.0, end_seconds=0.1),
        TimedWord(word="is", start_seconds=0.1, end_seconds=0.2),
        TimedWord(word="sexual", start_seconds=0.2, end_seconds=0.4),
        TimedWord(word="assault", start_seconds=0.4, end_seconds=0.7),
    ]

    plan = build_profanity_sfx_plan(
        timed_words=words,
        sound_pack=pack,
        profanity_words={"sexual assault"},
        pad_seconds=0.05,
    )

    assert plan.matches_found == 1
    assert len(plan.events) == 1
    assert plan.events[0].word == "sexual assault"
    assert plan.events[0].start_seconds == pytest.approx(0.07)
    assert plan.events[0].end_seconds == pytest.approx(0.75)


def test_analyze_profanity_lexicon_detects_duplicates_and_near_duplicates(
    tmp_path: Path,
) -> None:
    lexicon_file = tmp_path / "profanity_words.txt"
    lexicon_file.write_text(
        """# Comment
Fuck
fuck
Wet   Back
wet back
Big   Tits
big tits
""",
        encoding="utf-8",
    )

    report = analyze_profanity_lexicon(lexicon_file)

    assert report.path == lexicon_file.resolve()
    assert report.total_lines == 7
    assert report.active_lines == 6
    assert report.unique_normalized_entries == 3
    assert report.exact_duplicates == {"Fuck": 2}
    assert set(report.near_duplicates["wet back"]) == {"wet back", "Wet   Back"}
    assert set(report.near_duplicates["big tits"]) == {"Big   Tits", "big tits"}


def test_scan_text_for_profanity_returns_empty_for_none_lexicon() -> None:
    assert scan_text_for_profanity("hello world", None) == []


def test_scan_text_for_profanity_returns_empty_for_blank_text() -> None:
    assert scan_text_for_profanity("   ", {"badword"}) == []


def test_scan_text_for_profanity_matches_single_word() -> None:
    matched = scan_text_for_profanity("this is badword here", {"badword"})
    assert matched == ["badword"]


def test_scan_text_for_profanity_no_match_returns_empty() -> None:
    matched = scan_text_for_profanity("hello world", {"badword"})
    assert matched == []


def test_scan_text_for_profanity_matches_phrase() -> None:
    matched = scan_text_for_profanity("this is a bad phrase here", {"bad phrase"})
    assert matched == ["bad phrase"]


def test_scan_text_for_profanity_matches_leet_speak() -> None:
    # 'f4ck' normalises the same way as 'fack'
    matched = scan_text_for_profanity("you f4ck off", {"fack"})
    assert matched == ["fack"]


def test_scan_text_for_profanity_deduplicates_repeated_word() -> None:
    matched = scan_text_for_profanity("badword badword badword", {"badword"})
    assert matched == ["badword"]


def test_scan_text_for_profanity_prefers_longer_phrase_over_subset() -> None:
    matched = scan_text_for_profanity("bad phrase in text", {"bad", "bad phrase"})
    # 'bad phrase' (longer) is matched and counted once; 'bad' alone may also match
    assert "bad phrase" in matched


# ---------------------------------------------------------------------------
# Tiered sensitivity tests
# ---------------------------------------------------------------------------


def _make_pack(tmp_path: Path):
    from content_creator.profanity_sfx import SoundEffectAsset, SoundPack

    sound_file = tmp_path / "bleep.wav"
    sound_file.write_bytes(b"s")
    return SoundPack(
        name="pack",
        root_dir=tmp_path,
        target_mean_db=-18.0,
        assets=[
            SoundEffectAsset(
                path=sound_file,
                duration_seconds=0.3,
                mean_volume_db=-20.0,
                max_volume_db=-4.0,
            )
        ],
    )


# ---- scan_text_for_profanity: sensitivity ----


def test_scan_text_for_profanity_stem_catches_inflected_form() -> None:
    """Default sensitivity="stem" detects inflected forms like 'fucking'."""
    matched = scan_text_for_profanity("you're fucking stupid", {"fuck"})
    assert "fuck" in matched


def test_scan_text_for_profanity_exact_skips_inflected_form() -> None:
    """sensitivity="exact" must NOT match inflected forms."""
    matched = scan_text_for_profanity(
        "you're fucking stupid", {"fuck"}, sensitivity="exact"
    )
    assert matched == []


def test_scan_text_for_profanity_stem_catches_er_suffix() -> None:
    matched = scan_text_for_profanity("that fucker ran away", {"fuck"})
    assert "fuck" in matched


def test_scan_text_for_profanity_stem_catches_ed_suffix() -> None:
    matched = scan_text_for_profanity("he fucked up everything", {"fuck"})
    assert "fuck" in matched


def test_scan_text_for_profanity_stem_catches_ers_suffix() -> None:
    matched = scan_text_for_profanity("bunch of motherfuckers here", {"motherfuck"})
    assert "motherfuck" in matched


def test_scan_text_for_profanity_exact_matches_base_form_unchanged() -> None:
    """sensitivity="exact" still matches the un-inflected base form."""
    matched = scan_text_for_profanity("go to hell", {"hell"}, sensitivity="exact")
    assert "hell" in matched


# ---- build_profanity_sfx_plan: sensitivity ----


def test_build_plan_stem_catches_inflected_form(tmp_path: Path) -> None:
    """With default sensitivity='stem', 'fucking' triggers a bleep for lexicon entry 'fuck'."""
    pack = _make_pack(tmp_path)
    words = [
        TimedWord(word="you're", start_seconds=0.0, end_seconds=0.3),
        TimedWord(word="fucking", start_seconds=0.35, end_seconds=0.65),
        TimedWord(word="kidding", start_seconds=0.7, end_seconds=1.0),
    ]
    plan = build_profanity_sfx_plan(
        timed_words=words, sound_pack=pack, profanity_words={"fuck"}, pad_seconds=0.05
    )
    assert plan.matches_found == 1
    assert plan.events[0].word == "fucking"


def test_build_plan_exact_skips_inflected_form(tmp_path: Path) -> None:
    """sensitivity='exact' must NOT bleep inflected forms."""
    pack = _make_pack(tmp_path)
    words = [
        TimedWord(word="you're", start_seconds=0.0, end_seconds=0.3),
        TimedWord(word="fucking", start_seconds=0.35, end_seconds=0.65),
        TimedWord(word="kidding", start_seconds=0.7, end_seconds=1.0),
    ]
    plan = build_profanity_sfx_plan(
        timed_words=words,
        sound_pack=pack,
        profanity_words={"fuck"},
        pad_seconds=0.05,
        sensitivity="exact",
    )
    assert plan.matches_found == 0


def test_build_plan_exact_still_matches_base_form(tmp_path: Path) -> None:
    pack = _make_pack(tmp_path)
    words = [
        TimedWord(word="oh", start_seconds=0.0, end_seconds=0.2),
        TimedWord(word="fuck", start_seconds=0.3, end_seconds=0.5),
    ]
    plan = build_profanity_sfx_plan(
        timed_words=words,
        sound_pack=pack,
        profanity_words={"fuck"},
        sensitivity="exact",
    )
    assert plan.matches_found == 1


# ---- spelled-out detection ----


def test_build_plan_detects_spelled_out_profanity(tmp_path: Path) -> None:
    """Single-letter tokens 'f u c k' within gap threshold trigger a bleep."""
    pack = _make_pack(tmp_path)
    words = [
        TimedWord(word="f", start_seconds=0.0, end_seconds=0.15),
        TimedWord(word="u", start_seconds=0.2, end_seconds=0.35),
        TimedWord(word="c", start_seconds=0.4, end_seconds=0.55),
        TimedWord(word="k", start_seconds=0.6, end_seconds=0.75),
    ]
    plan = build_profanity_sfx_plan(
        timed_words=words, sound_pack=pack, profanity_words={"fuck"}
    )
    assert plan.matches_found == 1


def test_build_plan_ignores_spelled_out_with_large_gap(tmp_path: Path) -> None:
    """Spelled-out letters separated by > 0.8 s are NOT collapsed."""
    pack = _make_pack(tmp_path)
    words = [
        TimedWord(word="f", start_seconds=0.0, end_seconds=0.15),
        TimedWord(word="u", start_seconds=1.1, end_seconds=1.3),  # > 0.8 s gap
        TimedWord(word="c", start_seconds=1.4, end_seconds=1.55),
        TimedWord(word="k", start_seconds=1.6, end_seconds=1.75),
    ]
    plan = build_profanity_sfx_plan(
        timed_words=words, sound_pack=pack, profanity_words={"fuck"}
    )
    assert plan.matches_found == 0


def test_build_plan_stem_does_not_double_count(tmp_path: Path) -> None:
    """The exact pass already covers 'fuck'; stem pass must not add a duplicate."""
    pack = _make_pack(tmp_path)
    words = [TimedWord(word="fuck", start_seconds=0.0, end_seconds=0.4)]
    plan = build_profanity_sfx_plan(
        timed_words=words, sound_pack=pack, profanity_words={"fuck"}
    )
    assert plan.matches_found == 1
