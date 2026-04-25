from __future__ import annotations

from pathlib import Path

import pytest

from content_creator.hf_client import TimedWord
from content_creator.profanity_sfx import (
    analyze_profanity_lexicon,
    build_profanity_sfx_plan,
    load_profanity_words,
    load_sound_pack,
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
