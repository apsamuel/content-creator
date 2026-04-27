from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from content_creator.media import CinematicIntroCard, MediaAssembler
from content_creator.media import AudioOverlayEvent
from content_creator.planner import Scene


class Completed:
    def __init__(self, stdout: str = ""):
        self.stdout = stdout


def test_get_audio_duration_parses_ffprobe_json(monkeypatch, tmp_path: Path) -> None:
    assembler = MediaAssembler(width=1280, height=720, fps=24)

    def _run(command, check, capture_output, text):
        assert command[0] == "ffprobe"
        payload = {"format": {"duration": "3.75"}}
        return Completed(stdout=json.dumps(payload))

    monkeypatch.setattr(subprocess, "run", _run)

    duration = assembler.get_audio_duration(tmp_path / "audio.wav")

    assert duration == 3.75


def test_render_video_builds_concat_and_invokes_ffmpeg(
    monkeypatch, tmp_path: Path
) -> None:
    assembler = MediaAssembler(width=1280, height=720, fps=24)
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    image1 = tmp_path / "img1.png"
    image2 = tmp_path / "img2.png"
    image1.write_bytes(b"x")
    image2.write_bytes(b"y")

    scenes = [
        Scene(index=1, prompt="A", duration_seconds=1.0),
        Scene(index=2, prompt="B", duration_seconds=1.5),
    ]
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    output_path = tmp_path / "final.mp4"

    calls: list[list[str]] = []

    def _run(command, check, capture_output, text):
        calls.append(command)
        if command[0] == "ffmpeg":
            # Simulate ffmpeg writing output files for each invocation.
            Path(command[-1]).write_bytes(b"out")
        return Completed(stdout="")

    monkeypatch.setattr(subprocess, "run", _run)

    result = assembler.render_video(
        images=[image1, image2],
        scenes=scenes,
        audio_path=audio_path,
        output_path=output_path,
        work_dir=work_dir,
    )

    assert result == output_path
    assert output_path.exists()

    concat_path = work_dir / "concat.txt"
    assert concat_path.exists()
    concat_text = concat_path.read_text(encoding="utf-8")
    assert "scene_01.mp4" in concat_text
    assert "scene_02.mp4" in concat_text

    ffmpeg_calls = [command for command in calls if command[0] == "ffmpeg"]
    assert len(ffmpeg_calls) == 4
    final_mux_call = ffmpeg_calls[-1]
    assert "-af" in final_mux_call
    assert (
        "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,pan=stereo|c0=c0|c1=c0"
        in final_mux_call
    )
    assert "-ar" in final_mux_call
    assert final_mux_call[final_mux_call.index("-ar") + 1] == "48000"
    assert "-ac" in final_mux_call
    assert final_mux_call[final_mux_call.index("-ac") + 1] == "2"
    assert "-b:a" in final_mux_call
    assert final_mux_call[final_mux_call.index("-b:a") + 1] == "192k"
    assert "-profile:a" in final_mux_call
    assert final_mux_call[final_mux_call.index("-profile:a") + 1] == "aac_low"
    assert "+faststart" in final_mux_call


def test_render_video_supports_multiple_images_per_scene(
    monkeypatch, tmp_path: Path
) -> None:
    assembler = MediaAssembler(width=1280, height=720, fps=24)
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    image1 = tmp_path / "img1.png"
    image2 = tmp_path / "img2.png"
    image3 = tmp_path / "img3.png"
    image4 = tmp_path / "img4.png"
    image1.write_bytes(b"1")
    image2.write_bytes(b"2")
    image3.write_bytes(b"3")
    image4.write_bytes(b"4")

    scenes = [
        Scene(index=1, prompt="A", duration_seconds=1.0),
        Scene(index=2, prompt="B", duration_seconds=1.0),
    ]
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    output_path = tmp_path / "final-seq.mp4"

    calls: list[list[str]] = []

    def _run(command, check, capture_output, text):
        calls.append(command)
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"out")
        return Completed(stdout="")

    monkeypatch.setattr(subprocess, "run", _run)

    result = assembler.render_video(
        images=[[image1, image2], [image3, image4]],
        scenes=scenes,
        audio_path=audio_path,
        output_path=output_path,
        work_dir=work_dir,
    )

    assert result == output_path
    assert output_path.exists()
    ffmpeg_calls = [command for command in calls if command[0] == "ffmpeg"]
    # 4 segment renders + 2 per-scene sequence concats + 1 scene-stitch concat + 1 final mux
    assert len(ffmpeg_calls) == 8


def test_render_video_with_cinematic_intro_prepends_title_card(
    monkeypatch, tmp_path: Path
) -> None:
    assembler = MediaAssembler(width=1280, height=720, fps=24)
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    image1 = tmp_path / "img1.png"
    image2 = tmp_path / "img2.png"
    image1.write_bytes(b"x")
    image2.write_bytes(b"y")

    scenes = [
        Scene(index=1, prompt="A", duration_seconds=1.0),
        Scene(index=2, prompt="B", duration_seconds=1.5),
    ]
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    output_path = tmp_path / "final-cinematic.mp4"

    calls: list[list[str]] = []

    def _run(command, check, capture_output, text):
        calls.append(command)
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"out")
        return Completed(stdout="")

    monkeypatch.setattr(subprocess, "run", _run)

    result = assembler.render_video(
        images=[image1, image2],
        scenes=scenes,
        audio_path=audio_path,
        output_path=output_path,
        work_dir=work_dir,
        cinematic_intro=CinematicIntroCard(
            title="Quarterly Chaos, Now in Widescreen",
            description="A very official update that accidentally became a comedy special.",
            duration_seconds=5.8,
        ),
    )

    assert result == output_path
    assert output_path.exists()
    ffmpeg_calls = [command for command in calls if command[0] == "ffmpeg"]
    assert len(ffmpeg_calls) == 6
    assert any(
        "concat_intro.txt" in " ".join(command)
        for command in ffmpeg_calls
        if "-f" in command and "concat" in command
    )
    final_mux_call = ffmpeg_calls[-1]
    assert "-af" in final_mux_call
    filter_value = final_mux_call[final_mux_call.index("-af") + 1]
    assert "adelay=5800:all=1" in filter_value
    assert "loudnorm=I=-16:TP=-1.5:LRA=11" in filter_value


def test_render_video_with_cinematic_transitions_uses_xfade(
    monkeypatch, tmp_path: Path
) -> None:
    assembler = MediaAssembler(width=1280, height=720, fps=24)
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    image1 = tmp_path / "img1.png"
    image2 = tmp_path / "img2.png"
    image1.write_bytes(b"x")
    image2.write_bytes(b"y")

    scenes = [
        Scene(index=1, prompt="A", duration_seconds=1.0),
        Scene(index=2, prompt="B", duration_seconds=1.5),
    ]
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    output_path = tmp_path / "final-cinematic-transitions.mp4"

    calls: list[list[str]] = []

    def _run(command, check, capture_output, text):
        calls.append(command)
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"out")
        return Completed(stdout="")

    monkeypatch.setattr(subprocess, "run", _run)

    result = assembler.render_video(
        images=[image1, image2],
        scenes=scenes,
        audio_path=audio_path,
        output_path=output_path,
        work_dir=work_dir,
        cinematic_transitions=True,
    )

    assert result == output_path
    assert output_path.exists()

    ffmpeg_calls = [command for command in calls if command[0] == "ffmpeg"]
    # 2 scene clips + 1 xfade stitch + 1 final mux
    assert len(ffmpeg_calls) == 4
    stitched_call = ffmpeg_calls[2]
    assert "-filter_complex" in stitched_call
    filter_graph = stitched_call[stitched_call.index("-filter_complex") + 1]
    assert "xfade=transition=" in filter_graph


def test_chunk_audio_invokes_ffmpeg_and_returns_segments(
    monkeypatch, tmp_path: Path
) -> None:
    assembler = MediaAssembler(width=1280, height=720, fps=24)
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"audio")
    output_dir = tmp_path / "chunks"

    calls: list[list[str]] = []

    def _run(command, check, capture_output, text):
        calls.append(command)
        Path(output_dir / "chunk_0000.wav").write_bytes(b"a")
        Path(output_dir / "chunk_0001.wav").write_bytes(b"b")
        return Completed(stdout="")

    monkeypatch.setattr(subprocess, "run", _run)

    chunks = assembler.chunk_audio(
        audio_path=input_audio, output_dir=output_dir, chunk_seconds=30.0
    )

    assert [chunk.name for chunk in chunks] == ["chunk_0000.wav", "chunk_0001.wav"]
    assert calls[0][0] == "ffmpeg"
    assert "-segment_time" in calls[0]


def test_chunk_audio_rejects_non_positive_chunk_size(tmp_path: Path) -> None:
    assembler = MediaAssembler(width=1280, height=720, fps=24)
    with pytest.raises(ValueError, match="chunk_seconds"):
        assembler.chunk_audio(
            audio_path=tmp_path / "input.wav",
            output_dir=tmp_path / "chunks",
            chunk_seconds=0,
        )


def test_overlay_sound_effects_builds_filter_graph(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assembler = MediaAssembler(width=1280, height=720, fps=24)
    audio_path = tmp_path / "audio.wav"
    sfx_path = tmp_path / "button.wav"
    output_path = tmp_path / "censored.m4a"
    audio_path.write_bytes(b"audio")
    sfx_path.write_bytes(b"sfx")

    calls: list[list[str]] = []

    def _run(command, check, capture_output, text):
        calls.append(command)
        Path(command[-1]).write_bytes(b"rendered")
        return Completed(stdout="")

    monkeypatch.setattr(subprocess, "run", _run)

    result = assembler.overlay_sound_effects(
        audio_path=audio_path,
        output_path=output_path,
        events=[
            AudioOverlayEvent(
                start_seconds=0.4,
                end_seconds=0.8,
                sfx_path=sfx_path,
                sfx_duration_seconds=0.2,
                sfx_gain_db=2.5,
            )
        ],
        duck_db=-16.0,
    )

    assert result == output_path
    assert output_path.exists()
    command = calls[0]
    assert command[0] == "ffmpeg"
    assert "-stream_loop" in command
    assert command[command.index("-stream_loop") + 1] == "-1"
    assert "-filter_complex" in command
    filter_graph = command[command.index("-filter_complex") + 1]
    assert "between(t,0.400,0.800)" in filter_graph
    assert "adelay=400|400" in filter_graph
    assert "-profile:a" in command
    assert command[command.index("-profile:a") + 1] == "aac_low"
    assert "-ar" in command
    assert command[command.index("-ar") + 1] == "48000"
    assert "-ac" in command
    assert command[command.index("-ac") + 1] == "2"
