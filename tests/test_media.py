from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from content_creator.media import MediaAssembler
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
    assert "+faststart" in final_mux_call


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
