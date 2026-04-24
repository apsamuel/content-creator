from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable

from content_creator.planner import Scene


class MediaAssembler:
    def __init__(self, *, width: int, height: int, fps: int):
        self._width = width
        self._height = height
        self._fps = fps

    def get_audio_duration(self, audio_path: Path) -> float:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(audio_path),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
        return float(payload["format"]["duration"])

    def chunk_audio(
        self, *, audio_path: Path, output_dir: Path, chunk_seconds: float
    ) -> list[Path]:
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be greater than 0")

        output_dir.mkdir(parents=True, exist_ok=True)
        pattern = output_dir / "chunk_%04d.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-f",
                "segment",
                "-segment_time",
                str(chunk_seconds),
                "-c:a",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(pattern),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        chunks = sorted(output_dir.glob("chunk_*.wav"))
        if not chunks:
            raise RuntimeError("ffmpeg produced no audio chunks")
        return chunks

    def render_video(
        self,
        *,
        images: Iterable[Path],
        scenes: list[Scene],
        audio_path: Path,
        output_path: Path,
        work_dir: Path,
    ) -> Path:
        clips_dir = work_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        clip_paths: list[Path] = []

        for scene, image_path in zip(scenes, images, strict=True):
            clip_path = clips_dir / f"scene_{scene.index:02d}.mp4"
            self._render_scene_clip(
                image_path=image_path,
                duration=scene.duration_seconds,
                output_path=clip_path,
            )
            clip_paths.append(clip_path)

        concat_list = work_dir / "concat.txt"
        concat_list.write_text(
            "\n".join(f"file '{path.as_posix()}'" for path in clip_paths),
            encoding="utf-8",
        )

        stitched = work_dir / "stitched.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c",
                "copy",
                str(stitched),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(stitched),
                "-i",
                str(audio_path),
                "-c:v",
                "copy",
                "-af",
                "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,pan=stereo|c0=c0|c1=c0",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                "-shortest",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return output_path

    def _render_scene_clip(
        self, *, image_path: Path, duration: float, output_path: Path
    ) -> None:
        zoom_expr = (
            "zoom='min(zoom+0.0008,1.08)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        )
        vf = (
            f"scale={self._width}:{self._height}:force_original_aspect_ratio=increase,"
            f"crop={self._width}:{self._height},"
            f"zoompan={zoom_expr}:d={int(duration * self._fps)}:s={self._width}x{self._height}:fps={self._fps},"
            "format=yuv420p"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loop",
                "1",
                "-i",
                str(image_path),
                "-t",
                str(duration),
                "-vf",
                vf,
                "-r",
                str(self._fps),
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
