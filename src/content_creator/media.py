from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from content_creator.planner import Scene


@dataclass(slots=True)
class AudioOverlayEvent:
    start_seconds: float
    end_seconds: float
    sfx_path: Path
    sfx_duration_seconds: float
    sfx_gain_db: float = 0.0


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

    def overlay_sound_effects(
        self,
        *,
        audio_path: Path,
        output_path: Path,
        events: list[AudioOverlayEvent],
        duck_db: float,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not events:
            output_path.write_bytes(audio_path.read_bytes())
            return output_path

        command: list[str] = ["ffmpeg", "-y", "-i", str(audio_path)]
        for event in events:
            command.extend(["-stream_loop", "-1", "-i", str(event.sfx_path)])

        duck_gain = 10 ** (duck_db / 20.0)
        filter_parts: list[str] = []

        current_base = "[0:a]"
        for index, event in enumerate(events, start=1):
            next_base = f"[base_{index}]"
            filter_parts.append(
                f"{current_base}volume={duck_gain:.6f}:enable='between(t,{event.start_seconds:.3f},{event.end_seconds:.3f})'{next_base}"
            )
            current_base = next_base

        overlay_labels: list[str] = []
        for index, event in enumerate(events, start=1):
            delay_ms = max(0, int(event.start_seconds * 1000))
            label = f"[sfx_{index}]"
            overlay_labels.append(label)
            filter_parts.append(
                f"[{index}:a]atrim=0:{event.sfx_duration_seconds:.3f},asetpts=PTS-STARTPTS,"
                f"volume={event.sfx_gain_db:.2f}dB,adelay={delay_ms}|{delay_ms}{label}"
            )

        amix_inputs = current_base + "".join(overlay_labels)
        filter_parts.append(
            f"{amix_inputs}amix=inputs={1 + len(overlay_labels)}:normalize=0:dropout_transition=0[out]"
        )

        command.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                "[out]",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "48000",
                "-ac",
                "2",
                str(output_path),
            ]
        )
        subprocess.run(command, check=True, capture_output=True, text=True)
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
