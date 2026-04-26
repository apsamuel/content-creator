from __future__ import annotations

import json
import subprocess
import tempfile
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
        images: Iterable[Path] | Iterable[Iterable[Path]],
        scenes: list[Scene],
        audio_path: Path,
        output_path: Path,
        work_dir: Path,
    ) -> Path:
        clips_dir = work_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        clip_paths: list[Path] = []

        image_items = list(images)
        scene_image_sequences: list[list[Path]]
        if image_items and all(isinstance(item, Path) for item in image_items):
            scene_image_sequences = [[item] for item in image_items]
        else:
            scene_image_sequences = [
                [Path(path) for path in image_group] for image_group in image_items
            ]

        if len(scene_image_sequences) != len(scenes):
            raise ValueError(
                "Number of scene image groups must match number of scenes "
                f"({len(scene_image_sequences)} != {len(scenes)})"
            )

        for scene, scene_images in zip(scenes, scene_image_sequences, strict=True):
            if not scene_images:
                raise ValueError(f"Scene {scene.index} has no images to render")
            clip_path = clips_dir / f"scene_{scene.index:02d}.mp4"
            if len(scene_images) == 1:
                self._render_scene_clip(
                    image_path=scene_images[0],
                    duration=scene.duration_seconds,
                    output_path=clip_path,
                )
            else:
                self._render_scene_sequence_clip(
                    image_paths=scene_images,
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
                "-profile:a",
                "aac_low",
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
                "-profile:a",
                "aac_low",
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
        frame_count = max(1, int(round(duration * self._fps)))
        zoom_expr = (
            "zoom='min(zoom+0.0008,1.08)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        )
        vf = (
            f"scale={self._width}:{self._height}:force_original_aspect_ratio=increase,"
            f"crop={self._width}:{self._height},"
            f"zoompan={zoom_expr}:d={frame_count}:s={self._width}x{self._height}:fps={self._fps},"
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

    def _render_scene_sequence_clip(
        self, *, image_paths: list[Path], duration: float, output_path: Path
    ) -> None:
        segment_count = len(image_paths)
        if segment_count <= 0:
            raise ValueError("image_paths must contain at least one image")

        min_duration = 1.0 / max(1, self._fps)
        split_duration = max(min_duration, duration / segment_count)

        with tempfile.TemporaryDirectory(prefix="scene_seq_") as tmp_str:
            tmp_dir = Path(tmp_str)
            partial_clips: list[Path] = []
            for index, image_path in enumerate(image_paths, start=1):
                partial_path = tmp_dir / f"segment_{index:02d}.mp4"
                self._render_scene_clip(
                    image_path=image_path,
                    duration=split_duration,
                    output_path=partial_path,
                )
                partial_clips.append(partial_path)

            concat_list = tmp_dir / "concat.txt"
            concat_list.write_text(
                "\n".join(f"file '{path.as_posix()}'" for path in partial_clips),
                encoding="utf-8",
            )
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
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
