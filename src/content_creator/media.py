from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
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


@dataclass(slots=True)
class CinematicIntroCard:
    title: str
    description: str
    duration_seconds: float


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
        cinematic_intro: CinematicIntroCard | None = None,
        cinematic_transitions: bool = False,
    ) -> Path:
        clips_dir = work_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        clip_paths: list[Path] = []

        image_items = list(images)
        scene_image_sequences: list[list[Path]] = []
        for item in image_items:
            if isinstance(item, Path):
                scene_image_sequences.append([item])
            else:
                scene_image_sequences.append([Path(path) for path in item])

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

        stitched = work_dir / "stitched.mp4"
        if cinematic_transitions and len(clip_paths) > 1:
            self._stitch_with_cinematic_transitions(
                clip_paths=clip_paths, scenes=scenes, output_path=stitched
            )
        else:
            concat_list = work_dir / "concat.txt"
            concat_list.write_text(
                "\n".join(f"file '{path.as_posix()}'" for path in clip_paths),
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
                    str(stitched),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        final_visual_path = stitched
        intro_delay_seconds = 0.0
        if cinematic_intro is not None:
            intro_clip = work_dir / "intro_card.mp4"
            self._render_intro_card(intro_card=cinematic_intro, output_path=intro_clip)

            intro_concat_list = work_dir / "concat_intro.txt"
            intro_concat_list.write_text(
                "\n".join(
                    [f"file '{intro_clip.as_posix()}'", f"file '{stitched.as_posix()}'"]
                ),
                encoding="utf-8",
            )
            stitched_with_intro = work_dir / "stitched_with_intro.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(intro_concat_list),
                    "-c",
                    "copy",
                    str(stitched_with_intro),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            final_visual_path = stitched_with_intro
            intro_delay_seconds = max(0.0, float(cinematic_intro.duration_seconds))

        audio_filter = (
            "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,pan=stereo|c0=c0|c1=c0"
        )
        if intro_delay_seconds > 0:
            delay_ms = int(round(intro_delay_seconds * 1000))
            audio_filter = f"adelay={delay_ms}:all=1,{audio_filter}"

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(final_visual_path),
                "-i",
                str(audio_path),
                "-c:v",
                "copy",
                "-af",
                audio_filter,
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

    def _stitch_with_cinematic_transitions(
        self, *, clip_paths: list[Path], scenes: list[Scene], output_path: Path
    ) -> None:
        command: list[str] = ["ffmpeg", "-y"]
        for clip_path in clip_paths:
            command.extend(["-i", str(clip_path)])

        minimum_frame_seconds = 1.0 / max(1, self._fps)
        accumulated_duration = max(minimum_frame_seconds, scenes[0].duration_seconds)
        current_label = "[0:v]"
        filter_parts: list[str] = []

        for index in range(1, len(clip_paths)):
            transition = scenes[index - 1].transition_to_next
            transition_name = self._map_transition_name(
                transition_type=(
                    transition.transition_type if transition is not None else None
                )
            )
            requested_duration = self._resolve_transition_duration_seconds(transition)
            next_scene_duration = max(
                minimum_frame_seconds, scenes[index].duration_seconds
            )

            max_duration = max(
                minimum_frame_seconds,
                min(accumulated_duration, next_scene_duration) - minimum_frame_seconds,
            )
            transition_duration = min(requested_duration, max_duration)
            offset = max(0.0, accumulated_duration - transition_duration)
            output_label = f"[v{index}]"
            filter_parts.append(
                f"{current_label}[{index}:v]"
                f"xfade=transition={transition_name}:"
                f"duration={transition_duration:.3f}:offset={offset:.3f}"
                f"{output_label}"
            )
            current_label = output_label
            accumulated_duration = (
                accumulated_duration + next_scene_duration - transition_duration
            )

        command.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                current_label,
                "-c:v",
                "libx264",
                "-r",
                str(self._fps),
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(output_path),
            ]
        )
        subprocess.run(command, check=True, capture_output=True, text=True)

    def _resolve_transition_duration_seconds(self, transition: object | None) -> float:
        frames = 12
        if transition is not None:
            raw_frames = getattr(transition, "duration_frames", 12)
            if isinstance(raw_frames, (int, float)):
                frames = int(raw_frames)
        seconds = frames / max(1, self._fps)
        return max(0.2, min(1.2, float(seconds)))

    def _map_transition_name(self, *, transition_type: str | None) -> str:
        if not transition_type:
            return "fade"
        transition_map = {
            "dissolve": "dissolve",
            "match_cut": "fade",
            "whip_pan": "wipeleft",
            "focus_shift": "fadeblack",
            "color_match": "fade",
            "light_leak": "fadewhite",
            "tracking": "slideleft",
            "bokeh": "circleopen",
        }
        return transition_map.get(transition_type, "fade")

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

    def _render_intro_card(
        self, *, intro_card: CinematicIntroCard, output_path: Path
    ) -> None:
        duration = max(2.5, float(intro_card.duration_seconds))
        fade_window = min(1.2, duration * 0.22)
        title_hold_end = max(fade_window, duration - fade_window)

        description_text = self._wrap_intro_description(intro_card.description)
        title_text = self._escape_drawtext(intro_card.title)
        description_escaped = self._escape_drawtext(description_text)

        title_alpha = (
            f"if(lt(t,{fade_window:.2f}),t/{fade_window:.2f},"
            f"if(lt(t,{title_hold_end:.2f}),1,"
            f"if(lt(t,{duration:.2f}),({duration:.2f}-t)/{fade_window:.2f},0)))"
        )

        description_fade_in_start = min(duration * 0.24, duration - (fade_window * 2))
        description_fade_in_end = min(
            description_fade_in_start + 0.8, duration - (fade_window * 1.3)
        )
        description_fade_out_start = max(
            description_fade_in_end + 0.6, duration - (fade_window * 1.7)
        )
        description_alpha = (
            f"if(lt(t,{description_fade_in_start:.2f}),0,"
            f"if(lt(t,{description_fade_in_end:.2f}),"
            f"(t-{description_fade_in_start:.2f})/{max(0.2, description_fade_in_end - description_fade_in_start):.2f},"
            f"if(lt(t,{description_fade_out_start:.2f}),1,"
            f"if(lt(t,{duration:.2f}),({duration:.2f}-t)/{max(0.2, duration - description_fade_out_start):.2f},0))))"
        )

        font_arg = self._resolve_intro_font_arg()
        filter_parts = [
            f"color=c=#090b12:s={self._width}x{self._height}:d={duration:.3f}",
            "format=rgba",
            (
                "drawbox=x=0:y=0:w=iw:h=ih:color=#0f1628@0.35:t=fill,"
                "drawbox=x=0:y=ih*0.12:w=iw:h=ih*0.76:color=#000000@0.42:t=fill"
            ),
            (
                "drawtext="
                f"{font_arg}:"
                f"text='{title_text}':"
                "fontcolor=white:"
                "fontsize=min(h*0.09\\,94):"
                "x=(w-text_w)/2:"
                "y=h*0.28:"
                "line_spacing=10:"
                "shadowcolor=#000000@0.85:"
                "shadowx=2:shadowy=2:"
                f"alpha='{title_alpha}'"
            ),
            (
                "drawtext="
                f"{font_arg}:"
                f"text='{description_escaped}':"
                "fontcolor=#f1f4ff:"
                "fontsize=min(h*0.042\\,40):"
                "x=(w-text_w)/2:"
                "y=h*0.56:"
                "line_spacing=8:"
                "shadowcolor=#000000@0.78:"
                "shadowx=1:shadowy=1:"
                f"alpha='{description_alpha}'"
            ),
            "format=yuv420p",
        ]

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                filter_parts[0],
                "-vf",
                ",".join(filter_parts[1:]),
                "-t",
                f"{duration:.3f}",
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

    def _resolve_intro_font_arg(self) -> str:
        candidates = [
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/System/Library/Fonts/Supplemental/Helvetica.ttc"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return f"fontfile={self._escape_drawtext(str(candidate))}"
        return "font=Sans"

    def _wrap_intro_description(self, text: str) -> str:
        lines = textwrap.wrap(text.strip(), width=52)
        if not lines:
            return ""
        return "\n".join(lines[:3])

    def _escape_drawtext(self, value: str) -> str:
        escaped = value.replace("\\", "\\\\")
        escaped = escaped.replace(":", "\\:")
        escaped = escaped.replace("'", "\\'")
        escaped = escaped.replace("%", "\\%")
        escaped = escaped.replace("\n", "\\n")
        return escaped

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
