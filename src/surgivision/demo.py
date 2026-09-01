"""Synthetic video generation for software smoke testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def create_demo_video(
    output_path: str | Path,
    duration_s: float = 6.0,
    fps: float = 12.0,
    width: int = 640,
    height: int = 360,
) -> Path:
    """Create a clearly synthetic moving-shape video with several visual scenes."""

    if duration_s <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("Duration, frame rate, width, and height must be positive")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(destination), codec, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("The local OpenCV build cannot create an MP4 video")

    frame_total = max(1, round(duration_s * fps))
    try:
        for frame_index in range(frame_total):
            progress = frame_index / max(1, frame_total - 1)
            scene = min(2, int(progress * 3))
            backgrounds = [(26, 32, 42), (42, 26, 35), (24, 43, 36)]
            frame = np.full((height, width, 3), backgrounds[scene], dtype=np.uint8)

            grid_color = tuple(channel + 24 for channel in backgrounds[scene])
            for x_coordinate in range(0, width, 40):
                cv2.line(frame, (x_coordinate, 0), (x_coordinate, height), grid_color, 1)
            for y_coordinate in range(0, height, 40):
                cv2.line(frame, (0, y_coordinate), (width, y_coordinate), grid_color, 1)

            center_x = int(70 + progress * (width - 140))
            center_y = int(height / 2 + np.sin(progress * np.pi * 4) * height * 0.18)
            cv2.circle(frame, (center_x, center_y), 36, (70, 210, 235), -1)
            cv2.rectangle(
                frame,
                (width - center_x - 45, height - center_y - 25),
                (width - center_x + 45, height - center_y + 25),
                (190, 110, 245),
                -1,
            )
            cv2.putText(
                frame,
                "SYNTHETIC SOFTWARE TEST VIDEO",
                (24, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (235, 239, 245),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"scene {scene + 1}  frame {frame_index:03d}",
                (24, height - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (215, 221, 230),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()

    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError("Demo video was not written successfully")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/demo.mp4", help="Output MP4 path")
    parser.add_argument("--duration", type=float, default=6.0, help="Duration in seconds")
    parser.add_argument("--fps", type=float, default=12.0, help="Frames per second")
    args = parser.parse_args()
    output = create_demo_video(args.output, duration_s=args.duration, fps=args.fps)
    print(f"Created synthetic demo video: {output}")


if __name__ == "__main__":
    main()
