from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    path = tmp_path / "sample.avi"
    fps = 10.0
    size = (96, 64)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        size,
    )
    if not writer.isOpened():
        pytest.skip("OpenCV build does not provide the MJPG encoder")
    for frame_index in range(30):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        scene_color = (35, 60, 120) if frame_index < 15 else (150, 45, 30)
        frame[:] = scene_color
        cv2.circle(frame, (10 + frame_index * 2, 32), 8, (240, 240, 240), -1)
        cv2.putText(
            frame,
            str(frame_index),
            (3, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        writer.write(frame)
    writer.release()
    return path
