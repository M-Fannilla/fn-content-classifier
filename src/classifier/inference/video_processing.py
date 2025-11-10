from PIL import Image
from pathlib import Path
import numpy as np
import cv2
from .image_processing import ImageProcessor


class VideoProcessor(ImageProcessor):
    def __init__(self, target_size: int, batch_workers: int, video_interval: int = 24):
        super().__init__(target_size=target_size)
        self.batch_workers = batch_workers
        self.video_interval = video_interval

    def process(self, file_path: Path) -> np.ndarray:
        frames = self._extract_frames(file_path)
        print(f"[→] Processing {len(frames)} frames...")

        processed_frames = []
        for frame in frames:
            processed_frames.append(self._process_frame(frame))

        processed_frames = [frame for frame in processed_frames if frame is not None]

        return np.vstack(processed_frames)

    def _extract_frames(self, file_path: Path) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(file_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {str(file_path)}")

        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Process frame at specified intervals
            if frame_count % self.video_interval == 0:
                frames.append(frame)

            frame_count += 1

        cap.release()

        print(f"[✓] Extracted {len(frames)} frames from {total_frames} total frames (interval: {self.video_interval})")

        return frames

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return super()._process_image(image=Image.fromarray(frame_rgb))