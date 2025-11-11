from functools import cached_property
from PIL import Image
from pathlib import Path
import numpy as np
import cv2


class ImageProcessor:
    def __init__(self, target_size: int):
        self._target_size = target_size

    def process(self, file_path: Path) -> np.ndarray:
        with Image.open(file_path) as image:
            return self._process_pil_image(image=image)

    def _process_pil_image(self, image: Image.Image) -> np.ndarray:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize image proportionally and add black padding
        original_width, original_height = image.size

        # Calculate scale factor to fit within target_size while maintaining aspect ratio
        scale = min(
            self._target_size / original_width,
            self._target_size / original_height
        )
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize image proportionally
        image = image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR,
        )

        # Create black canvas
        canvas = self._padding_canvas_image

        # Calculate position to paste (center the image)
        paste_x = (self._target_size - new_width) // 2
        paste_y = (self._target_size - new_height) // 2

        # Paste resized image onto black canvas
        canvas.paste(image, (paste_x, paste_y))

        # Convert to numpy array and normalize to [0, 1]
        image_array = np.array(canvas).astype(np.float32) / 255.0

        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std

        # Convert HWC to CHW format
        image_array = np.transpose(image_array, (2, 0, 1))

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    @cached_property
    def _padding_canvas_image(self) -> Image.Image:
        return Image.new(
            'RGB',
            (self._target_size, self._target_size),
            (0, 0, 0)
        )


class VideoProcessor(ImageProcessor):
    def __init__(self, target_size: int, batch_workers: int, video_interval: int = 24):
        super().__init__(target_size=target_size)
        # keep the name that describes purpose in video batching
        self.workers = batch_workers
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
        return super()._process_pil_image(
            image=Image.fromarray(frame_rgb)
        )