from pathlib import Path

from functools import cached_property
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

class ImageProcessor:
    def __init__(self, target_size: int, batch_workers: int):
        self.target_size = target_size
        self.batch_workers = batch_workers

    def process(
            self,
            image_path: Path,
    ) -> np.ndarray:
        with Image.open(image_path) as image:
            # Ensure RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize image proportionally and add black padding
            original_width, original_height = image.size

            # Calculate scale factor to fit within target_size while maintaining aspect ratio
            scale = min(self.target_size / original_width, self.target_size / original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            # Resize image proportionally
            image = image.resize((new_width, new_height), Image.BILINEAR)

            # Create black canvas
            canvas = self._canvas

            # Calculate position to paste (center the image)
            paste_x = (self.target_size - new_width) // 2
            paste_y = (self.target_size - new_height) // 2

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
    def _canvas(self) -> Image.Image:
        return Image.new(
            'RGB',
            (self.target_size, self.target_size),
            (0, 0, 0)
        )

    def process_batch(
            self,
            images: list[Path],
    ) -> np.ndarray:
        processed_images = []
        for image in images:
            processed_images.append(self.process(image))
        processed_images = [img for img in processed_images if img is not None]
        print(f"[✓] Image batch preprocessing complete with {len(processed_images)}/{len(images)}.")
        return np.vstack(processed_images)