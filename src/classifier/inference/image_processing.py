from pathlib import Path

import numpy as np
from PIL import Image
from functools import cached_property

class ImageProcessor:
    def __init__(self, target_size: int):
        self._target_size = target_size

    def process(self, file_path: Path) -> np.ndarray:
        with Image.open(file_path) as image:
            return self._process_image(image=image)

    def _process_image(self, image: Image.Image) -> np.ndarray:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize image proportionally and add black padding
        original_width, original_height = image.size

        # Calculate scale factor to fit within target_size while maintaining aspect ratio
        scale = min(self._target_size / original_width, self._target_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize image proportionally
        image = image.resize((new_width, new_height), Image.BILINEAR)

        # Create black canvas
        canvas = self._padding_canvas

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
    def _padding_canvas(self) -> Image.Image:
        return Image.new(
            'RGB',
            (self._target_size, self._target_size),
            (0, 0, 0)
        )