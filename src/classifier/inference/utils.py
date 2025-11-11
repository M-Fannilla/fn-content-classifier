from collections import Counter
from pathlib import Path
from itertools import chain
from pydantic import BaseModel


def label_frequencies(frames: list[list[str]], min_count: int = 1, min_percent: float = 0.0):
    flattened = list(chain.from_iterable(frames))
    total = len(flattened)
    counts = Counter(flattened)

    freq = {
        label: count / total
        for label, count in counts.items()
        if count >= min_count and (count / total) >= min_percent
    }

    return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))

def flatten_labels(results: list[list[list[str]]]) -> list[list[str]]:
    flattened_results = []
    num_images = len(results[0])

    for i in range(num_images):
        image_labels = set()
        for model_result in results:
            image_labels.update(model_result[i])
        flattened_results.append(list(image_labels))

    return flattened_results


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv"}

def split_files_by_type(
        files: list[Path]
) -> tuple[list[Path], ...]:
    images = []
    videos = []
    unsupported = []

    for f in files:
        p = Path(f)
        ext = p.suffix.lower()

        if ext in IMAGE_EXTS:
            images.append(p)
        elif ext in VIDEO_EXTS:
            videos.append(p)
        else:
            unsupported.append(p)

    return images, videos, unsupported

class MediaClassificationModel(BaseModel):
    media_id: str
    file_path: str

class ContentClassificationEvent(BaseModel):
    results: list[MediaClassificationModel]