from collections import Counter
from itertools import chain


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