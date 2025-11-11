import sys
import json
import logging
from pathlib import Path
from .utils import split_files_by_type
from .download import GCPMediaDownloader
from .configs import InferenceConfig
from .processing import ImageProcessor, VideoProcessor
from .inference import ImageClassifier, VideoClassifier
from .model_loader import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fn-content-classifier.job_main")

def process_cli_args():
    logger.debug(f"CLI args: {sys.argv}")
    if len(sys.argv) < 2:
        raise ValueError("No URLs argument provided to job.")
    return json.loads(sys.argv[1])

def classify_images(
        classifier: ImageClassifier,
        images_to_predict: list[Path],
) -> dict[str, list[str]]:
    images_results = classifier.classify(file_paths=images_to_predict)
    return images_results

def classify_videos(
        classifier: VideoClassifier,
        videos_to_predict: list[Path]
) -> dict[str, list[str]]:
    all_results = {}
    for video_path in videos_to_predict:
        video_results = classifier.classify(file_path=video_path)
        all_results.update(video_results)

    return all_results

def clean_temp_folder_from_files(
        temp_dir: Path,
        result: dict[str, list[str]]
) -> dict[str, list[str]]:
    out = {}
    for file_name, labels in result.items():
        file_name_clean = file_name.removeprefix(str(temp_dir.resolve()) + '/')
        out[file_name_clean] = labels

    return out

def main():
    config = InferenceConfig()
    downloader = GCPMediaDownloader()
    model_manager = ModelManager()
    model_manager.load_all()

    # urls = process_cli_args()
    urls = [
        'fn-ai-datasets/content-classification/v0/images/1000.jpg',
        'fn-raw-user-content-eu-dev/u/test_user/v/c/of_short.mp4',
    ]

    logger.info(f"Received {len(urls)} URLs for inference")

    temp_files = downloader.download_images(*urls)

    image_classifier = ImageClassifier(
        inference_config=config,
        processor=ImageProcessor(target_size=model_manager.get_image_size()),
        model_manager=model_manager,
        apply_threshold=True,
        threshold_offset=0.2,
    )
    video_classifier = VideoClassifier(
        inference_config=config,
        processor=VideoProcessor(
            target_size=model_manager.get_image_size(),
            batch_workers=config.IMAGE_PROCESSING_WORKERS
        ),
        model_manager=model_manager,
        apply_threshold=True,
        threshold_offset=0.2,
    )

    logger.info(f"Downloaded {len(temp_files)} files for classification")
    images, videos, unsupported = split_files_by_type(files=temp_files)

    image_classification_results = classify_images(
        classifier=image_classifier,
        images_to_predict=images,
    )

    video_classification_results = classify_videos(
        classifier=video_classifier,
        videos_to_predict=videos,
    )

    image_classification_results.update(video_classification_results)

    clean_results = clean_temp_folder_from_files(
        temp_dir=downloader.download_dir,
        result=image_classification_results,
    )

    logger.info(f"Final combined results:\n{json.dumps(clean_results, indent=2)}")

if __name__ == "__main__":
    main()