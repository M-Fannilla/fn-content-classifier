from pathlib import Path
from google.cloud.storage import Bucket, transfer_manager

class GCPImageLoader:
    def __init__(self, download_dir: Path = Path('./temp')):
        self.download_dir = download_dir

    def download_images(self, *file_paths: str, bucket: Bucket):
        transfer_manager.download_many_to_path(
            bucket, file_paths,
            destination_directory=str(self.download_dir),
        )
