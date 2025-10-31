import os
from google.cloud.storage.transfer_manager import THREAD

from pathlib import Path
from google.cloud.storage import Bucket, transfer_manager

class GCPImageLoader:
    def __init__(self, download_dir: Path = Path('./temp')):
        self.download_dir = download_dir

    def download_images(self, *file_paths: str, bucket: Bucket):
        self.download_dir.mkdir(parents=True, exist_ok=True)
        transfer_manager.download_many_to_path(
            bucket=bucket,
            blob_names=file_paths,
            destination_directory=str(self.download_dir),
            max_workers=os.cpu_count() // 2,
            worker_type=THREAD,
            skip_if_exists=True,
        )
