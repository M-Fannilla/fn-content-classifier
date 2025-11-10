from functools import cached_property

import os
from google.cloud.storage.transfer_manager import THREAD

from pathlib import Path
from google.cloud.storage import Bucket, transfer_manager, Client

class GCPImageLoader:
    def __init__(self, download_dir: Path = Path('./temp')):
        self.download_dir = download_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def download_images(self, *file_paths: str) -> list[Path]:
        bucket_name = file_paths[0].split('/')[0]
        bucket = Bucket(client=self._gcp_client, name=bucket_name)

        transfer_manager.download_many_to_path(
            bucket=bucket,
            blob_names=[f.removeprefix(bucket_name + "/") for f in file_paths],
            destination_directory=str(self.download_dir / bucket_name),
            max_workers=os.cpu_count() // 2,
            worker_type=THREAD,
            skip_if_exists=True,
        )

        return [(self.download_dir / f).resolve() for f in file_paths]

    @cached_property
    def _gcp_client(self) -> Client:
        return Client()