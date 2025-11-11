import os
from pathlib import Path
from google.cloud.storage import Bucket, transfer_manager, Client
from functools import cached_property

class GCPMediaDownloader:
    def __init__(self, download_dir: Path = Path('./temp')):
        self.download_dir = download_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def split_different_buckets(self, *file_paths: str) -> dict[str, list[str]]:
        buckets_dict: dict[str, list[str]] = {}

        for file_path in file_paths:
            bucket_name = file_path.split('/')[0]
            if bucket_name not in buckets_dict:
                buckets_dict[bucket_name] = []
            buckets_dict[bucket_name].append(file_path)

        return buckets_dict

    def download_images(self, *file_paths: str) -> list[Path]:
        bucket_split = self.split_different_buckets(*file_paths)

        for bucket_name, paths in bucket_split.items():
            bucket = Bucket(client=self._gcp_client, name=bucket_name)

            transfer_manager.download_many_to_path(
                bucket=bucket,
                blob_names=[f.removeprefix(bucket_name + "/") for f in file_paths],
                destination_directory=str(self.download_dir / bucket_name),
                max_workers=os.cpu_count() // 2,
                worker_type=transfer_manager.THREAD,
                skip_if_exists=True,
            )

        return [(self.download_dir / f).resolve() for f in file_paths]

    @cached_property
    def _gcp_client(self) -> Client:
        return Client()