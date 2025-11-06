import os
from huggingface_hub import HfApi

if __name__ == "__main__":
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_large_folder(
        repo_id="fnmilosz/porncom-images",
        folder_path="/Users/milosz/Projects/fn-content-dataset/images",
        repo_type="dataset",
        print_report=True,
        print_report_every=10
    )
