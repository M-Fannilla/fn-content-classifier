import json
from google.cloud import run_v2

if __name__ == "__main__":
    client = run_v2.JobsClient()

    name = client.job_path("fannilla-dev", "europe-west1", "fn-content-classifier")

    urls = json.dumps([
        "https://cdni.pornpics.com/1280/7/154/71529151/71529151_054_d402.jpg",
        "https://www.sigmapic.com/images/xxxpornimages.net/365/985_Mary_hardcore.jpg",
        "https://cdni.pornpics.com/460/1/349/36736495/36736495_004_ebc2.jpg",
        "https://cdni.pornpics.com/460/7/108/73346357/73346357_030_3d3f.jpg",
        "https://www.porndos.com/images/thumb/7255.webp",
        "https://s86.erome.com/1637/cZQdg5uU/FkzU6ldv.jpeg",
    ])

    request = run_v2.RunJobRequest(
        name=name,
        overrides={
            "container_overrides": [
                {
                    "args": [urls],
                }
            ]
        }
    )

    operation = client.run_job(request=request)
    print("Job started — waiting for it to finish…")
    result = operation.result()
    print("✅ Job completed successfully:", result)