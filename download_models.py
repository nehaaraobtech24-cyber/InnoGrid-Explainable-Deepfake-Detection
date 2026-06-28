import os
from huggingface_hub import hf_hub_download

MODEL_DIR = "models"

REPO_ID = "CarrotSalad/innogrid-models"

MODELS = [
    "best_model.pth",
    "best_branchB_final.pth"
]


def download_models():

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("Downloading models from Hugging Face...")
    print("=" * 60)

    for filename in MODELS:

        destination = os.path.join(MODEL_DIR, filename)

        if os.path.exists(destination):
            print(f"{filename} already exists.")
            continue

        downloaded_file = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )

        print("Downloaded:", downloaded_file)

        if not os.path.exists(destination):
            raise RuntimeError(f"{filename} failed to download.")

        print(filename, "downloaded successfully.")

    print("=" * 60)
    print("Done downloading models.")
    print("=" * 60)