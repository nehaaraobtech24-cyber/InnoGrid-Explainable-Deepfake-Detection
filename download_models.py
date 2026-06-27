import os
import gdown

MODEL_DIR = "models"

MODELS = {
    "best_model.pth": "1bqk-FG1e-SDNgvfJeFLVeexvsEUFRB1U",
    "best_branchB_final.pth": "1EolEgPnn3io2KZHMbtqAa50gysx6P_HT"
}


def download_models():

    os.makedirs(MODEL_DIR, exist_ok=True)

    for filename, file_id in MODELS.items():

        destination = os.path.join(MODEL_DIR, filename)

        if os.path.exists(destination):
            print(f"✓ {filename} already exists.")
            continue

        print(f"Downloading {filename}...")

        url = f"https://drive.google.com/uc?id={file_id}"

        try:
            gdown.download(
                url=url,
                output=destination,
                quiet=False,
                fuzzy=True
            )

            print(f"✓ {filename} downloaded successfully.")

        except Exception as e:
            raise RuntimeError(
                f"Failed to download {filename}\n{e}"
            )

download_models()