import os
import gdown

MODEL_DIR = "models"

MODELS = {
    "best_model.pth": "1EolEgPnn3io2KZHMbtqAa50gysx6P_HT",
    "best_branchB_final.pth": "1bqk-FG1e-SDNgvfJeFLVeexvsEUFRB1U"
}

def download_models():

    os.makedirs(MODEL_DIR, exist_ok=True)

    for filename, file_id in MODELS.items():

        destination = os.path.join(MODEL_DIR, filename)

        if os.path.exists(destination):
            print(f"{filename} already exists.")
            continue

        print(f"Downloading {filename}...")

        url = f"https://drive.google.com/uc?id={file_id}"
        print("Filename :", filename)
        print("Google Drive ID :", file_id)

        gdown.download(
            url,
            destination,
            quiet=False
        )

download_models()