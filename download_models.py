import os
import gdown

MODEL_DIR = "models"

MODELS = {
    "best_model.pth": "1EolEgPnn3io2KZHMbtqAa50gysx6P_HT",
    "best_branchB_final.pth": "1bqk-FG1e-SDNgvfJeFLVeexvsEUFRB1U"
}


def download_models():

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("Downloading models...")
    print("Current directory:", os.getcwd())
    print("Models folder:", os.path.abspath(MODEL_DIR))
    print("=" * 60)

    for filename, file_id in MODELS.items():

        destination = os.path.join(MODEL_DIR, filename)

        if os.path.exists(destination):
            print(f"{filename} already exists.")
            print(f"Size: {os.path.getsize(destination)} bytes")
            continue

        print(f"\nDownloading {filename}")
        print(f"Google Drive ID: {file_id}")

        url = f"https://drive.google.com/uc?id={file_id}"

        output = gdown.download(
            url=url,
            output=destination,
            quiet=False,
            fuzzy=True
        )

        print("gdown returned:", output)

        if not os.path.exists(destination):
            raise RuntimeError(
                f"{filename} was NOT downloaded."
            )

        size = os.path.getsize(destination)

        print(f"{filename} downloaded successfully.")
        print(f"Size: {size} bytes")

        if size < 100000:
            raise RuntimeError(
                f"{filename} looks invalid (only {size} bytes). "
                "Google Drive probably returned an HTML page instead of the model."
            )

    print("=" * 60)
    print("Final contents of models folder:")
    print(os.listdir(MODEL_DIR))
    print("=" * 60)


