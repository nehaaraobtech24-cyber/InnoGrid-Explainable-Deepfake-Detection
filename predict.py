from download_models import download_models
import os

# -----------------------------------------
# Download models first
# -----------------------------------------

download_models()

print("Files in models folder:")
print(os.listdir("models"))

import torch
import timm
import torch.nn.functional as F

from utils import device, LABELS


# -------------------------------------------------
# Model Paths
# -------------------------------------------------

BRANCH_A_PATH = "models/best_model.pth"
BRANCH_B_PATH = "models/best_branchB_final.pth"

print("=" * 60)
print("BRANCH A PATH:", BRANCH_A_PATH)
print("BRANCH B PATH:", BRANCH_B_PATH)
print("=" * 60)


# -------------------------------------------------
# Load EfficientNet
# -------------------------------------------------

def load_model(model_path):

    model = timm.create_model(
        "efficientnet_b4",
        pretrained=False,
        num_classes=2
    )

    print("Loading:", model_path)
    print("Exists:", os.path.exists(model_path))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found!")

    print("Size:", os.path.getsize(model_path))

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    # Branch A checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


# -------------------------------------------------
# Load Models
# -------------------------------------------------

branchA = load_model(BRANCH_A_PATH)
branchB = load_model(BRANCH_B_PATH)


# -------------------------------------------------
# Prediction Function
# -------------------------------------------------

def predict(model, input_tensor):

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = F.softmax(output, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    return {
        "prediction": LABELS[prediction.item()],
        "confidence": confidence.item(),
        "class_index": prediction.item()
    }