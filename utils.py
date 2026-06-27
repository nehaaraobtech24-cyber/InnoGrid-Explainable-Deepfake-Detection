import torch
from PIL import Image
from torchvision import transforms


# -------------------------------------------------
# Device
# -------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# Labels
# -------------------------------------------------

LABELS = [
    "Real",
    "Fake"
]


# -------------------------------------------------
# Image Transform
# -------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------------------------------
# Load Image
# -------------------------------------------------

def load_image(image_file):
    """
    Loads uploaded image as RGB PIL Image.
    """
    image = Image.open(image_file).convert("RGB")
    return image


# -------------------------------------------------
# Preprocess Image
# -------------------------------------------------

def preprocess_image(image):
    """
    Converts PIL image into model input tensor.
    """
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)
    return tensor