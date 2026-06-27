import torch
from transformers import CLIPProcessor, CLIPModel

from utils import device


# -------------------------------------------------
# Load CLIP
# -------------------------------------------------

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)


# -------------------------------------------------
# CLIP Semantic Verification
# -------------------------------------------------

def clip_predict(image):

    prompts = [
        "A photograph of a real human face.",
        "A photograph of an AI-generated deepfake face."
    ]

    inputs = clip_processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = clip_model(**inputs)

    probabilities = outputs.logits_per_image.softmax(dim=1)

    real_score = probabilities[0][0].item()
    fake_score = probabilities[0][1].item()

    if fake_score > real_score:
        prediction = "Fake"
        confidence = fake_score
    else:
        prediction = "Real"
        confidence = real_score

    return {
        "prediction": prediction,
        "confidence": confidence,
        "real_score": real_score,
        "fake_score": fake_score
    }