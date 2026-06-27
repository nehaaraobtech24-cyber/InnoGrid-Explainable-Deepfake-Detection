import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_gradcam(model, image, input_tensor):
    """
    Generate Grad-CAM visualization for EfficientNet-B4.
    Returns the heatmap image as a NumPy array.
    """

    # Last convolution layer
    target_layers = [model.conv_head]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    rgb_img = np.array(
        image.resize((224, 224))
    ).astype(np.float32) / 255.0

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    return visualization