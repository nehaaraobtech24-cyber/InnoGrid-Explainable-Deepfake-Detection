import streamlit as st
import matplotlib.pyplot as plt

from utils import load_image, preprocess_image
from predict import branchA, branchB, predict
from clip_module import clip_predict
from gradcam import generate_gradcam
from fusion import fusion_prediction, generation_method

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Explainable Deepfake Detection",
    layout="wide"
)

st.title("🛡️ Explainable Deepfake Detection")
st.write(
    "Multi-model Deepfake Detection using "
    "EfficientNet-B4, CLIP and Grad-CAM"
)

# ---------------------------------------------------
# Upload Image
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    st.write("✅ Step 1 - File uploaded")

    image = load_image(uploaded_file)

    st.write("✅ Step 2 - Image loaded")

    input_tensor = preprocess_image(image)

    st.write("✅ Step 3 - Image preprocessed")

    st.image(image, caption="Uploaded Image", width=350)

    branchA_result = predict(branchA, input_tensor)

    st.write("✅ Step 4 - Branch A done")

    branchB_result = predict(branchB, input_tensor)

    st.write("✅ Step 5 - Branch B done")

    clip_result = clip_predict(image)

    st.write("✅ Step 6 - CLIP done")

    fusion_result = fusion_prediction(
        branchA_result,
        branchB_result,
        clip_result
    )

    st.write("✅ Step 7 - Fusion done")

    method_result = generation_method(
        branchA_result,
        branchB_result,
        fusion_result
    )

    st.write("✅ Step 8 - Generation method done")

    heatmapA = generate_gradcam(
        branchA,
        image,
        input_tensor
    )

    st.write("✅ Step 9 - GradCAM A done")

    heatmapB = generate_gradcam(
        branchB,
        image,
        input_tensor
    )

    st.write("✅ Step 10 - GradCAM B done")

    # -----------------------------------------
    # Branch Results
    # -----------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Branch A")

        st.write(
            f"Prediction: **{branchA_result['prediction']}**"
        )

        st.write(
            f"Confidence: **{branchA_result['confidence']*100:.2f}%**"
        )

        st.image(heatmapA)

    with col2:

        st.subheader("Branch B")

        st.write(
            f"Prediction: **{branchB_result['prediction']}**"
        )

        st.write(
            f"Confidence: **{branchB_result['confidence']*100:.2f}%**"
        )

        st.image(heatmapB)

    st.divider()

    st.subheader("CLIP Semantic Verification")

    st.write(
        f"Prediction: **{clip_result['prediction']}**"
    )

    st.write(
        f"Real Similarity: **{clip_result['real_score']*100:.2f}%**"
    )

    st.write(
        f"Fake Similarity: **{clip_result['fake_score']*100:.2f}%**"
    )

    st.divider()

    st.subheader("Fusion Engine")

    st.success(
        f"Final Prediction: {fusion_result['prediction']}"
    )

    st.write(
        f"Fusion Confidence: "
        f"**{fusion_result['confidence']*100:.2f}%**"
    )

    st.divider()

    st.subheader("Generation Method")

    st.write(
        f"Method: **{method_result['method']}**"
    )

    st.write(
        f"Reliability: **{method_result['reliability']}**"
    )

    st.info(method_result["reason"])