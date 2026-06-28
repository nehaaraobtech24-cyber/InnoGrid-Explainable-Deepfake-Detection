import streamlit as st

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

if uploaded_file is not None:

    with st.spinner("Running Deepfake Detection... Please wait."):

        image = load_image(uploaded_file)
        input_tensor = preprocess_image(image)

        st.image(image, caption="Uploaded Image", width=350)

        # -----------------------------------------
        # Predictions
        # -----------------------------------------

        branchA_result = predict(branchA, input_tensor)
        branchB_result = predict(branchB, input_tensor)
        clip_result = clip_predict(image)

        fusion_result = fusion_prediction(
            branchA_result,
            branchB_result,
            clip_result
        )

        method_result = generation_method(
            branchA_result,
            branchB_result,
            fusion_result
        )

        # -----------------------------------------
        # GradCAM
        # -----------------------------------------

        heatmapA = generate_gradcam(
            branchA,
            image,
            input_tensor
        )

        heatmapB = generate_gradcam(
            branchB,
            image,
            input_tensor
        )

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

    # -----------------------------------------
    # CLIP
    # -----------------------------------------

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

    # -----------------------------------------
    # Fusion
    # -----------------------------------------

    st.divider()

    st.subheader("Fusion Engine")

    st.success(
        f"Final Prediction: {fusion_result['prediction']}"
    )

    st.write(
        f"Fusion Confidence: "
        f"**{fusion_result['confidence']*100:.2f}%**"
    )

    # -----------------------------------------
    # Generation Method
    # -----------------------------------------

    st.divider()

    st.subheader("Generation Method")

    st.write(
        f"Method: **{method_result['method']}**"
    )

    st.write(
        f"Reliability: **{method_result['reliability']}**"
    )

    st.info(method_result["reason"])