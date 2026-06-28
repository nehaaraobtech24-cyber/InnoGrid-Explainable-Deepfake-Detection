import streamlit as st
from PIL import Image

st.title("Upload Test")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.success("Image received!")
    st.image(image)