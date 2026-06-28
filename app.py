import streamlit as st

st.set_page_config(page_title="Upload Test")

st.title("Upload Test")

st.write("App Loaded Successfully")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.success("File received!")
    st.write(uploaded_file.name)
    st.write(uploaded_file.size)