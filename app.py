import streamlit as st
import os
import zipfile
import shutil
from model import scan_for_plagiarism

st.set_page_config(
    page_title="Academic Plagiarism Detector 🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

def main():
    st.title("Academic Plagiarism Detector 🤖")
    st.markdown("Upload a ZIP file containing handwritten submission images (JPG/PNG).")

    uploaded_file = st.file_uploader("Choose a ZIP file", type="zip")
    use_level4 = st.checkbox("Enable Level 4 – Paraphrasing Detection (slower)")

    if uploaded_file is not None:
        if not zipfile.is_zipfile(uploaded_file):
            st.error("Uploaded file is not a valid ZIP file.")
            return

        folder_name, _ = os.path.splitext(uploaded_file.name)
        extraction_dest = "/tmp/plagiarism_uploads"
        destination_folder = os.path.join(extraction_dest, folder_name)

        # Clean previous run
        if os.path.exists(destination_folder):
            shutil.rmtree(destination_folder)

        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            os.makedirs(extraction_dest, exist_ok=True)
            zip_ref.extractall(extraction_dest)

        if st.button("🔍 Detect Plagiarism"):
            scan_for_plagiarism(destination_folder, use_level4)

if __name__ == "__main__":
    main()
