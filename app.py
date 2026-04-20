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

        # Fixed extraction destination in the /tmp directory
        extraction_dest = "/tmp/plagiarism_uploads"
        
        # Clean previous run to avoid mixing data
        if os.path.exists(extraction_dest):
            shutil.rmtree(extraction_dest)
        os.makedirs(extraction_dest, exist_ok=True)

        # Extract the ZIP
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(extraction_dest)

        # Handle nested folders (e.g., if the user zipped a folder instead of files)
        content = os.listdir(extraction_dest)
        if len(content) == 1 and os.path.isdir(os.path.join(extraction_dest, content[0])):
            destination_folder = os.path.join(extraction_dest, content[0])
        else:
            destination_folder = extraction_dest

        if st.button("🔍 Detect Plagiarism"):
            # Trigger the detection model
            scan_for_plagiarism(destination_folder, use_level4)

if __name__ == "__main__":
    main()