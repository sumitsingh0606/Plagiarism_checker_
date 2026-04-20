"""
Core plagiarism detection model.
- Level 1 & 2 : HOG-based visual (UDP) similarity
- Level 3     : Semantic text similarity via sentence-transformers
- Level 4     : Paraphrasing detection (also sentence-transformers, lower threshold)
"""

import os
import streamlit as st
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, util

from udp import generate_digital_pattern, compare_patterns
from ocr import extract_text

# Load model once at import time (cached by Streamlit)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def scan_for_plagiarism(submission_folder: str, use_level4: bool):
    # SAFETY CHECK: Ensure the folder exists before listing files
    if not os.path.exists(submission_folder):
        st.error(f"❌ Error: The directory {submission_folder} was not found.")
        return

    model = load_model()

    # Filter to image files only to prevent processing system files like .DS_Store
    all_files = os.listdir(submission_folder)
    submissions = [
        f for f in all_files
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"))
    ]

    if len(submissions) < 2:
        st.warning("Please upload a ZIP with at least 2 image files.")
        return

    st.info(f"Found {len(submissions)} submission(s). Running detection…")

    # ── Pre-extract all text and visual patterns ──────────────────────────────
    with st.spinner("Processing submissions (OCR & Visual Analysis)…"):
        text_list = []
        emb_list = []
        udp_list = []
        
        for fname in submissions:
            path = os.path.join(submission_folder, fname)
            
            # Extract text via OCR
            text = extract_text(path)
            text_list.append(text)
            
            # Generate Semantic Embeddings
            if text.strip():
                emb = model.encode(text, convert_to_tensor=True)
            else:
                emb = None
            emb_list.append(emb)
            
            # Generate Visual Digital Pattern (UDP)
            try:
                udp = generate_digital_pattern(path)
                udp_list.append(udp)
            except Exception:
                udp_list.append(None)

    # ── Comparison Logic ──────────────────────────────────────────────────────
    cp_list = [] # Complete Plagiarism
    pp_list = [] # Potential Plagiarism
    level4_list = []

    with st.spinner("Comparing pairs..."):
        for i in range(len(submissions)):
            for j in range(i + 1, len(submissions)):
                p1_name = os.path.splitext(submissions[i])[0]
                p2_name = os.path.splitext(submissions[j])[0]
                
                # Visual Check (UDP)
                visual_sim = 0.0
                if udp_list[i] is not None and udp_list[j] is not None:
                    visual_sim = compare_patterns(udp_list[i], udp_list[j])
                
                # Text Check
                text_sim = 0.0
                if emb_list[i] is not None and emb_list[j] is not None:
                    text_sim = float(util.cos_sim(emb_list[i], emb_list[j]).item())
                
                # Level 1: Visual Match (>95%)
                if visual_sim >= 0.95:
                    st.error(f"🔴 Level 1 – Complete Plagiarism: **{p1_name}** ↔ **{p2_name}** (Visual={visual_sim*100:.1f}%)")
                    cp_list.append((p1_name, p2_name))
                
                # Level 2: Mixed Match
                elif visual_sim >= 0.55 and text_sim >= 0.85:
                    st.error(f"🔴 Level 2 – Complete Plagiarism: **{p1_name}** ↔ **{p2_name}** (Mixed similarity)")
                    cp_list.append((p1_name, p2_name))
                
                # Level 3: Text Match (>70%)
                elif text_sim >= 0.70:
                    st.warning(f"🟡 Level 3 – Potential Plagiarism: **{p1_name}** ↔ **{p2_name}** (Text={text_sim*100:.1f}%)")
                    pp_list.append((p1_name, p2_name))
                
                # Level 4: Paraphrasing (Optional)
                elif use_level4 and text_sim >= 0.55:
                    st.info(f"🟠 Level 4 – Paraphrasing Suspected: **{p1_name}** ↔ **{p2_name}** (Text={text_sim*100:.1f}%)")
                    level4_list.append((p1_name, p2_name))

    # ── Summary ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 Complete", len(cp_list))
    col2.metric("🟡 Potential", len(pp_list))
    if use_level4:
        col3.metric("🟠 Paraphrasing", len(level4_list))

    if cp_list:
        with st.expander("🔴 View Complete Plagiarism Pairs"):
            for a, b in cp_list:
                st.write(f"• {a} ↔ {b}")