"""
Core plagiarism detection model.
- Level 1 & 2 : HOG-based visual (UDP) similarity
- Level 3     : Semantic text similarity via sentence-transformers
- Level 4     : Paraphrasing detection (also sentence-transformers, lower threshold)

No paid APIs required — 100% free.
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
    model = load_model()

    # Filter to image files only
    all_files = os.listdir(submission_folder)
    submissions = [
        f for f in all_files
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"))
    ]

    if len(submissions) < 2:
        st.warning("Please upload a ZIP with at least 2 image files.")
        return

    st.info(f"Found {len(submissions)} submission(s). Running detection…")

    # ── Pre-extract all text ──────────────────────────────────────────────────
    with st.spinner("Extracting text from all submissions (OCR)…"):
        text_list = []
        emb_list = []
        for fname in submissions:
            path = os.path.join(submission_folder, fname)
            text = extract_text(path)
            text_list.append(text)
            emb = model.encode(text) if text else None
            emb_list.append(emb)

    # ── Pairwise comparison ───────────────────────────────────────────────────
    cp_list, pp_list, level4_list = [], [], []
    excluded = set()

    progress = st.progress(0)
    total_pairs = len(submissions) * (len(submissions) - 1) // 2
    done = 0

    with st.expander("📋 Live Processing Log", expanded=True):
        for i in range(len(submissions)):
            if submissions[i] in excluded:
                continue

            for j in range(i + 1, len(submissions)):
                done += 1
                progress.progress(done / total_pairs)

                name1 = os.path.splitext(submissions[i])[0]
                name2 = os.path.splitext(submissions[j])[0]
                path1 = os.path.join(submission_folder, submissions[i])
                path2 = os.path.join(submission_folder, submissions[j])

                # ── Level 1 / 2 : UDP visual similarity ──────────────────────
                pat1 = generate_digital_pattern(path1)
                pat2 = generate_digital_pattern(path2)
                min_len = min(len(pat1), len(pat2))
                udp_sim = compare_patterns(pat1[:min_len], pat2[:min_len])

                if udp_sim > 0.95:
                    st.error(f"🔴 Level 1 – Complete Plagiarism: **{name1}** ↔ **{name2}**  (UDP={udp_sim*100:.1f}%)")
                    cp_list.append((name1, name2))
                    excluded.add(submissions[j])
                    continue

                # ── Level 2 / 3 : Semantic text similarity ───────────────────
                e1, e2 = emb_list[i], emb_list[j]

                if e1 is not None and e2 is not None:
                    cos_sim = float(util.cos_sim(e1, e2).item())
                else:
                    cos_sim = 0.0

                if udp_sim > 0.55:
                    if cos_sim >= 0.85:
                        st.error(
                            f"🔴 Level 2 – Complete Plagiarism: **{name1}** ↔ **{name2}**  "
                            f"(UDP={udp_sim*100:.1f}%, Text={cos_sim*100:.1f}%)"
                        )
                        cp_list.append((name1, name2))
                        excluded.add(submissions[j])
                    else:
                        st.warning(
                            f"🟡 Level 2 – Potential Plagiarism: **{name1}** ↔ **{name2}**  "
                            f"(UDP={udp_sim*100:.1f}%, Text={cos_sim*100:.1f}%)"
                        )
                        pp_list.append((name1, name2))
                else:
                    if cos_sim >= 0.85:
                        st.error(f"🔴 Level 3 – Complete Plagiarism: **{name1}** ↔ **{name2}**  (Text={cos_sim*100:.1f}%)")
                        cp_list.append((name1, name2))
                    elif cos_sim >= 0.70:
                        st.warning(f"🟡 Level 3 – Potential Plagiarism: **{name1}** ↔ **{name2}**  (Text={cos_sim*100:.1f}%)")
                        pp_list.append((name1, name2))
                    else:
                        st.success(f"✅ No significant similarity: **{name1}** ↔ **{name2}**  (Text={cos_sim*100:.1f}%)")

    # ── Level 4 : Paraphrasing (lower threshold, slower) ─────────────────────
    if use_level4:
        st.subheader("Level 4 – Paraphrasing Detection")
        with st.spinner("Running paraphrasing check…"):
            for i in range(len(submissions)):
                if submissions[i] in excluded:
                    continue
                for j in range(i + 1, len(submissions)):
                    pair = (os.path.splitext(submissions[i])[0], os.path.splitext(submissions[j])[0])
                    if pair in cp_list or pair in pp_list:
                        continue  # Already flagged
                    e1, e2 = emb_list[i], emb_list[j]
                    if e1 is not None and e2 is not None:
                        cos_sim = float(util.cos_sim(e1, e2).item())
                        if cos_sim >= 0.55:
                            st.warning(f"🟠 Level 4 – Possible Paraphrasing: **{pair[0]}** ↔ **{pair[1]}**  (Text={cos_sim*100:.1f}%)")
                            level4_list.append(pair)

    # ── Summary ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 Complete Plagiarism", len(cp_list))
    col2.metric("🟡 Potential Plagiarism", len(pp_list))
    if use_level4:
        col3.metric("🟠 Paraphrasing Suspected", len(level4_list))

    if cp_list:
        with st.expander("🔴 Complete Plagiarism Pairs"):
            for a, b in cp_list:
                st.write(f"• {a}  ↔  {b}")

    if pp_list:
        with st.expander("🟡 Potential Plagiarism Pairs"):
            for a, b in pp_list:
                st.write(f"• {a}  ↔  {b}")

    if use_level4 and level4_list:
        with st.expander("🟠 Paraphrasing Suspected Pairs"):
            for a, b in level4_list:
                st.write(f"• {a}  ↔  {b}")
