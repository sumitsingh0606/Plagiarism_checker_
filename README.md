# Academic Plagiarism Checker — Handwritten Submissions 📝

Detects plagiarism in handwritten assignment images using a multi-level approach:

| Level | Method | Trigger |
|-------|--------|---------|
| 1 | HOG visual pattern (UDP) | UDP similarity > 95% |
| 2 | UDP + Semantic text | UDP > 55%, text > 85% |
| 3 | Semantic text only | Text similarity > 70% |
| 4 *(optional)* | Paraphrasing via sentence-transformers | Text similarity > 55% |

**100% free — no paid APIs required.**

---

## 🚀 Deploy for Free on Streamlit Community Cloud

1. **Fork or push this repo** to your GitHub account.

2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.

3. Click **"New app"** → select your repo → set:
   - **Branch:** `main`
   - **Main file path:** `app.py`

4. Click **Deploy** — that's it! Streamlit Cloud reads `packages.txt` automatically to install Tesseract.

---

## 💻 Run Locally

```bash
# 1. Install Tesseract (system dependency)
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr
# macOS:
brew install tesseract

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

---

## 📁 How to Use

1. Put all student submission images (JPG/PNG) into a single folder.
2. ZIP that folder.
3. Upload the ZIP in the app and click **Detect Plagiarism**.

---

## Changes from Original

| Original | This version |
|----------|-------------|
| Paid apilayer OCR API | `pytesseract` (free, open-source) |
| OpenAI API for Level 4 | `sentence-transformers` (free, local) |
| Writes to script directory | Writes to `/tmp` (safe for cloud) |
| `streamlit.py` | `app.py` (avoids name collision) |
