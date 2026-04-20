"""
OCR module using pytesseract (free, open-source).
Replaces the original paid apilayer OCR API.

Requires system package: tesseract-ocr  (listed in packages.txt)
"""

import pytesseract
from PIL import Image
import numpy as np


def extract_text(image_path: str) -> str:
    """
    Extract text from a handwritten image using Tesseract OCR.
    Returns an empty string if extraction fails.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        # PSM 6 = assume a single uniform block of text (good for assignments)
        text = pytesseract.image_to_string(img, config="--psm 6")
        return text.strip()
    except Exception as e:
        print(f"[OCR] Error processing {image_path}: {e}")
        return ""


def compare_text_content(text1: str, text2: str) -> float:
    """
    Simple word-overlap (Jaccard) similarity as a lightweight fallback.
    The main model.py uses sentence-transformers for semantic similarity.
    """
    if not text1 or not text2:
        return 0.0
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0.0
