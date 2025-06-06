import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
import os
import pandas as pd
import logging
import re
import unicodedata
from datetime import datetime
from difflib import get_close_matches

# Configuration de Tesseract
tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

# Logs
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Paramètres
default_params = {
    "dpi": 300,
    "language": "fra",
    "resize_factor": 2.0,
    "thresholding": True,
    "blur": False,
    "min_conf": 0,
    "unwanted_chars_pattern": r"[@©]",
    "bin_width": 50,
    "peak_min_count": 20,
    "tol_y": 15,
    "row_exclude_patterns": [
        r"\bReader\b",
        r"\bZINIO\b",
        r"https?://",
        r"\d{2}/\d{2}/\d{4}",
        r"\d{2}:\d{2}"
    ],
    # Rotation fixe pour en-têtes diagonaux
    "header_angle": 45,
    # Hauteur de la zone header (10% de la page)
    "header_detect_height_ratio": 0.1,
    # Liste des entêtes attendus (test)
    "expected_headers": [
        "Activité", "Durée de séjour", "Obstétricien de garde", "Anesthésiste de garde", 
        "Pédiatre de garde", "Césariennes", "Péridurales", "Episiotomies", 
        "Extractions Instru.", "Naissances multiples", "Autoanalgésies", "Allaitement", 
        "Bloc opératoire", "Réanimation néonatale", "Psychologue", "Unité kangourou", 
        "Unité démédicalisée", "chambres à 1 lit", "chambres avec sdb", 
        "Suivi à domicile", "Note / 20"
    ],
    "export_excel": True,
    "output_basename": "extraction_tableaux"
}
params = default_params.copy()
file_counter = 1


def extract_pdf_images(pdf_path, output_dir="images"):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    logging.info(f"{len(doc)} page(s) détectée(s)")
    paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=params["dpi"])
        image_path = os.path.join(output_dir, f"page_{i+1}.png")
        pix.save(image_path)
        paths.append(image_path)
    return paths


def preprocess(img):
    if params["resize_factor"] != 1.0:
        img = cv2.resize(img, None, fx=params["resize_factor"], fy=params["resize_factor"])
    if params["blur"]:
        img = cv2.GaussianBlur(img, (5,5), 0)
    if params["thresholding"]:
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


def full_ocr(gray):
    df = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DATAFRAME, lang=params["language"])
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[df.conf > params.get("min_conf", 0)].copy()
    df['text'] = df['text'].astype(str).fillna("")
    df = df[~df['text'].str.contains(params['unwanted_chars_pattern'], regex=True)]
    return df.reset_index(drop=True)


def detect_rotated_headers(gray):
    """
    Extrait et redresse la zone header en rotation fixe, puis OCR.
    """
    h, w = gray.shape
    h_header = int(h * params['header_detect_height_ratio'])
    header_roi = gray[0:h_header, :]
    # rotation fixe
    angle = params.get('header_angle', 45)
    M = cv2.getRotationMatrix2D((w/2, h_header/2), -angle, 1)
    rotated = cv2.warpAffine(header_roi, M, (w, h_header), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    df = pytesseract.image_to_data(rotated, output_type=pytesseract.Output.DATAFRAME, lang=params['language'])
    if df is None or df.empty:
        return []
    df = df[df.conf > params.get('min_conf',0)].copy()
    df['text'] = df['text'].astype(str)
    df = df[~df['text'].str.contains(params['unwanted_chars_pattern'], regex=True)]
    # nettoyage et normalisation
    words = []
    for txt in df['text'].unique():
        txt = txt.strip()
        if txt:
            # normaliser accents
            txt_norm = ''.join(c for c in unicodedata.normalize('NFD', txt) if unicodedata.category(c) != 'Mn')
            words.append(txt_norm)
    return words


def normalize(s):
    s = s.strip().lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s


def find_header_matches(extracted, expected):
    """
    Compare listes normalisées et retourne mapping et manquants.
    """
    norm_extr = [normalize(w) for w in extracted]
    norm_exp = [normalize(w) for w in expected]
    matches = {}
    for exp, norm_e in zip(expected, norm_exp):
        # chercher meilleur match
        match = get_close_matches(norm_e, norm_extr, n=1, cutoff=0.5)
        matches[exp] = match[0] if match else None
    missing = [exp for exp, m in matches.items() if m is None]
    return matches, missing


def analyze_pdf(pdf_path):
    global file_counter
    images = extract_pdf_images(pdf_path)
    for i, img_path in enumerate(images):
        logging.info(f"--- Page {i+1} ---")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        pre = preprocess(img)

        # 1. Détection headers
        headers = detect_rotated_headers(pre)
        # 2. Tester correspondance
        mapping, missing = find_header_matches(headers, params['expected_headers'])
        logging.info(f"Headers extraits: {headers}")
        logging.info(f"Mapping attendu->extrait: {mapping}")
        if missing:
            logging.warning(f"Headers manquants: {missing}")

        # Ici on continue avec données si besoin...

if __name__ == '__main__':
    pdf = 'scan/1.pdf'
    if os.path.isfile(pdf):
        analyze_pdf(pdf)
    else:
        logging.error(f"PDF introuvable: {pdf}")
