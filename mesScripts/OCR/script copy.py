import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
import os
import pandas as pd
import logging
import re
from datetime import datetime

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
    "header_detect_height_ratio": 0.1,
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
        logging.debug(f"Page {i+1} sauvegardée -> {image_path}")
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
    try:
        df = df[~df['text'].str.contains(params['unwanted_chars_pattern'], regex=True)]
    except Exception as e:
        logging.warning(f"Erreur filtre unwanted_chars: {e}")
    return df.reset_index(drop=True)


def detect_rotated_headers(gray):
    """
    Détecte et OCR les textes tournés dans la zone header (haut de la page).
    """
    h, w = gray.shape
    h_header = int(h * params['header_detect_height_ratio'])
    header_roi = gray[0:h_header, :]
    # OSD detection pour orientation
    try:
        osd = pytesseract.image_to_osd(header_roi)
        angle = int(re.search(r'Rotate: (\d+)', osd).group(1))
    except Exception:
        angle = 0
    if angle != 0:
        M = cv2.getRotationMatrix2D((w/2, h_header/2), -angle, 1)
        header_roi = cv2.warpAffine(header_roi, M, (w, h_header), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    df_header = pytesseract.image_to_data(header_roi, output_type=pytesseract.Output.DATAFRAME, lang=params['language'])
    if df_header is None or df_header.empty:
        return []
    df_header = df_header[df_header.conf > params.get('min_conf',0)].copy()
    # filtrage des mots annexes
    df_header['text'] = df_header['text'].astype(str)
    df_header = df_header[~df_header['text'].str.contains(params['unwanted_chars_pattern'], regex=True)]
    # récupérer mots et position moyenne
    words = list(df_header['text'])
    return words


def detect_column_centers(df):
    xs = df['left'].values if 'left' in df else np.array([])
    if xs.size == 0:
        return []
    min_x, max_x = xs.min(), xs.max()
    bins = np.arange(min_x, max_x + params['bin_width'], params['bin_width'])
    hist, edges = np.histogram(xs, bins=bins)
    centers = []
    for i in range(1, len(hist)-1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] >= params['peak_min_count']:
            centers.append((edges[i] + edges[i+1]) / 2)
    centers = sorted(centers)
    logging.debug(f"Colonnes détectées en X: {centers}")
    return centers


def assign_columns(df, centers):
    if 'left' not in df or not centers:
        df['col'] = 0
        return df
    df['col'] = df['left'].apply(lambda x: int(np.argmin([abs(x - c) for c in centers])))
    return df


def group_by_row_and_col(df):
    if df.empty:
        return []
    df_sorted = df.sort_values(['top', 'col'])
    rows, buffer, current_y = [], [], None
    for _, r in df_sorted.iterrows():
        top = r['top']
        if current_y is None or abs(top - current_y) <= params['tol_y']:
            buffer.append(r)
            current_y = top if current_y is None else (current_y + top)/2
        else:
            rows.append(buffer)
            buffer, current_y = [r], r['top']
    if buffer: rows.append(buffer)
    n_cols = int(df['col'].max())+1 if 'col' in df else 1
    table = []
    for row in rows:
        cells = ["" for _ in range(n_cols)]
        for cell in row:
            c = int(cell['col']) if 'col' in cell else 0
            cells[c] += (" "+cell['text']).strip()
        table.append(cells)
    return table


def filter_rows(table):
    final = []
    for row in table:
        line = " ".join(row).strip()
        if not line: continue
        if any(re.search(pat, line, re.IGNORECASE) for pat in params.get('row_exclude_patterns', [])): continue
        final.append(row)
    return final


def analyze_pdf(pdf_path):
    global file_counter
    images = extract_pdf_images(pdf_path)
    for i, img_path in enumerate(images):
        logging.info(f"--- Page {i+1} ---")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        pre = preprocess(img)

        # 1. Détection et OCR en-têtes diagonaux
        headers = detect_rotated_headers(pre)

        # 2. OCR complet des données
        df = full_ocr(pre)
        centers = detect_column_centers(df)
        df = assign_columns(df, centers)
        table = group_by_row_and_col(df)
        table = filter_rows(table)

        # Préparation export avec entêtes
        rows_out = []
        if headers:
            # aligner nombre d'entêtes à n_cols
            n_cols = len(table[0]) if table else len(headers)
            header_cells = headers[:n_cols] + [""]*(n_cols-len(headers))
            rows_out.append(header_cells)
        rows_out.extend(table)

        df_out = pd.DataFrame(rows_out)
        ext = 'xlsx' if params['export_excel'] else 'csv'
        out_file = f"{params['output_basename']}_{file_counter}.{ext}"
        try:
            if params['export_excel']:
                df_out.to_excel(out_file, index=False, header=False)
            else:
                df_out.to_csv(out_file, index=False, header=False)
            logging.info(f"Fichier généré: {out_file}")
        except Exception as e:
            logging.error(f"Erreur export fichier: {e}")
        file_counter += 1

if __name__ == '__main__':
    pdf = 'scan/1.pdf'
    if os.path.isfile(pdf):
        analyze_pdf(pdf)
    else:
        logging.error(f"PDF introuvable: {pdf}")
