import pymupdf as fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
import os
import pandas as pd
import logging
import re
import unicodedata
from datetime import datetime

# Configuration de Tesseract
tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

# Logs\ n
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
    "min_conf": 30,
    "unwanted_chars_pattern": r"[@©]",
    "bin_width": 50,
    "peak_min_count": 20,
    "tol_y": 8,
    "row_exclude_patterns": [
        r"\bReader\b",
        r"\bZINIO\b",
        r"https?://",
        r"\d{2}/\d{2}/\d{4}",
        r"\d{2}:\d{2}"
    ],
    # Crop ratios
    "crop_top_ratio": 0.35,    # décale le haut de 5 % de la hauteur
    "crop_bottom_ratio": 0.95,
    "crop_left_ratio": 0.05,
    "crop_right_ratio": 0.95,
    # Slices\ n    "n_slices": 1,
    "save_debug": True,
    "export_excel": True,
    "output_basename": "extraction_tableaux"
}
params = default_params.copy()

# Regex utilitaires
rgx_rank_leading = re.compile(r"^(\d+)(?:er|e|re|r|°|ᵉ)?", re.I)
rgx_numeric      = re.compile(r"^\d+([\.,]\d+)?$")
rgx_collé        = re.compile(r"^(.*\)\s*)(\d{2,})$")

# Fonctions de traitement

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
    logging.debug(f"Images PDF extraites : {paths}")
    return paths


def preprocess(img):
    logging.debug(f"Prétraitement image: resize={params['resize_factor']}, blur={params['blur']}, threshold={params['thresholding']}")
    if params["resize_factor"] != 1.0:
        img = cv2.resize(img, None, fx=params["resize_factor"], fy=params["resize_factor"])
    if params["blur"]:
        img = cv2.GaussianBlur(img, (5,5), 0)
    if params["thresholding"]:
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    logging.debug(f"Image prétraitée shape={img.shape}")
    return img


def full_ocr(gray):
    df = pytesseract.image_to_data(
        gray,
        output_type=pytesseract.Output.DATAFRAME,
        lang=params["language"],
        config="--oem 1 --psm 6"
    )
    logging.debug(f"full_ocr: OCR ret {len(df)} composants" if df is not None else "full_ocr: df None")
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[df.conf > params.get("min_conf", 0)].copy()
    df['text'] = df['text'].astype(str).fillna("")
    df = df[~df['text'].str.contains(params['unwanted_chars_pattern'], regex=True)]
    logging.debug(f"full_ocr: after filter conf & unwanted, {len(df)} composants")
    return df.reset_index(drop=True)


def crop_table_region(gray):
    h, w = gray.shape
    y1 = int(h * params["crop_top_ratio"])
    y2 = int(h * params["crop_bottom_ratio"])
    x1 = int(w * params["crop_left_ratio"])
    x2 = int(w * params["crop_right_ratio"])
    logging.debug(f"Crop ROI: x1={x1}, x2={x2}, y1={y1}, y2={y2}")
    return gray[y1:y2, x1:x2]


def split_roi_horizontally(roi, slice_height=None, n_slices=None):
    h, w = roi.shape
    if slice_height:
        step = slice_height
        count = (h + step - 1) // step
    elif n_slices:
        count = n_slices
        step = h // n_slices
    else:
        return [(0, h, roi)]
    logging.debug(f"Découpage horizontal en {count} slices (hauteur={step})")
    slices = []
    for i in range(count):
        y1 = i * step
        y2 = min(h, (i + 1) * step)
        sub = roi[y1:y2, :]
        slices.append((y1, y2, sub))
        logging.debug(f"Slice {i}: y1={y1}, y2={y2}, hauteur={y2-y1}")
    return slices


def detect_column_centers(df):
    """
    Calcule les centres X des colonnes à partir des coordonnées OCR.

    - bin_width    : largeur d’histogramme (px)
    - peak_min_count : seuil brut (ancien)
    - min_words_for_col : seuil adaptatif = max(peak_min_count, 1 % du total de mots)
    """
    # 1. récupère les abscisses
    xs = df['left'].values if 'left' in df else np.array([])
    if xs.size == 0:
        return []

    # 2. histogramme
    min_x, max_x = xs.min(), xs.max()
    bins = np.arange(min_x, max_x + params['bin_width'], params['bin_width'])
    hist, edges = np.histogram(xs, bins=bins)

    # 3. seuil adaptatif : 1 % du nombre total de mots ou peak_min_count
    total_words = len(df)
    min_words_for_col = max(params['peak_min_count'],
                            int(total_words * 0.01))   # 1 % du texte

    # 4. détection des pics (creux / pic / creux) ET ≥ seuil
    centers = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] \
           and hist[i] >= min_words_for_col:
            centers.append((edges[i] + edges[i + 1]) / 2)

    centers = sorted(centers)
    logging.debug(f"[COL] Centres détectés = {centers} (seuil {min_words_for_col} mots)")
    return centers

    

    return sorted(centers)


def assign_columns(df, centers):
    if 'left' not in df or not centers:
        df['col'] = 0
        return df
    df['col'] = df['left'].apply(lambda x: int(np.argmin([abs(x-c) for c in centers])))
    return df

def compute_dynamic_tol_y(df_lines):
    """
    Calcule la médiane de l'écart vertical entre mots successifs
    afin de fixer automatiquement tol_y.
    """
    if df_lines.empty:
        return params["tol_y"]              # valeur par défaut
    ys = df_lines.sort_values("top")["top"].to_numpy()
    diffs = np.diff(ys)
    diffs = diffs[diffs > 0]               # ignore 0
    if diffs.size == 0:
        return params["tol_y"]
    median_gap = np.median(diffs)
    new_tol = max(6, int(median_gap * 1.2))   # 120 % de la médiane (ou min 6 px) 
    logging.debug(f"[TOL] tol_y dynamique = {new_tol} (médiane écarts = {median_gap:.1f}px)")
    return new_tol

def group_by_row_and_col(df):
    if df.empty:
        return []
    df_sorted = df.sort_values(['top','col'])
    rows=[]; buffer=[]; y0=None
    for _,r in df_sorted.iterrows():
        if y0 is None or abs(r.top-y0)<=params['tol_y']:
            buffer.append(r)
            y0 = r.top if y0 is None else (y0+r.top)/2
        else:
            rows.append(buffer); buffer=[r]; y0=r.top
    if buffer: rows.append(buffer)
    n_cols=int(df['col'].max())+1 if 'col' in df else 1
    table=[]
    for row in rows:
        cells=["" for _ in range(n_cols)]
        for cell in row:
            c=int(cell['col']) if 'col' in cell else 0
            cells[c] += (" "+cell['text']).strip()
        table.append(cells)
    return table


def filter_rows(table):
    final=[]
    for row in table:
        line=" ".join(row).strip()
        if not line: continue
        if any(re.search(pat,line,re.IGNORECASE) for pat in params['row_exclude_patterns']): continue
        final.append(row)
    return final


def recombine_rank_and_hospital(table_rows):
    out=[]
    for row in table_rows:
        cells=list(row)+[""]
        rang=""; nom_parts=[]
        first=cells[0].strip()
        m=rgx_rank_leading.match(first)
        if m:
            rang=m.group(1)
            reste=first[m.end():].lstrip(" ,-")
            cells[0]=reste
            if not reste: cells.pop(0)
        while cells and not rgx_numeric.fullmatch(cells[0].strip()):
            nom_parts.append(cells.pop(0).strip())
        nom=" ".join(nom_parts)
        m2=rgx_collé.match(nom)
        if m2:
            nom=m2.group(1).strip(); cells.insert(0,m2.group(2))
        out.append([rang,nom]+cells)
    return out


def detect_and_replace_outliers(table):
    # Déterminer le nombre maximal de colonnes
    max_cols = max(len(r) for r in table)
    for col in range(2, max_cols):
        nums = []
        # Collecte des valeurs numériques
        for row in table:
            if len(row) > col:
                val = row[col]
                try:
                    num = float(val.replace(',', '.'))
                    nums.append(num)
                except:
                    pass
        if not nums:
            logging.debug(f"Colonne {col}: aucune valeur numérique détectée.")
            continue
        mu = np.mean(nums)
        sigma = np.std(nums)
        logging.debug(f"Colonne {col}: moyenne={mu:.2f}, ecart-type={sigma:.2f}")
        # Remplacement des outliers
        for i, row in enumerate(table):
            if len(row) > col:
                val = row[col]
                try:
                    num = float(val.replace(',', '.'))
                    if abs(num - mu) > 3 * sigma:
                        logging.info(f"Outlier détecté ligne {i+1} col {col}: {num} (remplacé)")
                        table[i][col] = 'à remplir'
                except:
                    pass
    return table

# ─────────── Détection cercles rouge/vert ──────────

def detect_color_circles(roi_color, row_boxes, centers):
    """
    Détecte dans chaque cellule si un cercle vert ou rouge est présent.
    - roi_color: image BGR du tableau rogné
    - row_boxes: liste de (y1,y2) délimitant chaque ligne
    - centers: liste de positions x centrales de colonnes
    Retourne une matrice de même dimension que table, avec '1' pour vert, '0' pour rouge, '' sinon.
    """
    hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))  # vert
    mask_red1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))     # rouge bas
    mask_red2 = cv2.inRange(hsv, (160,50,50), (180,255,255))      # rouge haut
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    detection = []
    for (y1,y2) in row_boxes:
        row_detect = []
        for x in centers:
            # zone carrée centrée
            xi = int(x)
            yi1 = max(0, y1)
            yi2 = min(roi_color.shape[0], y2)
            xi1 = max(0, xi-10)
            xi2 = min(roi_color.shape[1], xi+10)
            cell_h = mask_green[yi1:yi2, xi1:xi2]
            cell_r = mask_red[yi1:yi2, xi1:xi2]
            if cv2.countNonZero(cell_h) > 50:
                row_detect.append('1')
            elif cv2.countNonZero(cell_r) > 50:
                row_detect.append('0')
            else:
                row_detect.append("")
        detection.append(row_detect)
    return detection
def strip_symbols(txt):
    return re.sub(r"[>%©,…;]", "", txt).strip()
def compare_with_reference(extracted_path: str, reference_path: str):
    """
    Compare cellule à cellule la table extraite et la table de référence, et loggues :
      - exact_matches    : mêmes chaînes
      - false_values     : mêmes positions, valeurs différentes
      - aberrant_values  : différences numériques > 3*écart-type (pour colonnes numériques)
      - pattern_errors   : valeurs qui ne respectent pas le format attendu (lettres/chiffres)
    """
    # 1) lecture des deux fichiers
    df_ext = pd.read_excel(extracted_path, header=None, dtype=str)
    df_ref = pd.read_excel(reference_path, header=None, dtype=str)

    # s’assurer même shape
    max_rows = max(df_ext.shape[0], df_ref.shape[0])
    max_cols = max(df_ext.shape[1], df_ref.shape[1])
    df_ext = df_ext.reindex(index=range(max_rows), columns=range(max_cols), fill_value="")
    df_ref = df_ref.reindex(index=range(max_rows), columns=range(max_cols), fill_value="")

    # initialisation
    exact, false, aberrant, pattern = [], [], [], []
    num_regex = re.compile(r"^\d+(\.\d+)?$")

    # 2) parcourir chaque cellule
    for i in range(max_rows):
        for j in range(max_cols):
            raw_ext = df_ext.iat[i, j]
            val_ext = "" if pd.isna(raw_ext) else str(raw_ext).strip()

            raw_ref = df_ref.iat[i, j]
            val_ref = "" if pd.isna(raw_ref) else str(raw_ref).strip()
            val_ext = strip_symbols(val_ext)
            val_ref = strip_symbols(val_ref)
            # 2.a) exact match
            if val_ext == val_ref:
                exact.append((i+1, j+1, val_ext))
                continue

            # 2.b) si les deux sont numériques, tester aberrance
            m_ext = num_regex.fullmatch(val_ext)
            m_ref = num_regex.fullmatch(val_ref)
            if m_ext and m_ref:
                # on recompute mean et std de la colonne de référence
                col_vals = pd.to_numeric(df_ref[j].dropna().astype(str).str.replace(",", "."),
                                         errors="coerce").dropna()
                if len(col_vals) >= 2:
                    mu, sigma = col_vals.mean(), col_vals.std()
                    diff = abs(float(val_ext) - float(val_ref))
                    if diff > 3 * sigma:
                        aberrant.append((i+1, j+1, val_ext, val_ref, mu, sigma))
                        continue
                # sinon c’est un simple false value
            # 2.c) pattern error : lettres dans chiffres ou chiffres dans textes
            if (m_ext and not m_ref) or (not m_ext and m_ref):
                pattern.append((i+1, j+1, val_ext, val_ref))
                continue

            # 2.d) sinon c’est juste une valeur fausse
            false.append((i+1, j+1, val_ext, val_ref))

    # 3) synthèse
    logging.info(f"[TEST] Exacts      : {len(exact)} cellules")
    logging.info(f"[TEST] Fausse(s)   : {len(false)} cellules -> positions & valeurs différentes")
    logging.info(f"[TEST] Aberrante(s): {len(aberrant)} valeurs hors 3σ")
    logging.info(f"[TEST] Patterns err:{len(pattern)} erreurs de format")

    # ← NOUVEAU : pourcentage d’erreur
    total_cells = max_rows * max_cols
    error_cells = len(false) + len(aberrant) + len(pattern)
    error_pct = (error_cells / total_cells) * 100 if total_cells else 0.0
    logging.info(f"[TEST] Pourcentage d'erreur : {error_pct:.2f}% ({error_cells}/{total_cells})")

    # 4) détail (optionnel : dumper dans un CSV)
    # Construire la liste de dictionnaires en adaptant exact (3-tuples) vs les autres
    rows = []

    # exact (row, col, val) -> ref = val
    for r, c, v in exact:
        rows.append({"type":"exact", "row": r, "col": c, "ext": v, "ref": v})

    # false (row, col, ext, ref)
    for r, c, v_ext, v_ref in false:
        rows.append({"type":"false", "row": r, "col": c, "ext": v_ext, "ref": v_ref})

    # aberrant (row, col, ext, ref, mean, std)
    for r, c, v_ext, v_ref, mu, sigma in aberrant:
        rows.append({
            "type":    "aberrant",
            "row":     r,
            "col":     c,
            "ext":     v_ext,
            "ref":     v_ref,
            "mean":    mu,
            "std":     sigma
        })

    # pattern (row, col, ext, ref)
    for r, c, v_ext, v_ref in pattern:
        rows.append({"type":"pattern", "row": r, "col": c, "ext": v_ext, "ref": v_ref})

    df_report = pd.DataFrame(rows)

    report_file = "test_report.csv"
    df_report.to_csv(report_file, index=False)
    logging.info(f"[TEST] Rapport détaillé généré : {report_file}")


def analyze_pdf(pdf_path):
    global file_counter
    images = extract_pdf_images(pdf_path)
    for i, img_path in enumerate(images):
        logging.info(f"--- Traitement Page {i+1} : {img_path} ---")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        logging.debug(f"Image chargée shape={img.shape}")
        pre = preprocess(img)
        roi = crop_table_region(pre)
        if params.get('save_debug', False):
            dbg = f"debug_crop_page{i+1}.png"
            cv2.imwrite(dbg, roi)
            logging.debug(f"Image cropée sauvegardée → {dbg}")

        # 1. Découpage horizontal de la zone cropée
        n_slices = params.get('n_slices', 1)
        slices = split_roi_horizontally(roi, n_slices=n_slices)
        logging.debug(f"Nombre de slices générées: {len(slices)}")
        df_list = []
        for idx, (y1, y2, sub_img) in enumerate(slices):
            if params.get('save_debug', False):
                dbg_slice = f"debug_slice{idx+1}_page{i+1}.png"
                cv2.imwrite(dbg_slice, sub_img)
                logging.debug(f"Slice {idx+1} sauvegardée → {dbg_slice}")
            df_sub = full_ocr(sub_img)
            logging.debug(f"Slice {idx+1} OCR ret {len(df_sub)} composants")
            if not df_sub.empty:
                df_sub['top'] += y1
                df_list.append(df_sub)
        df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
        logging.debug(f"Total OCR concaténé: {len(df)} composants")

        centers = detect_column_centers(df)
        logging.debug(f"Centers détectés: {centers}")
        df = assign_columns(df, centers)
        logging.debug(f"Assign colonnes done, unique columns: {df['col'].unique().tolist()}")
        params["tol_y"] = compute_dynamic_tol_y(df)    # ajuste tolérance ligne
        table = group_by_row_and_col(df)
        logging.info(f"[ROWS] Lignes obtenues : {len(table)} (attendu ≈ 50)")
        if not (45 <= len(table) <= 55):
            logging.warning("[ROWS] Nombre de lignes anormal – revoir tol_y ou zone de crop")
        logging.debug(f"Group by rows : {len(table)} lignes brutes")
        table = filter_rows(table)
        logging.debug(f"Après filter_rows : {len(table)} lignes")
        table = recombine_rank_and_hospital(table)
        logging.debug(f"Après recombine rank+hospi : {len(table)} lignes")
        table = detect_and_replace_outliers(table)
        logging.debug(f"Après détection outliers : {len(table)} lignes finales")

        # Export
        df_out = pd.DataFrame(table)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = 'xlsx' if params['export_excel'] else 'csv'
        out_file = f"{params['output_basename']}_{ts}.{ext}"
        try:
            if params['export_excel']:
                df_out.to_excel(out_file, index=False, header=False)
            else:
                df_out.to_csv(out_file, index=False, header=False)
            logging.info(f"Fichier généré: {out_file}")
        except Exception as e:
            logging.error(f"Erreur export fichier: {e}")

        # … dans analyze_pdf, juste après l’export :
        if params['export_excel']:
            df_out.to_excel(out_file, index=False, header=False)
        else:
            df_out.to_csv(out_file, index=False, header=False)

        # ← ici, on déclenche le mode test :
        ref_file = "ext_LP-167_Acc_Risque.xlsx"
        compare_with_reference(out_file, ref_file)



if __name__ == '__main__':
    pdf = 'scan/1.pdf'
    if os.path.isfile(pdf):
        analyze_pdf(pdf)
    else:
        logging.error(f"PDF introuvable: {pdf}")
