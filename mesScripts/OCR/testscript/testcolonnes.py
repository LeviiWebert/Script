# ────────────────────────────  testcolonnes.py  ─────────────────────────────
import os, logging, unicodedata, difflib, tempfile
from datetime import datetime

import fitz                  # PyMuPDF
import cv2
import numpy as np
import pandas as pd
import pytesseract
import re

_word_re = re.compile(r"^[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s\-]{2,}$")  # ≥3 caractères alpha

# ───────────── Configuration explicite du binaire Tesseract ─────────
tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

# ───────────────────────── PARAMÈTRES GLOBAUX ──────────────────────────
params = {
    "dpi": 300,
    "language": "fra",                # pack Tesseract installé
    "header_top_ratio": 0.28,         # 26 %  (≈ 220 px @72 dpi)
    "header_bottom_ratio": 0.52,      # 36 %  (≈ 301 px)
    "angles_deg": [-45], 
    "fuzzy_cutoff": 0.50,             # seuil pour difflib
    "tol_y": 15,                      # tolérance (px) regroupement par ligne
    "export_excel": True,
    "out_basename": "extraction_tableaux",
    "tesseract_cfg": "--psm 6",      # OCR « bloc de texte »
    "save_debug": True,              # True → écrit des PNG de debug
}

# ────────────────────────────  LOGGING  ─────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,                                  # DEBUG pour voir tous les essais
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ────────────────────────  OUTILS D’OCR / MATCH  ───────────────────────
def strip_accents(txt: str) -> str:
    """Supprime accents et caractères combinants, met en minuscules."""
    txt = ''.join(c for c in unicodedata.normalize("NFKD", txt)
                  if not unicodedata.combining(c))
    return txt.lower()



"""def ocr_header_band(gray: np.ndarray, angle_deg: float, page_num: int = 1) -> list[str]:
    #Retourne la liste de *mots complets* détectés dans la bande d’en-tête
    if angle_deg is None:
        raise ValueError("angle_deg ne peut pas être None")

    h, w = gray.shape
    top = int(params["header_top_ratio"] * h)
    bot = int(params["header_bottom_ratio"] * h)
    band = gray[top:bot, :]

    band = cv2.resize(band, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # rotation bande seule
    M = cv2.getRotationMatrix2D((band.shape[1]//2, band.shape[0]//2), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        band, M, (band.shape[1], band.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_DEFAULT, borderValue=255  # bord blanc, plus de stries
    )

    # recadrage auto pour enlever le blanc
    ys, xs = np.where(rotated < 250)
    if ys.size and xs.size:
        rotated = rotated[ys.min():ys.max()+1, xs.min():xs.max()+1]

    if params["save_debug"]:
        cv2.imwrite(f"debug_rot_{int(angle_deg):+d}deg_p{page_num}.png", rotated)

    # OCR → texte brut
    txt = pytesseract.image_to_string(
        rotated, lang=params["language"], config=params["tesseract_cfg"]
    )

    # nettoyage → mots complets uniques
    words = []
    for line in txt.splitlines():
        line = line.strip()
        if _word_re.match(line):
            words.append(line)

    logging.info(f"[OCR] mots complets captés = {len(words)} : {words}")
    return words
"""

def ocr_header_band(gray: np.ndarray,
                    angle_deg: float = -45,
                    page_num: int = 1) -> list[str]:
    """
    Extrait l’entête inclinée de la page PDF et renvoie la liste des titres
    reconnus, filtrés de façon souple (≥ 2 mots, présence d’au moins 1 majuscule).

    - Coupe la bande verticale 26-36 % de la hauteur.
    - Agrandit ×2 pour rendre les lettres plus lisibles.
    - Tourne la bande de `angle_deg` (bord blanc, pas de stries).
    - OCR détaillé (Tesseract OEM 1, PSM 6 → TSV par mot).
    - Regroupe les mots par lignes avec une tolérance de 5 px.
    - Filtre les lignes bruitées.
    - Sauvegarde les PNG de debug si `params["save_debug"]` est True.
    """
    if angle_deg is None:
        raise ValueError("angle_deg ne peut pas être None")

    h, w = gray.shape
    top = int(params["header_top_ratio"] * h)
    bot = int(params["header_bottom_ratio"] * h)
    band = gray[top:bot, :]

    # ↑  agrandissement (améliore la reconnaissance sur petits caractères)
    band = cv2.resize(band, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # (optionnel) contraste – le laissait commenté si ça dégrade
    # band = boost_contrast(band)

    # Rotation de la bande seule
    M = cv2.getRotationMatrix2D((band.shape[1] // 2, band.shape[0] // 2),
                                angle_deg, 1.0)
    rotated = cv2.warpAffine(
        band, M, (band.shape[1], band.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255      # bord blanc ⟹ plus de rayures
    )

    # Recadrage automatique (on enlève le bord blanc)
    ys, xs = np.where(rotated < 250)         # zones “encre”
    if ys.size and xs.size:
        rotated = rotated[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # Debug visuel
    if params["save_debug"]:
        cv2.imwrite(f"debug_rot_{int(angle_deg):+d}deg_p{page_num}.png", rotated)

    # ───── OCR détaillé ───────────────────────────────────────────────
    df = pytesseract.image_to_data(
        rotated,
        output_type=pytesseract.Output.DATAFRAME,
        lang=params["language"],
        config="--oem 1 --psm 6",
    )

    if df.empty:
        logging.info("[OCR] Aucun mot détecté dans la bande.")
        return []

    df = df[df.conf > 30].copy()               # ← ①   seuil abaissé
    df["text"] = df["text"].astype(str)

    # Regroupement par ligne – tolérance verticale
    lignes, courant, y0 = [], [], None
    for _, r in df.sort_values("top").iterrows():
        if y0 is None or abs(r.top - y0) < 8:   # ← ②  5 → 8 px
            courant.append(r.text)
        else:
            lignes.append(" ".join(courant))
            courant = [r.text]
        y0 = r.top
    if courant:
        lignes.append(" ".join(courant))

    # ─── Post-traitement & filtre souple ──────────────────────────────
    titres_net, vu = [], set()
    for ln in lignes:
        mots = re.findall(r"[A-Za-zÀ-ÿ]{2,}", ln)   # on collecte les blocs alphabétiques
        if len(mots) < 2:                           # toujours au moins 2 « vrais » mots
            continue
        titre = " ".join(mots).title()
        ratio = sum(c.isalpha() for c in ln) / len(ln)   # ← ③  ratio sur la ligne brute
        if ratio < 0.5:                                 # 0,6 → 0,5
            continue
        k = titre.lower()
        if k not in vu:
            titres_net.append(titre)
            vu.add(k)

    logging.info(f"[OCR] titres filtrés = {len(titres_net)} : {titres_net}")
    return titres_net

# ──────────────────────────────────────────────────────────────────────
def score_headers(words: list[str]) -> set[str]:
    """Renvoie l’ensemble des entêtes attendus trouvés dans *words* (fuzzy)."""
    found = set()
    for tgt in expected_headers:
        norm_t = strip_accents(tgt)
        if difflib.get_close_matches(norm_t, words, n=1, cutoff=params["fuzzy_cutoff"]):
            found.add(tgt)
    return found

def tune_header_detection(gray):
    """Teste la grille d’angles, retourne l’angle optimal + sets found/missing."""
    best = {"angle": None, "found": set()}
    for ang in params["angles_deg"]:
        words = ocr_header_band(gray, ang)
        found = score_headers(words)
        logging.debug(f"[AUTO] angle={ang:+}°  trouvés={len(found)}/{len(expected_headers)} → {sorted(found)}")
        if len(found) > len(best["found"]):
            best = {"angle": ang, "found": found}
        # early break si on a tout trouvé
        if len(best["found"]) == len(expected_headers):
            break

    missing = set(expected_headers) - best["found"]
    logging.info(f"[AUTO] Meilleur angle={best['angle']}°   headers trouvés={len(best['found'])}/"
                 f"{len(expected_headers)}   manquants={len(missing)}")
    if missing:
        logging.info(f"[AUTO] Headers manquants : {', '.join(missing)}")
    return best["angle"], best["found"], missing

# ─────────────────────────── UTILITAIRES PDF ───────────────────────────
def pdf_page_to_gray(pdf_path: str, page_idx: int = 0) -> np.ndarray:
    """Charge une page PDF en niveaux de gris numpy (uint8)."""
    doc = fitz.open(pdf_path)
    pix = doc[page_idx].get_pixmap(dpi=params["dpi"])
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height,
                                                             pix.width, pix.n)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if pix.n == 3 else
                             cv2.COLOR_BGR2GRAY if pix.n == 4 else
                             cv2.COLOR_GRAY2BGR)
    return gray

# ────────────────────────────  MAIN FLOW  ──────────────────────────────
def analyze_pdf(pdf_path: str) -> list[str]:
    gray = pdf_page_to_gray(pdf_path)
    titles = ocr_header_band(gray, -45)
    logging.info(f"Titres extraits ({len(titles)}): {titles}")
    return titles

# ──────────────────────────  EXECUTION ────────────────────────────────
if __name__ == "__main__":
    pdf_file = "scan/1.pdf"      # ← adapte le chemin
    if not os.path.isfile(pdf_file):
        logging.critical(f"PDF introuvable : {pdf_file}")
    else:
        analyze_pdf(pdf_file)
# ───────────────────────────────────────────────────────────────────────
