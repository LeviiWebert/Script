#!/usr/bin/env python3
"""table_cropper_cv.py

Détection *purement visuelle* des tableaux dans les scans du magazine **Le Point** –
aucun OCR requis.

Principe
--------
1. **Crop grossier optionnel** (`--crop`) : on enlève en‑tête, pied de page, marges.
2. **Seuillage adaptatif inversé** pour révéler toutes les lignes.
3. **Extraction des lignes horizontales & verticales** :
   - noyau rect. horizontal : `(largeur / kernel_scale, 1)`.
   - noyau rect. vertical : `(1, hauteur / kernel_scale)`.
4. **Fusion & dilatation** (`--dilate_iter`) pour combler les césures de grille.
5. **Contours externes** → une boîte englobante par tableau.
6. **Filtre + fusion** : aire ≥ `--min_area`, largeur & hauteur ≥ 50 px.
   Boîtes qui se chevauchent sont fusionnées (``merge_boxes``).
7. **Sauvegarde** des crops côté image d’origine – suffixe `_tableN`.

Usage
-----
```bash
pip install opencv-python numpy tqdm
python table_cropper_cv.py /chemin/dossier \
       --min_area 10000 \
       --kernel_scale 35 \
       --save_debug
```
Options clés :
* `--kernel_scale` : plus petit → détecte des lignes plus fines.
* `--dilate_iter` : si la grille est morcelée, augmentez‑le.
* `--crop` : ratios *top bottom left right* pour ignorer les bandeaux parasites.

Le script écrit également (si `--save_debug`) une version annotée en rouge des tables
détectées.
"""

from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple
from tqdm import tqdm

# ----------------------
# PARAMÈTRES PAR DÉFAUT
# ----------------------
DEFAULT_CROP = (0.10, 0.95, 0.05, 0.95)  # top, bottom, left, right ratios
EXTS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'}

# ------------------
#  UTILITAIRES
# ------------------

def coarse_crop(img: np.ndarray, ratios) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Retourne le ROI et le décalage (left, top)."""
    h, w = img.shape[:2]
    t = int(h * ratios[0]); b = int(h * ratios[1])
    l = int(w * ratios[2]); r = int(w * ratios[3])
    return img[t:b, l:r], (l, t)


def merge_boxes(boxes: List[Tuple[int, int, int, int]], overlap_thresh: float = 0.15):
    if not boxes:
        return []
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    area = boxes[:, 2] * boxes[:, 3]
    idxs = np.argsort(y1)
    keep = []
    while len(idxs):
        i = idxs[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / area[idxs[1:]]
        idxs = idxs[np.where(overlap < overlap_thresh)[0] + 1]
    return [tuple(boxes[i]) for i in keep]

# ------------------
#  CŒUR DE L’ALGOS
# ------------------

def detect_tables(img: np.ndarray, p):
    roi, offset = coarse_crop(img, p.crop)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 15, 9)

    # Extraction des lignes horizontales et verticales
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, roi.shape[1] // p.kernel_scale), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, roi.shape[0] // p.kernel_scale)))

    horiz = cv2.erode(thr, h_kernel, iterations=1)
    horiz = cv2.dilate(horiz, h_kernel, iterations=1)

    vert = cv2.erode(thr, v_kernel, iterations=1)
    vert = cv2.dilate(vert, v_kernel, iterations=1)

    grid = cv2.bitwise_or(horiz, vert)

    # Fermeture pour combler les lacunes
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=p.dilate_iter)

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < p.min_area or w < 50 or h < 50:
            continue
        # coordonnées plein‑cadre
        x += offset[0]; y += offset[1]
        boxes.append((x, y, w, h))

    boxes = merge_boxes(boxes, p.overlap_thresh)
    return boxes

# -------------------------------
#  ENREGISTREMENT & DEBUG
# -------------------------------

def save_crops(img_path: Path, img: np.ndarray, boxes):
    for i, (x, y, w, h) in enumerate(boxes, 1):
        crop = img[y:y + h, x:x + w]
        out = img_path.with_name(f"{img_path.stem}_table{i}{img_path.suffix}")
        cv2.imwrite(str(out), crop)


def save_debug(img_path: Path, img: np.ndarray, boxes):
    dbg = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 3)
    dbg_path = img_path.with_name(img_path.stem + '_debug_bbox' + img_path.suffix)
    cv2.imwrite(str(dbg_path), dbg)

# ------------------
#  BOUCLE PRINCIPALE
# ------------------

def process_dir(root: Path, p):
    imgs = [f for f in root.rglob('*') if f.suffix.lower() in EXTS]
    for img_path in tqdm(imgs, desc='Détection tableaux'):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        boxes = detect_tables(img, p)
        if not boxes:
            continue
        save_crops(img_path, img, boxes)
        if p.save_debug:
            save_debug(img_path, img, boxes)

# --------------
#  ARGUMENTS
# --------------

class Params:
    def __init__(self, a):
        self.min_area = a.min_area
        self.kernel_scale = a.kernel_scale
        self.dilate_iter = a.dilate_iter
        self.crop = a.crop
        self.save_debug = a.save_debug
        self.overlap_thresh = a.overlap_thresh


def get_args():
    ap = argparse.ArgumentParser(description='Rogne automatiquement les tableaux sans OCR.')
    ap.add_argument('root', type=Path, help='Dossier racine des images')
    ap.add_argument('--min_area', type=int, default=8000, help='Aire minimale d’un tableau en px²')
    ap.add_argument('--kernel_scale', type=int, default=30, help='Diviseur pour la taille des noyaux struct. (plus petit → noyau plus grand)')
    ap.add_argument('--dilate_iter', type=int, default=2, help='Itérations de dilatation pour fusionner les lignes')
    ap.add_argument('--crop', type=float, nargs=4, metavar=('TOP','BOTTOM','LEFT','RIGHT'), default=DEFAULT_CROP, help='Fenêtre de travail (ratios de 0 à 1)')
    ap.add_argument('--overlap_thresh', type=float, default=0.15, help='Seuil de fusion des boîtes (IoU inversé)')
    ap.add_argument('--save_debug', action='store_true', help='Sauvegarde une image annotée')
    return ap.parse_args()


# --------------
#  MAIN
# --------------

def main():
    args = get_args()
    if not args.root.exists():
        raise SystemExit(f'Le dossier {args.root} est introuvable.')
    process_dir(args.root, Params(args))
    if args.save_debug:
        print('Images debug enregistrées.')


if __name__ == '__main__':
    main()
