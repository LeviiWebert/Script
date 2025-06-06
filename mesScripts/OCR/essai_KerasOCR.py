import glob
import cv2
import keras_ocr
import pandas as pd
from matplotlib import pyplot as plt

# 1. Pipeline Keras-OCR (téléchargement des poids si nécessaire)
pipeline = keras_ocr.pipeline.Pipeline()

# 2. Chargement & redimensionnement
def load_and_resize(path, max_dim=1024):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Impossible de lire {path}")
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

# 3. Chargement des images locales
def main():
    image_paths = sorted(glob.glob('images/*.*'))
    images = [load_and_resize(p) for p in image_paths]

    # 4. OCR image par image
    prediction_groups = []
    for img in images:
        preds = pipeline.recognize([img])[0]
        prediction_groups.append(preds)

    # 5. Export des résultats en Excel
    rows = []
    for path, preds in zip(image_paths, prediction_groups):
        for word, box in preds:
            text = ''.join(word) if isinstance(word, (list, tuple)) else word
            x_center = float(box[:, 0].mean())
            y_center = float(box[:, 1].mean())
            rows.append({
                'image': path,
                'text': text,
                'x_center': x_center,
                'y_center': y_center
            })
    df = pd.DataFrame(rows)
    df.to_excel('ocr_predictions.xlsx', index=False)
    print(f"✅ {len(rows)} entrées exportées vers 'ocr_predictions.xlsx'")

    # 6. Visualisation
    fig, axs = plt.subplots(nrows=len(images), figsize=(20, 5 * len(images)))
    for ax, image, predictions in zip(axs, images, prediction_groups):
        keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
