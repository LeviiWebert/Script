# Configuration pour title_banner_analyzer.py
# Remplacez les valeurs ci-dessous selon votre environnement

import os

# =============================================================================
# CONFIGURATION GEMINI API
# =============================================================================
# Obtenez votre clé API Gemini sur: https://makersuite.google.com/app/apikey
GEMINI_API_KEY = "AIzaSyBfQjj1pNx0yDlXUSo4tdWUe5RcE35ON6o"

# =============================================================================
# CONFIGURATION DES CHEMINS
# =============================================================================
# Chemin vers le dossier contenant tous les sous-dossiers d'images
IMAGES_BASE_PATH = r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\tri_image_script\images_backup_20250625_145044"

# Dossier cache temporaire (utilise le dossier temp de l'utilisateur)
CACHE_DIR = os.path.join(os.path.expanduser("~"), "temp_banner_analyzer_cache")

# =============================================================================
# CONFIGURATION DE LA DÉTECTION
# =============================================================================
# Dimensions des bannières à extraire
BANNER_WIDTH = 2290
BANNER_HEIGHT = 86

# Seuil de confiance pour la détection de template (0.0 à 1.0)
DETECTION_THRESHOLD = 0.8

# Marge au-dessus du tableau en pixels
MARGIN_ABOVE_TABLE = 50

# =============================================================================
# INSTRUCTIONS D'INSTALLATION
# =============================================================================
"""
Pour utiliser ce script, vous devez installer les dépendances suivantes:

pip install opencv-python
pip install pillow
pip install pandas
pip install openpyxl
pip install google-generativeai
pip install numpy

Puis modifiez la clé API Gemini dans ce fichier.
"""
