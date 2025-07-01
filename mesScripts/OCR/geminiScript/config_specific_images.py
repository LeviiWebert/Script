# Configuration spécifique pour traiter uniquement les images restantes
# Ce fichier permet de traiter seulement les images qui n'étaient pas dans le dossier backup

import os

# =============================================================================
# CONFIGURATION GEMINI API
# =============================================================================
# Obtenez votre clé API Gemini sur: https://makersuite.google.com/app/apikey
GEMINI_API_KEY = "AIzaSyBfQjj1pNx0yDlXUSo4tdWUe5RcE35ON6o"

# =============================================================================
# CONFIGURATION DES CHEMINS - DOSSIER IMAGES PRINCIPAL
# =============================================================================
# Chemin vers le dossier "images" principal (pas le backup)
IMAGES_BASE_PATH = r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\tri_image_script\images"

# Dossier cache temporaire (utilise le dossier temp de l'utilisateur)
CACHE_DIR = os.path.join(os.path.expanduser("~"), "temp_banner_analyzer_cache_specific")

# =============================================================================
# LISTE DES IMAGES SPÉCIFIQUES À TRAITER
# =============================================================================
# Liste des dossiers/images à traiter spécifiquement
SPECIFIC_FOLDERS_TO_PROCESS = [
    "APPENDICITE",
    "AUDITION", 
    "CANCERDELATHYRODE",
    "CANCERDELOVAIRE",
    "CANCEROR",
    "CHIRURGIEDELARETINE",
    "CHIRURGIEDENTAIREETORALEDELENFANTETDELADOLESCENT",
    "CHIRURGIEDUNEZETDESSINUS",
    "GLANDESSALIVAIRES",
    "INFARCTUSDUMYOCARDE",
    "PROCTOLOGIE",
    "SCHIZOPHRENIE",
    "UCARDIOLOGIEINTERVENTIONNELLE",
    "UCHIRURGIEDESTESTICULESDELENFANTETDELADOLESCENT",
    "UCHIRURGIEDUDOSDELENFANTETDELADOLESCENT",
    "UCHIRURGIEMAXILLOFACIALE",
    "ULEUCEMIEDELADULTE",
    "UTUMEURSDUCERVEAUDELENFANTETDELADOLESCENT"
]

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
# INSTRUCTIONS D'UTILISATION
# =============================================================================
"""
Ce fichier de configuration est spécialement conçu pour traiter uniquement 
les images qui n'étaient pas dans le dossier backup.

Pour utiliser cette configuration :
1. Copiez le script title_banner_analyzer.py vers title_banner_analyzer_specific.py
2. Modifiez l'import pour utiliser ce fichier de config
3. Lancez le script spécifique

Les dossiers qui seront traités :
- APPENDICITE
- AUDITION
- CANCERDELATHYRODE
- CANCERDELOVAIRE
- CANCEROR
- CHIRURGIEDELARETINE
- CHIRURGIEDENTAIREETORALEDELENFANTETDELADOLESCENT
- CHIRURGIEDUNEZETDESSINUS
- GLANDESSALIVAIRES
- INFARCTUSDUMYOCARDE
- PROCTOLOGIE
- SCHIZOPHRENIE
- UCARDIOLOGIEINTERVENTIONNELLE
- UCHIRURGIEDESTESTICULESDELENFANTETDELADOLESCENT
- UCHIRURGIEDUDOSDELENFANTETDELADOLESCENT
- UCHIRURGIEMAXILLOFACIALE
- ULEUCEMIEDELADULTE
- UTUMEURSDUCERVEAUDELENFANTETDELADOLESCENT
"""
