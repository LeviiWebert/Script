# OCR Medical Document Processor

Ce projet permet d'extraire et de traiter automatiquement les données de tableaux dans des documents médicaux PDF en utilisant la reconnaissance optique de caractères (OCR).

## Structure du projet

```
OCR/
├── main.py                    # Point d'entrée principal
├── config.py                  # Configuration et paramètres
├── image_processing.py        # Traitement d'images et PDF
├── ocr_processing.py         # Extraction OCR et traitement du texte
├── validation.py             # Validation et métriques de qualité
├── file_operations.py        # Opérations de fichiers et export
├── requirements.txt          # Dépendances Python
└── README.md                 # Ce fichier
```

## Fonctionnalités principales

### 1. Traitement des images
- **Extraction PDF** : Conversion des pages PDF en images haute résolution
- **Préprocessing** : Redimensionnement, flou gaussien, binarisation
- **Rogbage intelligent** : Détection et extraction de la zone de tableau
- **Découpage horizontal** : Division en tranches pour améliorer l'OCR

### 2. Reconnaissance OCR
- **Extraction de texte** : Utilisation de Tesseract avec configuration optimisée
- **Détection de colonnes** : Analyse par histogramme des positions X
- **Reconstruction de tableau** : Regroupement intelligent des mots en lignes et colonnes
- **Tolérance dynamique** : Ajustement automatique des paramètres selon le contenu

### 3. Traitement des données
- **Nettoyage du texte** : Suppression des caractères indésirables
- **Filtrage des lignes** : Exclusion des en-têtes et pieds de page
- **Recombinaison** : Assemblage des rangs et noms d'hôpitaux séparés
- **Détection d'outliers** : Identification et marquage des valeurs aberrantes

### 4. Validation et qualité
- **Comparaison avec référence** : Évaluation cellule par cellule
- **Métriques de précision** : Calcul du pourcentage d'erreur
- **Rapports détaillés** : Génération de rapports CSV avec types d'erreurs
- **Validation structurelle** : Vérification du nombre de lignes et colonnes

## Installation

### Prérequis
- Python 3.8+
- Tesseract OCR installé sur le système

### Installation des dépendances
```bash
pip install -r requirements.txt
```

### Configuration Tesseract
Assurez-vous que Tesseract est installé et accessible. Le chemin par défaut est :
```
C:\Program Files\Tesseract-OCR\tesseract.exe
```

## Utilisation

### Utilisation de base
```python
from main import MedicalDocumentProcessor

# Créer le processeur
processor = MedicalDocumentProcessor()

# Traiter un document
results = processor.process_document('chemin/vers/document.pdf')

if results['success']:
    print(f"Traitement réussi ! {results['statistics']['successful_pages']} pages traitées")
else:
    print(f"Erreur : {results['error']}")
```

### Ligne de commande
```bash
python main.py chemin/vers/document.pdf
```

### Configuration personnalisée
```python
config_personnalisee = {
    'dpi': 400,
    'language': 'fra',
    'n_slices': 2,
    'crop_right_ratio': 0.85,
    'export_excel': True
}

processor = MedicalDocumentProcessor(config_personnalisee)
```

## Configuration

### Paramètres OCR
- `dpi` : Résolution d'extraction PDF (défaut: 300)
- `language` : Langue OCR (défaut: "fra")
- `min_conf` : Confiance minimale OCR (défaut: 30)
- `resize_factor` : Facteur de redimensionnement (défaut: 2.0)

### Paramètres de détection
- `bin_width` : Largeur des bins d'histogramme (défaut: 50)
- `peak_min_count` : Seuil minimum pour détection de pic (défaut: 20)
- `tol_y` : Tolérance verticale pour groupage (défaut: 8)

### Paramètres de rogbage
- `crop_top_ratio` : Ratio de rogbage haut (défaut: 0.0)
- `crop_bottom_ratio` : Ratio de rogbage bas (défaut: 0.0)
- `crop_left_ratio` : Ratio de rogbage gauche (défaut: 0.0)
- `crop_right_ratio` : Ratio de rogbage droite (défaut: 0.80)

### Paramètres de sortie
- `save_debug` : Sauvegarder images de debug (défaut: True)
- `export_excel` : Exporter en Excel vs CSV (défaut: True)
- `output_basename` : Nom de base des fichiers (défaut: "extraction_tableaux")

## Architecture modulaire

### `config.py`
- Paramètres globaux et configuration
- Patterns regex réutilisables
- Configuration du logging

### `image_processing.py`
- `PDFProcessor` : Extraction d'images depuis PDF
- `ImageProcessor` : Préprocessing et manipulation d'images
- `ColorCircleDetector` : Détection de cercles colorés (fonctionnalité future)

### `ocr_processing.py`
- `OCRProcessor` : Extraction de texte avec Tesseract
- `ColumnDetector` : Détection et assignation de colonnes
- `TableReconstructor` : Reconstruction de la structure tabulaire
- `TextProcessor` : Nettoyage et traitement du texte

### `validation.py`
- `QualityAssessment` : Évaluation de la qualité par comparaison
- `ValidationMetrics` : Métriques de validation structurelle

### `file_operations.py`
- `FileExporter` : Export vers Excel/CSV
- `ReportGenerator` : Génération de rapports
- `ConfigManager` : Gestion des fichiers de configuration

## Fichiers de sortie

### Fichiers principaux
- `extraction_tableaux_YYYYMMDD_HHMMSS.xlsx` : Tableau extrait
- `test_report.csv` : Rapport de comparaison détaillé
- `processing_summary.txt` : Résumé du traitement

### Fichiers de debug (si activé)
- `debug_crop_pageX.png` : Zone rognée par page
- `debug_sliceX_pageY.png` : Tranches découpées
- `page_X.png` : Images extraites du PDF

## Métriques de qualité

Le système génère plusieurs métriques :
- **Correspondances exactes** : Cellules identiques
- **Valeurs fausses** : Cellules avec contenu différent
- **Valeurs aberrantes** : Valeurs numériques hors 3σ
- **Erreurs de pattern** : Incompatibilités de format
- **Pourcentage de précision** : Précision globale

## Dépannage

### Problèmes courants

**Tesseract non trouvé**
```
Solution : Vérifier l'installation et le chemin dans config.py
```

**Aucune colonne détectée**
```
Solution : Ajuster bin_width et peak_min_count
```

**Nombre de lignes incorrect**
```
Solution : Modifier tol_y ou les paramètres de rogbage
```

**Qualité OCR faible**
```
Solution : Augmenter DPI, resize_factor, ou activer le préprocessing
```

### Mode debug
Activez `save_debug: True` pour générer les images intermédiaires et diagnostiquer les problèmes.

## Évolutions futures

- Interface graphique utilisateur
- Support de formats PDF complexes
- Détection automatique de la zone de tableau
- Machine learning pour améliorer la précision
- Support multi-langues étendu

## Contribution

Pour contribuer au projet :
1. Fork le repository
2. Créez une branche feature
3. Committez vos changements
4. Poussez vers la branche
5. Créez une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
