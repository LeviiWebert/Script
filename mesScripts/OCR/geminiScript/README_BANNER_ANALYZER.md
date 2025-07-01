# 📊 Title Banner Analyzer

Script d'analyse automatique des bannières de titres dans les classements médicaux.

## 🎯 Fonctionnalités

- **🔍 Détection automatique** des tableaux dans les images originales
- **✂️ Découpage précis** des bannières (2290×86 px) au-dessus des tableaux
- **🤖 Analyse Gemini** pour extraire les titres de classements
- **🏥 Détection du type** d'établissement (Hôpitaux/Cliniques)
- **📊 Export Excel** avec résultats détaillés et statistiques

## 🛠️ Installation

### 1. Installer les dépendances
```bash
pip install opencv-python pillow pandas openpyxl google-generativeai numpy
```

### 2. Configuration de l'API Gemini
1. Obtenez votre clé API sur : https://makersuite.google.com/app/apikey
2. Modifiez le fichier `config_banner_analyzer.py`
3. Remplacez `YOUR_GEMINI_API_KEY_HERE` par votre vraie clé

### 3. Configuration des chemins
Modifiez `IMAGES_BASE_PATH` dans `config_banner_analyzer.py` selon votre structure :
```python
IMAGES_BASE_PATH = r"C:\votre\chemin\vers\images"
```

## 📁 Structure attendue des dossiers

```
images/
├── ABLATIONDESVARICES/
│   ├── img00001.jpg                 # Image originale
│   ├── ABLATIONDESVARICES_1.jpg     # Image tableau (avec numéro)
│   └── ABLATIONDESVARICES_2.jpg     # Image tableau (multiple)
├── CANCER_PROSTATE/
│   ├── img00002.jpg                 # Image originale
│   └── CANCER_PROSTATE.jpg          # Image tableau (unique)
└── ...
```

### 📝 Formats d'images supportés

**Images originales :**
- `img00001.jpg`, `img00002.jpg`, etc.

**Images de tableaux :**
- **Avec numéro :** `DOSSIER_1.jpg`, `DOSSIER_2.jpg`, `DOSSIER_3.jpg`
- **Sans numéro :** `DOSSIER.jpg` (pour les cas d'un seul tableau par dossier)

## 🚀 Utilisation

```bash
python title_banner_analyzer.py
```

## 📋 Résultats Excel

Le script génère un fichier Excel avec 2 feuilles :

### Feuille "Résultats_Analysis"
| Colonne | Description |
|---------|-------------|
| `Dossier` | Nom du dossier source |
| `Image_Originale` | Nom de l'image originale |
| `Image_Tableau` | Nom de l'image tableau |
| `Position_Detectee` | OUI/NON si tableau détecté |
| `Titre_Classement` | Titre extrait par Gemini |
| `Type_Etablissement` | HÔPITAUX/CLINIQUES/MIXTE |
| `Date_Traitement` | Horodatage du traitement |

### Feuille "Statistiques"
- Total de paires traitées
- Taux de détection réussie
- Nombre de titres extraits
- Performances globales

## ⚙️ Configuration avancée

Dans `config_banner_analyzer.py` :

```python
# Dimensions des bannières
BANNER_WIDTH = 2290
BANNER_HEIGHT = 86

# Seuil de confiance (0.0 à 1.0)
DETECTION_THRESHOLD = 0.8

# Marge au-dessus du tableau
MARGIN_ABOVE_TABLE = 50
```

## 🤖 Prompts Gemini utilisés

### Pour l'extraction du titre :
```
Analyse cette image qui contient la partie supérieure d'un tableau médical. 
Identifie et extrait UNIQUEMENT le titre principal du classement médical 
qui se trouve généralement dans cette bannière, souvent en lettres capitales.
Retourne seulement le titre exact, sans formatage ni explication.
```

### Pour le type d'établissement :
```
Analyse ce tableau médical et détermine s'il s'agit d'un classement de:
- HÔPITAUX (établissements publics)
- CLINIQUES (établissements privés) 
- MIXTE (les deux types)
Retourne uniquement: "HÔPITAUX", "CLINIQUES", "MIXTE" ou "NON DÉTECTÉ".
```

## 🗂️ Gestion du cache

- Les bannières extraites sont temporairement stockées dans `./cache/`
- Le dossier est automatiquement nettoyé après traitement
- En cas d'erreur, supprimez manuellement le dossier `cache/`

## ⚠️ Limitations

- **Maximum 2 requêtes Gemini par image tableau** (respect des quotas)
- Nécessite une correspondance visuelle entre image originale et tableau
- Seuil de détection ajustable selon la qualité des images
- Connexion internet requise pour l'API Gemini

## 🔧 Dépannage

### Erreur "Gemini API non configurée"
- Vérifiez votre clé API dans `config_banner_analyzer.py`
- Assurez-vous d'avoir un compte Google AI Studio actif

### Aucune paire d'images trouvée
- Vérifiez le chemin `IMAGES_BASE_PATH`
- Assurez-vous que la structure des dossiers est correcte

### Faible taux de détection
- Réduisez `DETECTION_THRESHOLD` (ex: 0.6)
- Vérifiez la qualité des images
- Ajustez `MARGIN_ABOVE_TABLE`

## 📞 Support

En cas de problème, vérifiez :
1. ✅ Installation des dépendances
2. ✅ Configuration de la clé API Gemini  
3. ✅ Chemins des dossiers d'images
4. ✅ Structure des noms de fichiers

---

*Script développé pour l'analyse automatique des classements médicaux*
