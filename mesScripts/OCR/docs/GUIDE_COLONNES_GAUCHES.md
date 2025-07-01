# 📊 Guide d'utilisation - Processeur Colonnes Gauches

## 🎯 Vue d'ensemble

Ce processeur spécialisé traite **automatiquement** tous les images d'un dossier (et sous-dossiers) qui contiennent **"EDIT"** dans leur nom, pour extraire uniquement les **2 premières colonnes** des tableaux (40% de la largeur gauche).

## 📁 Structure des fichiers

```
📦 Nouveaux fichiers créés:
├── left_columns_processor.py     # 🎯 Processeur principal
├── test_left_columns.py          # 🧪 Utilitaires de test
├── exemple_colonnes_gauches.py   # 📚 Exemples d'utilisation
└── GUIDE_COLONNES_GAUCHES.md     # 📖 Ce guide
```

## 🚀 Utilisation rapide

### 1. **Traitement d'un dossier complet**
```bash
python left_columns_processor.py "C:\MonDossier\Images"
```

### 2. **Tests interactifs**
```bash
python test_left_columns.py
```

### 3. **Tests automatiques**
```bash
# Test d'un dossier
python test_left_columns.py "C:\MonDossier"

# Test d'une image
python test_left_columns.py "image_EDIT_001.png"
```

## 🔧 Configuration

### Configuration par défaut
- **Zone extraite**: 40% de la largeur (côté gauche), 100% de la hauteur
- **Filtre images**: Contient "EDIT" dans le nom de fichier
- **Colonnes**: Maximum 2 colonnes extraites
- **Format sortie**: Excel (.xlsx)
- **Images debug**: Activées

### Personnalisation
```python
from left_columns_processor import LeftColumnsProcessor

config_custom = {
    'crop_right_ratio': 0.35,      # 35% au lieu de 40%
    'dpi': 400,                    # Résolution plus haute
    'min_conf': 50,                # Confiance OCR plus stricte
    'save_debug': True             # Images de débogage
}

processor = LeftColumnsProcessor(config_custom)
```

## 📷 Types d'images supportées

### ✅ Images traitées
- **Extensions**: .png, .jpg, .jpeg, .tiff, .tif, .bmp
- **Critère**: Nom contient "EDIT" (insensible à la casse)
- **Exemples valides**:
  - `tableau_EDIT_01.png`
  - `scan_edit_final.jpg`
  - `Document_Edit_Version2.tiff`

### ❌ Images ignorées
- Pas d'extension image
- Ne contient pas "EDIT" dans le nom
- Fichiers corrompus

## 📊 Zone d'extraction

```
Image originale (100% x 100%)
┌─────────────────────────────┐
│ 🎯 ZONE EXTRAITE │          │
│    (40% x 100%)   │   Ignoré │
│                   │          │
│ Colonne 1│Colonne2│ Col3│Col4│
│ Rang     │Hôpital │ ... │ ...│
│ 1        │CHU A   │ ... │ ...│
│ 2        │CHU B   │ ... │ ...│
└─────────────────────────────┘
```

## 📁 Structure de sortie

### Fichiers générés
```
📁 Dossier de travail/
├── 📄 colonnes_gauches_YYYYMMDD_HHMMSS.xlsx    # Données consolidées
├── 📄 rapport_colonnes_gauches.txt              # Rapport détaillé
├── 🖼️ debug_left_cols_[nom_image].png           # Images de débogage
└── 📄 processing_summary.txt                    # Résumé du traitement
```

### Format du fichier Excel
| Image Source | Colonne 1 | Colonne 2 |
|--------------|-----------|-----------|
| image_EDIT_1 | 1         | CHU Nord  |
| image_EDIT_1 | 2         | CHU Sud   |
| (ligne vide) |           |           |
| image_EDIT_2 | 1         | Clinique A|
| image_EDIT_2 | 2         | Clinique B|

## 🧪 Tests et débogage

### Mode test interactif
```bash
python test_left_columns.py
```

Menu disponible:
1. 🔍 **Scanner dossier** - Recherche images EDIT
2. 📷 **Test image unique** - Traite une image
3. ✂️ **Visualiser rogbage** - Voit la zone extraite
4. 🚀 **Traitement complet** - Processus entier
5. ❌ **Quitter**

### Images de débogage
Si `save_debug: True`, génère automatiquement:
- `debug_left_cols_[nom].png` : Zone des colonnes gauches extraite
- `test_original_[nom].png` : Image originale (pour tests)
- `test_preprocessed_[nom].png` : Après prétraitement
- `test_left_columns_[nom].png` : Zone finale extraite

## 📊 Métriques et rapports

### Informations affichées
```
✅ Traitement réussi !
📊 Images traitées: 15/18
📝 Lignes extraites: 247
📁 Fichier généré: colonnes_gauches_20250623_154530.xlsx
⚠️ Images échouées: 3
```

### Rapport détaillé
Le fichier `rapport_colonnes_gauches.txt` contient:
- Statistiques globales
- Liste des images traitées/échouées
- Nombre de lignes par image
- Temps de traitement

## 🔧 Résolution de problèmes

### ❌ "Aucune image EDIT trouvée"
**Solutions**:
- Vérifiez que vos images contiennent "EDIT" dans le nom
- Vérifiez les extensions (png, jpg, etc.)
- Testez avec `test_left_columns.py` mode scanner

### ❌ "Aucun texte extrait"
**Solutions**:
- Augmentez la résolution (`dpi: 400`)
- Diminuez la confiance OCR (`min_conf: 20`)
- Vérifiez les images de debug
- Ajustez `resize_factor: 2.5`

### ❌ "Mauvaise zone extraite"
**Solutions**:
- Modifiez `crop_right_ratio` (0.3 à 0.5)
- Utilisez le test de visualisation du rogbage
- Vérifiez les images debug

### ❌ "Colonnes mal détectées"
**Solutions**:
- Ajustez `bin_width: 30` (plus petit)
- Modifiez `peak_min_count: 15` (plus bas)
- Augmentez la résolution de l'image

## 💡 Conseils d'optimisation

### Pour de meilleurs résultats
1. **Images de qualité**: Scan en 300 DPI minimum
2. **Contraste élevé**: Texte noir sur fond blanc
3. **Tableaux alignés**: Colonnes bien définies
4. **Noms clairs**: Images avec "EDIT" visible

### Paramètres recommandés par type
```python
# Pour scans de haute qualité
config_hq = {'dpi': 400, 'resize_factor': 1.0, 'min_conf': 60}

# Pour scans de qualité moyenne
config_medium = {'dpi': 300, 'resize_factor': 2.0, 'min_conf': 40}

# Pour scans de faible qualité
config_lq = {'dpi': 200, 'resize_factor': 3.0, 'min_conf': 20}
```

## 🚀 Exemple d'utilisation complète

```python
from left_columns_processor import LeftColumnsProcessor

# 1. Configuration
config = {
    'crop_right_ratio': 0.45,  # 45% de largeur
    'save_debug': True,        # Images debug
    'min_conf': 35            # Confiance OCR
}

# 2. Création du processeur
processor = LeftColumnsProcessor(config)

# 3. Traitement
results = processor.process_directory("C:/Mes_Images_Medicales")

# 4. Résultats
if results['success']:
    print(f"✅ {results['processed_images']} images traitées")
    print(f"📁 Fichier: {results['output_file']}")
else:
    print(f"❌ Erreur: {results['error']}")
```

## 📞 Support

- 🔍 **Debug**: Activez `save_debug: True` et examinez les images
- 🧪 **Tests**: Utilisez `test_left_columns.py` pour diagnostiquer
- ⚙️ **Config**: Modifiez les paramètres selon vos images
- 📖 **Docs**: Consultez les docstrings dans le code

---

**🎉 Processeur spécialisé prêt à l'emploi pour extraction automatique des colonnes gauches !**
