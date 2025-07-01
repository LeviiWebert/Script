# 🏥 SYSTÈME COLONNES GAUCHES - VERSION AMÉLIORÉE

## 🎯 MODIFICATIONS APPORTÉES

### ✅ **1. EXTRACTION SIMPLIFIÉE**
- **Avant**: Extraction de toutes les colonnes avec données supplémentaires
- **Maintenant**: **Seulement le rang et le nom d'hôpital** (2 colonnes)
- **Format**: `[rang] [nom hôpital]` (sans ville, département, etc.)

### ✅ **2. ORGANISATION EN SOUS-DOSSIERS**
- **Avant**: Tous les fichiers dans le dossier principal
- **Maintenant**: Structure organisée avec sous-dossiers

```
📁 Votre dossier de travail/
└── outputs/
    ├── 📊 excel_files/         # Fichiers Excel avec données
    ├── 🖼️ debug_images/        # Images de debug/contrôle
    └── 📋 reports/             # Rapports de traitement
```

## 🚀 UTILISATION

### **Commande Simple**
```bash
python left_columns_processor.py "C:\MonDossier\Images"
```

### **Test des Nouvelles Fonctionnalités**
```bash
python test_nouvelles_fonctionnalites.py
```

## 📊 EXEMPLE DE RÉSULTAT

### **Fichier Excel Généré** (`outputs/excel_files/colonnes_gauches_YYYYMMDD_HHMMSS.xlsx`)

| Image         | Rang | Nom Hôpital           |
|---------------|------|-----------------------|
| scan_EDIT_01  | 1    | CHU Marseille        |
| scan_EDIT_01  | 2    | CHU Lyon             |
| scan_EDIT_01  | 3    | Clinique Nord        |
| (ligne vide)  |      |                      |
| doc_EDIT_02   | 1    | Hôpital Saint-Joseph |
| doc_EDIT_02   | 2    | Centre Médical Sud   |

### **Images de Debug** (`outputs/debug_images/`)
- `debug_left_cols_scan_EDIT_01.png` - Zone extraite de la première image
- `debug_left_cols_doc_EDIT_02.png` - Zone extraite de la seconde image
- `debug_analysis_[image].png` - Analyse de la structure

### **Rapport Détaillé** (`outputs/reports/rapport_colonnes_gauches.txt`)
```
=== RAPPORT TRAITEMENT OCR - COLONNES GAUCHES ===

Généré le: 2025-06-23 14:30:15

STATISTIQUES DE TRAITEMENT:
------------------------------
📁 Dossier traité: C:\MonDossier\Images
📊 Images trouvées: 5
✅ Images réussies: 4
❌ Images échouées: 1
📝 Lignes extraites: 28

DÉTAILS PAR IMAGE:
--------------------
✅ scan_EDIT_01.png: 12 lignes extraites
✅ doc_EDIT_02.jpg: 8 lignes extraites
✅ image_EDIT_03.png: 6 lignes extraites
✅ test_EDIT_04.jpg: 2 lignes extraites
❌ prob_EDIT_05.png: 0 lignes extraites
    Erreur: Image trop floue pour OCR
```

## 🔧 AVANTAGES DE LA NOUVELLE VERSION

### 🎯 **Données Plus Propres**
- Élimination des données parasites (villes, départements, codes)
- Focus sur l'essentiel : **rang** et **nom d'hôpital**
- Format cohérent et exploitable

### 📁 **Meilleure Organisation**
- Séparation claire des types de fichiers
- Plus facile à naviguer et gérer
- Idéal pour archivage et partage

### 🔍 **Contrôle Qualité Amélioré**
- Images de debug dans un dossier dédié
- Rapports détaillés avec statistiques
- Traçabilité complète du traitement

### 📈 **Workflow Professionnel**
- Structure industrielle
- Facilite l'automatisation
- Prêt pour intégration dans d'autres systèmes

## 🛠️ CONFIGURATION AVANCÉE

### **Personnalisation des Paramètres**
```python
from left_columns_processor import LeftColumnsProcessor

config = {
    'crop_right_ratio': 0.35,      # 35% au lieu de 40%
    'min_conf': 40,                # Confiance OCR plus stricte
    'save_debug': True,            # Toujours activer debug
    'export_excel': True,          # Format Excel préféré
    'enable_dynamic_analysis': True # Analyse intelligente
}

processor = LeftColumnsProcessor(config)
results = processor.process_directory("MonDossier")
```

## 🎯 CAS D'USAGE TYPIQUES

### **1. Traitement de Lot Mensuel**
```bash
# Tous les documents du mois
python left_columns_processor.py "C:\Documents\2025-06\Scans"
# Résultats automatiquement organisés dans outputs/
```

### **2. Validation et Contrôle**
```bash
# Test sur quelques échantillons
python test_nouvelles_fonctionnalites.py
# Vérification des images debug dans outputs/debug_images/
```

### **3. Archivage Organisé**
```bash
# Traitement puis déplacement vers archives
python left_columns_processor.py "Nouveau_Batch"
move outputs "Archives\Batch_2025_06_23"
```

## 🔍 RÉSOLUTION DE PROBLÈMES

### **Problème**: Images pas trouvées
**Solution**: Vérifier que les noms contiennent "EDIT"

### **Problème**: Extraction incomplète
**Solution**: Ajuster `crop_right_ratio` (0.35 à 0.45)

### **Problème**: OCR de mauvaise qualité
**Solution**: Augmenter `min_conf` (30 à 50)

### **Problème**: Sous-dossiers pas créés
**Solution**: Vérifier les permissions d'écriture

## 📞 **SUPPORT ET AIDE**

1. **Test initial**: `python test_nouvelles_fonctionnalites.py`
2. **Images debug**: Consultez `outputs/debug_images/`
3. **Rapports détaillés**: Lisez `outputs/reports/`
4. **Configuration**: Ajustez les paramètres selon vos besoins

---

## 🎉 **PRÊT À UTILISER !**

Le système est maintenant **optimisé** pour :
- ✅ Extraire **seulement** rang + nom d'hôpital
- ✅ Organiser automatiquement tous les fichiers
- ✅ Fournir un contrôle qualité complet
- ✅ Gérer des volumes importants de documents

```bash
# Commande pour commencer immédiatement :
python left_columns_processor.py "VotreDossierImages"
```

**Vos données seront propres, organisées et prêtes à l'emploi ! 🎯**
