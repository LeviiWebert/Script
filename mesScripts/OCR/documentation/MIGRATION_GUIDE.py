"""
Guide de migration vers la nouvelle architecture OCR modulaire
==============================================================

🚀 NOUVELLE ARCHITECTURE DISPONIBLE !

Le code a été refactorisé en modules pour améliorer la lisibilité, la maintenabilité et la réutilisabilité.

📁 STRUCTURE DES NOUVEAUX FICHIERS
==================================

📦 Architecture modulaire :
├── main.py                    # 🎯 Point d'entrée principal (UTILISEZ CELUI-CI)
├── config.py                  # ⚙️ Configuration et paramètres
├── image_processing.py        # 🖼️ Traitement PDF et images
├── ocr_processing.py         # 📝 Extraction OCR et colonnes
├── validation.py             # ✅ Validation et métriques qualité
├── file_operations.py        # 💾 Export et gestion fichiers
├── test_utils.py             # 🧪 Utilitaires de test
├── requirements.txt          # 📋 Dépendances Python
├── ocr_config.json          # 🔧 Configuration JSON
└── README.md                 # 📖 Documentation complète

🔄 MIGRATION RAPIDE
==================

ANCIEN CODE (tesseract.py) :
```python
pdf = 'scan/1.pdf'
if os.path.isfile(pdf):
    analyze_pdf(pdf)
```

NOUVEAU CODE (main.py) :
```python
from main import MedicalDocumentProcessor

# Utilisation de base
processor = MedicalDocumentProcessor()
results = processor.process_document('scan/1.pdf')

if results['success']:
    print(f"✅ Succès ! {results['statistics']['successful_pages']} pages traitées")
else:
    print(f"❌ Erreur : {results['error']}")
```

🎨 AVANTAGES DE LA NOUVELLE ARCHITECTURE
=======================================

✅ **Modularité** : Code organisé en modules spécialisés
✅ **Lisibilité** : Classes et fonctions bien documentées
✅ **Testabilité** : Tests unitaires pour chaque composant
✅ **Configurabilité** : Configuration JSON externalisée
✅ **Robustesse** : Gestion d'erreurs améliorée
✅ **Évolutivité** : Facile d'ajouter de nouvelles fonctionnalités
✅ **Documentation** : README détaillé et commentaires complets

🚀 DÉMARRAGE RAPIDE
==================

1. **Installation des dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

2. **Traitement d'un document** :
   ```bash
   python main.py scan/1.pdf
   ```

3. **Tests des composants** :
   ```bash
   python test_utils.py scan/1.pdf
   ```

4. **Configuration personnalisée** :
   - Modifiez `ocr_config.json`
   - Ou passez un dictionnaire de config

🔧 CONFIGURATION AVANCÉE
=======================

```python
from main import MedicalDocumentProcessor

# Configuration personnalisée
config = {
    'dpi': 400,                    # Résolution plus élevée
    'n_slices': 2,                 # Découpage en 2 tranches
    'crop_right_ratio': 0.85,      # Rogbage plus agressif
    'save_debug': True,            # Images de debug
    'export_excel': True           # Export Excel
}

processor = MedicalDocumentProcessor(config)
results = processor.process_document('document.pdf')
```

🧪 TESTS ET VALIDATION
=====================

Le nouveau système inclut des outils de test complets :

```bash
# Test complet de tous les composants
python test_utils.py scan/1.pdf full_suite

# Test spécifique d'un composant
python test_utils.py scan/1.pdf ocr_extraction image.png
```

📊 MÉTRIQUES DE QUALITÉ
======================

La nouvelle architecture fournit des métriques détaillées :
- Pourcentage de précision
- Comparaison avec fichier de référence
- Détection d'outliers automatique
- Rapports d'erreurs détaillés

📖 DOCUMENTATION COMPLÈTE
========================

Consultez README.md pour :
- Guide d'installation détaillé
- Explication de tous les paramètres
- Exemples d'utilisation avancée
- Guide de dépannage
- Architecture technique

🔄 COMPATIBILITÉ
===============

L'ancien fichier `tesseract.py` reste disponible mais est maintenant déprécié.
Il est fortement recommandé de migrer vers la nouvelle architecture pour :
- Bénéficier des améliorations
- Faciliter la maintenance
- Accéder aux nouvelles fonctionnalités

💡 SUPPORT
=========

En cas de problème avec la migration :
1. Consultez README.md
2. Utilisez test_utils.py pour diagnostiquer
3. Vérifiez la configuration dans ocr_config.json
4. Activez save_debug pour voir les images intermédiaires

🎯 COMMENCEZ MAINTENANT !
========================

```bash
# Migration en une ligne :
python main.py scan/1.pdf
```

Profitez de la nouvelle architecture modulaire ! 🚀
"""

if __name__ == "__main__":
    print(__doc__)
