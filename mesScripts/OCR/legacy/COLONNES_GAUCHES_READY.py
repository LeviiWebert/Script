"""
🎉 PROCESSEUR DE COLONNES GAUCHES - PRÊT !
==========================================

✅ Votre nouveau système spécialisé pour l'extraction des colonnes gauches est opérationnel !

📦 NOUVEAUX FICHIERS SPÉCIALISÉS
================================

1. 🎯 **left_columns_processor.py** - Processeur principal
   - Traite les images contenant "EDIT" dans un dossier
   - Extrait les 40% gauche (2 premières colonnes)
   - Consolide tous les résultats en un fichier Excel

2. 🧪 **test_left_columns.py** - Suite de tests
   - Mode interactif pour tester étape par étape
   - Visualisation du rogbage et des zones extraites
   - Tests automatiques sur images ou dossiers

3. 📚 **exemple_colonnes_gauches.py** - Exemples d'usage
   - Code d'exemple prêt à copier-coller
   - Configurations personnalisées
   - Différents cas d'utilisation

4. 📖 **GUIDE_COLONNES_GAUCHES.md** - Documentation complète
   - Guide utilisateur détaillé
   - Résolution de problèmes
   - Optimisations et conseils

🎯 FONCTIONNALITÉS SPÉCIALISÉES
==============================

✅ **Traitement par lot** : Dossiers entiers avec sous-dossiers
✅ **Filtrage intelligent** : Seulement les images "EDIT"
✅ **Zone ciblée** : 40% largeur gauche, 100% hauteur
✅ **Limite colonnes** : Maximum 2 colonnes extraites
✅ **Consolidation** : Tous résultats dans un seul fichier Excel
✅ **Debug visuel** : Images de chaque étape sauvegardées
✅ **Rapports détaillés** : Statistiques et métriques

🚀 UTILISATION IMMÉDIATE
========================

1. **Traitement automatique d'un dossier** :
   ```bash
   python left_columns_processor.py "C:\MonDossier\Images"
   ```

2. **Tests et validation** :
   ```bash
   python test_left_columns.py
   ```

3. **Avec configuration personnalisée** :
   ```python
   from left_columns_processor import LeftColumnsProcessor
   
   config = {'crop_right_ratio': 0.35, 'min_conf': 50}
   processor = LeftColumnsProcessor(config)
   results = processor.process_directory("MonDossier")
   ```

📊 EXEMPLE DE WORKFLOW TYPIQUE
==============================

1. **Préparation** :
   - Placez vos images dans un dossier
   - Assurez-vous qu'elles contiennent "EDIT" dans le nom
   - Vérifiez qu'elles sont au format PNG, JPG, etc.

2. **Test initial** :
   ```bash
   python test_left_columns.py MonDossier
   ```

3. **Traitement complet** :
   ```bash
   python left_columns_processor.py MonDossier
   ```

4. **Résultats** :
   - Fichier Excel avec toutes les données consolidées
   - Images de debug pour vérification
   - Rapport détaillé du traitement

🔧 PARAMÈTRES CLÉS
=================

| Paramètre | Valeur défaut | Description |
|-----------|---------------|-------------|
| `crop_right_ratio` | 0.40 | Largeur extraite (40%) |
| `crop_left_ratio` | 0.0 | Début extraction (gauche) |
| `min_conf` | 30 | Confiance OCR minimum |
| `save_debug` | True | Sauver images debug |
| `export_excel` | True | Export en Excel |

📁 STRUCTURE DES FICHIERS GÉNÉRÉS
=================================

```
📁 Votre dossier de travail/
├── 📄 colonnes_gauches_YYYYMMDD_HHMMSS.xlsx    # 📊 Données consolidées
├── 📄 rapport_colonnes_gauches.txt              # 📋 Rapport détaillé
├── 🖼️ debug_left_cols_[image].png               # 🔍 Images debug
└── 📄 processing_summary.txt                    # 📈 Résumé traitement
```

🎯 AVANTAGES SPÉCIFIQUES
=======================

✨ **Spécialisé** : Conçu uniquement pour les colonnes gauches
✨ **Filtrage automatique** : Trouve automatiquement les images EDIT
✨ **Traitement par lot** : Dossiers entiers en une commande
✨ **Zone précise** : Exactement 40% de largeur comme demandé
✨ **Consolidation** : Toutes les données dans un seul fichier
✨ **Validation visuelle** : Images debug pour contrôle qualité
✨ **Robuste** : Gestion d'erreurs et rapports détaillés

🔍 EXEMPLE DE RÉSULTAT
======================

Fichier Excel généré :
```
| Source Image    | Colonne 1 | Colonne 2      |
|----------------|-----------|----------------|
| scan_EDIT_01   | 1         | CHU Marseille  |
| scan_EDIT_01   | 2         | CHU Lyon       |
| scan_EDIT_01   | 3         | CHU Toulouse   |
| (ligne vide)   |           |                |
| doc_EDIT_02    | 1         | Clinique Nord  |
| doc_EDIT_02    | 2         | Clinique Sud   |
```

💡 CONSEILS D'UTILISATION
=========================

🎯 **Pour de meilleurs résultats** :
- Images de bonne qualité (300 DPI+)
- Noms contenant clairement "EDIT"
- Tableaux bien alignés et contrastés

🔧 **En cas de problème** :
- Utilisez `test_left_columns.py` pour diagnostiquer
- Vérifiez les images debug générées
- Ajustez `crop_right_ratio` si nécessaire (0.35 à 0.45)
- Modifiez `min_conf` selon la qualité des images

📞 **Support** :
- Consultez `GUIDE_COLONNES_GAUCHES.md` pour les détails
- Utilisez le mode test interactif pour explorer
- Activez `save_debug: True` pour voir les étapes

🎉 FÉLICITATIONS !
=================

Vous disposez maintenant d'un système spécialisé et automatisé pour :

✅ Traiter automatiquement tous vos dossiers d'images EDIT
✅ Extraire précisément les 2 premières colonnes (40% largeur)
✅ Consolider tous les résultats en un seul fichier Excel
✅ Obtenir des rapports détaillés et des images de contrôle

🚀 **PRÊT À UTILISER DÈS MAINTENANT !**

```bash
# Commande rapide pour commencer :
python left_columns_processor.py "VotreDossierImages"
```

Le système est optimisé pour votre cas d'usage spécifique et prêt pour la production ! 🎯
"""

print(__doc__)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎯 COMMANDES RAPIDES")
    print("="*60)
    print("➡️  python left_columns_processor.py MonDossier")
    print("➡️  python test_left_columns.py")
    print("➡️  python test_left_columns.py MonImage.png")
    print("="*60)
    print("\n💡 Consultez GUIDE_COLONNES_GAUCHES.md pour plus de détails")
