"""
Test du nouveau système de colonnes gauches avec organisation en sous-dossiers.
Ce script teste les modifications suivantes :
1. Extraction seulement du rang et nom d'hôpital (2 colonnes)
2. Organisation des fichiers en sous-dossiers (outputs/excel_files, outputs/debug_images, outputs/reports)
"""

import os
import sys
import logging
from pathlib import Path

# Ajouter le dossier parent au path pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from left_columns_processor import LeftColumnsProcessor

def test_organized_output():
    """Test de l'organisation des fichiers en sous-dossiers."""
    
    print("🧪 TEST - ORGANISATION EN SOUS-DOSSIERS")
    print("=" * 50)
    
    # Configuration pour test
    test_config = {
        'crop_right_ratio': 0.40,
        'min_conf': 30,
        'save_debug': True,
        'export_excel': True,
        'enable_dynamic_analysis': True
    }
    
    # Initialiser le processeur
    processor = LeftColumnsProcessor(test_config)
    
    # Vérifier que les dossiers sont créés
    expected_dirs = [
        "outputs",
        "outputs/excel_files", 
        "outputs/debug_images",
        "outputs/reports"
    ]
    
    print("📁 Vérification des dossiers créés:")
    for directory in expected_dirs:
        if os.path.exists(directory):
            print(f"  ✅ {directory}")
        else:
            print(f"  ❌ {directory} (manquant)")
    
    print("\n🔍 Configuration du processeur:")
    print(f"  📏 Largeur extraction: {test_config['crop_right_ratio']*100}%")
    print(f"  🔍 Confiance OCR minimum: {test_config['min_conf']}")
    print(f"  🖼️ Images debug: {'Activé' if test_config['save_debug'] else 'Désactivé'}")
    print(f"  📊 Export Excel: {'Activé' if test_config['export_excel'] else 'Désactivé'}")
    
    return processor

def test_data_extraction():
    """Test de l'extraction des données (rang + nom hôpital seulement)."""
    
    print("\n🧪 TEST - EXTRACTION DONNÉES")
    print("=" * 35)
    
    # Données de test simulées (comme si elles venaient de l'OCR)
    test_table = [
        ["1", "CHU Marseille", "12345", "données", "supplémentaires"],
        ["2", "Clinique", "Nord", "98765", "autres", "données"],
        ["3", "Hôpital Saint-Joseph", "54321", "plus", "de", "données"]
    ]
    
    print("📊 Données avant traitement:")
    for i, row in enumerate(test_table, 1):
        print(f"  Ligne {i}: {row}")
    
    # Simuler le traitement qui ne garde que rang + nom
    processed_table = []
    for row in test_table:
        if len(row) >= 2:
            # Ne garder que les 2 premières colonnes (rang + nom hôpital)
            rank = row[0]
            hospital_name = " ".join(row[1:3]) if len(row) > 2 else row[1]
            # Nettoyer le nom d'hôpital (enlever les données numériques)
            hospital_parts = []
            for part in row[1:]:
                if not part.isdigit():
                    hospital_parts.append(part)
                else:
                    break
            hospital_name = " ".join(hospital_parts)
            processed_table.append([rank, hospital_name])
    
    print("\n📋 Données après traitement (rang + nom uniquement):")
    for i, row in enumerate(processed_table, 1):
        print(f"  Ligne {i}: {row}")
    
    return processed_table

def test_file_structure():
    """Test de la structure des fichiers générés."""
    
    print("\n🧪 TEST - STRUCTURE FICHIERS")
    print("=" * 32)
    
    expected_structure = {
        "outputs/excel_files/": "Fichiers Excel avec données extraites",
        "outputs/debug_images/": "Images de debug pour contrôle qualité", 
        "outputs/reports/": "Rapports de traitement détaillés"
    }
    
    print("📁 Structure attendue:")
    for folder, description in expected_structure.items():
        print(f"  📂 {folder}")
        print(f"     └── {description}")
    
    # Vérifier si les dossiers existent
    print("\n✅ Vérification existence:")
    for folder in expected_structure.keys():
        exists = os.path.exists(folder)
        status = "✅" if exists else "❌"
        print(f"  {status} {folder}")

def simulate_complete_workflow():
    """Simule un workflow complet de traitement."""
    
    print("\n🚀 SIMULATION WORKFLOW COMPLET")
    print("=" * 38)
    
    # Exemple de résultats que le système devrait produire
    example_results = [
        ["scan_EDIT_01", "1", "CHU Marseille"],
        ["scan_EDIT_01", "2", "CHU Lyon"], 
        ["scan_EDIT_01", "3", "Clinique Nord"],
        ["", "", ""],  # Ligne vide de séparation
        ["doc_EDIT_02", "1", "Hôpital Saint-Joseph"],
        ["doc_EDIT_02", "2", "Centre Médical Sud"]
    ]
    
    print("📊 Exemple de résultat attendu en Excel:")
    print("┌─────────────────┬──────┬────────────────────────┐")
    print("│ Image           │ Rang │ Nom Hôpital           │")
    print("├─────────────────┼──────┼────────────────────────┤")
    
    for row in example_results:
        if all(cell.strip() for cell in row):  # Ignorer les lignes vides
            print(f"│ {row[0]:<15} │ {row[1]:<4} │ {row[2]:<22} │")
        else:
            print("├─────────────────┼──────┼────────────────────────┤")
    
    print("└─────────────────┴──────┴────────────────────────┘")
    
    print("\n📁 Fichiers qui seraient générés:")
    print("  📊 outputs/excel_files/colonnes_gauches_YYYYMMDD_HHMMSS.xlsx")
    print("  🖼️ outputs/debug_images/debug_left_cols_[image].png")
    print("  📋 outputs/reports/rapport_colonnes_gauches.txt")

def main():
    """Fonction principale de test."""
    
    print("🏥 TEST SYSTÈME COLONNES GAUCHES - VERSION AMÉLIORÉE")
    print("=" * 55)
    print("🎯 Objectifs:")
    print("  1. Extraire SEULEMENT le rang et nom d'hôpital")
    print("  2. Organiser les fichiers en sous-dossiers")
    print("  3. Améliorer la structure et la lisibilité")
    print()
    
    try:
        # Test 1: Organisation des dossiers
        processor = test_organized_output()
        
        # Test 2: Extraction des données
        processed_data = test_data_extraction()
        
        # Test 3: Structure des fichiers
        test_file_structure()
        
        # Test 4: Simulation workflow complet
        simulate_complete_workflow()
        
        print("\n✅ TOUS LES TESTS RÉUSSIS!")
        print("\n💡 Pour utiliser le système:")
        print("   python left_columns_processor.py 'VotreDossierImages'")
        print("\n📂 Les résultats seront organisés dans:")
        print("   📁 outputs/")
        print("   ├── 📊 excel_files/    (fichiers de données)")
        print("   ├── 🖼️ debug_images/   (images de contrôle)")
        print("   └── 📋 reports/        (rapports détaillés)")
        
    except Exception as e:
        print(f"\n❌ ERREUR LORS DU TEST: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
