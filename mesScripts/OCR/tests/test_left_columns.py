"""
Utilitaire de test pour le processeur de colonnes gauches.
Permet de tester le traitement sur un échantillon d'images.
"""

import os
import sys
from pathlib import Path
import logging

# Ajouter le répertoire courant au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from left_columns_processor import LeftColumnsProcessor
from config import setup_logging


def test_single_image(processor, image_path: str) -> None:
    """
    Test le traitement d'une seule image.
    
    Args:
        processor: Instance du processeur
        image_path: Chemin vers l'image à tester
    """
    print(f"\n🧪 TEST IMAGE UNIQUE")
    print("=" * 30)
    print(f"📷 Image: {os.path.basename(image_path)}")
    
    result = processor.process_single_image(image_path)
    
    if result['success']:
        print(f"✅ Succès !")
        print(f"📊 Lignes extraites: {result['row_count']}")
        print(f"🏛️ Colonnes détectées: {result['column_count']}")
        
        if result.get('debug_image'):
            print(f"🖼️ Image debug: {result['debug_image']}")
        
        # Afficher un échantillon des données
        if result.get('table_data'):
            print("\n📋 Échantillon des données:")
            for i, row in enumerate(result['table_data'][:5]):  # Premières 5 lignes
                print(f"   {i+1}: {row}")
            
            if len(result['table_data']) > 5:
                print(f"   ... et {len(result['table_data'])-5} autres lignes")
    else:
        print(f"❌ Échec: {result.get('error', 'Erreur inconnue')}")


def test_directory_scan(processor, root_directory: str) -> None:
    """
    Test la recherche d'images EDIT dans un dossier.
    
    Args:
        processor: Instance du processeur
        root_directory: Dossier à scanner
    """
    print(f"\n🔍 TEST RECHERCHE IMAGES EDIT")
    print("=" * 35)
    print(f"📁 Dossier: {root_directory}")
    
    edit_images = processor.find_edit_images(root_directory)
    
    print(f"🎯 Résultat: {len(edit_images)} image(s) trouvée(s)")
    
    if edit_images:
        print("\n📷 Images trouvées:")
        for i, img_path in enumerate(edit_images[:10], 1):  # Afficher max 10
            rel_path = os.path.relpath(img_path, root_directory)
            print(f"   {i}: {rel_path}")
        
        if len(edit_images) > 10:
            print(f"   ... et {len(edit_images)-10} autres images")
    else:
        print("⚠️ Aucune image EDIT trouvée")
        print("💡 Vérifiez que vos images contiennent 'EDIT' dans le nom")


def test_cropping_visualization(processor, image_path: str) -> None:
    """
    Test et visualise le rogbage des colonnes gauches.
    
    Args:
        processor: Instance du processeur
        image_path: Chemin vers l'image à tester
    """
    print(f"\n✂️ TEST ROGBAGE COLONNES GAUCHES")
    print("=" * 38)
    print(f"📷 Image: {os.path.basename(image_path)}")
    
    try:
        import cv2
        
        # Charger l'image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("❌ Impossible de charger l'image")
            return
        
        print(f"📏 Dimensions originales: {image.shape[1]}x{image.shape[0]} (L x H)")
        
        # Prétraitement
        preprocessed = processor.image_processor.preprocess(image)
        print(f"📏 Après prétraitement: {preprocessed.shape[1]}x{preprocessed.shape[0]}")
        
        # Rogbage colonnes gauches
        left_columns = processor.crop_left_columns(preprocessed)
        print(f"📏 Colonnes gauches: {left_columns.shape[1]}x{left_columns.shape[0]}")
        
        # Calculer les pourcentages
        width_ratio = (left_columns.shape[1] / preprocessed.shape[1]) * 100
        height_ratio = (left_columns.shape[0] / preprocessed.shape[0]) * 100
        
        print(f"📐 Ratio largeur: {width_ratio:.1f}%")
        print(f"📐 Ratio hauteur: {height_ratio:.1f}%")
        
        # Sauvegarder les images de comparaison
        base_name = Path(image_path).stem
        
        cv2.imwrite(f"test_original_{base_name}.png", image)
        cv2.imwrite(f"test_preprocessed_{base_name}.png", preprocessed)
        cv2.imwrite(f"test_left_columns_{base_name}.png", left_columns)
        
        print(f"💾 Images sauvegardées:")
        print(f"   - test_original_{base_name}.png")
        print(f"   - test_preprocessed_{base_name}.png")
        print(f"   - test_left_columns_{base_name}.png")
        
        print("✅ Test de rogbage terminé avec succès")
        
    except Exception as e:
        print(f"❌ Erreur durant le test: {e}")


def interactive_test():
    """Test interactif pour choisir le type de test."""
    print("🧪 TESTEUR COLONNES GAUCHES - MODE INTERACTIF")
    print("=" * 50)
    
    while True:
        print("\n🎯 Choisissez un test:")
        print("1. 🔍 Scanner un dossier (recherche images EDIT)")
        print("2. 📷 Tester une image unique")
        print("3. ✂️ Visualiser le rogbage (colonnes gauches)")
        print("4. 🚀 Traitement complet d'un dossier")
        print("5. ❌ Quitter")
        
        choice = input("\nVotre choix (1-5): ").strip()
        
        if choice == '5':
            print("👋 Au revoir !")
            break
        
        if choice == '1':
            folder = input("📁 Chemin du dossier: ").strip()
            if os.path.exists(folder):
                processor = LeftColumnsProcessor()
                test_directory_scan(processor, folder)
            else:
                print("❌ Dossier introuvable")
        
        elif choice == '2':
            image_path = input("📷 Chemin de l'image: ").strip()
            if os.path.exists(image_path):
                processor = LeftColumnsProcessor()
                test_single_image(processor, image_path)
            else:
                print("❌ Image introuvable")
        
        elif choice == '3':
            image_path = input("📷 Chemin de l'image: ").strip()
            if os.path.exists(image_path):
                processor = LeftColumnsProcessor()
                test_cropping_visualization(processor, image_path)
            else:
                print("❌ Image introuvable")
        
        elif choice == '4':
            folder = input("📁 Chemin du dossier: ").strip()
            if os.path.exists(folder):
                processor = LeftColumnsProcessor()
                print("\n🚀 Lancement du traitement complet...")
                results = processor.process_directory(folder)
                
                print(f"\n📊 RÉSULTATS FINAUX:")
                print(f"   Images trouvées: {results['total_images']}")
                print(f"   Images traitées: {results['processed_images']}")
                print(f"   Images échouées: {results['failed_images']}")
                print(f"   Lignes consolidées: {results['consolidated_rows']}")
                
                if results.get('output_file'):
                    print(f"   Fichier généré: {results['output_file']}")
            else:
                print("❌ Dossier introuvable")
        
        else:
            print("❌ Choix invalide")


def main():
    """Point d'entrée principal du testeur."""
    setup_logging()
    
    if len(sys.argv) == 1:
        # Mode interactif
        interactive_test()
    elif len(sys.argv) == 2:
        # Test avec un argument (dossier ou image)
        path = sys.argv[1]
        
        if not os.path.exists(path):
            print(f"❌ Chemin introuvable: {path}")
            return 1
        
        processor = LeftColumnsProcessor()
        
        if os.path.isdir(path):
            # Test d'un dossier
            print("🧪 TEST AUTOMATIQUE - DOSSIER")
            test_directory_scan(processor, path)
        else:
            # Test d'une image
            print("🧪 TEST AUTOMATIQUE - IMAGE")
            test_single_image(processor, path)
            test_cropping_visualization(processor, path)
    
    else:
        print("Usage:")
        print("  python test_left_columns.py                    # Mode interactif")
        print("  python test_left_columns.py <dossier>          # Test dossier")
        print("  python test_left_columns.py <image>            # Test image")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
