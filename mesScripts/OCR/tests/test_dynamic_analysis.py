#!/usr/bin/env python3
"""
Script de test pour l'analyse dynamique des colonnes.
Teste sur quelques images pour voir l'efficacité du système.
"""

import os
import sys
import logging
from pathlib import Path

# Ajouter le répertoire OCR au path
sys.path.append(os.path.dirname(__file__))

from left_columns_processor import LeftColumnsProcessor
from dynamic_column_analyzer import DynamicColumnAnalyzer
import cv2


def test_dynamic_analysis():
    """Teste l'analyse dynamique sur quelques images."""
    
    print("🔬 TEST D'ANALYSE DYNAMIQUE DES COLONNES")
    print("=" * 60)
    
    # Configuration avec analyse dynamique activée
    config = {
        'enable_dynamic_analysis': True,
        'save_debug': True,
        'min_column_width': 50,
        'max_column_width': 800,
        'text_density_threshold': 0.1
    }
    
    # Créer le processeur avec analyse dynamique
    processor = LeftColumnsProcessor(config)
    
    # Dossier d'images de test
    test_dir = r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\tri_image_automatique\images_renommees"
    
    # Trouver quelques images EDIT pour tester
    edit_images = processor.find_edit_images(test_dir)
    
    if not edit_images:
        print("❌ Aucune image EDIT trouvée pour les tests")
        return
    
    # Tester sur les 3 premières images
    test_images = edit_images[:3]
    
    print(f"📊 Test sur {len(test_images)} images:")
    for i, img_path in enumerate(test_images, 1):
        print(f"   {i}. {os.path.basename(img_path)}")
    
    print("\n" + "=" * 60)
    
    results = []
    
    for i, image_path in enumerate(test_images, 1):
        print(f"\n🔍 Test {i}/{len(test_images)}: {os.path.basename(image_path)}")
        print("-" * 40)
        
        try:
            # Charger l'image pour analyse directe
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print("❌ Impossible de charger l'image")
                continue
            
            # Analyse avec l'analyseur dynamique
            analyzer = DynamicColumnAnalyzer(config)
            analysis = analyzer.analyze_image_structure(image)
              # Afficher les résultats
            print(f"📏 Dimensions: {analysis['image_dimensions']}")
            print(f"� Lignes médicales: {analysis['medical_lines_found']}")
            print(f"� Ratio optimal: {analysis['optimal_crop_ratio']:.2f}")
            print(f"🔧 Méthode: {analysis['analysis_method']}")
            
            # Détails des lignes médicales trouvées
            if analysis.get('medical_lines'):
                print(f"\n📋 Exemples de lignes médicales détectées:")
                for j, line in enumerate(analysis['medical_lines'][:3]):
                    pattern_status = "✅" if line.get('pattern_match', False) else "🔶"
                    print(f"   {pattern_status} Ligne {j+1}: {line['text'][:60]}...")
                    if line.get('rang') and line.get('hopital'):
                        print(f"      → Rang: {line['rang']}, Hôpital: {line['hopital'][:30]}...")
            
            # Info sur la zone optimale
            if analysis.get('optimal_zone'):
                zone = analysis['optimal_zone']
                print(f"\n📊 Zone optimale:")
                print(f"   - Position: {zone['left_x']} → {zone['right_x']} px")
                print(f"   - Largeur: {zone['width']} px")
                print(f"   - Confiance: {zone['confidence']:.2f}")
                print(f"   - Méthode: {zone['method']}")
            
            # Créer la visualisation de debug
            debug_path = f"test_analysis_{i}_{Path(image_path).stem}.png"
            analyzer.create_debug_visualization(image, analysis, debug_path)
            print(f"🖼️  Debug sauvé: {debug_path}")
              # Traitement complet avec le processeur
            print("\n🚀 Traitement complet...")
            result = processor.process_single_image(image_path)
            
            if result['success']:
                print(f"✅ Succès: {result['row_count']} lignes extraites")
            else:
                print(f"❌ Échec: {result.get('error', 'Erreur inconnue')}")
            
            results.append({
                'image': os.path.basename(image_path),
                'optimal_ratio': analysis['optimal_crop_ratio'],
                'medical_lines_found': analysis['medical_lines_found'],
                'processing_success': result['success'],
                'rows_extracted': result.get('row_count', 0)
            })
            
        except Exception as e:
            print(f"❌ Erreur lors du test: {e}")
            import traceback
            traceback.print_exc()
    
    # Résumé des résultats
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    if results:
        print(f"{'Image':<25} {'Ratio':<8} {'Lignes Méd':<10} {'Succès':<7} {'Lignes':<7}")
        print("-" * 65)
        
        for result in results:
            success_icon = "✅" if result['processing_success'] else "❌"
            print(f"{result['image']:<25} {result['optimal_ratio']:<8.2f} "
                  f"{result['medical_lines_found']:<10} {success_icon:<7} {result['rows_extracted']:<7}")
        
        # Statistiques
        avg_ratio = sum(r['optimal_ratio'] for r in results) / len(results)
        success_rate = sum(1 for r in results if r['processing_success']) / len(results) * 100
        total_rows = sum(r['rows_extracted'] for r in results)
        
        print("\n📈 Statistiques:")
        print(f"   - Ratio moyen optimal: {avg_ratio:.2f}")
        print(f"   - Taux de succès: {success_rate:.1f}%")
        print(f"   - Total lignes extraites: {total_rows}")
    else:
        print("❌ Aucun résultat à afficher")
    
    print("\n🎯 Test terminé!")


if __name__ == "__main__":
    # Configurer le logging pour voir les détails
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    test_dynamic_analysis()
