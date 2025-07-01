#!/usr/bin/env python3
"""
Script de test pour vérifier la détection de la structure des fichiers
Teste la fonction find_image_pairs() sans faire appel à Gemini
"""

import os
import re
from pathlib import Path

# Import de la configuration
try:
    from config_banner_analyzer import IMAGES_BASE_PATH
except ImportError:
    print("⚠️  Fichier config_banner_analyzer.py non trouvé")
    IMAGES_BASE_PATH = r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\tri_image_script\images_backup_20250625_145044"

def test_find_image_pairs(images_base_path):
    """
    Teste la détection des paires image originale / tableau découpé
    Structure attendue : img*.jpg (originales) + DOSSIER_*.jpg (tableaux découpés)
    
    Args:
        images_base_path (str): Chemin vers le dossier d'images
        
    Returns:
        list: Liste de tuples (dossier, image_originale, image_tableau, folder_path)
    """
    pairs = []
    
    print(f"🔍 Analyse de la structure dans: {images_base_path}")
    print("="*70)
    
    if not os.path.exists(images_base_path):
        print(f"❌ Chemin non trouvé: {images_base_path}")
        return pairs
    
    for root, dirs, files in os.walk(images_base_path):
        folder_name = os.path.basename(root)
        
        # Skip le dossier racine
        if root == images_base_path:
            print(f"📁 Dossier racine: {root}")
            print(f"   Sous-dossiers trouvés: {dirs}")
            continue
        
        print(f"\n📁 Analyse du dossier: {folder_name}")
        print(f"   Chemin complet: {root}")
        
        # Séparer les images originales des images de tableaux
        original_images = [f for f in files if f.lower().startswith('img') and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Chercher les images de tableaux avec pattern DOSSIER_*.jpg
        table_images = []
        other_images = []
        
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Vérifier si c'est un fichier DOSSIER_X.jpg
                if re.match(r'^dossier_\d+\.jpg$', f.lower()):
                    table_images.append(f)
                elif not f.lower().startswith('img'):
                    other_images.append(f)
        
        print(f"   📷 Images originales (img*): {len(original_images)}")
        for img in original_images:
            print(f"      - {img}")
        
        print(f"   📋 Images tableaux (DOSSIER_*): {len(table_images)}")
        for img in table_images:
            print(f"      - {img}")
        
        if other_images:
            print(f"   ❓ Autres images: {len(other_images)}")
            for img in other_images:
                print(f"      - {img}")
        
        # Associer chaque image de tableau à une image originale
        if table_images and original_images:
            for table_img in table_images:
                # Prendre la première image originale disponible
                matching_original = original_images[0]
                
                pairs.append((folder_name, matching_original, table_img, root))
                print(f"   ✅ Paire créée: {matching_original} ↔ {table_img}")
        elif table_images and not original_images:
            print(f"   ⚠️  {len(table_images)} tableau(x) trouvé(s) mais aucune image originale")
        elif original_images and not table_images:
            print(f"   ⚠️  {len(original_images)} image(s) originale(s) trouvée(s) mais aucun tableau")
        else:
            print(f"   ℹ️  Dossier vide ou sans images compatibles")
    
    print("\n" + "="*70)
    print(f"🎯 RÉSUMÉ: {len(pairs)} paires d'images trouvées au total")
    
    if pairs:
        print("\n📋 LISTE DES PAIRES:")
        for i, (folder, orig, table, path) in enumerate(pairs, 1):
            print(f"{i:2d}. {folder:<20} | {orig:<25} ↔ {table}")
    
    return pairs

def test_file_existence(pairs):
    """Vérifie que tous les fichiers détectés existent bien"""
    print("\n🔍 VÉRIFICATION DE L'EXISTENCE DES FICHIERS:")
    print("="*70)
    
    missing_files = []
    
    for folder, orig, table, path in pairs:
        orig_path = os.path.join(path, orig)
        table_path = os.path.join(path, table)
        
        print(f"\n📁 {folder}:")
        
        if os.path.exists(orig_path):
            size_orig = os.path.getsize(orig_path) / (1024*1024)  # MB
            print(f"   ✅ {orig} ({size_orig:.1f} MB)")
        else:
            print(f"   ❌ {orig} - FICHIER MANQUANT")
            missing_files.append(orig_path)
        
        if os.path.exists(table_path):
            size_table = os.path.getsize(table_path) / (1024*1024)  # MB
            print(f"   ✅ {table} ({size_table:.1f} MB)")
        else:
            print(f"   ❌ {table} - FICHIER MANQUANT")
            missing_files.append(table_path)
    
    if missing_files:
        print(f"\n❌ {len(missing_files)} fichier(s) manquant(s) détecté(s)")
    else:
        print(f"\n✅ Tous les fichiers sont présents")
    
    return missing_files

def main():
    """Fonction principale de test"""
    print("🧪 TEST DE DÉTECTION DE STRUCTURE D'IMAGES")
    print("="*70)
    
    # Tester la détection des paires
    pairs = test_find_image_pairs(IMAGES_BASE_PATH)
    
    if pairs:
        # Vérifier l'existence des fichiers
        missing = test_file_existence(pairs)
        
        print(f"\n🎯 RÉSULTAT FINAL:")
        print(f"   • {len(pairs)} paires détectées")
        print(f"   • {len(missing)} fichiers manquants")
        
        if len(missing) == 0:
            print("   ✅ Structure valide - prête pour l'analyse")
        else:
            print("   ⚠️  Problèmes détectés - vérifiez les fichiers manquants")
    else:
        print("\n❌ Aucune paire valide détectée")
        print("💡 Vérifiez:")
        print("   • Le chemin dans config_banner_analyzer.py")
        print("   • La présence d'images img*.jpg et DOSSIER_*.jpg")
        print("   • La structure des dossiers")

if __name__ == "__main__":
    main()
