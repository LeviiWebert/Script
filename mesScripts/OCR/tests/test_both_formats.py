#!/usr/bin/env python3
"""
Script de test pour vérifier la détection des deux formats :
- DOSSIER_X.jpg (avec numéro)
- DOSSIER.jpg (sans numéro)
"""
import os
import re
from pathlib import Path

def test_format_detection():
    """
    Simule la logique de détection des images de tableaux
    """
    
    # Test avec différents types de fichiers
    test_files = [
        # Format avec numéro (supporté avant)
        "UTUMEURSDUCERVEAUDELENFANTETDELADOLESCENT_1.jpg",
        "UTUMEURSDUCERVEAUDELENFANTETDELADOLESCENT_2.jpg",
        "CANCER_PROSTATE_1.jpg",
        
        # Format sans numéro (nouvelle fonctionnalité)
        "UTUMEURSDUCERVEAUDELENFANTETDELADOLESCENT.jpg",
        "CANCER_PROSTATE.jpg",
        "ABLATIONDESVARICES.jpg",
        
        # Images originales (ne doivent pas être détectées comme tableaux)
        "img00001.jpg",
        "img00002.jpg",
        
        # Autres formats qui ne doivent pas être détectés
        "readme.txt",
        "config.py"
    ]
    
    print("🔍 Test de détection des formats d'images de tableaux")
    print("=" * 60)
    
    table_images = []
    original_images = []
    
    for f in test_files:
        print(f"📄 Fichier: {f}")
        
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Images originales
            if f.lower().startswith('img'):
                original_images.append(f)
                print(f"   ✅ Détecté comme IMAGE ORIGINALE")
                
            # Pattern 1: DOSSIER_X.jpg (avec numéro)
            elif re.match(r'^.+_\d+\.jpg$', f.lower()):
                table_images.append(f)
                print(f"   ✅ Détecté comme TABLEAU (format avec numéro)")
                
            # Pattern 2: DOSSIER.jpg (sans numéro, mais pas img*.jpg)
            elif not f.lower().startswith('img') and f.lower().endswith('.jpg'):
                # Vérifier que ce n'est pas une image originale
                if not re.match(r'^img\d+\.jpg$', f.lower()):
                    table_images.append(f)
                    print(f"   ✅ Détecté comme TABLEAU (format sans numéro)")
                else:
                    print(f"   ❌ Ignoré (image originale)")
            else:
                print(f"   ❌ Ignoré (ne correspond à aucun pattern)")
        else:
            print(f"   ❌ Ignoré (pas une image)")
        
        print()
    
    print("📊 RÉSULTATS FINAUX")
    print("=" * 60)
    print(f"🖼️  Images originales trouvées ({len(original_images)}):")
    for img in original_images:
        print(f"   - {img}")
    
    print(f"\n📋 Images de tableaux trouvées ({len(table_images)}):")
    for img in table_images:
        print(f"   - {img}")
    
    print(f"\n🎯 Total de paires possibles: {min(len(original_images), len(table_images))}")

if __name__ == "__main__":
    test_format_detection()
