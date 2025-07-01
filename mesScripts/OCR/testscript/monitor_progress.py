#!/usr/bin/env python3
"""
Moniteur de progression pour le traitement des images EDIT.
Affiche l'avancement en temps réel.
"""

import time
import os
import glob
from datetime import datetime

def monitor_progress():
    """Surveille les fichiers générés pour estimer la progression."""
    
    # Dossier de base
    base_dir = r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\tri_image_automatique\images_renommees"
    ocr_dir = os.getcwd()
    
    print("🔍 MONITEUR DE PROGRESSION - TRAITEMENT IMAGES EDIT")
    print("=" * 60)
    print(f"📁 Dossier surveillé: {base_dir}")
    print(f"📊 Dossier OCR: {ocr_dir}")
    print("=" * 60)
    
    start_time = datetime.now()
    
    while True:
        try:
            # Compter les images debug générées (indicateur de progression)
            debug_files = glob.glob(os.path.join(ocr_dir, "debug_left_cols_*.png"))
            
            # Chercher les fichiers Excel récents
            excel_files = glob.glob(os.path.join(ocr_dir, "colonnes_gauches_*.xlsx"))
            
            # Chercher les rapports
            report_files = glob.glob(os.path.join(ocr_dir, "rapport_colonnes_gauches*.txt"))
            
            current_time = datetime.now()
            elapsed = current_time - start_time
            
            # Affichage de l'état
            print(f"\r⏱️  {elapsed.total_seconds():.0f}s | "
                  f"🖼️  Debug: {len(debug_files)} | "
                  f"📊 Excel: {len(excel_files)} | "
                  f"📝 Rapports: {len(report_files)}", end="", flush=True)
            
            # Si on a un rapport récent, on peut s'arrêter
            if report_files:
                latest_report = max(report_files, key=os.path.getctime)
                report_time = datetime.fromtimestamp(os.path.getctime(latest_report))
                
                # Si le rapport est très récent (moins de 30s), le traitement est probablement fini
                if (current_time - report_time).total_seconds() < 30:
                    print(f"\n✅ Traitement probablement terminé!")
                    print(f"📄 Dernier rapport: {os.path.basename(latest_report)}")
                    break
            
            time.sleep(2)  # Attendre 2 secondes
            
        except KeyboardInterrupt:
            print("\n⏹️  Surveillance interrompue par l'utilisateur")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            break
    
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ FINAL:")
    print(f"   - Fichiers debug: {len(debug_files)}")
    print(f"   - Fichiers Excel: {len(excel_files)}")
    print(f"   - Rapports: {len(report_files)}")
    
    if excel_files:
        latest_excel = max(excel_files, key=os.path.getctime)
        print(f"   - Dernier Excel: {os.path.basename(latest_excel)}")

if __name__ == "__main__":
    monitor_progress()
