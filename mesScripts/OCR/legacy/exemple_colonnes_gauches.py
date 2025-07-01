"""
Exemple d'utilisation du processeur de colonnes gauches.
"""

from left_columns_processor import LeftColumnsProcessor

def exemple_simple():
    """Exemple simple d'utilisation."""
    
    # 1. Créer le processeur
    processor = LeftColumnsProcessor()
    
    # 2. Traiter un dossier complet
    dossier_images = "C:/MonDossier/Images"  # Remplacez par votre chemin
    
    print("🏥 TRAITEMENT COLONNES GAUCHES")
    print("=" * 35)
    print(f"📁 Dossier: {dossier_images}")
    print("🔍 Filtre: images contenant 'EDIT'")
    print("📏 Zone: 40% largeur gauche, 100% hauteur")
    
    # 3. Lancer le traitement
    resultats = processor.process_directory(dossier_images)
    
    # 4. Afficher les résultats
    if resultats['success']:
        print(f"\n✅ Traitement réussi !")
        print(f"📊 {resultats['processed_images']}/{resultats['total_images']} images traitées")
        print(f"📝 {resultats['consolidated_rows']} lignes extraites au total")
        
        if resultats['output_file']:
            print(f"📁 Fichier Excel généré: {resultats['output_file']}")
    else:
        print(f"\n❌ Échec: {resultats.get('error')}")


def exemple_avec_configuration():
    """Exemple avec configuration personnalisée."""
    
    # Configuration personnalisée
    config_personnalisee = {
        'dpi': 400,                    # Résolution plus élevée
        'resize_factor': 1.5,          # Moins de redimensionnement
        'min_conf': 40,                # Confiance OCR plus élevée
        'crop_right_ratio': 0.35,      # Seulement 35% de largeur
        'save_debug': True,            # Sauvegarder images debug
        'export_excel': True           # Export en Excel
    }
    
    # Créer le processeur avec configuration
    processor = LeftColumnsProcessor(config_personnalisee)
    
    # Traitement
    resultats = processor.process_directory("MonDossier")
    
    return resultats


def exemple_image_unique():
    """Exemple de traitement d'une seule image."""
    
    processor = LeftColumnsProcessor()
    
    # Traiter une image spécifique
    resultat = processor.process_single_image("image_EDIT_001.png")
    
    if resultat['success']:
        print(f"✅ Image traitée avec succès")
        print(f"📊 {resultat['row_count']} lignes extraites")
        print(f"🏛️ {resultat['column_count']} colonnes détectées")
        
        # Accéder aux données
        tableau = resultat['table_data']
        for i, ligne in enumerate(tableau):
            print(f"Ligne {i+1}: {ligne}")
    
    return resultat


if __name__ == "__main__":
    print("📚 EXEMPLES D'UTILISATION")
    print("=" * 30)
    
    # Remplacez par vos vrais chemins
    print("💡 Modifiez les chemins dans ce fichier avant exécution")
    print("   - Ligne 13: dossier_images = 'VOTRE_CHEMIN'")
    print("   - Ligne 53: processor.process_directory('VOTRE_CHEMIN')")
    print("   - Ligne 61: processor.process_single_image('VOTRE_IMAGE')")
