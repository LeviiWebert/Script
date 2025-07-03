# Script de test rapide pour extraction de tableaux
# Configuration de la clé API
GOOGLE_API_KEY = "AIzaSyBfQjj1pNx0yDlXUSo4tdWUe5RcE35ON6o"

from test_extraction_tableaux import TableauExtractorTester
from pathlib import Path

def test_rapide():
    """Test rapide avec des chemins pré-définis"""
    
    # 🔧 MODIFIEZ CES CHEMINS SELON VOS FICHIERS DE TEST
    image_path = r"C:\chemin\vers\votre\image_tableau.jpg"
    excel_path = r"C:\chemin\vers\votre\fichier_reference.xlsx"
    
    print("🧪 TEST RAPIDE D'EXTRACTION DE TABLEAU")
    print("="*50)
    
    # Vérifications
    if not Path(image_path).exists():
        print(f"❌ Image non trouvée: {image_path}")
        print("👉 Modifiez la variable 'image_path' dans ce script")
        return
    
    if not Path(excel_path).exists():
        print(f"❌ Excel non trouvé: {excel_path}")
        print("👉 Modifiez la variable 'excel_path' dans ce script")
        return
    
    try:
        # Création du testeur
        tester = TableauExtractorTester(GOOGLE_API_KEY)
        
        # Test
        print(f"🔍 Test de l'image: {Path(image_path).name}")
        print(f"📊 Avec la référence: {Path(excel_path).name}")
        
        results = tester.test_single_image(image_path, excel_path)
        
        print("\n✅ Test terminé!")
        print("📁 Consultez le dossier 'test_results/' pour les résultats détaillés")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    test_rapide()
