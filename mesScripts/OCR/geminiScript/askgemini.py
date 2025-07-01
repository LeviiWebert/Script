# Configuration de la clé API (recommandé)
GOOGLE_API_KEY="AIzaSyBfQjj1pNx0yDlXUSo4tdWUe5RcE35ON6o"

import google.generativeai as genai
from pathlib import Path
import re
from datetime import datetime

class GeminiImageAnalyzer:
    def __init__(self, api_key, base_folder_path, output_file="resultats_analyse.txt"):
        """
        Initialise l'analyseur d'images Gemini
        
        Args:
            api_key (str): Clé API Gemini
            base_folder_path (str): Chemin vers le dossier 'images'
            output_file (str): Nom du fichier de sortie
        """
        self.api_key = api_key
        self.base_folder_path = Path(base_folder_path)
        self.output_file = output_file
        self.model = None
        self.setup_gemini()
        
    def setup_gemini(self):
        """Configure l'API Gemini"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-pro')
            print("✅ API Gemini configurée avec succès")
        except Exception as e:
            print(f"❌ Erreur lors de la configuration de Gemini: {e}")
            raise
    
    def is_original_image(self, filename):
        """
        Détecte si c'est une image originale basée sur le pattern img00002 (25)
        
        Args:
            filename (str): Nom du fichier
            
        Returns:
            bool: True si c'est une image originale
        """
        # Pattern pour détecter les images originales : img suivi de chiffres et parenthèses
        pattern = r'^img\d+.*\(\d+\)'
        return bool(re.match(pattern, filename.lower()))
    
    def analyze_original_image(self, image_path):
        """
        Analyse une image originale pour extraire les titres de classement
        
        Args:
            image_path (Path): Chemin vers l'image
            
        Returns:
            str: Réponse de Gemini
        """
        prompt = """
        Analyse cette image et identifie s'il s'agit d'une page de classement ou d'index.
        Si c'est le cas, extrais TOUS les titres de classement visibles dans les nabbnières colorés.
        Format de réponse souhaité :
        TYPE: Image originale - Page de classement
        TITRES EXTRAITS:
        - [Titre 1]
        - [Titre 2]
        - [etc...]
        
        Si ce n'est pas une page de classement, réponds simplement :
        TYPE: Image originale - Autre contenu
        DESCRIPTION: [Brève description du contenu]
        """
        
        return self.send_image_to_gemini(image_path, prompt)
    
    def analyze_table_image(self, image_path):
        """
        Analyse une image de tableau pour extraire la liste des hôpitaux
        
        Args:
            image_path (Path): Chemin vers l'image
            
        Returns:
            str: Réponse de Gemini
        """
        prompt = """
        Analyse cette image qui contient un tableau de classement d'hopitaux ou de clinique.
        Extrais la liste complète de tous les hôpitaux ou/et les cliniques mentionnés dans ce tableau.
        
        Format de réponse souhaité :
        CLASSEMENT: Tableau d'hôpitaux(ou cliniques)
        HÔPITAUX IDENTIFIÉS:
        - [libelle hôpital 1]
        - [libelle hôpital 2]
        - [etc...]
        
        Si aucun hôpital n'est visible ou lisible, réponds :
        TYPE: Image tableau découpé
        STATUT: Aucun hôpital identifiable
        """
        
        return self.send_image_to_gemini(image_path, prompt)
    
    def send_image_to_gemini(self, image_path, prompt):
        """
        Envoie une image à Gemini avec un prompt
        
        Args:
            image_path (Path): Chemin vers l'image
            prompt (str): Prompt pour l'analyse
            
        Returns:
            str: Réponse de Gemini ou message d'erreur
        """
        try:
            # Lecture de l'image
            with open(image_path, 'rb') as img_file:
                image_data = img_file.read()
            
            # Préparation de l'image pour Gemini
            image_part = {
                "mime_type": "image/jpeg",  # Assumant des images JPEG
                "data": image_data
            }
            
            # Envoi à Gemini
            response = self.model.generate_content([prompt, image_part])
            return response.text
            
        except Exception as e:
            error_msg = f"❌ Erreur lors de l'analyse de {image_path.name}: {str(e)}"
            print(error_msg)
            return error_msg
    
    def process_folder(self):
        """
        Parcourt tous les dossiers et sous-dossiers pour analyser les images
        """
        if not self.base_folder_path.exists():
            print(f"❌ Le dossier {self.base_folder_path} n'existe pas")
            return
        
        # Création du fichier de sortie
        with open(self.output_file, 'w', encoding='utf-8') as output:
            output.write(f"=== ANALYSE D'IMAGES AVEC GEMINI ===\n")
            output.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            output.write(f"Dossier analysé: {self.base_folder_path}\n")
            output.write("="*60 + "\n\n")
        
        # Extensions d'images supportées
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        
        # Compteurs
        total_images = 0
        images_originales = 0
        images_tableaux = 0
        
        # Parcours des sous-dossiers NOMCLASSEMENT
        for classement_folder in self.base_folder_path.iterdir():
            if not classement_folder.is_dir():
                continue
            
            print(f"\n📁 Traitement du dossier: {classement_folder.name}")
            
            # Écriture du header dans le fichier
            with open(self.output_file, 'a', encoding='utf-8') as output:
                output.write(f"\n{'='*60}\n")
                output.write(f"DOSSIER: {classement_folder.name}\n")
                output.write(f"{'='*60}\n\n")
            
            # Parcours des images dans le dossier
            images_in_folder = []
            for file_path in classement_folder.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    images_in_folder.append(file_path)
            
            # Tri des images par nom
            images_in_folder.sort(key=lambda x: x.name)
            
            for image_path in images_in_folder:
                total_images += 1
                print(f"  🖼️  Analyse de: {image_path.name}")
                
                # Détermination du type d'image et analyse appropriée
                if self.is_original_image(image_path.name):
                    images_originales += 1
                    print(f"    ➡️  Image originale détectée")
                    response = self.analyze_original_image(image_path)
                else:
                    images_tableaux += 1
                    print(f"    ➡️  Image tableau détectée")
                    response = self.analyze_table_image(image_path)
                
                # Écriture du résultat dans le fichier
                with open(self.output_file, 'a', encoding='utf-8') as output:
                    output.write(f"FICHIER: {image_path.name}\n")
                    output.write(f"CHEMIN: {image_path}\n")
                    output.write(f"RÉPONSE GEMINI:\n{response}\n")
                    output.write("-" * 40 + "\n\n")
                
                # Respect des limites de taux (15 requêtes/minute pour Flash)
                #time.sleep(4.5)  # Attente de 4.5 secondes entre chaque requête
        
        # Statistiques finales
        stats = f"""
        === STATISTIQUES FINALES ===
        Total d'images analysées: {total_images}
        Images originales: {images_originales}
        Images de tableaux: {images_tableaux}
        Fichier de résultats: {self.output_file}
        """
        print(stats)
        
        with open(self.output_file, 'a', encoding='utf-8') as output:
            output.write(stats)

def main():
    """Fonction principale"""
    # Configuration
    API_KEY = GOOGLE_API_KEY 
    DOSSIER_IMAGES = r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\tri_image_script\images"  # Chemin vers votre dossier images
    FICHIER_SORTIE = "resultats_analyse_gemini.txt"
    
    # Vérification de la clé API
    if API_KEY != GOOGLE_API_KEY or not API_KEY:
        print("❌ Veuillez configurer votre clé API Gemini:")
        print("   - Soit dans la variable d'environnement GOOGLE_API_KEY")
        print("   - Soit en modifiant directement la variable API_KEY dans le code")
        return
    
    try:
        # Création et lancement de l'analyseur
        analyzer = GeminiImageAnalyzer(
            api_key=API_KEY,
            base_folder_path=DOSSIER_IMAGES,
            output_file=FICHIER_SORTIE
        )
        
        print("🚀 Début de l'analyse des images...")
        analyzer.process_folder()
        print("✅ Analyse terminée!")
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")

if __name__ == "__main__":
    main()