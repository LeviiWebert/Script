import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import google.generativeai as genai
from pathlib import Path
from datetime import datetime
import shutil
import re

# Import de la configuration spécifique
try:
    from config_specific_images import *
except ImportError:
    import os
    print("⚠️  Fichier config_specific_images.py non trouvé, utilisation des valeurs par défaut")
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
    IMAGES_BASE_PATH = r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\tri_image_script\images"
    BANNER_WIDTH = 2290
    BANNER_HEIGHT = 86
    DETECTION_THRESHOLD = 0.8
    MARGIN_ABOVE_TABLE = 50
    CACHE_DIR = os.path.join(os.path.expanduser("~"), "temp_banner_analyzer_cache_specific")
    SPECIFIC_FOLDERS_TO_PROCESS = []

class TitleBannerAnalyzerSpecific:
    def __init__(self, images_base_path, output_excel="titres_classements_specific_analysis.xlsx"):
        """
        Analyseur de bannières de titres pour images spécifiques
        
        Args:
            images_base_path (str): Chemin vers le dossier images contenant les sous-dossiers
            output_excel (str): Fichier Excel de sortie
        """
        self.images_base_path = images_base_path
        self.output_excel = output_excel
        self.cache_dir = CACHE_DIR
        self.results = []
        self.specific_folders = SPECIFIC_FOLDERS_TO_PROCESS
        
        # Configuration depuis le fichier config
        self.BANNER_WIDTH = BANNER_WIDTH
        self.BANNER_HEIGHT = BANNER_HEIGHT
        self.DETECTION_THRESHOLD = DETECTION_THRESHOLD
        self.MARGIN_ABOVE_TABLE = MARGIN_ABOVE_TABLE
        
        # Configuration Gemini
        self.setup_gemini()
        
        # Création du dossier cache
        self.setup_cache_directory()
    
    def setup_gemini(self):
        """Configure l'API Gemini"""
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-2.5-pro')
            print("✅ Gemini API configurée")
        except Exception as e:
            print(f"❌ Erreur configuration Gemini: {e}")
            print("💡 Vérifiez votre clé API dans config_specific_images.py")
            self.model = None
    
    def setup_cache_directory(self):
        """Crée le dossier cache temporaire"""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir)
        print(f"📁 Dossier cache créé: {self.cache_dir}")
    
    def find_image_pairs(self):
        """
        Trouve toutes les paires image originale / image tableau découpé
        SEULEMENT pour les dossiers spécifiés dans SPECIFIC_FOLDERS_TO_PROCESS
        
        Returns:
            list: Liste de tuples (dossier, image_originale, image_tableau, folder_path)
        """
        pairs = []
        
        print(f"🎯 Traitement spécifique de {len(self.specific_folders)} dossiers")
        print(f"📂 Dossiers ciblés: {self.specific_folders}")
        
        for root, dirs, files in os.walk(self.images_base_path):
            folder_name = os.path.basename(root)
            
            # Skip le dossier racine
            if root == self.images_base_path:
                continue
            
            # FILTRE : Ne traiter que les dossiers spécifiés
            if folder_name not in self.specific_folders:
                print(f"⏭️  Dossier ignoré (non dans la liste): {folder_name}")
                continue
            
            # Séparer les images originales des images de tableaux
            original_images = [f for f in files if f.lower().startswith('img') and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Chercher les images de tableaux avec pattern DOSSIER_*.jpg ou DOSSIER.jpg
            table_images = []
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Pattern 1: DOSSIER_X.jpg (avec numéro)
                    if re.match(r'^.+_\d+\.jpg$', f.lower()):
                        table_images.append(f)
                    # Pattern 2: DOSSIER.jpg (sans numéro, mais pas img*.jpg)
                    elif not f.lower().startswith('img') and f.lower().endswith('.jpg'):
                        # Vérifier que ce n'est pas une image originale
                        if not re.match(r'^img\d+\.jpg$', f.lower()):
                            table_images.append(f)
            
            print(f"📁 Dossier CIBLÉ: {folder_name}")
            print(f"   Images originales trouvées: {original_images}")
            print(f"   Images tableaux trouvées: {table_images}")
            
            # Associer chaque image de tableau à une image originale
            for table_img in table_images:
                if original_images:
                    # Prendre la première image originale disponible
                    matching_original = original_images[0]
                    
                    pairs.append((folder_name, matching_original, table_img, root))
                    print(f"📌 Paire créée: {folder_name} | {matching_original} ↔ {table_img}")
                else:
                    print(f"⚠️  Tableau {table_img} trouvé mais aucune image originale dans {folder_name}")
        
        print(f"🔍 {len(pairs)} paires d'images SPÉCIFIQUES trouvées au total")
        return pairs
    
    def detect_table_position(self, original_image_path, table_image_path):
        """
        Détecte la position du tableau découpé dans l'image originale
        
        Args:
            original_image_path (str): Chemin vers l'image originale
            table_image_path (str): Chemin vers l'image de tableau découpé (template)
            
        Returns:
            tuple: (x, y, width, height) ou None si pas trouvé
        """
        try:
            # Charger les images
            original = cv2.imread(original_image_path)
            template = cv2.imread(table_image_path)
            
            if original is None or template is None:
                print(f"❌ Impossible de charger les images: {original_image_path} ou {table_image_path}")
                return None
            
            # Conversion en niveaux de gris
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # Template matching
            result = cv2.matchTemplate(original_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            print(f"   🎯 Score de détection: {max_val:.3f}")
            
            if max_val >= self.DETECTION_THRESHOLD:
                x, y = max_loc
                h, w = template_gray.shape
                print(f"   ✅ Tableau détecté à: ({x}, {y}) avec taille ({w}, {h})")
                return (x, y, w, h)
            else:
                print(f"   ⚠️  Tableau non détecté (score trop bas: {max_val:.3f})")
                return None
                
        except Exception as e:
            print(f"❌ Erreur lors de la détection: {e}")
            return None
    
    def extract_banner_above_table(self, original_image_path, table_position, folder_name, original_filename):
        """
        Extrait 30% de la hauteur de l'image originale au-dessus du tableau détecté
        
        Args:
            original_image_path (str): Chemin vers l'image originale
            table_position (tuple): Position du tableau (x, y, w, h)
            folder_name (str): Nom du dossier
            original_filename (str): Nom du fichier original
            
        Returns:
            str: Chemin vers l'image de bannière sauvegardée ou None
        """
        try:
            x, y, w, h = table_position
            
            # Charger l'image originale
            original = cv2.imread(original_image_path)
            if original is None:
                return None
            
            img_height, img_width = original.shape[:2]
            
            # Calculer 30% de la hauteur de l'image originale
            banner_height = int(img_height * 0.30)
            
            # Zone de bannière : 30% de hauteur jusqu'au haut du tableau
            banner_y_start = max(0, y - banner_height)
            banner_y_end = y
            
            # S'assurer qu'on a une zone valide
            if banner_y_end <= banner_y_start:
                banner_y_start = 0
                banner_y_end = min(y, banner_height)
            
            # Extraire sur toute la largeur de l'image
            banner_x_start = 0
            banner_x_end = img_width
            
            # Extraire la région de bannière (30% de hauteur au-dessus du tableau)
            banner = original[banner_y_start:banner_y_end, banner_x_start:banner_x_end]
            
            # Sauvegarder la bannière dans le cache (sans redimensionnement)
            banner_filename = f"banner_{folder_name}_{original_filename}_{datetime.now().strftime('%H%M%S')}.jpg"
            banner_path = os.path.join(self.cache_dir, banner_filename)
            cv2.imwrite(banner_path, banner)
            
            actual_height, actual_width = banner.shape[:2]
            print(f"   📷 Zone titre extraite: {banner_path} ({actual_width}x{actual_height}px = 30% hauteur)")
            return banner_path
            
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction de la zone titre: {e}")
            return None
    
    def analyze_banner_with_gemini(self, banner_path):
        """
        Analyse la zone au-dessus du tableau avec Gemini pour extraire le premier titre depuis le bas
        
        Args:
            banner_path (str): Chemin vers l'image de la zone (30% hauteur au-dessus du tableau)
            
        Returns:
            str: Titre extrait ou None
        """
        if not self.model:
            return None
            
        try:
            # Charger l'image
            image = Image.open(banner_path)
            
            # Prompt pour l'extraction du titre
            prompt = """
            Analyse cette image qui contient la zone au-dessus d'un tableau médical.
            
            IMPORTANT : Cherche le premier titre de classement médical en partant du BAS de cette image vers le haut.
            Le titre est généralement en lettres capitales et se trouve juste au-dessus du tableau.
            
            Identifie et extrait UNIQUEMENT ce premier titre trouvé en remontant depuis le bas.
            Retourne seulement le titre exact, sans formatage, explication ou commentaire.
            Si aucun titre n'est visible, retourne "TITRE NON DÉTECTÉ".
            """
            
            response = self.model.generate_content([prompt, image])
            title = response.text.strip()
            
            print(f"   🤖 Titre extrait par Gemini: '{title}'")
            return title
            
        except Exception as e:
            print(f"❌ Erreur analyse Gemini (titre): {e}")
            return None
    
    def detect_hospital_type_with_gemini(self, table_image_path):
        """
        Détecte si le classement concerne des hôpitaux ou des cliniques
        
        Args:
            table_image_path (str): Chemin vers l'image de tableau découpé
            
        Returns:
            str: "HÔPITAUX", "CLINIQUES" ou "NON DÉTECTÉ"
        """
        if not self.model:
            return "NON DÉTECTÉ"
            
        try:
            # Charger l'image de tableau
            image = Image.open(table_image_path)
            
            # Prompt pour la détection du type d'établissement
            prompt = """
            Analyse ce tableau médical et détermine s'il s'agit d'un classement de:
            - HÔPITAUX (établissements publics)
            - CLINIQUES (établissements privés)
            
            Le titre est clairement indiqué dans le tableau en haut à gauche dans l'entête en gros caractère.
            Retourne uniquement un de ces mots: "HÔPITAUX", "CLINIQUES" ou "NON DÉTECTÉ".
            """
            
            response = self.model.generate_content([prompt, image])
            hospital_type = response.text.strip().upper()
            
            # Validation de la réponse
            valid_types = ["HÔPITAUX", "CLINIQUES", "NON DÉTECTÉ"]
            if hospital_type not in valid_types:
                hospital_type = "NON DÉTECTÉ"
            
            print(f"   🏥 Type détecté par Gemini: '{hospital_type}'")
            return hospital_type
            
        except Exception as e:
            print(f"❌ Erreur analyse Gemini (type): {e}")
            return "NON DÉTECTÉ"
    
    def cleanup_cache(self):
        """Supprime tous les fichiers temporaires du cache"""
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                print(f"🧹 Cache nettoyé: {self.cache_dir}")
        except Exception as e:
            print(f"⚠️  Erreur lors du nettoyage: {e}")
    
    def save_results_to_excel(self):
        """Sauvegarde les résultats dans un fichier Excel"""
        if not self.results:
            print("❌ Aucun résultat à sauvegarder")
            return
        
        try:
            # Créer le DataFrame
            df = pd.DataFrame(self.results)
            
            # Ajouter une feuille de statistiques
            stats_data = {
                'Métrique': [
                    'Total de paires traitées',
                    'Détections réussies',
                    'Titres extraits',
                    'Types détectés',
                    'Taux de réussite (%)'
                ],
                'Valeur': [
                    len(self.results),
                    len([r for r in self.results if r['Position_Detectee'] == 'OUI']),
                    len([r for r in self.results if r['Titre_Classement'] != 'NON DÉTECTÉ']),
                    len([r for r in self.results if r['Type_Etablissement'] != 'NON DÉTECTÉ']),
                    round((len([r for r in self.results if r['Position_Detectee'] == 'OUI']) / len(self.results)) * 100, 2)
                ]
            }
            df_stats = pd.DataFrame(stats_data)
            
            # Sauvegarder avec plusieurs feuilles
            with pd.ExcelWriter(self.output_excel, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Résultats_Analysis', index=False)
                df_stats.to_excel(writer, sheet_name='Statistiques', index=False)
            
            print(f"✅ Fichier Excel créé: {self.output_excel}")
            print(f"📊 {len(self.results)} résultats sauvegardés")
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde Excel: {e}")
    
    def process_all_images(self):
        """
        Lance le traitement complet de toutes les paires d'images SPÉCIFIQUES
        """
        print("🚀 Début de l'analyse des bannières de titres (IMAGES SPÉCIFIQUES)...")
        
        # Trouver toutes les paires d'images dans les dossiers spécifiés
        image_pairs = self.find_image_pairs()
        
        if not image_pairs:
            print("❌ Aucune paire d'images trouvée dans les dossiers spécifiés")
            return
        
        # Traiter chaque paire
        for i, (folder_name, original_filename, table_filename, folder_path) in enumerate(image_pairs, 1):
            print(f"\n📂 [{i}/{len(image_pairs)}] Traitement: {folder_name}")
            print(f"   Original: {original_filename}")
            print(f"   Tableau: {table_filename}")
            
            # Chemins complets
            original_path = os.path.join(folder_path, original_filename)
            table_path = os.path.join(folder_path, table_filename)
            
            # Initialiser le résultat
            result = {
                'Dossier': folder_name,
                'Image_Originale': original_filename,
                'Image_Tableau': table_filename,
                'Position_Detectee': 'NON',
                'Titre_Classement': 'NON DÉTECTÉ',
                'Type_Etablissement': 'NON DÉTECTÉ',
                'Score_Detection': 0.0,
                'Date_Traitement': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            try:
                # 1. Détecter la position du tableau
                table_position = self.detect_table_position(original_path, table_path)
                
                if table_position:
                    result['Position_Detectee'] = 'OUI'
                    
                    # 2. Extraire la bannière au-dessus du tableau
                    banner_path = self.extract_banner_above_table(
                        original_path, table_position, folder_name, original_filename
                    )
                    
                    if banner_path:
                        # 3. Analyser la bannière avec Gemini (1ère requête)
                        title = self.analyze_banner_with_gemini(banner_path)
                        if title and title != "TITRE NON DÉTECTÉ":
                            result['Titre_Classement'] = title
                        
                        # Supprimer la bannière temporaire
                        try:
                            os.remove(banner_path)
                        except:
                            pass
                    
                    # 4. Analyser le type d'établissement avec Gemini (2ème requête)
                    hospital_type = self.detect_hospital_type_with_gemini(table_path)
                    result['Type_Etablissement'] = hospital_type
                
            except Exception as e:
                print(f"   ❌ Erreur lors du traitement: {e}")
            
            # Ajouter le résultat
            self.results.append(result)
            
            # Affichage du résumé
            print(f"   📋 Résultat: {result['Titre_Classement']} | {result['Type_Etablissement']}")
        
        # Sauvegarder les résultats
        print(f"\n💾 Sauvegarde des résultats...")
        self.save_results_to_excel()
        
        # Nettoyage final
        self.cleanup_cache()
        
        print("✅ Analyse terminée!")
        self.print_final_summary()
    
    def print_final_summary(self):
        """Affiche un résumé final des résultats"""
        if not self.results:
            return
            
        print("\n" + "="*70)
        print("RÉSUMÉ FINAL DE L'ANALYSE (IMAGES SPÉCIFIQUES)")
        print("="*70)
        
        total = len(self.results)
        detected = len([r for r in self.results if r['Position_Detectee'] == 'OUI'])
        titles_found = len([r for r in self.results if r['Titre_Classement'] != 'NON DÉTECTÉ'])
        types_found = len([r for r in self.results if r['Type_Etablissement'] != 'NON DÉTECTÉ'])
        
        print(f"📊 Total de paires traitées: {total}")
        print(f"🎯 Détections réussies: {detected}/{total} ({(detected/total)*100:.1f}%)")
        print(f"📝 Titres extraits: {titles_found}/{total} ({(titles_found/total)*100:.1f}%)")
        print(f"🏥 Types détectés: {types_found}/{total} ({(types_found/total)*100:.1f}%)")
        print(f"📄 Fichier Excel: {self.output_excel}")

def main():
    """Fonction principale"""
    # Configuration depuis le fichier config
    OUTPUT_FILE = f"titres_classements_SPECIFIC_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    # Vérification du chemin des images
    if not os.path.exists(IMAGES_BASE_PATH):
        print(f"❌ Chemin des images non trouvé: {IMAGES_BASE_PATH}")
        print("💡 Modifiez la variable IMAGES_BASE_PATH dans config_specific_images.py")
        return
    
    try:
        # Création et lancement de l'analyseur spécifique
        analyzer = TitleBannerAnalyzerSpecific(
            images_base_path=IMAGES_BASE_PATH,
            output_excel=OUTPUT_FILE
        )
        
        analyzer.process_all_images()
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")

if __name__ == "__main__":
    main()
