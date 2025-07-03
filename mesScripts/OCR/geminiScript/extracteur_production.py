# Configuration de la clé API
GOOGLE_API_KEY = "AIzaSyBfQjj1pNx0yDlXUSo4tdWUe5RcE35ON6o"

import google.generativeai as genai
import pandas as pd
from pathlib import Path
import json
import re
from datetime import datetime
import logging
from PIL import Image
import numpy as np
import io
import time

class TableauExtractorProduction:
    def __init__(self, api_key, enable_image_splitting=True, split_threshold=2000, output_dir="extractions_tableaux"):
        """
        Extracteur de données de tableaux pour production
        
        Args:
            api_key (str): Clé API Gemini
            enable_image_splitting (bool): Activer le découpage d'images
            split_threshold (int): Seuil de largeur pour découper l'image
            output_dir (str): Dossier de sortie principal
        """
        self.api_key = api_key
        self.model = None
        self.enable_image_splitting = enable_image_splitting
        self.split_threshold = split_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.setup_gemini()
        self.setup_logging()
        
        # Statistiques globales
        self.stats = {
            'total_images': 0,
            'images_traitees': 0,
            'images_ignorees': 0,
            'erreurs': 0,
            'dossiers_traites': 0,
            'temps_debut': datetime.now()
        }
        
    def setup_gemini(self):
        """Configure l'API Gemini"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-pro')
            print("✅ API Gemini configurée avec succès")
        except Exception as e:
            print(f"❌ Erreur lors de la configuration de Gemini: {e}")
            raise
    
    def setup_logging(self):
        """Configure le système de logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = self.output_dir / f"extraction_production_log_{timestamp}.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.log_file = log_filename
        
    def is_original_image(self, filename):
        """
        Détecte si c'est une image originale à ignorer (pattern img*.jpg)
        
        Args:
            filename (str): Nom du fichier
            
        Returns:
            bool: True si c'est une image originale à ignorer
        """
        # Pattern pour détecter les images originales : img suivi de n'importe quoi
        pattern = r'^img.*\.(jpg|jpeg|png|gif|bmp|tiff|webp)$'
        return bool(re.match(pattern, filename.lower()))
    
    def extract_headers_from_top(self, image_path):
        """
        Extrait les headers depuis la partie haute de l'image
        
        Args:
            image_path (Path): Chemin vers l'image
            
        Returns:
            list: Liste des headers détectés
        """
        try:
            self.logger.info(f"🔍 Extraction headers: {image_path.name}")
            
            # Charger l'image
            img = Image.open(image_path)
            width, height = img.size
            
            # Prendre seulement les 20% du haut pour les headers
            header_height = int(height * 0.2)
            header_img = img.crop((0, 0, width, header_height))
            
            # Convertir pour Gemini
            image_data = self.image_to_base64(header_img)
            image_part = {
                "mime_type": "image/jpeg",
                "data": image_data
            }
            
            # Prompt spécialisé pour les headers
            header_prompt = """
            Analyse cette image qui contient UNIQUEMENT la ligne des HEADERS/TITRES de colonnes d'un tableau médical.
            
            MISSION CRITIQUE: Extrais TOUS les titres de colonnes visibles de gauche à droite.
            
            INSTRUCTIONS PRÉCISES:
            1. Lis CHAQUE titre de colonne de gauche à droite
            2. Inclus même les titres partiellement visibles ou coupés
            3. Respecte l'ordre exact des colonnes
            4. Préserve l'orthographe exacte des titres
            5. Si un titre est illisible, note "HEADER_ILLISIBLE"
            
            FORMAT DE RÉPONSE OBLIGATOIRE:
            ```json
            {
                "headers_detectes": ["Titre1", "Titre2", "Titre3", ...],
                "nombre_colonnes": X,
                "qualite_detection": "EXCELLENTE/BONNE/MOYENNE/FAIBLE",
                "notes": "observations sur la qualité de détection des headers"
            }
            ```
            """
            
            # Envoyer à Gemini
            response = self.model.generate_content([header_prompt, image_part])
            response_text = response.text
            
            # Extraire le JSON
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                header_data = json.loads(json_str)
                headers = header_data.get('headers_detectes', [])
                self.logger.info(f"✅ Headers détectés: {len(headers)} colonnes")
                return headers
            else:
                self.logger.warning("⚠️ Échec extraction headers")
                return []
                
        except Exception as e:
            self.logger.error(f"❌ Erreur extraction headers: {e}")
            return []
    
    def split_image_intelligently(self, image_path):
        """
        Découpe intelligemment une image de tableau en plusieurs parties
        """
        try:
            self.logger.info(f"🔪 Découpage: {image_path.name}")
            
            img = Image.open(image_path)
            img_array = np.array(img)
            height, width = img_array.shape[:2]
            
            if width <= self.split_threshold:
                return [{'image': img, 'position': 'complete', 'coords': (0, 0, width, height)}]
            
            parts = []
            num_parts = max(2, width // self.split_threshold)
            part_width = width // num_parts
            overlap = int(part_width * 0.1)
            
            for i in range(num_parts):
                start_x = max(0, i * part_width - overlap)
                end_x = min(width, (i + 1) * part_width + overlap)
                
                if i == num_parts - 1:
                    end_x = width
                
                part_img = img.crop((start_x, 0, end_x, height))
                
                part_info = {
                    'image': part_img,
                    'position': f'partie_{i+1}_sur_{num_parts}',
                    'coords': (start_x, 0, end_x, height),
                    'width': end_x - start_x
                }
                
                parts.append(part_info)
            
            self.logger.info(f"✅ Découpé en {len(parts)} parties")
            return parts
            
        except Exception as e:
            self.logger.error(f"❌ Erreur découpage: {e}")
            return [{'image': Image.open(image_path), 'position': 'complete_fallback', 'coords': (0, 0, 0, 0)}]
    
    def image_to_base64(self, pil_image):
        """Convertit une image PIL en base64 pour Gemini"""
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=95)
        return buffer.getvalue()
    
    def create_part_extraction_prompt(self, part_info, detected_headers, total_parts=1):
        """
        Crée un prompt spécialisé pour une partie d'image
        """
        position = part_info['position']
        
        if total_parts == 1:
            base_prompt = """
            Analyse cette image complète qui contient un tableau de données médicales/hospitalières.
            """
        else:
            base_prompt = f"""
            Analyse cette PARTIE d'un tableau de données médicales/hospitalières.
            Position: {position}
            """
        
        if detected_headers:
            base_prompt += f"""
            
            HEADERS IDENTIFIÉS:
            {detected_headers}
            
            INSTRUCTIONS SPÉCIALES:
            - Utilise EXACTEMENT ces headers comme colonnes
            - Concentre-toi uniquement sur l'extraction des DONNÉES sous ces colonnes
            - Respecte l'ordre exact des headers fournis
            """
        
        base_prompt += f"""
        INSTRUCTIONS CRITIQUES:
        1. Extrais chaque ligne de données complète
        2. Respecte l'ordre des colonnes de gauche à droite
        3. Pour les cellules vides: utilise ""
        4. Pour les cellules illisibles: utilise "ILLISIBLE"
        5. Préserve les formats exacts (chiffres, noms, scores)
        6. Assure-toi que chaque ligne a le même nombre de valeurs que de headers
        
        FORMAT DE RÉPONSE STRICT:
        ```json
        {{
            "headers_detectes": {detected_headers if detected_headers else '["Header1", "Header2", ...]'},
            "nombre_lignes": X,
            "nombre_colonnes": Y,
            "position_partie": "{position}",
            "donnees": [
                ["val1_L1", "val2_L1", "val3_L1", ...],
                ["val1_L2", "val2_L2", "val3_L2", ...],
                ...
            ],
            "qualite_extraction": "EXCELLENTE/BONNE/MOYENNE/FAIBLE",
            "problemes_detectes": ["problème1", "problème2", ...],
            "confiance_globale": "XX%",
            "notes_partie": "observations spécifiques"
        }}
        ```
        """
        
        return base_prompt
    
    def extract_data_from_image(self, image_path):
        """
        Extrait les données d'une image de tableau
        """
        try:
            self.logger.info(f"🔍 Traitement: {image_path.name}")
            
            # Étape 1: Extraction des headers
            detected_headers = self.extract_headers_from_top(image_path)
            
            # Étape 2: Découpage si nécessaire
            if self.enable_image_splitting:
                image_parts = self.split_image_intelligently(image_path)
            else:
                img = Image.open(image_path)
                image_parts = [{'image': img, 'position': 'complete', 'coords': (0, 0, img.width, img.height)}]
            
            # Étape 3: Extraction des données de chaque partie
            all_parts_data = []
            
            for i, part_info in enumerate(image_parts):
                self.logger.info(f"📤 Partie {i+1}/{len(image_parts)}")
                
                image_data = self.image_to_base64(part_info['image'])
                image_part = {
                    "mime_type": "image/jpeg",
                    "data": image_data
                }
                
                prompt = self.create_part_extraction_prompt(part_info, detected_headers, len(image_parts))
                
                response = self.model.generate_content([prompt, image_part])
                response_text = response.text
                
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(1)
                    part_data = json.loads(json_str)
                    
                    if detected_headers:
                        part_data['headers_detectes'] = detected_headers
                    
                    all_parts_data.append(part_data)
                    self.logger.info(f"✅ Partie {i+1}: {len(part_data.get('donnees', []))} lignes")
                else:
                    self.logger.warning(f"⚠️ Échec partie {i+1}")
                    all_parts_data.append({'erreur': f'JSON non trouvé pour partie {i+1}'})
                
                # Pause pour respecter les limites de l'API
                time.sleep(1)
            
            # Étape 4: Fusion des résultats
            if len(all_parts_data) == 1:
                merged_data = all_parts_data[0]
            else:
                merged_data = self.merge_parts_data(all_parts_data, detected_headers)
            
            # Métadonnées finales
            merged_data.update({
                'image_path': str(image_path),
                'extraction_timestamp': datetime.now().isoformat(),
                'nombre_parties': len(image_parts),
                'headers_pre_detectes': detected_headers
            })
            
            return merged_data
                
        except Exception as e:
            error_msg = f"❌ Erreur extraction {image_path.name}: {str(e)}"
            self.logger.error(error_msg)
            return {'erreur': error_msg, 'image_path': str(image_path)}
    
    def merge_parts_data(self, parts_data, detected_headers):
        """Fusionne les données de plusieurs parties"""
        valid_parts = [part for part in parts_data if 'donnees' in part]
        
        if not valid_parts:
            return {'erreur': 'Aucune partie extraite avec succès'}
        
        all_headers = detected_headers if detected_headers else []
        if not all_headers:
            for part in valid_parts:
                for header in part.get('headers_detectes', []):
                    if header not in all_headers:
                        all_headers.append(header)
        
        max_rows = max(len(part.get('donnees', [])) for part in valid_parts)
        merged_data = []
        
        for row_idx in range(max_rows):
            merged_row = []
            
            for header_idx, header in enumerate(all_headers):
                cell_value = ""
                
                for part in valid_parts:
                    part_data = part.get('donnees', [])
                    
                    if detected_headers and row_idx < len(part_data):
                        if header_idx < len(part_data[row_idx]):
                            potential_value = part_data[row_idx][header_idx]
                            if potential_value and potential_value not in ["", "N/A"]:
                                cell_value = potential_value
                                break
                    else:
                        part_headers = part.get('headers_detectes', [])
                        if header in part_headers:
                            col_idx = part_headers.index(header)
                            if row_idx < len(part_data) and col_idx < len(part_data[row_idx]):
                                potential_value = part_data[row_idx][col_idx]
                                if potential_value and potential_value not in ["", "N/A"]:
                                    cell_value = potential_value
                                    break
                
                merged_row.append(cell_value)
            
            merged_data.append(merged_row)
        
        # Calcul qualité
        total_cells = len(all_headers) * max_rows
        filled_cells = sum(1 for row in merged_data for cell in row if cell and cell != "")
        fill_rate = (filled_cells / total_cells * 100) if total_cells > 0 else 0
        
        return {
            'headers_detectes': all_headers,
            'nombre_lignes': max_rows,
            'nombre_colonnes': len(all_headers),
            'donnees': merged_data,
            'qualite_extraction': 'EXCELLENTE' if fill_rate > 90 else 'BONNE' if fill_rate > 70 else 'MOYENNE' if fill_rate > 50 else 'FAIBLE',
            'confiance_globale': f"{fill_rate:.1f}%",
            'taux_remplissage': f"{fill_rate:.1f}%"
        }
    
    def save_extraction_results(self, image_path, extracted_data, subfolder_name):
        """Sauvegarde les résultats d'extraction"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = image_path.stem
        
        # Créer un dossier pour ce sous-dossier source
        output_subfolder = self.output_dir / subfolder_name
        output_subfolder.mkdir(exist_ok=True)
        
        # Sauvegarder les données en Excel
        if 'donnees' in extracted_data and 'headers_detectes' in extracted_data:
            try:
                df = pd.DataFrame(
                    extracted_data['donnees'], 
                    columns=extracted_data['headers_detectes']
                )
                
                excel_file = output_subfolder / f"{image_name}_extracted_{timestamp}.xlsx"
                df.to_excel(excel_file, index=False)
                
                self.logger.info(f"💾 Excel sauvegardé: {excel_file.name}")
                
                # Sauvegarder aussi en JSON pour analyse
                json_file = output_subfolder / f"{image_name}_metadata_{timestamp}.json"
                
                # Nettoyer les données pour JSON
                clean_data = {
                    'image_name': image_name,
                    'subfolder': subfolder_name,
                    'extraction_timestamp': extracted_data.get('extraction_timestamp'),
                    'nombre_lignes': extracted_data.get('nombre_lignes', 0),
                    'nombre_colonnes': extracted_data.get('nombre_colonnes', 0),
                    'headers_detectes': extracted_data.get('headers_detectes', []),
                    'qualite_extraction': extracted_data.get('qualite_extraction', 'INCONNUE'),
                    'confiance_globale': extracted_data.get('confiance_globale', '0%'),
                    'taux_remplissage': extracted_data.get('taux_remplissage', '0%'),
                    'nombre_parties': extracted_data.get('nombre_parties', 1),
                    'headers_pre_detectes': bool(extracted_data.get('headers_pre_detectes'))
                }
                
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(clean_data, f, indent=2, ensure_ascii=False)
                
                return excel_file, json_file
                
            except Exception as e:
                self.logger.error(f"❌ Erreur sauvegarde: {e}")
                return None, None
        else:
            self.logger.warning(f"⚠️ Données incomplètes pour {image_name}")
            return None, None
    
    def process_folder_structure(self, base_folder_path):
        """
        Traite toute la structure de dossiers
        
        Args:
            base_folder_path (str): Chemin vers le dossier racine
        """
        base_path = Path(base_folder_path)
        
        if not base_path.exists():
            self.logger.error(f"❌ Dossier inexistant: {base_path}")
            return
        
        self.logger.info(f"🚀 DÉBUT TRAITEMENT: {base_path}")
        self.logger.info(f"📁 Sortie: {self.output_dir}")
        
        # Extensions d'images supportées
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        
        # Parcourir tous les sous-dossiers
        for subfolder in base_path.iterdir():
            if not subfolder.is_dir():
                continue
            
            self.stats['dossiers_traites'] += 1
            self.logger.info(f"\n📂 DOSSIER: {subfolder.name}")
            
            # Trouver toutes les images dans ce sous-dossier
            images_in_folder = []
            for file_path in subfolder.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    # Ignorer les images originales (pattern img*.jpg)
                    if self.is_original_image(file_path.name):
                        self.logger.info(f"⏭️ Ignoré (image originale): {file_path.name}")
                        self.stats['images_ignorees'] += 1
                    else:
                        images_in_folder.append(file_path)
                        self.stats['total_images'] += 1
            
            if not images_in_folder:
                self.logger.info(f"  📭 Aucune image de tableau trouvée")
                continue
            
            self.logger.info(f"  🖼️ {len(images_in_folder)} images à traiter")
            
            # Traiter chaque image
            for image_path in sorted(images_in_folder):
                try:
                    self.logger.info(f"\n  🔄 {image_path.name}")
                    
                    # Extraire les données
                    extracted_data = self.extract_data_from_image(image_path)
                    
                    if 'erreur' not in extracted_data:
                        # Sauvegarder les résultats
                        excel_file, json_file = self.save_extraction_results(
                            image_path, extracted_data, subfolder.name
                        )
                        
                        if excel_file:
                            self.stats['images_traitees'] += 1
                            self.logger.info(f"  ✅ Succès: {excel_file.name}")
                        else:
                            self.stats['erreurs'] += 1
                    else:
                        self.stats['erreurs'] += 1
                        self.logger.error(f"  ❌ Échec: {extracted_data.get('erreur', 'Erreur inconnue')}")
                
                except Exception as e:
                    self.stats['erreurs'] += 1
                    self.logger.error(f"  ❌ Exception: {e}")
        
        # Statistiques finales
        self.print_final_stats()
    
    def print_final_stats(self):
        """Affiche les statistiques finales"""
        temps_total = datetime.now() - self.stats['temps_debut']
        
        print("\n" + "="*80)
        print("📊 STATISTIQUES FINALES")
        print("="*80)
        print(f"⏱️ Temps total: {temps_total}")
        print(f"📁 Dossiers traités: {self.stats['dossiers_traites']}")
        print(f"🖼️ Images trouvées: {self.stats['total_images']}")
        print(f"✅ Images traitées avec succès: {self.stats['images_traitees']}")
        print(f"⏭️ Images ignorées (originales): {self.stats['images_ignorees']}")
        print(f"❌ Erreurs: {self.stats['erreurs']}")
        
        if self.stats['total_images'] > 0:
            taux_succes = (self.stats['images_traitees'] / self.stats['total_images']) * 100
            print(f"📈 Taux de succès: {taux_succes:.1f}%")
        
        print(f"📁 Résultats dans: {self.output_dir}")
        print(f"📝 Log détaillé: {self.log_file}")
        print("="*80)

def main():
    """Fonction principale"""
    print("🏭 EXTRACTEUR DE TABLEAUX - MODE PRODUCTION")
    print("="*60)
    
    # Configuration
    API_KEY = GOOGLE_API_KEY
    
    if not API_KEY:
        print("❌ Veuillez configurer votre clé API Gemini")
        return
    
    # Paramètres de traitement
    print("\n🔧 CONFIGURATION:")
    
    # Dossier source
    dossier_source = input("📂 Chemin vers le dossier racine contenant les sous-dossiers: ").strip().strip('"')
    
    if not Path(dossier_source).exists():
        print(f"❌ Dossier non trouvé: {dossier_source}")
        return
    
    # Options de découpage
    print("\n📐 DÉCOUPAGE D'IMAGES:")
    print("1. Activé (recommandé pour grandes images)")
    print("2. Désactivé")
    
    choix_decoupage = input("Choix (1 ou 2): ").strip()
    enable_splitting = choix_decoupage == "1"
    
    if enable_splitting:
        seuil = input("Seuil de largeur pour découpage (défaut: 2000px): ").strip()
        split_threshold = int(seuil) if seuil.isdigit() else 2000
    else:
        split_threshold = 10000
    
    # Dossier de sortie
    dossier_sortie = input("📁 Dossier de sortie (défaut: extractions_tableaux): ").strip()
    if not dossier_sortie:
        dossier_sortie = "extractions_tableaux"
    
    try:
        # Création de l'extracteur
        extracteur = TableauExtractorProduction(
            api_key=API_KEY,
            enable_image_splitting=enable_splitting,
            split_threshold=split_threshold,
            output_dir=dossier_sortie
        )
        
        print(f"\n🚀 DÉBUT DU TRAITEMENT...")
        print(f"📂 Source: {dossier_source}")
        print(f"📁 Sortie: {dossier_sortie}")
        print(f"🔪 Découpage: {'Activé' if enable_splitting else 'Désactivé'}")
        
        # Lancement du traitement
        extracteur.process_folder_structure(dossier_source)
        
        print(f"\n✅ TRAITEMENT TERMINÉ!")
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")

if __name__ == "__main__":
    main()
