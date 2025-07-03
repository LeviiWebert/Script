# Configuration de la clé API
GOOGLE_API_KEY = "AIzaSyBfQjj1pNx0yDlXUSo4tdWUe5RcE35ON6o"

import google.generativeai as genai
import pandas as pd
from pathlib import Path
import json
import re
from datetime import datetime
import logging
# Nouveaux imports pour le traitement d'images
from PIL import Image
import numpy as np
import cv2
import base64
import io

class TableauExtractorTester:
    def __init__(self, api_key, enable_image_splitting=True, split_threshold=2000):
        """
        Testeur d'extraction de données de tableaux avec Gemini
        
        Args:
            api_key (str): Clé API Gemini
            enable_image_splitting (bool): Activer le découpage d'images
            split_threshold (int): Seuil de largeur pour découper l'image
        """
        self.api_key = api_key
        self.model = None
        self.enable_image_splitting = enable_image_splitting
        self.split_threshold = split_threshold
        self.setup_gemini()
        self.setup_logging()
        
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
        """Configure le système de logging pour améliorer le code"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"test_extraction_log_{timestamp}.txt"
        
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
        
    def split_image_intelligently(self, image_path):
        """
        Découpe intelligemment une image de tableau en plusieurs parties
        
        Args:
            image_path (str): Chemin vers l'image
            
        Returns:
            list: Liste des parties d'image (PIL Images) avec métadonnées
        """
        try:
            self.logger.info(f"🔪 Découpage intelligent de l'image: {image_path}")
            
            # Charger l'image
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Informations sur l'image
            height, width = img_array.shape[:2]
            self.logger.info(f"📐 Dimensions image: {width}x{height}")
            
            # Si l'image est petite, pas besoin de découper
            if width <= self.split_threshold:
                self.logger.info("📏 Image suffisamment petite, pas de découpage nécessaire")
                return [{'image': img, 'position': 'complete', 'coords': (0, 0, width, height)}]
            
            # Découpage intelligent basé sur la détection de lignes verticales
            parts = []
            
            # Méthode 1: Découpage par colonnes fixes (plus simple et robuste)
            num_parts = max(2, width // self.split_threshold)
            part_width = width // num_parts
            overlap = int(part_width * 0.1)  # 10% de chevauchement
            
            for i in range(num_parts):
                start_x = max(0, i * part_width - overlap)
                end_x = min(width, (i + 1) * part_width + overlap)
                
                # Si c'est la dernière partie, prendre jusqu'au bout
                if i == num_parts - 1:
                    end_x = width
                
                # Découper la partie
                part_img = img.crop((start_x, 0, end_x, height))
                
                part_info = {
                    'image': part_img,
                    'position': f'partie_{i+1}_sur_{num_parts}',
                    'coords': (start_x, 0, end_x, height),
                    'width': end_x - start_x
                }
                
                parts.append(part_info)
                self.logger.info(f"✂️ Partie {i+1}: {start_x}-{end_x} (largeur: {end_x-start_x}px)")
            
            self.logger.info(f"✅ Image découpée en {len(parts)} parties")
            return parts
            
        except Exception as e:
            self.logger.error(f"❌ Erreur découpage image: {e}")
            # En cas d'erreur, retourner l'image complète
            return [{'image': Image.open(image_path), 'position': 'complete_fallback', 'coords': (0, 0, 0, 0)}]
    
    def image_to_base64(self, pil_image):
        """
        Convertit une image PIL en base64 pour Gemini
        
        Args:
            pil_image: Image PIL
            
        Returns:
            bytes: Données image pour Gemini
        """
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=95)
        return buffer.getvalue()
    
    def create_part_extraction_prompt(self, part_info, expected_columns=None, total_parts=1, detected_headers=None):
        """
        Crée un prompt spécialisé pour une partie d'image
        
        Args:
            part_info (dict): Informations sur la partie d'image
            expected_columns (list): Colonnes attendues
            total_parts (int): Nombre total de parties
            detected_headers (list): Headers déjà détectés depuis la partie haute
            
        Returns:
            str: Prompt optimisé pour cette partie
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
            
            ATTENTION: Ceci est une PARTIE ({position}) d'un tableau plus large.
            """
        
        # Si on a déjà détecté les headers, les utiliser
        if detected_headers:
            base_prompt += f"""
            
            HEADERS DÉJÀ IDENTIFIÉS:
            {detected_headers}
            
            INSTRUCTIONS SPÉCIALES:
            - Utilise EXACTEMENT ces headers comme colonnes
            - N'essaie PAS de détecter de nouveaux headers
            - Concentre-toi uniquement sur l'extraction des DONNÉES sous ces colonnes
            - Respecte l'ordre exact des headers fournis
            """
        
        base_prompt += """
        INSTRUCTIONS CRITIQUES:
        1. Extrais chaque ligne de données complète pour cette partie
        2. Respecte rigoureusement l'ordre des colonnes de gauche à droite
        3. Pour les cellules vides: utilise ""
        4. Pour les cellules illisibles: utilise "ILLISIBLE"
        5. Préserve les formats exacts (chiffres, noms, scores)
        6. Assure-toi que chaque ligne a le même nombre de valeurs que de headers
        
        FORMAT DE RÉPONSE STRICT:
        ```json
        {"""
        
        if detected_headers:
            base_prompt += f'''
            "headers_detectes": {detected_headers},'''
        else:
            base_prompt += '''
            "headers_detectes": ["Header1", "Header2", "Header3", ...],'''
            
        base_prompt += f'''
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
            "notes_partie": "observations spécifiques à cette partie"
        }}
        ```
        '''
        
        if expected_columns and total_parts > 1 and not detected_headers:
            base_prompt += f"""
            
            COLONNES ATTENDUES GLOBALEMENT:
            {expected_columns}
            
            NOTE: Cette partie peut ne contenir qu'un sous-ensemble de ces colonnes.
            Concentre-toi sur les colonnes visibles dans cette partie uniquement.
            """
        
        return base_prompt
    
    def create_extraction_prompt(self, expected_columns=None):
        """
        Crée un prompt optimisé pour l'extraction de données de tableau
        
        Args:
            expected_columns (list): Liste des colonnes attendues du fichier Excel de référence
            
        Returns:
            str: Prompt optimisé
        """
        base_prompt = """
        Analyse cette image qui contient un tableau de données médicales/hospitalières.
        Extrais TOUTES les données visibles sous forme de tableau structuré.
        
        INSTRUCTIONS IMPORTANTES:
        1. Identifie d'abord les colonnes (headers) du tableau
        2. Extrais chaque ligne de données complète
        3. Respecte l'ordre des colonnes
        4. Si une cellule est vide ou illisible, note "N/A"
        5. Préserve les numéros de classement, noms d'hôpitaux, scores, etc.
        
        FORMAT DE RÉPONSE REQUIS:
        ```json
        {
            "headers_detectes": ["Colonne1", "Colonne2", "Colonne3", ...],
            "nombre_lignes": X,
            "donnees": [
                ["valeur1_ligne1", "valeur2_ligne1", "valeur3_ligne1", ...],
                ["valeur1_ligne2", "valeur2_ligne2", "valeur3_ligne2", ...],
                ...
            ],
            "qualite_extraction": "EXCELLENTE/BONNE/MOYENNE/FAIBLE",
            "problemes_detectes": ["problème1", "problème2", ...],
            "confiance_globale": "90%" 
        }
        ```
        """
        
        if expected_columns:
            base_prompt += f"""
        
        COLONNES ATTENDUES (pour référence):
        {expected_columns}
        
        Essaie de faire correspondre tes headers détectés avec ces colonnes attendues.
        """
        
        return base_prompt
    
    def extract_data_from_image(self, image_path, expected_columns=None):
        """
        Extrait les données d'une image de tableau avec découpage intelligent
        
        Args:
            image_path (str): Chemin vers l'image
            expected_columns (list): Colonnes attendues
            
        Returns:
            dict: Données extraites et métadonnées
        """
        try:
            self.logger.info(f"🔍 Début extraction avec découpage optimisé: {image_path}")
            
            # Étape 0: Extraction prioritaire des headers depuis la partie haute
            detected_headers = self.extract_headers_from_top(image_path)
            if detected_headers:
                self.logger.info(f"📋 Headers pré-détectés: {len(detected_headers)} colonnes")
            else:
                self.logger.warning("⚠️ Aucun header pré-détecté, utilisation du mode classique")
            
            # Étape 1: Découper l'image si nécessaire
            if self.enable_image_splitting:
                image_parts = self.split_image_intelligently(image_path)
            else:
                # Mode classique sans découpage
                img = Image.open(image_path)
                image_parts = [{'image': img, 'position': 'complete', 'coords': (0, 0, img.width, img.height)}]
            
            # Étape 2: Extraire les données de chaque partie
            all_parts_data = []
            
            for i, part_info in enumerate(image_parts):
                self.logger.info(f"📤 Analyse partie {i+1}/{len(image_parts)}: {part_info['position']}")
                
                # Convertir l'image en format pour Gemini
                image_data = self.image_to_base64(part_info['image'])
                image_part = {
                    "mime_type": "image/jpeg",
                    "data": image_data
                }
                
                # Créer le prompt spécialisé avec les headers pré-détectés
                prompt = self.create_part_extraction_prompt(
                    part_info, 
                    expected_columns, 
                    len(image_parts),
                    detected_headers
                )
                
                # Envoyer à Gemini
                response = self.model.generate_content([prompt, image_part])
                response_text = response.text
                
                # Extraire le JSON
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(1)
                    part_data = json.loads(json_str)
                    part_data['response_brute'] = response_text
                    part_data['part_info'] = part_info
                    
                    # Forcer l'utilisation des headers pré-détectés si disponibles
                    if detected_headers:
                        part_data['headers_detectes'] = detected_headers
                        part_data['headers_source'] = 'pre_detected'
                    
                    all_parts_data.append(part_data)
                    self.logger.info(f"✅ Partie {i+1} analysée: {len(part_data.get('donnees', []))} lignes")
                else:
                    self.logger.warning(f"⚠️ Aucun JSON dans partie {i+1}")
                    all_parts_data.append({
                        'erreur': f'JSON non trouvé pour partie {i+1}',
                        'response_brute': response_text,
                        'part_info': part_info
                    })
            
            # Étape 3: Fusionner les résultats des parties
            if len(all_parts_data) == 1:
                # Une seule partie = résultat direct
                merged_data = all_parts_data[0]
            else:
                # Plusieurs parties = fusion intelligente
                merged_data = self.merge_parts_data(all_parts_data, detected_headers)
            
            # Ajout de métadonnées finales
            merged_data['image_path'] = str(image_path)
            merged_data['extraction_timestamp'] = datetime.now().isoformat()
            merged_data['nombre_parties'] = len(image_parts)
            merged_data['decoupage_active'] = self.enable_image_splitting
            merged_data['headers_pre_detectes'] = detected_headers
            
            return merged_data
                
        except Exception as e:
            error_msg = f"❌ Erreur extraction {image_path}: {str(e)}"
            self.logger.error(error_msg)
            return {'erreur': error_msg, 'image_path': str(image_path)}
    
    def merge_parts_data(self, parts_data, detected_headers=None):
        """
        Fusionne intelligemment les données extraites de plusieurs parties
        
        Args:
            parts_data (list): Liste des données extraites de chaque partie
            detected_headers (list): Headers pré-détectés depuis la partie haute
            
        Returns:
            dict: Données fusionnées
        """
        self.logger.info(f"🔀 Fusion de {len(parts_data)} parties...")
        
        # Filtrer les parties avec erreur
        valid_parts = [part for part in parts_data if 'donnees' in part and 'headers_detectes' in part]
        
        if not valid_parts:
            return {
                'erreur': 'Aucune partie extraite avec succès',
                'parties_avec_erreur': len(parts_data)
            }
        
        # Utiliser les headers pré-détectés si disponibles, sinon fusionner comme avant
        if detected_headers:
            self.logger.info(f"📋 Utilisation des headers pré-détectés: {detected_headers}")
            all_headers = detected_headers
        else:
            # Fusionner les headers (colonnes) - méthode classique
            all_headers = []
            for part in valid_parts:
                for header in part.get('headers_detectes', []):
                    if header not in all_headers:
                        all_headers.append(header)
        
        # Déterminer le nombre de lignes (prendre le maximum)
        max_rows = max(len(part.get('donnees', [])) for part in valid_parts)
        
        # Reconstruire le tableau complet
        merged_data = []
        
        for row_idx in range(max_rows):
            merged_row = []
            
            # Pour chaque colonne dans l'ordre des headers
            for header_idx, header in enumerate(all_headers):
                cell_value = ""
                
                # Chercher cette colonne dans les parties
                for part in valid_parts:
                    part_headers = part.get('headers_detectes', [])
                    part_data = part.get('donnees', [])
                    
                    # Si on utilise les headers pré-détectés, utiliser l'index direct
                    if detected_headers and row_idx < len(part_data):
                        if header_idx < len(part_data[row_idx]):
                            potential_value = part_data[row_idx][header_idx]
                            if potential_value and potential_value != "" and potential_value != "N/A":
                                cell_value = potential_value
                                break
                    # Sinon, chercher par nom de header
                    elif header in part_headers:
                        col_idx = part_headers.index(header)
                        if row_idx < len(part_data) and col_idx < len(part_data[row_idx]):
                            potential_value = part_data[row_idx][col_idx]
                            if potential_value and potential_value != "" and potential_value != "N/A":
                                cell_value = potential_value
                                break
                
                merged_row.append(cell_value)
            
            merged_data.append(merged_row)
        
        # Calculer la qualité de fusion
        total_cells = len(all_headers) * max_rows
        filled_cells = sum(1 for row in merged_data for cell in row if cell and cell != "")
        fill_rate = (filled_cells / total_cells * 100) if total_cells > 0 else 0
        
        # Résultat fusionné
        result = {
            'headers_detectes': all_headers,
            'nombre_lignes': max_rows,
            'nombre_colonnes': len(all_headers),
            'donnees': merged_data,
            'qualite_extraction': 'EXCELLENTE' if fill_rate > 90 else 'BONNE' if fill_rate > 70 else 'MOYENNE' if fill_rate > 50 else 'FAIBLE',
            'confiance_globale': f"{fill_rate:.1f}%",
            'fusion_info': {
                'parties_fusionnees': len(valid_parts),
                'parties_avec_erreur': len(parts_data) - len(valid_parts),
                'taux_remplissage': f"{fill_rate:.1f}%",
                'headers_pre_detectes': detected_headers is not None
            },
            'parties_details': parts_data
        }
        
        self.logger.info(f"✅ Fusion terminée: {len(all_headers)} colonnes, {max_rows} lignes, {fill_rate:.1f}% rempli")
        
        return result
    
    def load_reference_excel(self, excel_path):
        """
        Charge le fichier Excel de référence
        
        Args:
            excel_path (str): Chemin vers le fichier Excel
            
        Returns:
            tuple: (DataFrame, liste des colonnes, métadonnées)
        """
        try:
            self.logger.info(f"📊 Chargement Excel référence: {excel_path}")
            
            df = pd.read_excel(excel_path)
            columns = list(df.columns)
            
            metadata = {
                'nombre_lignes': len(df),
                'nombre_colonnes': len(columns),
                'colonnes': columns,
                'shape': df.shape
            }
            
            self.logger.info(f"✅ Excel chargé: {metadata['shape']}")
            return df, columns, metadata
            
        except Exception as e:
            error_msg = f"❌ Erreur chargement Excel: {str(e)}"
            self.logger.error(error_msg)
            return None, [], {'erreur': error_msg}
    
    def compare_with_reference(self, extracted_data, reference_df):
        """
        Compare les données extraites avec le fichier Excel de référence
        
        Args:
            extracted_data (dict): Données extraites de l'image
            reference_df (DataFrame): DataFrame de référence
            
        Returns:
            dict: Rapport de comparaison détaillé
        """
        self.logger.info("🔍 Début comparaison avec référence...")
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'extraction_reussie': 'donnees' in extracted_data,
            'scores': {},
            'differences': {},
            'recommandations': []
        }
        
        if not comparison['extraction_reussie']:
            comparison['erreur'] = 'Extraction échouée'
            return comparison
        
        # Comparaison des dimensions
        ref_rows, ref_cols = reference_df.shape
        ext_rows = len(extracted_data.get('donnees', []))
        ext_cols = len(extracted_data.get('headers_detectes', []))
        
        comparison['dimensions'] = {
            'reference': {'lignes': ref_rows, 'colonnes': ref_cols},
            'extraite': {'lignes': ext_rows, 'colonnes': ext_cols},
            'diff_lignes': abs(ref_rows - ext_rows),
            'diff_colonnes': abs(ref_cols - ext_cols)
        }
        
        # Score de précision dimensionnelle
        dim_score = max(0, 100 - (comparison['dimensions']['diff_lignes'] * 5) - 
                       (comparison['dimensions']['diff_colonnes'] * 10))
        comparison['scores']['precision_dimensionnelle'] = dim_score
        
        # Comparaison des headers
        ref_columns = list(reference_df.columns)
        ext_headers = extracted_data.get('headers_detectes', [])
        
        # Correspondance fuzzy des headers
        from fuzzywuzzy import fuzz
        header_matches = []
        for ext_header in ext_headers:
            best_match = max(ref_columns, key=lambda x: fuzz.ratio(str(ext_header), str(x)))
            score = fuzz.ratio(str(ext_header), str(best_match))
            header_matches.append({
                'extrait': ext_header,
                'reference': best_match,
                'score': score
            })
        
        comparison['correspondance_headers'] = header_matches
        avg_header_score = sum(match['score'] for match in header_matches) / len(header_matches) if header_matches else 0
        comparison['scores']['correspondance_headers'] = avg_header_score
        
        # Analyse du contenu (première ligne pour test)
        if ext_rows > 0 and ref_rows > 0:
            first_extracted_row = extracted_data['donnees'][0]
            first_ref_row = reference_df.iloc[0].tolist()
            
            cell_scores = []
            for i, (ext_val, ref_val) in enumerate(zip(first_extracted_row, first_ref_row)):
                if i < len(first_ref_row):
                    cell_score = fuzz.ratio(str(ext_val), str(ref_val))
                    cell_scores.append(cell_score)
            
            comparison['scores']['premiere_ligne'] = sum(cell_scores) / len(cell_scores) if cell_scores else 0
        
        # Score global
        scores = [score for score in comparison['scores'].values() if isinstance(score, (int, float))]
        comparison['scores']['global'] = sum(scores) / len(scores) if scores else 0
        
        # Recommandations basées sur les scores
        if comparison['scores']['global'] < 50:
            comparison['recommandations'].append("❌ Score global faible - Revoir le prompt d'extraction")
        if comparison['scores']['correspondance_headers'] < 70:
            comparison['recommandations'].append("⚠️ Headers mal détectés - Améliorer la détection des colonnes")
        if comparison['dimensions']['diff_lignes'] > 2:
            comparison['recommandations'].append("📊 Différence de lignes importante - Vérifier la détection des lignes")
        
        self.logger.info(f"✅ Comparaison terminée - Score global: {comparison['scores']['global']:.1f}%")
        
        return comparison
    
    def test_single_image(self, image_path, excel_path, output_dir="test_results"):
        """
        Test complet sur une image avec son Excel de référence
        
        Args:
            image_path (str): Chemin vers l'image
            excel_path (str): Chemin vers l'Excel de référence
            output_dir (str): Dossier de sortie pour les résultats
            
        Returns:
            dict: Résultats complets du test
        """
        self.logger.info(f"🚀 DÉBUT TEST - Image: {image_path}")
        self.logger.info(f"📊 Excel référence: {excel_path}")
        
        # Création du dossier de sortie
        Path(output_dir).mkdir(exist_ok=True)
        
        # Étape 1: Charger l'Excel de référence
        reference_df, ref_columns, ref_metadata = self.load_reference_excel(excel_path)
        
        if reference_df is None:
            return {'erreur': 'Impossible de charger l\'Excel de référence'}
        
        # Étape 2: Extraire les données de l'image
        extracted_data = self.extract_data_from_image(image_path, ref_columns)
        
        # Étape 3: Comparer avec la référence
        comparison = self.compare_with_reference(extracted_data, reference_df)
        
        # Étape 4: Créer le rapport complet
        test_results = {
            'test_info': {
                'image_path': str(image_path),
                'excel_path': str(excel_path),
                'timestamp': datetime.now().isoformat()
            },
            'reference_metadata': ref_metadata,
            'extracted_data': extracted_data,
            'comparison': comparison,
            'log_file': self.log_file
        }
        
        # Sauvegarde des résultats (nettoyer les objets non-sérialisables)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(output_dir) / f"test_results_{timestamp}.json"
        
        # Créer une copie propre pour la sérialisation JSON
        clean_results = {
            'test_info': test_results['test_info'],
            'reference_metadata': test_results['reference_metadata'],
            'comparison': test_results['comparison'],
            'log_file': test_results['log_file']
        }
        
        # Nettoyer les données extraites
        if 'extracted_data' in test_results:
            clean_extracted = {}
            for key, value in test_results['extracted_data'].items():
                if key != 'parties_details':  # Éviter les objets Image dans parties_details
                    clean_extracted[key] = value
            clean_results['extracted_data'] = clean_extracted
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        # Sauvegarde des données extraites en Excel pour comparaison visuelle
        if 'donnees' in extracted_data:
            try:
                ext_df = pd.DataFrame(
                    extracted_data['donnees'], 
                    columns=extracted_data.get('headers_detectes', [])
                )
                excel_output = Path(output_dir) / f"donnees_extraites_{timestamp}.xlsx"
                
                with pd.ExcelWriter(excel_output) as writer:
                    ext_df.to_excel(writer, sheet_name='Données_Extraites', index=False)
                    reference_df.to_excel(writer, sheet_name='Référence', index=False)
                
                self.logger.info(f"💾 Données sauvegardées: {excel_output}")
                
            except Exception as e:
                self.logger.error(f"Erreur sauvegarde Excel: {e}")
        
        # Affichage du résumé
        self.print_test_summary(test_results)
        
        return test_results
    
    def print_test_summary(self, test_results):
        """Affiche un résumé du test"""
        print("\n" + "="*60)
        print("📋 RÉSUMÉ DU TEST D'EXTRACTION")
        print("="*60)
        
        comparison = test_results.get('comparison', {})
        scores = comparison.get('scores', {})
        
        print(f"📂 Image testée: {Path(test_results['test_info']['image_path']).name}")
        print(f"📊 Excel référence: {Path(test_results['test_info']['excel_path']).name}")
        
        if 'global' in scores:
            print(f"\n🎯 SCORE GLOBAL: {scores['global']:.1f}%")
            
            print(f"\n📏 Scores détaillés:")
            for metric, score in scores.items():
                if metric != 'global':
                    print(f"  • {metric}: {score:.1f}%")
        
        # Dimensions
        dims = comparison.get('dimensions', {})
        if dims:
            print(f"\n📊 Comparaison dimensions:")
            print(f"  • Référence: {dims['reference']['lignes']} lignes, {dims['reference']['colonnes']} colonnes")
            print(f"  • Extraite: {dims['extraite']['lignes']} lignes, {dims['extraite']['colonnes']} colonnes")
        
        # Recommandations
        recommendations = comparison.get('recommandations', [])
        if recommendations:
            print(f"\n💡 RECOMMANDATIONS:")
            for rec in recommendations:
                print(f"  {rec}")
        
        print(f"\n📝 Log détaillé: {self.log_file}")
        print("="*60)

    def extract_headers_from_top(self, image_path):
        """
        Extrait d'abord les headers depuis la partie haute de l'image
        
        Args:
            image_path (str): Chemin vers l'image
            
        Returns:
            list: Liste des headers détectés
        """
        try:
            self.logger.info(f"🔍 Extraction des headers depuis la partie haute: {image_path}")
            
            # Charger l'image
            img = Image.open(image_path)
            width, height = img.size
            
            # Prendre seulement les 15-20% du haut de l'image pour les headers
            header_height = int(height * 0.2)  # 20% du haut
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
            
            EXEMPLE:
            Si tu vois: "Rang | Nom Hôpital | Score | Ville"
            Réponds: {"headers_detectes": ["Rang", "Nom Hôpital", "Score", "Ville"], "nombre_colonnes": 4}
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
                self.logger.info(f"✅ Headers détectés: {headers}")
                return headers
            else:
                self.logger.warning("⚠️ Échec extraction headers")
                return []
                
        except Exception as e:
            self.logger.error(f"❌ Erreur extraction headers: {e}")
            return []

def main():
    """Fonction principale pour tester le script"""
    print("🧪 TESTEUR D'EXTRACTION DE TABLEAUX AVEC DÉCOUPAGE")
    print("="*60)
    
    # Configuration
    API_KEY = GOOGLE_API_KEY
    
    if not API_KEY:
        print("❌ Veuillez configurer votre clé API Gemini")
        return
    
    # Options de découpage
    print("\n🔧 CONFIGURATION DU DÉCOUPAGE:")
    print("1. Avec découpage automatique (recommandé pour grandes images)")
    print("2. Sans découpage (image complète)")
    
    choix = input("Votre choix (1 ou 2): ").strip()
    enable_splitting = choix == "1"
    
    if enable_splitting:
        seuil = input("Seuil de largeur pour découpage (défaut: 2000px): ").strip()
        split_threshold = int(seuil) if seuil.isdigit() else 2000
    else:
        split_threshold = 10000  # Très grand pour éviter le découpage
    
    # Chemins des fichiers
    image_test = input("\n📂 Chemin vers l'image de tableau à tester: ").strip().strip('"')
    excel_test = input("📊 Chemin vers l'Excel de référence: ").strip().strip('"')
    
    if not Path(image_test).exists():
        print(f"❌ Image non trouvée: {image_test}")
        return
    
    if not Path(excel_test).exists():
        print(f"❌ Excel non trouvé: {excel_test}")
        return
    
    try:
        # Création du testeur avec options
        tester = TableauExtractorTester(
            API_KEY, 
            enable_image_splitting=enable_splitting,
            split_threshold=split_threshold
        )
        
        print(f"\n🚀 Lancement du test...")
        print(f"📐 Découpage: {'Activé' if enable_splitting else 'Désactivé'}")
        if enable_splitting:
            print(f"🔪 Seuil: {split_threshold}px")
        
        # Lancement du test
        results = tester.test_single_image(image_test, excel_test)
        
        print(f"\n✅ Test terminé! Résultats sauvegardés dans 'test_results/'")
        
        # Affichage d'informations sur le découpage
        if 'nombre_parties' in results.get('extracted_data', {}):
            nb_parties = results['extracted_data']['nombre_parties']
            print(f"🔪 Image découpée en {nb_parties} partie(s)")
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")

if __name__ == "__main__":
    main()
