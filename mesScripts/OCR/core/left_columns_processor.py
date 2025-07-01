"""
Processeur OCR spécialisé pour colonnes gauches des tableaux médicaux.
Traite les images d'un dossier (avec sous-dossiers) contenant "EDIT" dans le nom.
Version améliorée avec mémoire des patterns, validation des classements et support multi-tableaux.
"""

import os
import logging
import cv2
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Import custom modules
from config import DEFAULT_PARAMS, setup_logging
from image_processing import ImageProcessor
from ocr_processing import OCRProcessor, ColumnDetector, TableReconstructor, TextProcessor
from file_operations import FileExporter, ReportGenerator
from dynamic_column_analyzer import DynamicColumnAnalyzer
from pattern_memory import PatternMemory, RankingValidator, MultiTableDetector


class LeftColumnsProcessor:
    """
    Processeur spécialisé pour extraire les colonnes des tableaux médicaux.
    Version améliorée avec mémoire des patterns et validation des classements.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise le processeur pour colonnes gauches avec améliorations.
        
        Args:
            config: Configuration optionnelle
        """
        setup_logging()
        
        # Configuration adaptée pour colonnes gauches
        self.config = DEFAULT_PARAMS.copy()
        if config:
            self.config.update(config)
          # Paramètres spécifiques pour colonnes gauches
        self.config.update({
            'crop_left_ratio': 0.0,     # Commencer à gauche
            'crop_right_ratio': 0.40,   # Prendre 40% de la largeur (valeur par défaut)
            'crop_top_ratio': 0.0,      # Toute la hauteur
            'crop_bottom_ratio': 1.0,   # Toute la hauteur
            'export_excel': True,
            'save_debug': True,
            'enable_dynamic_analysis': True,  # Activer l'analyse dynamique
            'enable_pattern_memory': True,    # Nouvelle fonctionnalité
            'enable_ranking_validation': True, # Validation des classements
            'enable_multi_table_detection': True  # Détection multi-tableaux
        })
        
        # Initialiser les nouveaux composants
        self.pattern_memory = PatternMemory() if self.config.get('enable_pattern_memory') else None
        self.ranking_validator = RankingValidator() if self.config.get('enable_ranking_validation') else None
        self.multi_table_detector = MultiTableDetector() if self.config.get('enable_multi_table_detection') else None
        
        # Initialiser l'analyseur dynamique de colonnes
        self.dynamic_analyzer = DynamicColumnAnalyzer(self.config)
        
        # Initialiser les composants
        self._initialize_components()
        
        logging.info("🚀 Left Columns Processor initialisé avec améliorations")
        if self.pattern_memory:
            logging.info("🧠 Mémoire des patterns activée")
        if self.ranking_validator:
            logging.info("✅ Validation des classements activée")
        if self.multi_table_detector:
            logging.info("🔍 Détection multi-tableaux activée")
    
    def _initialize_components(self) -> None:
        """Initialise les composants de traitement."""
        
        # Traitement d'images
        self.image_processor = ImageProcessor(
            resize_factor=self.config['resize_factor'],
            enable_blur=self.config['blur'],
            enable_thresholding=self.config['thresholding']
        )
        
        # OCR
        from config import TESSERACT_CMD
        self.ocr_processor = OCRProcessor(
            tesseract_cmd=TESSERACT_CMD,
            language=self.config['language'],
            min_conf=self.config['min_conf']
        )
        
        # Détection colonnes et reconstruction
        self.column_detector = ColumnDetector(
            bin_width=self.config['bin_width'],
            peak_min_count=self.config['peak_min_count']
        )
        
        self.table_reconstructor = TableReconstructor(
            tol_y=self.config['tol_y']
        )
        
        # Traitement texte
        self.text_processor = TextProcessor(
            exclude_patterns=self.config['row_exclude_patterns']
        )
        
        # Export
        self.file_exporter = FileExporter("colonnes_gauches")
        self.report_generator = ReportGenerator()
    
    def find_edit_images(self, root_directory: str) -> List[str]:
        """
        Trouve toutes les images contenant "EDIT" dans un dossier et ses sous-dossiers.
        
        Args:
            root_directory: Dossier racine à scanner
            
        Returns:
            Liste des chemins d'images trouvées
        """
        edit_images = []
        
        # Extensions d'images supportées
        image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
        
        logging.info(f"Recherche d'images EDIT dans: {root_directory}")
        
        # Parcourir récursivement le dossier
        root_path = Path(root_directory)
        
        for file_path in root_path.rglob("*"):
            if file_path.is_file():
                # Vérifier l'extension
                if file_path.suffix.lower() in image_extensions:
                    # Vérifier si "EDIT" est dans le nom
                    if "EDIT" in file_path.name.upper():
                        edit_images.append(str(file_path))
                        logging.debug(f"Image EDIT trouvée: {file_path}")
        
        logging.info(f"🔍 {len(edit_images)} image(s) EDIT trouvée(s)")
        return edit_images
    
    def crop_left_columns(self, image: any) -> any:
        """
        Rogne l'image pour ne garder que les colonnes de gauche (40% largeur).
        
        Args:
            image: Image en niveaux de gris
            
        Returns:
            Image rognée (colonnes gauches)
        """
        height, width = image.shape
        
        # Calculer les coordonnées de rogbage
        x1 = int(width * self.config['crop_left_ratio'])      # 0% = début
        x2 = int(width * self.config['crop_right_ratio'])     # 40% = fin
        y1 = int(height * self.config['crop_top_ratio'])      # 0% = haut
        y2 = int(height * self.config['crop_bottom_ratio'])   # 100% = bas
          # Corriger si nécessaire
        y2 = height if y2 > height else y2
        x2 = width if x2 > width else x2
        
        logging.debug(f"Rogbage colonnes gauches: x1={x1}, x2={x2}, y1={y1}, y2={y2}")
        logging.debug(f"Dimensions originales: {width}x{height} -> Nouvelles: {x2-x1}x{y2-y1}")
        
        return image[y1:y2, x1:x2]
    
    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        Traite une seule image pour extraire les colonnes gauches avec analyse dynamique.
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Résultats du traitement
        """
        try:
            logging.info(f"📷 Traitement: {os.path.basename(image_path)}")
            
            # Charger l'image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return {'success': False, 'error': 'Impossible de charger l\'image'}
            
            logging.debug(f"Image chargée: {image.shape}")
            
            # Prétraitement
            preprocessed = self.image_processor.preprocess(image)
              # Analyse dynamique des colonnes si activée
            dynamic_crop_ratio = self.config['crop_right_ratio']  # Valeur par défaut
            analysis_results = None
            
            if self.config.get('enable_dynamic_analysis', True):
                try:
                    analysis_results = self.dynamic_analyzer.analyze_image_structure(preprocessed)
                    dynamic_crop_ratio = analysis_results['optimal_crop_ratio']
                    
                    logging.info(f"🔍 Analyse simplifiée: ratio optimal = {dynamic_crop_ratio:.2f}")
                    logging.info(f"📋 Lignes médicales détectées: {analysis_results['medical_lines_found']}")
                    
                    # Afficher quelques exemples de lignes détectées
                    if analysis_results.get('medical_lines'):
                        for i, line in enumerate(analysis_results['medical_lines'][:3]):
                            logging.debug(f"Ligne {i+1}: {line['text'][:50]}...")
                    
                    # Créer une visualisation de debug pour l'analyse
                    if self.config.get('save_debug', False):
                        debug_analysis_name = f"debug_analysis_{Path(image_path).stem}.png"
                        self.dynamic_analyzer.create_debug_visualization(
                            preprocessed, analysis_results, debug_analysis_name
                        )
                        logging.debug(f"Debug analyse sauvé: {debug_analysis_name}")
                        
                except Exception as e:
                    logging.warning(f"Échec de l'analyse dynamique: {e}")
                    logging.info("Utilisation du ratio fixe par défaut")
            
            # Appliquer le ratio dynamique
            original_ratio = self.config['crop_right_ratio']
            self.config['crop_right_ratio'] = dynamic_crop_ratio
            
            # Rogner pour les colonnes optimales
            left_columns = self.crop_left_columns(preprocessed)
            
            # Restaurer le ratio original
            self.config['crop_right_ratio'] = original_ratio
              # Sauvegarder l'image de debug si demandé
            if self.config.get('save_debug', False):
                debug_name = f"debug_left_cols_{Path(image_path).stem}.png"
                debug_path = os.path.join("outputs", "debug_images", debug_name)
                # Créer le dossier si nécessaire
                os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                cv2.imwrite(debug_path, left_columns)
                logging.debug(f"🖼️ Image debug sauvée: {debug_path}")
            
            # Extraction OCR
            df_ocr = self.ocr_processor.extract_text(left_columns)
            
            if df_ocr.empty:
                return {'success': False, 'error': 'Aucun texte extrait'}
            
            logging.debug(f"OCR: {len(df_ocr)} éléments extraits")
            
            # Détection des colonnes (limité à 2 colonnes max)
            column_centers = self.column_detector.detect_centers(df_ocr)
            
            # Limiter à 2 colonnes maximum
            if len(column_centers) > 2:
                column_centers = column_centers[:2]
                logging.info(f"Limitation à 2 colonnes: {column_centers}")
            
            df_ocr = self.column_detector.assign_columns(df_ocr, column_centers)
            
            # Reconstruction du tableau
            dynamic_tolerance = self.table_reconstructor.compute_dynamic_tolerance(df_ocr)
            self.table_reconstructor.tol_y = dynamic_tolerance
            
            table = self.table_reconstructor.group_by_rows_and_columns(df_ocr)
            
            logging.debug(f"Tableau reconstruit: {len(table)} lignes")
            
            # Traitement du texte
            table = self.text_processor.filter_rows(table)
            table = self.text_processor.recombine_rank_and_hospital(table)
            
            # Garder seulement les 2 premières colonnes si plus de colonnes détectées
            if table and len(table[0]) > 2:
                table = [row[:2] for row in table]
                logging.debug("Tableau limité aux 2 premières colonnes")
            
            logging.info(f"✅ Extraction réussie: {len(table)} lignes, {len(table[0]) if table else 0} colonnes")
            
            return {
                'success': True,
                'image_path': image_path,
                'row_count': len(table),
                'column_count': min(len(column_centers), 2),
                'table_data': table,
                'debug_image': debug_name if self.config.get('save_debug') else None
            }
            
        except Exception as e:
            logging.error(f"Erreur traitement {image_path}: {e}")
            return {'success': False, 'error': str(e), 'image_path': image_path}
    
    def process_directory(self, root_directory: str) -> Dict[str, Any]:
        """
        Traite tous les images EDIT d'un dossier et ses sous-dossiers.
        
        Args:
            root_directory: Dossier racine à traiter
            
        Returns:
            Résultats globaux du traitement
        """
        logging.info(f"🚀 Démarrage traitement dossier: {root_directory}")
        
        # Trouver toutes les images EDIT
        edit_images = self.find_edit_images(root_directory)
        
        if not edit_images:
            return {
                'success': False, 
                'error': 'Aucune image contenant "EDIT" trouvée',
                'total_images': 0,
                'processed_images': 0
            }
        
        # Traiter chaque image
        all_results = []
        successful_count = 0
        failed_count = 0
        
        for i, image_path in enumerate(edit_images, 1):
            logging.info(f"📊 Progression: {i}/{len(edit_images)}")
            
            result = self.process_single_image(image_path)
            all_results.append(result)
            
            if result['success']:
                successful_count += 1
            else:
                failed_count += 1
                logging.warning(f"Échec: {result.get('error', 'Erreur inconnue')}")
        
        # Consolider tous les résultats en un seul tableau
        consolidated_table = self._consolidate_results(all_results)
        
        # Exporter le tableau consolidé
        output_file = None
        if consolidated_table:
            output_file = self.file_exporter.export_table(
                consolidated_table,
                export_excel=self.config.get('export_excel', True),
                timestamp=True
            )
        
        # Générer rapport de traitement
        self._generate_processing_report(all_results, root_directory)
        
        # Résultats finaux
        results = {
            'success': successful_count > 0,
            'total_images': len(edit_images),
            'processed_images': successful_count,
            'failed_images': failed_count,
            'output_file': output_file,
            'consolidated_rows': len(consolidated_table) if consolidated_table else 0,
            'individual_results': all_results
        }
        
        logging.info(f"🎯 Traitement terminé: {successful_count}/{len(edit_images)} images traitées")
        
        return results
    
    def _consolidate_results(self, all_results: List[Dict[str, Any]]) -> List[List[str]]:
        """
        Consolide tous les tableaux extraits en un seul.
        
        Args:
            all_results: Liste de tous les résultats
            
        Returns:
            Tableau consolidé
        """
        consolidated = []
        
        for result in all_results:
            if result['success'] and result.get('table_data'):
                table_data = result['table_data']
                image_name = Path(result['image_path']).stem
                
                # Ajouter le nom de l'image comme préfixe ou colonne séparée
                for row in table_data:
                    # Ajouter nom d'image comme première colonne
                    consolidated_row = [image_name] + row
                    consolidated.append(consolidated_row)
                
                # Ajouter une ligne vide entre les images
                consolidated.append([''] * (len(table_data[0]) + 1 if table_data else 3))
        
        logging.info(f"Consolidation: {len(consolidated)} lignes au total")
        return consolidated
    
    def _generate_processing_report(self, all_results: List[Dict[str, Any]], 
                                  root_directory: str) -> None:
        """
        Génère un rapport détaillé du traitement.
        
        Args:
            all_results: Tous les résultats de traitement
            root_directory: Dossier traité
        """
        try:
            report_data = {
                'root_directory': root_directory,
                'total_images': len(all_results),
                'successful_images': sum(1 for r in all_results if r['success']),
                'failed_images': sum(1 for r in all_results if not r['success']),
                'total_rows_extracted': sum(r.get('row_count', 0) for r in all_results if r['success']),
                'processing_details': all_results
            }
            
            self.report_generator.generate_processing_summary(
                report_data, 
                "rapport_colonnes_gauches.txt"
            )
            
        except Exception as e:
            logging.error(f"Erreur génération rapport: {e}")


def main():
    """Point d'entrée principal pour le traitement des colonnes gauches."""
    import sys
    
    # Dossier par défaut configuré
    default_directory = r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\tri_image_automatique\images_renommees\IMG00001 (8)"
    
    if len(sys.argv) < 2:
        print("Usage: python left_columns_processor.py [dossier_racine]")
        print(f"Dossier par défaut: {default_directory}")
        print("Exemple: python left_columns_processor.py C:/MonDossier/Images")
        root_directory = default_directory
    else:
        root_directory = sys.argv[1]
    
    if not os.path.exists(root_directory):
        print(f"❌ Dossier introuvable: {root_directory}")
        return 1
    
    print("🏥 PROCESSEUR COLONNES GAUCHES - IMAGES EDIT")
    print("=" * 50)
    print(f"📁 Dossier: {root_directory}")
    print("🔍 Recherche: images contenant 'EDIT'")
    print("📏 Zone: 40% largeur (colonnes gauches), 100% hauteur")
    print("=" * 50)
    
    # Créer et lancer le processeur
    processor = LeftColumnsProcessor()
    results = processor.process_directory(root_directory)
    
    # Afficher les résultats
    if results['success']:
        print(f"\n✅ Traitement réussi !")
        print(f"📊 Images traitées: {results['processed_images']}/{results['total_images']}")
        print(f"📝 Lignes extraites: {results['consolidated_rows']}")
        
        if results['output_file']:
            print(f"📁 Fichier généré: {results['output_file']}")
        
        if results['failed_images'] > 0:
            print(f"⚠️  Images échouées: {results['failed_images']}")
    else:
        print(f"\n❌ Échec du traitement: {results.get('error', 'Erreur inconnue')}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
