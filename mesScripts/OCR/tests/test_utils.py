"""
Utilitaires de test pour le processeur OCR médical.
Permet de tester individuellement les différents composants.
"""

import os
import sys
import logging
from typing import Dict, Any, List

# Ajouter le répertoire parent au path pour importer les modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import setup_logging, DEFAULT_PARAMS
from image_processing import PDFProcessor, ImageProcessor
from ocr_processing import OCRProcessor, ColumnDetector, TableReconstructor, TextProcessor
from validation import QualityAssessment, ValidationMetrics
from file_operations import FileExporter


class ComponentTester:
    """Testeur pour les composants individuels du système OCR."""
    
    def __init__(self):
        """Initialise le testeur avec la configuration par défaut."""
        setup_logging()
        self.config = DEFAULT_PARAMS.copy()
        
    def test_pdf_extraction(self, pdf_path: str) -> bool:
        """
        Test l'extraction d'images depuis un PDF.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            True si le test réussit
        """
        print("🔍 Test extraction PDF...")
        
        if not os.path.exists(pdf_path):
            print(f"❌ Fichier PDF introuvable: {pdf_path}")
            return False
            
        try:
            processor = PDFProcessor(self.config['dpi'])
            images = processor.extract_images(pdf_path, "test_images")
            
            if images:
                print(f"✅ {len(images)} image(s) extraite(s)")
                for img in images:
                    print(f"   - {img}")
                return True
            else:
                print("❌ Aucune image extraite")
                return False
                
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction: {e}")
            return False
    
    def test_image_preprocessing(self, image_path: str) -> bool:
        """
        Test le préprocessing d'image.
        
        Args:
            image_path: Chemin vers l'image à traiter
            
        Returns:
            True si le test réussit
        """
        print("🖼️  Test préprocessing image...")
        
        if not os.path.exists(image_path):
            print(f"❌ Image introuvable: {image_path}")
            return False
            
        try:
            import cv2
            
            # Charger l'image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print("❌ Impossible de charger l'image")
                return False
                
            print(f"📏 Image originale: {image.shape}")
            
            # Préprocessing
            processor = ImageProcessor(
                resize_factor=self.config['resize_factor'],
                enable_blur=self.config['blur'],
                enable_thresholding=self.config['thresholding']
            )
            
            processed = processor.preprocess(image)
            print(f"📏 Image traitée: {processed.shape}")
            
            # Test du rogbage
            crop_ratios = {
                'crop_top_ratio': self.config['crop_top_ratio'],
                'crop_bottom_ratio': self.config['crop_bottom_ratio'],
                'crop_left_ratio': self.config['crop_left_ratio'],
                'crop_right_ratio': self.config['crop_right_ratio']
            }
            
            cropped = processor.crop_table_region(processed, crop_ratios)
            print(f"✂️  Image rognée: {cropped.shape}")
            
            # Sauvegarder les résultats de test
            cv2.imwrite("test_preprocessed.png", processed)
            cv2.imwrite("test_cropped.png", cropped)
            print("💾 Images de test sauvegardées: test_preprocessed.png, test_cropped.png")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du préprocessing: {e}")
            return False
    
    def test_ocr_extraction(self, image_path: str) -> bool:
        """
        Test l'extraction OCR.
        
        Args:
            image_path: Chemin vers l'image à traiter
            
        Returns:
            True si le test réussit
        """
        print("📝 Test extraction OCR...")
        
        if not os.path.exists(image_path):
            print(f"❌ Image introuvable: {image_path}")
            return False
            
        try:
            import cv2
            from config import TESSERACT_CMD
            
            # Charger l'image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print("❌ Impossible de charger l'image")
                return False
            
            # Extraction OCR
            ocr_processor = OCRProcessor(
                tesseract_cmd=TESSERACT_CMD,
                language=self.config['language'],
                min_conf=self.config['min_conf']
            )
            
            df = ocr_processor.extract_text(image)
            
            if df.empty:
                print("❌ Aucun texte extrait")
                return False
            
            print(f"✅ {len(df)} éléments de texte extraits")
            print(f"📊 Colonnes disponibles: {list(df.columns)}")
            
            # Afficher quelques échantillons
            if len(df) > 0:
                print("📋 Échantillon de texte extrait:")
                for i, row in df.head(5).iterrows():
                    print(f"   {i}: '{row.get('text', '')}' (conf: {row.get('conf', 0)})")
            
            # Sauvegarder les résultats
            df.to_csv("test_ocr_results.csv", index=False)
            print("💾 Résultats OCR sauvegardés: test_ocr_results.csv")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction OCR: {e}")
            return False
    
    def test_column_detection(self, ocr_data_path: str) -> bool:
        """
        Test la détection de colonnes.
        
        Args:
            ocr_data_path: Chemin vers le fichier CSV des données OCR
            
        Returns:
            True si le test réussit
        """
        print("📊 Test détection colonnes...")
        
        if not os.path.exists(ocr_data_path):
            print(f"❌ Fichier de données OCR introuvable: {ocr_data_path}")
            return False
            
        try:
            import pandas as pd
            
            # Charger les données OCR
            df = pd.read_csv(ocr_data_path)
            
            if df.empty or 'left' not in df.columns:
                print("❌ Données OCR invalides (colonne 'left' manquante)")
                return False
            
            # Détection des colonnes
            detector = ColumnDetector(
                bin_width=self.config['bin_width'],
                peak_min_count=self.config['peak_min_count']
            )
            
            centers = detector.detect_centers(df)
            
            if not centers:
                print("❌ Aucune colonne détectée")
                return False
            
            print(f"✅ {len(centers)} colonne(s) détectée(s)")
            print(f"📍 Centres des colonnes: {centers}")
            
            # Assignation des colonnes
            df_with_cols = detector.assign_columns(df, centers)
            print(f"🔢 Colonnes assignées: {sorted(df_with_cols['col'].unique())}")
            
            # Validation
            is_valid = ValidationMetrics.assess_column_detection(centers)
            print(f"✅ Validation: {'Réussie' if is_valid else 'Échouée'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la détection de colonnes: {e}")
            return False
    
    def test_table_reconstruction(self, ocr_data_path: str) -> bool:
        """
        Test la reconstruction de tableau.
        
        Args:
            ocr_data_path: Chemin vers le fichier CSV des données OCR
            
        Returns:
            True si le test réussit
        """
        print("📋 Test reconstruction tableau...")
        
        try:
            import pandas as pd
            
            # Charger et préparer les données
            df = pd.read_csv(ocr_data_path)
            
            # Simulation de l'assignation des colonnes si nécessaire
            if 'col' not in df.columns:
                detector = ColumnDetector()
                centers = detector.detect_centers(df)
                df = detector.assign_columns(df, centers)
            
            # Reconstruction du tableau
            reconstructor = TableReconstructor(self.config['tol_y'])
            
            # Calcul de la tolérance dynamique
            dynamic_tol = reconstructor.compute_dynamic_tolerance(df)
            reconstructor.tol_y = dynamic_tol
            print(f"🎯 Tolérance dynamique: {dynamic_tol}px")
            
            # Groupage en lignes et colonnes
            table = reconstructor.group_by_rows_and_columns(df)
            
            print(f"✅ Tableau reconstruit: {len(table)} lignes")
            
            if table:
                print(f"📏 Colonnes par ligne: {[len(row) for row in table[:5]]}")
                print("📋 Échantillon de lignes:")
                for i, row in enumerate(table[:3]):
                    print(f"   Ligne {i+1}: {row}")
            
            # Validation du nombre de lignes
            is_valid = ValidationMetrics.assess_row_count(len(table))
            print(f"✅ Validation: {'Réussie' if is_valid else 'Échouée'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la reconstruction: {e}")
            return False
    
    def test_text_processing(self, table_data: List[List[str]]) -> bool:
        """
        Test le traitement de texte.
        
        Args:
            table_data: Données de tableau à traiter
            
        Returns:
            True si le test réussit
        """
        print("🔤 Test traitement texte...")
        
        try:
            processor = TextProcessor(self.config['row_exclude_patterns'])
            
            print(f"📊 Tableau initial: {len(table_data)} lignes")
            
            # Filtrage des lignes
            filtered_table = processor.filter_rows(table_data)
            print(f"🔍 Après filtrage: {len(filtered_table)} lignes")
            
            # Recombinaison rang/hôpital
            recombined_table = processor.recombine_rank_and_hospital(filtered_table)
            print(f"🔗 Après recombinaison: {len(recombined_table)} lignes")
            
            # Détection des outliers
            final_table = processor.detect_and_replace_outliers(recombined_table)
            print(f"📈 Après détection outliers: {len(final_table)} lignes")
            
            if final_table:
                print("📋 Échantillon final:")
                for i, row in enumerate(final_table[:3]):
                    print(f"   Ligne {i+1}: {row[:3]}...")  # Afficher les 3 premières colonnes
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement texte: {e}")
            return False
    
    def run_full_test_suite(self, pdf_path: str) -> Dict[str, bool]:
        """
        Exécute la suite complète de tests.
        
        Args:
            pdf_path: Chemin vers le fichier PDF à tester
            
        Returns:
            Dictionnaire avec les résultats de chaque test
        """
        print("🧪 SUITE DE TESTS COMPLÈTE")
        print("=" * 50)
        
        results = {}
        
        # Test 1: Extraction PDF
        results['pdf_extraction'] = self.test_pdf_extraction(pdf_path)
        print()
        
        # Test 2: Préprocessing (sur la première image extraite)
        if results['pdf_extraction'] and os.path.exists("test_images/page_1.png"):
            results['image_preprocessing'] = self.test_image_preprocessing("test_images/page_1.png")
        else:
            results['image_preprocessing'] = False
            print("⏭️  Test préprocessing ignoré (pas d'image disponible)")
        print()
        
        # Test 3: OCR (sur l'image rognée si disponible)
        ocr_image = "test_cropped.png" if os.path.exists("test_cropped.png") else "test_images/page_1.png"
        if results['image_preprocessing'] or os.path.exists(ocr_image):
            results['ocr_extraction'] = self.test_ocr_extraction(ocr_image)
        else:
            results['ocr_extraction'] = False
            print("⏭️  Test OCR ignoré (pas d'image disponible)")
        print()
        
        # Test 4: Détection colonnes
        if results['ocr_extraction'] and os.path.exists("test_ocr_results.csv"):
            results['column_detection'] = self.test_column_detection("test_ocr_results.csv")
        else:
            results['column_detection'] = False
            print("⏭️  Test colonnes ignoré (pas de données OCR)")
        print()
        
        # Test 5: Reconstruction tableau
        if results['column_detection']:
            results['table_reconstruction'] = self.test_table_reconstruction("test_ocr_results.csv")
        else:
            results['table_reconstruction'] = False
            print("⏭️  Test reconstruction ignoré")
        print()
        
        # Test 6: Traitement texte (données simulées)
        sample_table = [
            ["1", "Hôpital Test", "100", "200", "50"],
            ["2", "Clinique Exemple", "150", "250", "75"],
            ["", "Reader ZINIO", "999", "888", "777"]  # Ligne à filtrer
        ]
        results['text_processing'] = self.test_text_processing(sample_table)
        print()
        
        # Résumé
        print("📊 RÉSUMÉ DES TESTS")
        print("=" * 30)
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ RÉUSSI" if result else "❌ ÉCHOUÉ"
            print(f"{test_name:20} : {status}")
        
        print(f"\n🎯 Score global: {passed}/{total} ({(passed/total*100):.1f}%)")
        
        return results


def main():
    """Point d'entrée principal pour les tests."""
    print("🔬 TESTEUR DE COMPOSANTS OCR MÉDICAL")
    print("=" * 50)
    
    # Vérifier les arguments
    if len(sys.argv) < 2:
        print("Usage: python test_utils.py <chemin_pdf>")
        print("       python test_utils.py <chemin_pdf> <nom_test>")
        print("\nTests disponibles:")
        print("  - pdf_extraction")
        print("  - image_preprocessing")
        print("  - ocr_extraction")
        print("  - column_detection")
        print("  - table_reconstruction")
        print("  - text_processing")
        print("  - full_suite (défaut)")
        return 1
    
    pdf_path = sys.argv[1]
    test_name = sys.argv[2] if len(sys.argv) > 2 else "full_suite"
    
    # Créer le testeur
    tester = ComponentTester()
    
    # Exécuter le test demandé
    if test_name == "full_suite":
        results = tester.run_full_test_suite(pdf_path)
        return 0 if all(results.values()) else 1
    
    elif test_name == "pdf_extraction":
        success = tester.test_pdf_extraction(pdf_path)
    
    elif test_name == "image_preprocessing":
        if len(sys.argv) < 4:
            print("Usage pour ce test: python test_utils.py <pdf_path> image_preprocessing <image_path>")
            return 1
        success = tester.test_image_preprocessing(sys.argv[3])
    
    elif test_name == "ocr_extraction":
        if len(sys.argv) < 4:
            print("Usage pour ce test: python test_utils.py <pdf_path> ocr_extraction <image_path>")
            return 1
        success = tester.test_ocr_extraction(sys.argv[3])
    
    elif test_name == "column_detection":
        if len(sys.argv) < 4:
            print("Usage pour ce test: python test_utils.py <pdf_path> column_detection <csv_path>")
            return 1
        success = tester.test_column_detection(sys.argv[3])
    
    elif test_name == "table_reconstruction":
        if len(sys.argv) < 4:
            print("Usage pour ce test: python test_utils.py <pdf_path> table_reconstruction <csv_path>")
            return 1
        success = tester.test_table_reconstruction(sys.argv[3])
    
    elif test_name == "text_processing":
        sample_data = [["1", "Test Hospital", "100"], ["2", "Example Clinic", "200"]]
        success = tester.test_text_processing(sample_data)
    
    else:
        print(f"Test inconnu: {test_name}")
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
