"""
Main OCR Medical Document Processor
Orchestrates the entire OCR processing pipeline for medical documents.

This module provides a high-level interface to process PDF medical documents,
extract table data using OCR, and generate structured output files.
"""

import os
import logging
try:
    import cv2
except ImportError:
    print("❌ OpenCV non trouvé. Installez avec: pip install opencv-python")
    cv2 = None

try:
    import pandas as pd
except ImportError:
    print("❌ Pandas non trouvé. Installez avec: pip install pandas")
    pd = None
from typing import Dict, Any, List, Optional

# Import custom modules
from config import DEFAULT_PARAMS, TESSERACT_CMD, setup_logging
from image_processing import PDFProcessor, ImageProcessor, ColorCircleDetector
from ocr_processing import OCRProcessor, ColumnDetector, TableReconstructor, TextProcessor
from validation import QualityAssessment, ValidationMetrics
from file_operations import FileExporter, ReportGenerator, ConfigManager


class MedicalDocumentProcessor:
    """
    Main processor for medical document OCR analysis.
    
    This class orchestrates the entire pipeline:
    1. PDF extraction and image preprocessing
    2. OCR text extraction and column detection
    3. Table reconstruction and data cleaning
    4. Quality validation and output generation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the medical document processor.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        # Setup logging
        setup_logging()
        
        # Load and merge configuration
        user_config = ConfigManager.load_config() if config is None else config
        self.config = ConfigManager.merge_configs(DEFAULT_PARAMS, user_config)
        
        # Initialize processing components
        self._initialize_components()
        
        # Initialize utilities
        self.file_exporter = FileExporter(self.config['output_basename'])
        self.quality_assessor = QualityAssessment()
        self.report_generator = ReportGenerator()
        
        logging.info("Medical Document Processor initialized")
    
    def _initialize_components(self) -> None:
        """Initialize all processing components with configuration."""
        
        # PDF and image processing
        self.pdf_processor = PDFProcessor(self.config['dpi'])
        self.image_processor = ImageProcessor(
            resize_factor=self.config['resize_factor'],
            enable_blur=self.config['blur'],
            enable_thresholding=self.config['thresholding']
        )
        
        # OCR processing
        self.ocr_processor = OCRProcessor(
            tesseract_cmd=TESSERACT_CMD,
            language=self.config['language'],
            min_conf=self.config['min_conf']
        )
        
        # Column and table processing
        self.column_detector = ColumnDetector(
            bin_width=self.config['bin_width'],
            peak_min_count=self.config['peak_min_count']
        )
        
        self.table_reconstructor = TableReconstructor(
            tol_y=self.config['tol_y']
        )
        
        # Text processing
        self.text_processor = TextProcessor(
            exclude_patterns=self.config['row_exclude_patterns']
        )
        
        # Color detection (if needed)
        self.color_detector = ColorCircleDetector()
    
    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a complete medical document.
        
        Args:
            pdf_path: Path to the PDF file to process
            
        Returns:
            Dictionary with processing results and statistics
        """
        if not os.path.isfile(pdf_path):
            logging.error(f"PDF file not found: {pdf_path}")
            return {'success': False, 'error': 'File not found'}
        
        logging.info(f"Starting document processing: {pdf_path}")
        
        try:
            # Extract images from PDF
            image_paths = self.pdf_processor.extract_images(pdf_path)
            if not image_paths:
                return {'success': False, 'error': 'No images extracted from PDF'}
            
            # Process each page
            all_results = []
            processing_stats = {
                'total_pages': len(image_paths),
                'successful_pages': 0,
                'failed_pages': 0,
                'total_rows_detected': 0,
                'output_files': []
            }
            
            for page_num, image_path in enumerate(image_paths, 1):
                logging.info(f"Processing page {page_num}: {image_path}")
                
                page_result = self._process_page(image_path, page_num)
                
                if page_result['success']:
                    all_results.append(page_result)
                    processing_stats['successful_pages'] += 1
                    processing_stats['total_rows_detected'] += page_result['row_count']
                    
                    if page_result.get('output_file'):
                        processing_stats['output_files'].append(page_result['output_file'])
                else:
                    processing_stats['failed_pages'] += 1
                    logging.error(f"Failed to process page {page_num}: {page_result.get('error', 'Unknown error')}")
            
            # Generate final report
            if processing_stats['successful_pages'] > 0:
                self._generate_final_reports(processing_stats, all_results)
            
            logging.info(f"Document processing completed. "
                        f"Success: {processing_stats['successful_pages']}/{processing_stats['total_pages']} pages")
            
            return {
                'success': processing_stats['successful_pages'] > 0,
                'statistics': processing_stats,
                'results': all_results
            }
            
        except Exception as e:
            logging.error(f"Error processing document: {e}")
            return {'success': False, 'error': str(e)}
    
    def _process_page(self, image_path: str, page_num: int) -> Dict[str, Any]:
        """
        Process a single page image.
        
        Args:
            image_path: Path to the page image
            page_num: Page number for logging and naming
            
        Returns:
            Dictionary with page processing results
        """
        try:
            # Load and preprocess image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return {'success': False, 'error': 'Could not load image'}
            
            logging.debug(f"Image loaded, shape: {image.shape}")
            
            # Preprocess image
            preprocessed_image = self.image_processor.preprocess(image)
            
            # Crop table region
            crop_ratios = {
                'crop_top_ratio': self.config['crop_top_ratio'],
                'crop_bottom_ratio': self.config['crop_bottom_ratio'],
                'crop_left_ratio': self.config['crop_left_ratio'],
                'crop_right_ratio': self.config['crop_right_ratio']
            }
            roi = self.image_processor.crop_table_region(preprocessed_image, crop_ratios)
            
            # Save debug image if requested
            if self.config.get('save_debug', False):
                debug_crop_path = f"debug_crop_page{page_num}.png"
                cv2.imwrite(debug_crop_path, roi)
                logging.debug(f"Debug crop image saved: {debug_crop_path}")
            
            # Split ROI into slices if needed
            slices = self.image_processor.split_horizontally(
                roi, 
                n_slices=self.config.get('n_slices', 1)
            )
            
            # Process each slice with OCR
            ocr_results = []
            for slice_idx, (y1, y2, slice_image) in enumerate(slices):
                if self.config.get('save_debug', False):
                    debug_slice_path = f"debug_slice{slice_idx+1}_page{page_num}.png"
                    cv2.imwrite(debug_slice_path, slice_image)
                    logging.debug(f"Debug slice saved: {debug_slice_path}")
                
                # Extract text from slice
                df_slice = self.ocr_processor.extract_text(slice_image)
                
                if not df_slice.empty:
                    # Adjust coordinates to global ROI
                    df_slice['top'] += y1
                    ocr_results.append(df_slice)
            
            # Combine all OCR results
            if ocr_results:
                combined_df = pd.concat(ocr_results, ignore_index=True)
            else:
                combined_df = pd.DataFrame()
            
            logging.debug(f"Total OCR results: {len(combined_df)} components")
            
            if combined_df.empty:
                return {'success': False, 'error': 'No text extracted from page'}
            
            # Detect columns
            column_centers = self.column_detector.detect_centers(combined_df)
            combined_df = self.column_detector.assign_columns(combined_df, column_centers)
            
            logging.debug(f"Columns detected: {len(column_centers)}")
            
            # Validate column detection
            ValidationMetrics.assess_column_detection(column_centers)
            
            # Compute dynamic tolerance and reconstruct table
            dynamic_tolerance = self.table_reconstructor.compute_dynamic_tolerance(combined_df)
            self.table_reconstructor.tol_y = dynamic_tolerance
            
            table = self.table_reconstructor.group_by_rows_and_columns(combined_df)
            
            logging.info(f"Rows detected: {len(table)}")
            
            # Validate row count
            ValidationMetrics.assess_row_count(len(table))
            
            # Process and clean table
            table = self.text_processor.filter_rows(table)
            table = self.text_processor.recombine_rank_and_hospital(table)
            table = self.text_processor.detect_and_replace_outliers(table)
            
            logging.debug(f"Final processed table: {len(table)} rows")
            
            # Export results
            output_file = self.file_exporter.export_table(
                table, 
                export_excel=self.config.get('export_excel', True)
            )
            
            # Run quality assessment if reference file exists
            quality_results = None
            reference_file = "ext_LP-167_Acc_Risque.xlsx"
            if output_file and os.path.exists(reference_file):
                quality_results = self.quality_assessor.compare_with_reference(
                    output_file, reference_file
                )
            
            return {
                'success': True,
                'page_number': page_num,
                'row_count': len(table),
                'column_count': len(column_centers),
                'output_file': output_file,
                'quality_results': quality_results,
                'table_data': table
            }
            
        except Exception as e:
            logging.error(f"Error processing page {page_num}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_final_reports(self, processing_stats: Dict[str, Any], 
                               all_results: List[Dict[str, Any]]) -> None:
        """
        Generate final processing reports.
        
        Args:
            processing_stats: Overall processing statistics
            all_results: List of all page results
        """
        try:
            # Generate processing summary
            summary_stats = processing_stats.copy()
            
            # Add quality metrics if available
            quality_metrics = {}
            for result in all_results:
                if result.get('quality_results'):
                    quality_stats = result['quality_results'].get('statistics', {})
                    for key, value in quality_stats.items():
                        if key not in quality_metrics:
                            quality_metrics[key] = []
                        quality_metrics[key].append(value)
            
            # Average quality metrics
            if quality_metrics:
                avg_quality = {}
                for key, values in quality_metrics.items():
                    if values:
                        avg_quality[f"avg_{key}"] = sum(values) / len(values)
                summary_stats['quality_metrics'] = avg_quality
            
            # Generate reports
            self.report_generator.generate_processing_summary(summary_stats)
            
            logging.info("Final reports generated successfully")
            
        except Exception as e:
            logging.error(f"Error generating final reports: {e}")
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update processor configuration.
        
        Args:
            new_config: New configuration parameters
        """
        self.config.update(new_config)
        self._initialize_components()
        logging.info("Configuration updated and components reinitialized")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Current configuration dictionary
        """
        return self.config.copy()


def main():
    """
    Main entry point for the application.
    Processes PDF files or redirects to left columns processor for image directories.
    """
    import sys
    
    # Check if we have a command line argument
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        # Load default directory from config
        try:
            config_manager = ConfigManager()
            user_config = config_manager.load_config()
            default_path = user_config.get('default_directory', 'scan/1.pdf')
        except:
            default_path = 'scan/1.pdf'
        input_path = default_path
    
    # Check if input is a directory (for image processing) or PDF file
    if os.path.isdir(input_path):
        print(f"📁 Détection d'un dossier: {input_path}")
        print("🔄 Redirection vers le processeur de colonnes gauches...")
          # Import and run left columns processor
        try:
            from left_columns_processor import LeftColumnsProcessor
            processor = LeftColumnsProcessor()
            results = processor.process_directory(input_path)
            
            if results['success']:
                print(f"\n✅ Traitement des images terminé avec succès!")
                print(f"📊 Images traitées: {results['processed_images']}/{results['total_images']}")
                print(f"📝 Lignes extraites: {results['consolidated_rows']}")
                if results.get('output_file'):
                    print(f"📁 Fichier généré: {results['output_file']}")
                return 0
            else:
                print(f"\n❌ Échec du traitement: {results.get('error', 'Erreur inconnue')}")
                return 1
                
        except ImportError as e:
            print(f"❌ Erreur d'import: {e}")
            return 1
    else:
        # Process as PDF file
        print(f"📄 Traitement du fichier PDF: {input_path}")
        
        # Create processor and process document
        processor = MedicalDocumentProcessor()
        
        # Process the document
        results = processor.process_document(input_path)        
        if results['success']:
            print(f"\n✅ Processing completed successfully!")
            print(f"📊 Pages processed: {results['statistics']['successful_pages']}/{results['statistics']['total_pages']}")
            print(f"📝 Total rows detected: {results['statistics']['total_rows_detected']}")
            print(f"📁 Output files: {len(results['statistics']['output_files'])}")
            
            for output_file in results['statistics']['output_files']:
                print(f"   - {output_file}")
            return 0
        else:
            print(f"\n❌ Processing failed: {results.get('error', 'Unknown error')}")
            return 1


if __name__ == '__main__':
    exit(main())
