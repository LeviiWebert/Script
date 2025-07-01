"""
File operations and export utilities for OCR medical document processing.
Handles saving results in various formats and managing output files.
MODIFICATION: Organisation en sous-dossiers pour une meilleure structure.
"""

import pandas as pd
import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path


class FileExporter:
    """Handles exporting OCR results to various file formats with organized folder structure."""
    
    def __init__(self, output_basename: str = "extraction_tableaux"):
        """
        Initialize file exporter with organized folder structure.
        
        Args:
            output_basename: Base name for output files
        """
        self.output_basename = output_basename
        self._create_output_directories()
    
    def _create_output_directories(self) -> None:
        """Crée la structure de dossiers pour organiser les sorties."""
        directories = [
            "outputs",
            "outputs/excel_files",
            "outputs/debug_images",
            "outputs/reports"
        ]
        
        for directory in directories:            Path(directory).mkdir(parents=True, exist_ok=True)
            
        logging.info("Structure de dossiers créée: outputs/{excel_files, debug_images, reports}")
    
    def export_table(self, table: List[List[str]], export_excel: bool = True, 
                    timestamp: bool = True) -> str:
        """
        Export table to Excel or CSV format in organized folder structure.
        
        Args:
            table: 2D list representing the table (rank and hospital name only)
            export_excel: If True, export to Excel; otherwise CSV
            timestamp: If True, add timestamp to filename
            
        Returns:
            Path to the exported file
        """
        if not table:
            logging.warning("Cannot export empty table")
            return ""
        
        try:
            # Create DataFrame with appropriate headers
            df = pd.DataFrame(table, columns=["Image", "Rang", "Nom Hôpital"])
            
            # Generate filename with folder structure
            filename = self._generate_filename_with_folder(export_excel, timestamp)
              # Export file
            if export_excel:
                df.to_excel(filename, index=False)
                logging.info(f"📊 Excel exporté: {filename}")
            else:
                df.to_csv(filename, index=False)
                logging.info(f"📊 CSV exporté: {filename}")
            
            return filename
            
        except Exception as e:
            logging.error(f"Error exporting table: {e}")
            return ""
    
    def _generate_filename_with_folder(self, is_excel: bool, with_timestamp: bool) -> str:
        """Generate filename with organized folder structure."""
        extension = 'xlsx' if is_excel else 'csv'
        
        if with_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_basename}_{timestamp}.{extension}"
        else:
            filename = f"{self.output_basename}.{extension}"
        
        # Placer dans le sous-dossier approprié
        if is_excel:
            return os.path.join("outputs", "excel_files", filename)
        else:
            return os.path.join("outputs", "excel_files", filename)
    
    def export_debug_images(self, images: Dict[str, any], page_num: int) -> List[str]:
        """
        Export debug images for troubleshooting in organized folder structure.
        
        Args:
            images: Dictionary of image name to image data
            page_num: Page number for naming
            
        Returns:
            List of saved image paths
        """
        import cv2
        
        saved_paths = []
        
        try:
            for image_name, image_data in images.items():
                filename = f"debug_{image_name}_page{page_num}.png"
                # Placer dans le sous-dossier debug_images
                filepath = os.path.join("outputs", "debug_images", filename)
                cv2.imwrite(filepath, image_data)
                saved_paths.append(filepath)
                logging.debug(f"🖼️ Image debug sauvée: {filepath}")
            
            return saved_paths
            
        except Exception as e:
            logging.error(f"Error saving debug images: {e}")
            return []
    
    def _generate_filename(self, is_excel: bool, with_timestamp: bool) -> str:
        """
        Generate filename for export.
        
        Args:
            is_excel: If True, use .xlsx extension; otherwise .csv
            with_timestamp: If True, add timestamp to filename
            
        Returns:
            Generated filename
        """
        extension = 'xlsx' if is_excel else 'csv'
        
        if with_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{self.output_basename}_{timestamp}.{extension}"
        else:
            return f"{self.output_basename}.{extension}"
    
    @staticmethod
    def ensure_directory_exists(directory_path: str) -> bool:
        """
        Ensure that a directory exists, create if necessary.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            True if directory exists or was created successfully
        """
        try:
            os.makedirs(directory_path, exist_ok=True)
            return True
        except Exception as e:
            logging.error(f"Error creating directory {directory_path}: {e}")
            return False


class ReportGenerator:
    """Generates various types of reports for OCR processing."""
    
    def __init__(self):
        """Initialize report generator."""
        pass
    
    def generate_processing_summary(self, processing_stats: Dict[str, Any], 
                                  output_file: str = "processing_summary.txt") -> str:
        """
        Generate a text summary of processing results in organized folder structure.
        
        Args:
            processing_stats: Dictionary with processing statistics
            output_file: Output file path
            
        Returns:
            Path to generated summary file
        """
        try:
            # Placer le rapport dans le sous-dossier reports
            report_path = os.path.join("outputs", "reports", output_file)
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=== RAPPORT TRAITEMENT OCR - COLONNES GAUCHES ===\n\n")
                f.write(f"Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Statistiques de base
                f.write("STATISTIQUES DE TRAITEMENT:\n")
                f.write("-" * 30 + "\n")
                f.write(f"📁 Dossier traité: {processing_stats.get('root_directory', 'N/A')}\n")
                f.write(f"📊 Images trouvées: {processing_stats.get('total_images', 0)}\n")
                f.write(f"✅ Images réussies: {processing_stats.get('successful_images', 0)}\n")
                f.write(f"❌ Images échouées: {processing_stats.get('failed_images', 0)}\n")
                f.write(f"📝 Lignes extraites: {processing_stats.get('total_rows_extracted', 0)}\n\n")
                
                # Détails du traitement
                f.write("DÉTAILS PAR IMAGE:\n")
                f.write("-" * 20 + "\n")
                details = processing_stats.get('processing_details', [])
                for detail in details:
                    image_name = os.path.basename(detail.get('image_path', 'Unknown'))
                    success = "✅" if detail.get('success', False) else "❌"
                    rows = detail.get('row_count', 0)
                    f.write(f"{success} {image_name}: {rows} lignes extraites\n")
                    if not detail.get('success', False):
                        f.write(f"    Erreur: {detail.get('error', 'Inconnue')}\n")
                
                f.write(f"\n📄 Rapport sauvé: {report_path}\n")
                
            logging.info(f"📋 Rapport généré: {report_path}")
            return report_path
            
        except Exception as e:
            logging.error(f"Error generating processing summary: {e}")
            return ""
            
            logging.info(f"Processing summary generated: {output_file}")
            return output_file
            
        except Exception as e:
            logging.error(f"Error generating processing summary: {e}")
            return ""
    
    def generate_error_report(self, errors: List[Dict[str, Any]], 
                            output_file: str = "error_report.txt") -> str:
        """
        Generate a detailed error report.
        
        Args:
            errors: List of error dictionaries
            output_file: Output file path
            
        Returns:
            Path to generated error report
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=== ERROR REPORT ===\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total errors: {len(errors)}\n\n")
                
                if not errors:
                    f.write("No errors detected during processing.\n")
                else:
                    for i, error in enumerate(errors, 1):
                        f.write(f"ERROR #{i}:\n")
                        f.write(f"  Type: {error.get('type', 'Unknown')}\n")
                        f.write(f"  Message: {error.get('message', 'No message')}\n")
                        f.write(f"  Location: {error.get('location', 'Unknown')}\n")
                        f.write(f"  Timestamp: {error.get('timestamp', 'Unknown')}\n")
                        f.write("-" * 40 + "\n")
            
            logging.info(f"Error report generated: {output_file}")
            return output_file
            
        except Exception as e:
            logging.error(f"Error generating error report: {e}")
            return ""


class ConfigManager:
    """Manages configuration files and settings."""
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_file: str = "ocr_config.json") -> bool:
        """
        Save configuration to JSON file.
        
        Args:
            config: Configuration dictionary
            config_file: Path to configuration file
            
        Returns:
            True if saved successfully
        """
        import json
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Configuration saved: {config_file}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
            return False
    
    @staticmethod
    def load_config(config_file: str = "ocr_config.json") -> Optional[Dict[str, Any]]:
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Configuration dictionary or None if failed
        """
        import json
        
        if not os.path.exists(config_file):
            logging.warning(f"Configuration file not found: {config_file}")
            return None
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            logging.info(f"Configuration loaded: {config_file}")
            return config
            
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            return None
    
    @staticmethod
    def merge_configs(default_config: Dict[str, Any], 
                     user_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge user configuration with default configuration.
        
        Args:
            default_config: Default configuration dictionary
            user_config: User configuration dictionary (can be None)
            
        Returns:
            Merged configuration dictionary
        """
        if user_config is None:
            return default_config.copy()
        
        merged_config = default_config.copy()
        merged_config.update(user_config)
        
        logging.debug("Configuration merged with user settings")
        return merged_config
