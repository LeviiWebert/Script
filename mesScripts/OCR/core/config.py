"""
Configuration module for OCR medical document processing.
Contains all parameters and settings for the application.
"""

import logging
import re

# Tesseract configuration
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Logging configuration
def setup_logging():
    """Setup logging configuration for the application."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s] %(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Default parameters for OCR processing
DEFAULT_PARAMS = {
    # OCR settings
    "dpi": 300,
    "language": "fra",
    "resize_factor": 2.0,
    "thresholding": True,
    "blur": False,
    "min_conf": 30,
    
    # Text processing
    "unwanted_chars_pattern": r"[@©]",
    
    # Column detection
    "bin_width": 50,
    "peak_min_count": 20,
    
    # Row grouping
    "tol_y": 8,
    
    # Row filtering patterns
    "row_exclude_patterns": [
        r"\bReader\b",
        r"\bZINIO\b",
        r"https?://",
        r"\d{2}/\d{2}/\d{4}",
        r"\d{2}:\d{2}"
    ],
    
    # Image cropping ratios
    "crop_top_ratio": 0.0,
    "crop_bottom_ratio": 0.0,
    "crop_left_ratio": 0.0,
    "crop_right_ratio": 0.80,
    
    # Image slicing
    "n_slices": 1,
    
    # Output settings
    "save_debug": True,
    "export_excel": True,
    "output_basename": "extraction_tableaux"
}

# Regex patterns for text processing
class RegexPatterns:
    """Container for commonly used regex patterns."""
    
    RANK_LEADING = re.compile(r"^(\d+)(?:er|e|re|r|°|ᵉ)?", re.I)
    NUMERIC = re.compile(r"^\d+([\.,]\d+)?$")
    COLLE = re.compile(r"^(.*\)\s*)(\d{2,})$")
    SYMBOLS_TO_STRIP = re.compile(r"[>%©,…;]")
