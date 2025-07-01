"""
PDF and image processing utilities for OCR medical document processing.
Handles PDF extraction, image preprocessing, and ROI operations.
"""

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz  # Alternative import name
    except ImportError:
        print("❌ PyMuPDF non trouvé. Installez avec: pip install PyMuPDF")
        fitz = None
import cv2
import numpy as np
import os
import logging
from typing import List, Tuple


class PDFProcessor:
    """Handles PDF document processing and image extraction."""
    
    def __init__(self, dpi: int = 300):
        """
        Initialize PDF processor.
        
        Args:
            dpi: Resolution for PDF to image conversion
        """
        self.dpi = dpi
    
    def extract_images(self, pdf_path: str, output_dir: str = "images") -> List[str]:
        """
        Extract images from PDF pages.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images
            
        Returns:
            List of paths to extracted images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if fitz is None:
            logging.error("PyMuPDF not available. Cannot extract PDF images.")
            return []
            
        try:
            doc = fitz.open(pdf_path)
            logging.info(f"{len(doc)} page(s) detected in PDF")
            
            image_paths = []
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=self.dpi)
                image_path = os.path.join(output_dir, f"page_{i+1}.png")
                pix.save(image_path)
                image_paths.append(image_path)
            
            logging.debug(f"PDF images extracted: {image_paths}")
            return image_paths
            
        except Exception as e:
            logging.error(f"Error extracting PDF images: {e}")
            return []
        finally:
            if 'doc' in locals():
                doc.close()


class ImageProcessor:
    """Handles image preprocessing and manipulation."""
    
    def __init__(self, resize_factor: float = 2.0, enable_blur: bool = False, 
                 enable_thresholding: bool = True):
        """
        Initialize image processor.
        
        Args:
            resize_factor: Factor to resize images
            enable_blur: Whether to apply Gaussian blur
            enable_thresholding: Whether to apply binary thresholding
        """
        self.resize_factor = resize_factor
        self.enable_blur = enable_blur
        self.enable_thresholding = enable_thresholding
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to grayscale image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Preprocessed image
        """
        logging.debug(f"Preprocessing image: resize={self.resize_factor}, "
                     f"blur={self.enable_blur}, threshold={self.enable_thresholding}")
        
        processed_img = image.copy()
        
        # Resize if needed
        if self.resize_factor != 1.0:
            processed_img = cv2.resize(
                processed_img, None, 
                fx=self.resize_factor, 
                fy=self.resize_factor
            )
        
        # Apply Gaussian blur if enabled
        if self.enable_blur:
            processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
        
        # Apply binary thresholding if enabled
        if self.enable_thresholding:
            _, processed_img = cv2.threshold(
                processed_img, 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        
        logging.debug(f"Image preprocessed, shape={processed_img.shape}")
        return processed_img
    
    def crop_table_region(self, image: np.ndarray, crop_ratios: dict) -> np.ndarray:
        """
        Crop table region from image based on ratios.
        
        Args:
            image: Input grayscale image
            crop_ratios: Dictionary with crop ratios (top, bottom, left, right)
            
        Returns:
            Cropped image
        """
        height, width = image.shape
        
        y1 = int(height * crop_ratios.get("crop_top_ratio", 0.0))
        y2 = int(height * crop_ratios.get("crop_bottom_ratio", 1.0))
        x1 = int(width * crop_ratios.get("crop_left_ratio", 0.0))
        x2 = int(width * crop_ratios.get("crop_right_ratio", 1.0))
        
        # Ensure valid coordinates
        y2 = height if y2 <= y1 else y2
        x2 = width if x2 <= x1 else x2
        
        logging.debug(f"Cropping ROI: x1={x1}, x2={x2}, y1={y1}, y2={y2}")
        return image[y1:y2, x1:x2]
    
    def split_horizontally(self, roi: np.ndarray, n_slices: int = 1, 
                          slice_height: int = None) -> List[Tuple[int, int, np.ndarray]]:
        """
        Split ROI horizontally into slices.
        
        Args:
            roi: Region of interest image
            n_slices: Number of slices to create
            slice_height: Fixed height for each slice (overrides n_slices)
            
        Returns:
            List of tuples (y1, y2, sub_image)
        """
        height, width = roi.shape
        
        if slice_height:
            step = slice_height
            count = (height + step - 1) // step
        elif n_slices:
            count = n_slices
            step = height // n_slices
        else:
            return [(0, height, roi)]
        
        logging.debug(f"Splitting horizontally into {count} slices (height={step})")
        
        slices = []
        for i in range(count):
            y1 = i * step
            y2 = min(height, (i + 1) * step)
            sub_image = roi[y1:y2, :]
            slices.append((y1, y2, sub_image))
            logging.debug(f"Slice {i}: y1={y1}, y2={y2}, height={y2-y1}")
        
        return slices


class ColorCircleDetector:
    """Detects colored circles in table cells."""
    
    def __init__(self):
        """Initialize color circle detector with HSV ranges."""
        # HSV ranges for green and red circles
        self.green_range = ((40, 50, 50), (80, 255, 255))
        self.red_range_1 = ((0, 50, 50), (10, 255, 255))
        self.red_range_2 = ((160, 50, 50), (180, 255, 255))
        self.min_pixels = 50
    
    def detect_circles(self, roi_color: np.ndarray, row_boxes: List[Tuple[int, int]], 
                      centers: List[float]) -> List[List[str]]:
        """
        Detect colored circles in table cells.
        
        Args:
            roi_color: BGR image of cropped table
            row_boxes: List of (y1, y2) tuples delimiting each row
            centers: List of x-coordinates for column centers
            
        Returns:
            Matrix with '1' for green, '0' for red, '' for no circle
        """
        hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        
        # Create color masks
        mask_green = cv2.inRange(hsv, *self.green_range)
        mask_red1 = cv2.inRange(hsv, *self.red_range_1)
        mask_red2 = cv2.inRange(hsv, *self.red_range_2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        detection_matrix = []
        
        for y1, y2 in row_boxes:
            row_detection = []
            
            for x in centers:
                # Define cell region (square centered on column)
                xi = int(x)
                yi1 = max(0, y1)
                yi2 = min(roi_color.shape[0], y2)
                xi1 = max(0, xi - 10)
                xi2 = min(roi_color.shape[1], xi + 10)
                
                # Extract cell regions from masks
                cell_green = mask_green[yi1:yi2, xi1:xi2]
                cell_red = mask_red[yi1:yi2, xi1:xi2]
                
                # Count non-zero pixels and determine circle type
                if cv2.countNonZero(cell_green) > self.min_pixels:
                    row_detection.append('1')  # Green circle
                elif cv2.countNonZero(cell_red) > self.min_pixels:
                    row_detection.append('0')  # Red circle
                else:
                    row_detection.append('')   # No circle
            
            detection_matrix.append(row_detection)
        
        return detection_matrix
