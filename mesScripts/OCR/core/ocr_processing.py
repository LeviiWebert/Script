"""
OCR processing module for medical document analysis.
Handles text extraction, column detection, and table reconstruction.
"""

try:
    import pytesseract
except ImportError:
    print("❌ Tesseract non trouvé. Installez avec: pip install pytesseract")
    pytesseract = None
import pandas as pd
import numpy as np
import logging
import re
from typing import List, Tuple, Dict, Any
from config import RegexPatterns


class OCRProcessor:
    """Handles OCR text extraction and processing."""
    
    def __init__(self, tesseract_cmd: str, language: str = "fra", min_conf: int = 30):
        """
        Initialize OCR processor.
        
        Args:
            tesseract_cmd: Path to Tesseract executable
            language: OCR language code
            min_conf: Minimum confidence threshold
        """
        if pytesseract is None:
            raise ImportError("Tesseract not available")
            
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.language = language
        self.min_conf = min_conf
        self.unwanted_pattern = r"[@©]"
    
    def extract_text(self, image: np.ndarray) -> pd.DataFrame:
        """
        Extract text from image using OCR.
        
        Args:
            image: Grayscale image
            
        Returns:
            DataFrame with OCR results
        """
        try:
            df = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DATAFRAME,
                lang=self.language,
                config="--oem 1 --psm 6"
            )
            
            logging.debug(f"OCR extracted {len(df)} components" if df is not None else "OCR returned None")
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # Filter by confidence and clean text
            df = df[df.conf > self.min_conf].copy()
            df['text'] = df['text'].astype(str).fillna("")
            df = df[~df['text'].str.contains(self.unwanted_pattern, regex=True)]
            
            logging.debug(f"After filtering: {len(df)} components")
            return df.reset_index(drop=True)
            
        except Exception as e:
            logging.error(f"OCR extraction failed: {e}")
            return pd.DataFrame()


class ColumnDetector:
    """Detects column positions in OCR data."""
    
    def __init__(self, bin_width: int = 50, peak_min_count: int = 20):
        """
        Initialize column detector.
        
        Args:
            bin_width: Width of histogram bins in pixels
            peak_min_count: Minimum count for peak detection
        """
        self.bin_width = bin_width
        self.peak_min_count = peak_min_count
    
    def detect_centers(self, df: pd.DataFrame) -> List[float]:
        """
        Detect column centers from OCR data using histogram analysis.
        
        Args:
            df: DataFrame with OCR results containing 'left' column
            
        Returns:
            List of column center x-coordinates
        """
        if df.empty or 'left' not in df.columns:
            return []
        
        # Get x-coordinates
        x_coords = df['left'].values
        if x_coords.size == 0:
            return []
        
        # Create histogram
        min_x, max_x = x_coords.min(), x_coords.max()
        bins = np.arange(min_x, max_x + self.bin_width, self.bin_width)
        hist, edges = np.histogram(x_coords, bins=bins)
        
        # Adaptive threshold: 1% of total words or minimum count
        total_words = len(df)
        min_words_for_col = max(self.peak_min_count, int(total_words * 0.01))
        
        # Detect peaks (valley/peak/valley pattern above threshold)
        centers = []
        for i in range(1, len(hist) - 1):
            is_peak = (hist[i] > hist[i - 1] and 
                      hist[i] > hist[i + 1] and 
                      hist[i] >= min_words_for_col)
            
            if is_peak:
                center = (edges[i] + edges[i + 1]) / 2
                centers.append(center)
        
        centers = sorted(centers)
        logging.debug(f"Column centers detected: {centers} (threshold: {min_words_for_col} words)")
        return centers
    
    def assign_columns(self, df: pd.DataFrame, centers: List[float]) -> pd.DataFrame:
        """
        Assign column indices to OCR data based on centers.
        
        Args:
            df: DataFrame with OCR results
            centers: List of column center coordinates
            
        Returns:
            DataFrame with added 'col' column
        """
        if df.empty or 'left' not in df.columns or not centers:
            df['col'] = 0
            return df
        
        # Assign each word to nearest column center
        df['col'] = df['left'].apply(
            lambda x: int(np.argmin([abs(x - c) for c in centers]))
        )
        
        return df


class TableReconstructor:
    """Reconstructs table structure from OCR data."""
    
    def __init__(self, tol_y: int = 8):
        """
        Initialize table reconstructor.
        
        Args:
            tol_y: Vertical tolerance for grouping words into rows
        """
        self.tol_y = tol_y
    
    def compute_dynamic_tolerance(self, df: pd.DataFrame) -> int:
        """
        Compute dynamic vertical tolerance based on text spacing.
        
        Args:
            df: DataFrame with OCR results containing 'top' column
            
        Returns:
            Dynamic tolerance value
        """
        if df.empty or 'top' not in df.columns:
            return self.tol_y
        
        # Calculate median vertical gap between consecutive words
        y_coords = df.sort_values("top")["top"].to_numpy()
        diffs = np.diff(y_coords)
        diffs = diffs[diffs > 0]  # Remove zero differences
        
        if diffs.size == 0:
            return self.tol_y
        
        median_gap = np.median(diffs)
        new_tolerance = max(6, int(median_gap * 1.2))  # 120% of median or min 6px
        
        logging.debug(f"Dynamic tolerance: {new_tolerance} (median gap: {median_gap:.1f}px)")
        return new_tolerance
    
    def group_by_rows_and_columns(self, df: pd.DataFrame) -> List[List[str]]:
        """
        Group OCR data into table rows and columns.
        
        Args:
            df: DataFrame with OCR results
            
        Returns:
            2D list representing table structure
        """
        if df.empty:
            return []
        
        # Sort by vertical position and column
        df_sorted = df.sort_values(['top', 'col'])
        
        # Group words into rows based on vertical tolerance
        rows = []
        current_row = []
        current_y = None
        
        for _, word_data in df_sorted.iterrows():
            word_y = word_data['top']
            
            # Check if word belongs to current row
            if current_y is None or abs(word_y - current_y) <= self.tol_y:
                current_row.append(word_data)
                current_y = word_y if current_y is None else (current_y + word_y) / 2
            else:
                # Start new row
                if current_row:
                    rows.append(current_row)
                current_row = [word_data]
                current_y = word_y
        
        # Add final row
        if current_row:
            rows.append(current_row)
        
        # Convert to table structure
        n_cols = int(df['col'].max()) + 1 if 'col' in df.columns else 1
        table = []
        
        for row_words in rows:
            # Initialize empty cells for all columns
            cells = ["" for _ in range(n_cols)]
            
            # Fill cells with text from each column
            for word_data in row_words:
                col_idx = int(word_data.get('col', 0))
                if col_idx < n_cols:
                    current_text = cells[col_idx]
                    new_text = str(word_data['text']).strip()
                    cells[col_idx] = (current_text + " " + new_text).strip()
            
            table.append(cells)
        
        return table


class TextProcessor:
    """Handles text cleaning and processing."""
    
    def __init__(self, exclude_patterns: List[str] = None):
        """
        Initialize text processor.
        
        Args:
            exclude_patterns: List of regex patterns for row exclusion
        """
        self.exclude_patterns = exclude_patterns or []
    
    def filter_rows(self, table: List[List[str]]) -> List[List[str]]:
        """
        Filter out unwanted rows based on patterns.
        
        Args:
            table: 2D list representing table
            
        Returns:
            Filtered table
        """
        filtered_table = []
        
        for row in table:
            # Join row text for pattern matching
            row_text = " ".join(row).strip()
            
            # Skip empty rows
            if not row_text:
                continue
            
            # Check against exclude patterns
            should_exclude = any(
                re.search(pattern, row_text, re.IGNORECASE) 
                for pattern in self.exclude_patterns
            )            
            if not should_exclude:
                filtered_table.append(row)
        
        return filtered_table
    
    def recombine_rank_and_hospital(self, table: List[List[str]]) -> List[List[str]]:
        """
        Recombine rank numbers and hospital names that may have been split.
        MODIFICATION: Ne garde que le rang et le nom d'hôpital (pas les autres colonnes).
        
        Args:
            table: 2D list representing table
            
        Returns:
            Table with only rank and hospital names (2 columns max)
        """
        processed_table = []
        
        for row in table:
            if not row:
                continue
            
            cells = list(row) + [""]  # Add empty cell as buffer
            rank = ""
            hospital_parts = []
            
            # Extract rank from first cell
            first_cell = cells[0].strip()
            rank_match = RegexPatterns.RANK_LEADING.match(first_cell)
            
            if rank_match:
                rank = rank_match.group(1)
                remainder = first_cell[rank_match.end():].lstrip(" ,-")
                cells[0] = remainder
                
                # Remove first cell if it's now empty
                if not remainder:
                    cells.pop(0)
            
            # Collect hospital name parts (non-numeric cells)
            while cells and not RegexPatterns.NUMERIC.fullmatch(cells[0].strip()):
                hospital_parts.append(cells.pop(0).strip())
            
            # Combine hospital name
            hospital_name = " ".join(hospital_parts)
            
            # Handle cases where numeric data is stuck to hospital name
            colle_match = RegexPatterns.COLLE.match(hospital_name)
            if colle_match:
                hospital_name = colle_match.group(1).strip()
                # NOTE: On ignore les données numériques (colle_match.group(2))
            
            # MODIFICATION: Construire seulement rang et nom hôpital (2 colonnes)
            final_row = [rank, hospital_name]
            
            # Ne garder que les lignes avec rang ET nom d'hôpital non vides
            if rank.strip() and hospital_name.strip():
                processed_table.append(final_row)
        
        return processed_table
    
    def detect_and_replace_outliers(self, table: List[List[str]]) -> List[List[str]]:
        """
        Detect and replace outlier values in numeric columns.
        
        Args:
            table: 2D list representing table
            
        Returns:
            Table with outliers replaced
        """
        if not table:
            return table
        
        max_cols = max(len(row) for row in table)
        
        # Process each column starting from column 2 (skip rank and hospital name)
        for col_idx in range(2, max_cols):
            numeric_values = []
            
            # Collect numeric values from column
            for row in table:
                if len(row) > col_idx:
                    try:
                        value = float(row[col_idx].replace(',', '.'))
                        numeric_values.append(value)
                    except (ValueError, AttributeError):
                        pass
            
            if not numeric_values:
                logging.debug(f"Column {col_idx}: no numeric values detected")
                continue
            
            # Calculate statistics
            mean_val = np.mean(numeric_values)
            std_val = np.std(numeric_values)
            
            logging.debug(f"Column {col_idx}: mean={mean_val:.2f}, std={std_val:.2f}")
            
            # Replace outliers (values beyond 3 standard deviations)
            for i, row in enumerate(table):
                if len(row) > col_idx:
                    try:
                        value = float(row[col_idx].replace(',', '.'))
                        if abs(value - mean_val) > 3 * std_val:
                            logging.info(f"Outlier detected row {i+1} col {col_idx}: "
                                       f"{value} (replaced with 'à remplir')")
                            table[i][col_idx] = 'à remplir'
                    except (ValueError, AttributeError):
                        pass
        
        return table
    
    @staticmethod
    def strip_symbols(text: str) -> str:
        """
        Remove unwanted symbols from text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        return RegexPatterns.SYMBOLS_TO_STRIP.sub("", text).strip()
