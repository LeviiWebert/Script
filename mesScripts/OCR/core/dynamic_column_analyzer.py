"""
Analyseur dynamique de colonnes pour détection automatique des zones d'intérêt.
Utilise des techniques de vision par ordinateur pour identifier automatiquement
les colonnes pertinentes dans les tableaux médicaux.
"""

import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass


@dataclass
class ColumnRegion:
    """Représente une région de colonne détectée."""
    x_start: int
    x_end: int
    width: int
    confidence: float
    text_density: float
    column_index: int


class DynamicColumnAnalyzer:
    """
    Analyseur simplifié qui détecte automatiquement les colonnes pertinentes 
    en cherchant le pattern médical : [rang] [nom hopital], [ville] ([département])
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise l'analyseur dynamique simplifié.
        
        Args:
            config: Configuration optionnelle
        """
        self.config = config or {}
        
        # Pattern pour détecter les lignes d'hôpitaux
        self.hospital_pattern = r'^\s*(\d+)\s+(.+?),\s*(.+?)\s*\((\d{2})\)'
        
        # Paramètres simplifiés
        self.min_text_length = 10  # Longueur minimale de texte pour être considéré
        self.margin_ratio = 0.05   # Marge de sécurité (5%)
        
        logging.info("Dynamic Column Analyzer (simplifié) initialized")
    
    def analyze_image_structure(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyse simplifiée : fait un OCR rapide et cherche le pattern médical.
        
        Args:
            image: Image en niveaux de gris
            
        Returns:
            Dictionnaire contenant l'analyse structurelle
        """
        height, width = image.shape
        
        # 1. OCR rapide sur toute l'image pour détecter le pattern
        ocr_results = self._quick_ocr_scan(image)
        
        # 2. Chercher les lignes qui matchent le pattern médical
        medical_lines = self._find_medical_pattern_lines(ocr_results)
        
        # 3. Déterminer la zone optimale basée sur ces lignes
        optimal_zone = self._calculate_optimal_zone_from_pattern(medical_lines, width)
        
        # 4. Calculer le ratio de crop optimal
        optimal_ratio = optimal_zone['right_x'] / width if optimal_zone['right_x'] > 0 else 0.4
        
        # Limiter le ratio entre 0.3 et 0.9
        optimal_ratio = max(0.3, min(0.9, optimal_ratio))
        
        return {
            'image_dimensions': (width, height),
            'medical_lines_found': len(medical_lines),
            'medical_lines': medical_lines[:5],  # Garder seulement les 5 premières pour debug
            'optimal_zone': optimal_zone,
            'optimal_crop_ratio': optimal_ratio,
            'analysis_method': 'pattern_based_simplified'
        }
    
    def _quick_ocr_scan(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        OCR rapide pour détecter les lignes de texte.
        
        Args:
            image: Image en niveaux de gris
            
        Returns:
            Liste des éléments texte détectés avec leurs positions
        """
        import pytesseract
        
        try:
            # OCR avec données de position
            data = pytesseract.image_to_data(
                image, 
                lang='fra',
                config='--oem 1 --psm 6',
                output_type=pytesseract.Output.DICT
            )
            
            # Regrouper par lignes
            lines = {}
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30 and data['text'][i].strip():  # Confiance > 30%
                    line_num = data['line_num'][i]
                    if line_num not in lines:
                        lines[line_num] = {
                            'texts': [],
                            'left': data['left'][i],
                            'top': data['top'][i],
                            'right': data['left'][i] + data['width'][i],
                            'bottom': data['top'][i] + data['height'][i]
                        }
                    
                    lines[line_num]['texts'].append(data['text'][i])
                    lines[line_num]['left'] = min(lines[line_num]['left'], data['left'][i])
                    lines[line_num]['right'] = max(lines[line_num]['right'], data['left'][i] + data['width'][i])
            
            # Convertir en liste avec texte combiné
            result = []
            for line_num, line_data in lines.items():
                combined_text = ' '.join(line_data['texts']).strip()
                if len(combined_text) > self.min_text_length:
                    result.append({
                        'text': combined_text,
                        'left': line_data['left'],
                        'top': line_data['top'],
                        'right': line_data['right'],
                        'bottom': line_data['bottom'],
                        'line_num': line_num
                    })
            
            return result
            
        except Exception as e:
            logging.warning(f"OCR rapide échoué: {e}")
            return []
    
    def _find_medical_pattern_lines(self, ocr_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Trouve les lignes qui correspondent au pattern médical.
        
        Args:
            ocr_results: Résultats OCR
            
        Returns:
            Liste des lignes qui matchent le pattern médical
        """
        import re
        
        medical_lines = []
        
        for line_data in ocr_results:
            text = line_data['text']
            
            # Chercher le pattern exact: [rang] [hopital], [ville] ([dept])
            match = re.search(self.hospital_pattern, text)
            
            if match:
                rang, hopital, ville, dept = match.groups()
                medical_lines.append({
                    'text': text,
                    'rang': rang,
                    'hopital': hopital.strip(),
                    'ville': ville.strip(),
                    'departement': dept,
                    'left': line_data['left'],
                    'right': line_data['right'],
                    'top': line_data['top'],
                    'bottom': line_data['bottom'],
                    'pattern_match': True
                })
            else:
                # Chercher des patterns moins stricts (juste numéro + texte + ville)
                simple_pattern = r'^\s*(\d+)\s+(.+)'
                simple_match = re.search(simple_pattern, text)
                
                # Ou chercher des mots-clés d'hôpitaux
                hospital_keywords = ['CHU', 'Hôpital', 'Centre', 'Clinique', 'Institut']
                has_hospital_keyword = any(keyword.lower() in text.lower() for keyword in hospital_keywords)
                
                if simple_match and (has_hospital_keyword or len(text) > 20):
                    medical_lines.append({
                        'text': text,
                        'rang': simple_match.group(1) if simple_match else '',
                        'hopital': text,
                        'ville': '',
                        'departement': '',
                        'left': line_data['left'],
                        'right': line_data['right'],
                        'top': line_data['top'],
                        'bottom': line_data['bottom'],
                        'pattern_match': False
                    })
        
        # Trier par position verticale (top)
        medical_lines.sort(key=lambda x: x['top'])
        
        logging.debug(f"Lignes médicales trouvées: {len(medical_lines)}")
        return medical_lines
    
    def _calculate_optimal_zone_from_pattern(self, medical_lines: List[Dict[str, Any]], 
                                           image_width: int) -> Dict[str, Any]:
        """
        Calcule la zone optimale basée sur les lignes médicales détectées.
        
        Args:
            medical_lines: Lignes médicales détectées
            image_width: Largeur de l'image
            
        Returns:
            Zone optimale
        """
        if not medical_lines:
            # Aucune ligne détectée, utiliser valeurs par défaut
            return {
                'left_x': 0,
                'right_x': int(image_width * 0.4),  # 40% par défaut
                'confidence': 0.1,
                'method': 'default_fallback'
            }
        
        # Trouver les limites des lignes médicales
        min_left = min(line['left'] for line in medical_lines)
        max_right = max(line['right'] for line in medical_lines)
        
        # Ajouter une marge de sécurité
        margin = int(image_width * self.margin_ratio)
        
        optimal_left = max(0, min_left - margin)
        optimal_right = min(image_width, max_right + margin)
        
        # Assurer une largeur minimale
        min_width = int(image_width * 0.3)
        if (optimal_right - optimal_left) < min_width:
            optimal_right = min(image_width, optimal_left + min_width)
        
        # Confiance basée sur le nombre de lignes trouvées
        confidence = min(1.0, len(medical_lines) / 10.0)        
        return {
            'left_x': optimal_left,
            'right_x': optimal_right,
            'width': optimal_right - optimal_left,
            'confidence': confidence,
            'lines_analyzed': len(medical_lines),
            'method': 'pattern_based'
        }
    
    def _calculate_optimal_crop_ratio(self, analysis_results: Dict[str, Any], image_width: int) -> float:
        """
        Calcule le ratio de rogbage optimal basé sur l'analyse.
        
        Args:
            analysis_results: Résultats de l'analyse
            image_width: Largeur de l'image
            
        Returns:
            Ratio de rogbage optimal (0.0 à 1.0)
        """
        if 'optimal_zone' in analysis_results:
            ratio = analysis_results['optimal_zone']['right_x'] / image_width
            return max(0.3, min(0.9, ratio))
        
        return 0.4  # Valeur par défaut
    
    def create_debug_visualization(self, image: np.ndarray, analysis: Dict[str, Any], 
                                  output_path: str) -> None:
        """
        Crée une visualisation de debug simplifiée montrant les lignes détectées.
        
        Args:
            image: Image originale
            analysis: Résultats de l'analyse
            output_path: Chemin de sortie
        """
        # Créer une copie colorée de l'image
        debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Dessiner les lignes médicales détectées
        if 'medical_lines' in analysis:
            for i, line in enumerate(analysis['medical_lines']):
                # Rectangle pour chaque ligne médicale
                color = (0, 255, 0) if line.get('pattern_match', False) else (0, 255, 255)
                cv2.rectangle(debug_image, 
                            (line['left'], line['top']), 
                            (line['right'], line['bottom']), 
                            color, 2)
                
                # Numéro de ligne
                cv2.putText(debug_image, str(i+1), 
                           (line['left'], line['top']-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Dessiner la zone optimale
        if 'optimal_zone' in analysis:
            zone = analysis['optimal_zone']
            cv2.rectangle(debug_image, 
                        (zone['left_x'], 0), 
                        (zone['right_x'], image.shape[0]), 
                        (255, 0, 0), 3)
            
            # Texte informatif
            info_text = f"Zone: {zone['left_x']}-{zone['right_x']}px ({zone['width']}px)"
            cv2.putText(debug_image, info_text, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            ratio_text = f"Ratio: {analysis.get('optimal_crop_ratio', 0):.2f}"
            cv2.putText(debug_image, ratio_text, 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            lines_text = f"Lignes medicales: {analysis.get('medical_lines_found', 0)}"
            cv2.putText(debug_image, lines_text, 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Sauvegarder l'image de debug
        cv2.imwrite(output_path, debug_image)
        logging.debug(f"Debug visualization saved: {output_path}")
