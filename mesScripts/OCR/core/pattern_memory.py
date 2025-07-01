"""
Gestionnaire de mémoire des patterns de colonnes et validation des données médicales.
Améliore la précision en mémorisant les configurations réussies et valide les classements.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class ColumnPattern:
    """Pattern de colonnes mémorisé pour améliorer la détection."""
    width_ratio: float
    column_count: int
    success_count: int
    typical_structure: List[str]  # ["rang", "hopital", "ville", "departement"]
    confidence_score: float
    image_dimensions: Tuple[int, int]  # (largeur, hauteur) approximatives


class PatternMemory:
    """Gestionnaire de mémoire des patterns de colonnes réussis."""
    
    def __init__(self, memory_file: str = "outputs/patterns_memory.json"):
        """
        Initialise la mémoire des patterns.
        
        Args:
            memory_file: Fichier pour sauvegarder la mémoire
        """
        self.memory_file = memory_file
        self.patterns: List[ColumnPattern] = []
        self.load_memory()
        
        logging.info(f"🧠 Mémoire des patterns initialisée avec {len(self.patterns)} patterns")
    
    def load_memory(self) -> None:
        """Charge la mémoire depuis le fichier."""
        try:
            if Path(self.memory_file).exists():
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        pattern = ColumnPattern(
                            width_ratio=item['width_ratio'],
                            column_count=item['column_count'],
                            success_count=item['success_count'],
                            typical_structure=item['typical_structure'],
                            confidence_score=item['confidence_score'],
                            image_dimensions=tuple(item['image_dimensions'])
                        )
                        self.patterns.append(pattern)
                logging.info(f"📚 {len(self.patterns)} patterns chargés depuis {self.memory_file}")
        except Exception as e:
            logging.warning(f"Impossible de charger la mémoire: {e}")
    
    def save_memory(self) -> None:
        """Sauvegarde la mémoire dans le fichier."""
        try:
            Path(self.memory_file).parent.mkdir(parents=True, exist_ok=True)
            data = []
            for pattern in self.patterns:
                data.append({
                    'width_ratio': pattern.width_ratio,
                    'column_count': pattern.column_count, 
                    'success_count': pattern.success_count,
                    'typical_structure': pattern.typical_structure,
                    'confidence_score': pattern.confidence_score,
                    'image_dimensions': list(pattern.image_dimensions)
                })
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logging.debug(f"💾 Mémoire sauvegardée: {len(data)} patterns")
        except Exception as e:
            logging.error(f"Erreur sauvegarde mémoire: {e}")
    
    def find_best_pattern(self, image_dimensions: Tuple[int, int]) -> Optional[ColumnPattern]:
        """
        Trouve le meilleur pattern pour une image donnée.
        
        Args:
            image_dimensions: Dimensions de l'image (largeur, hauteur)
            
        Returns:
            Meilleur pattern ou None
        """
        if not self.patterns:
            return None
        
        # Trier par score de confiance et succès
        sorted_patterns = sorted(
            self.patterns, 
            key=lambda p: (p.confidence_score, p.success_count), 
            reverse=True
        )
        
        # Chercher un pattern avec des dimensions similaires
        width, height = image_dimensions
        tolerance = 0.2  # 20% de tolérance
        
        for pattern in sorted_patterns:
            p_width, p_height = pattern.image_dimensions
            width_diff = abs(width - p_width) / max(width, p_width)
            height_diff = abs(height - p_height) / max(height, p_height)
            
            if width_diff <= tolerance and height_diff <= tolerance:
                logging.info(f"🎯 Pattern trouvé: ratio={pattern.width_ratio:.2f}, score={pattern.confidence_score:.2f}")
                return pattern
        
        # Si aucun pattern similaire, retourner le meilleur général
        best_pattern = sorted_patterns[0]
        logging.info(f"🔄 Utilisation du meilleur pattern général: ratio={best_pattern.width_ratio:.2f}")
        return best_pattern
    
    def add_successful_pattern(self, width_ratio: float, column_count: int, 
                             structure: List[str], image_dimensions: Tuple[int, int],
                             confidence: float = 1.0) -> None:
        """
        Ajoute un pattern réussi à la mémoire.
        
        Args:
            width_ratio: Ratio de largeur utilisé
            column_count: Nombre de colonnes détectées
            structure: Structure typique détectée
            image_dimensions: Dimensions de l'image
            confidence: Score de confiance
        """
        # Chercher si un pattern similaire existe déjà
        for pattern in self.patterns:
            if (abs(pattern.width_ratio - width_ratio) < 0.05 and 
                pattern.column_count == column_count):
                # Mettre à jour le pattern existant
                pattern.success_count += 1
                pattern.confidence_score = (pattern.confidence_score + confidence) / 2
                logging.debug(f"📈 Pattern mis à jour: succès={pattern.success_count}")
                self.save_memory()
                return
        
        # Créer un nouveau pattern
        new_pattern = ColumnPattern(
            width_ratio=width_ratio,
            column_count=column_count,
            success_count=1,
            typical_structure=structure,
            confidence_score=confidence,
            image_dimensions=image_dimensions
        )
        self.patterns.append(new_pattern)
        logging.info(f"🆕 Nouveau pattern ajouté: ratio={width_ratio:.2f}, colonnes={column_count}")
        self.save_memory()


class RankingValidator:
    """Validateur de classements médicaux pour s'assurer de la cohérence."""
    
    def __init__(self):
        """Initialise le validateur de classements."""
        self.rank_pattern = re.compile(r'^\d+$')
        self.hospital_patterns = [
            re.compile(r'\b(CHU|CHR|CHRU|Clinique|Hôpital|Centre|Polyclinique)\b', re.IGNORECASE),
            re.compile(r'\b(Médical|Hospitalier|Universitaire|Régional)\b', re.IGNORECASE)
        ]
        
    def validate_ranking_sequence(self, extracted_data: List[List[str]]) -> Dict[str, Any]:
        """
        Valide une séquence de classement.
        
        Args:
            extracted_data: Données extraites [[rang, hopital, ville, dept], ...]
            
        Returns:
            Résultats de validation avec métriques
        """
        if not extracted_data:
            return {'valid': False, 'reason': 'Aucune donnée'}
        
        valid_rows = []
        issues = []
        expected_rank = 1
        
        for i, row in enumerate(extracted_data):
            if len(row) < 2:
                issues.append(f"Ligne {i+1}: Données insuffisantes")
                continue
            
            rank_str, hospital = row[0], row[1]
            
            # Validation du rang
            if not self.rank_pattern.match(rank_str.strip()):
                issues.append(f"Ligne {i+1}: Rang invalide '{rank_str}'")
                continue
            
            rank = int(rank_str.strip())
            
            # Validation de la séquence
            if rank != expected_rank:
                if rank == expected_rank + 1:
                    # Rang manqué, tolérable
                    issues.append(f"Ligne {i+1}: Rang {expected_rank} manqué")
                else:
                    issues.append(f"Ligne {i+1}: Séquence brisée, attendu {expected_rank}, trouvé {rank}")
            
            # Validation du nom d'hôpital
            if not self._validate_hospital_name(hospital):
                issues.append(f"Ligne {i+1}: Nom d'hôpital suspect '{hospital[:30]}...'")
            
            valid_rows.append(row)
            expected_rank = rank + 1
        
        validation_score = len(valid_rows) / len(extracted_data) if extracted_data else 0
        
        result = {
            'valid': validation_score >= 0.7,  # 70% de lignes valides minimum
            'score': validation_score,
            'valid_rows': valid_rows,
            'issues': issues,
            'total_rows': len(extracted_data),
            'valid_count': len(valid_rows),
            'ranking_complete': len(issues) == 0
        }
        
        logging.info(f"✅ Validation: {len(valid_rows)}/{len(extracted_data)} lignes valides ({validation_score:.1%})")
        if issues:
            logging.warning(f"⚠️ Problèmes détectés: {len(issues)}")
            for issue in issues[:3]:  # Afficher les 3 premiers problèmes
                logging.warning(f"   - {issue}")
        
        return result
    
    def _validate_hospital_name(self, hospital_name: str) -> bool:
        """
        Valide si un nom ressemble à un nom d'hôpital.
        
        Args:
            hospital_name: Nom à valider
            
        Returns:
            True si le nom semble valide
        """
        if not hospital_name or len(hospital_name.strip()) < 3:
            return False
        
        # Vérifier les patterns d'hôpitaux
        for pattern in self.hospital_patterns:
            if pattern.search(hospital_name):
                return True
        
        # Vérifier qu'il n'y a pas que des chiffres ou caractères spéciaux
        text_chars = re.sub(r'[^\w\s]', '', hospital_name)
        if len(text_chars) < 3:
            return False
        
        return True


class MultiTableDetector:
    """Détecteur de multiples tableaux dans une image."""
    
    def __init__(self):
        """Initialise le détecteur multi-tableaux."""
        self.min_table_width = 0.3  # Largeur minimum d'un tableau (30% de l'image)
        
    def detect_table_regions(self, image_shape: Tuple[int, int]) -> List[Dict[str, float]]:
        """
        Détecte les régions potentielles de tableaux dans une image.
        
        Args:
            image_shape: (hauteur, largeur) de l'image
            
        Returns:
            Liste des régions [{'left': 0.0, 'right': 0.4, 'label': 'gauche'}, ...]
        """
        height, width = image_shape
        regions = []
        
        # Région gauche (colonnes de gauche traditionnelles)
        regions.append({
            'left': 0.0,
            'right': 0.45,  # 45% pour capturer plus de colonnes si nécessaire
            'label': 'gauche',
            'priority': 1
        })
        
        # Région centre-droite (pour les tableaux multiples)
        regions.append({
            'left': 0.5,    # Commencer au centre
            'right': 0.95,  # Jusqu'à 95% pour éviter les bordures
            'label': 'centre-droite', 
            'priority': 2
        })
        
        logging.debug(f"🔍 Régions détectées: {len(regions)} zones à analyser")
        return regions
    
    def should_analyze_region(self, region: Dict[str, float], 
                            previous_results: List[Dict[str, Any]]) -> bool:
        """
        Détermine si une région doit être analysée selon les résultats précédents.
        
        Args:
            region: Région à analyser
            previous_results: Résultats des régions précédentes
            
        Returns:
            True si la région doit être analysée
        """
        # Toujours analyser la région gauche (priorité 1)
        if region['priority'] == 1:
            return True
        
        # Pour les autres régions, analyser seulement si la région gauche a donné des résultats
        left_results = [r for r in previous_results if r.get('region_label') == 'gauche']
        if left_results and left_results[0].get('success', False):
            valid_rows = left_results[0].get('row_count', 0)
            # Analyser la région droite si on a trouvé des données à gauche
            if valid_rows > 0:
                logging.info(f"🔄 Analyse région {region['label']} (région gauche: {valid_rows} lignes)")
                return True
        
        return False
