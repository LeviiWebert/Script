import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ExcelComparator:
    def __init__(self, file1_path, file2_path, similarity_threshold=80):
        """
        Comparateur Excel avancé
        
        Args:
            file1_path: Chemin vers le premier fichier Excel
            file2_path: Chemin vers le second fichier Excel
            similarity_threshold: Seuil de similitude pour considérer deux colonnes comme similaires (0-100)
        """
        self.file1_path = file1_path
        self.file2_path = file2_path
        self.similarity_threshold = similarity_threshold
        self.df1 = None
        self.df2 = None
        self.column_mapping = {}
        self.common_columns = []
        self.results = {}
        
    def load_files(self):
        """Charge les fichiers Excel"""
        try:
            print(f"Chargement de {self.file1_path}...")
            self.df1 = pd.read_excel(self.file1_path)
            print(f"Fichier 1 chargé: {self.df1.shape[0]} lignes, {self.df1.shape[1]} colonnes")
            
            print(f"Chargement de {self.file2_path}...")
            self.df2 = pd.read_excel(self.file2_path)
            print(f"Fichier 2 chargé: {self.df2.shape[0]} lignes, {self.df2.shape[1]} colonnes")
            
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des fichiers: {e}")
            return False
    
    def find_similar_columns(self):
        """Trouve les colonnes similaires entre les deux DataFrames"""
        columns1 = list(self.df1.columns)
        columns2 = list(self.df2.columns)
        
        print("\n=== ANALYSE DES COLONNES ===")
        print(f"Colonnes fichier 1: {columns1}")
        print(f"Colonnes fichier 2: {columns2}")
        
        # Colonnes exactement identiques
        exact_matches = set(columns1) & set(columns2)
        for col in exact_matches:
            self.column_mapping[col] = col
            self.common_columns.append(col)
        
        print(f"\nColonnes identiques trouvées: {list(exact_matches)}")
        
        # Colonnes similaires (utilisation de fuzzy matching)
        remaining_cols1 = [col for col in columns1 if col not in exact_matches]
        remaining_cols2 = [col for col in columns2 if col not in exact_matches]
        
        similar_pairs = []
        for col1 in remaining_cols1:
            best_match = process.extractOne(col1, remaining_cols2)
            if best_match and best_match[1] >= self.similarity_threshold:
                col2 = best_match[0]
                similarity_score = best_match[1]
                
                # Vérifier que ce n'est pas déjà mappé
                if col2 not in self.column_mapping.values():
                    self.column_mapping[col1] = col2
                    self.common_columns.append(col1)
                    similar_pairs.append((col1, col2, similarity_score))
                    remaining_cols2.remove(col2)
        
        print(f"\nColonnes similaires trouvées (seuil {self.similarity_threshold}%):")
        for col1, col2, score in similar_pairs:
            print(f"  '{col1}' ↔ '{col2}' (similitude: {score}%)")
        
        print(f"\nTotal colonnes communes: {len(self.common_columns)}")
        return len(self.common_columns) > 0
    
    def calculate_cell_similarity(self, val1, val2):
        """Calcule la similitude entre deux cellules"""
        # Gérer les valeurs manquantes
        if pd.isna(val1) and pd.isna(val2):
            return 100.0
        if pd.isna(val1) or pd.isna(val2):
            return 0.0
        
        # Convertir en string pour la comparaison
        str1 = str(val1).strip()
        str2 = str(val2).strip()
        
        # Si identiques
        if str1 == str2:
            return 100.0
        
        # Utiliser fuzzy matching pour les chaînes
        return fuzz.ratio(str1, str2)
    
    def calculate_row_similarity(self, row1, row2, columns):
        """Calcule la similitude moyenne d'une ligne"""
        similarities = []
        for col in columns:
            col2 = self.column_mapping.get(col, col)
            if col in row1.index and col2 in row2.index:
                sim = self.calculate_cell_similarity(row1[col], row2[col2])
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def calculate_column_similarity(self, col1_name):
        """Calcule la similitude globale d'une colonne"""
        col2_name = self.column_mapping.get(col1_name, col1_name)
        
        if col1_name not in self.df1.columns or col2_name not in self.df2.columns:
            return 0.0
        
        series1 = self.df1[col1_name]
        series2 = self.df2[col2_name]
        
        # Aligner les séries (prendre la longueur minimale pour la comparaison)
        min_len = min(len(series1), len(series2))
        
        similarities = []
        for i in range(min_len):
            sim = self.calculate_cell_similarity(series1.iloc[i], series2.iloc[i])
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def generate_detailed_comparison(self):
        """Génère une comparaison détaillée"""
        print("\n=== COMPARAISON DÉTAILLÉE ===")
        
        # 1. Similitude par colonne
        column_similarities = {}
        for col in self.common_columns:
            sim = self.calculate_column_similarity(col)
            column_similarities[col] = sim
            col2 = self.column_mapping.get(col, col)
            print(f"Colonne '{col}' ↔ '{col2}': {sim:.2f}%")
        
        # 2. Créer un DataFrame de comparaison détaillée
        max_rows = max(len(self.df1), len(self.df2))
        
        detailed_results = []
        row_similarities = []
        
        for i in range(max_rows):
            row_data = {'Ligne': i + 1}
            row_sim_values = []
            
            # Obtenir les lignes (ou NaN si index out of bounds)
            row1 = self.df1.iloc[i] if i < len(self.df1) else pd.Series(dtype=object)
            row2 = self.df2.iloc[i] if i < len(self.df2) else pd.Series(dtype=object)
            
            for col in self.common_columns:
                col2 = self.column_mapping.get(col, col)
                
                # Valeurs des cellules
                val1 = row1.get(col, np.nan) if not row1.empty else np.nan
                val2 = row2.get(col2, np.nan) if not row2.empty else np.nan
                
                # Similitude de la cellule
                cell_sim = self.calculate_cell_similarity(val1, val2)
                
                row_data[f'{col}_Fichier1'] = val1
                row_data[f'{col}_Fichier2'] = val2
                row_data[f'{col}_Similitude'] = f"{cell_sim:.1f}%"
                
                row_sim_values.append(cell_sim)
            
            # Similitude moyenne de la ligne
            row_similarity = np.mean(row_sim_values) if row_sim_values else 0.0
            row_data['Similitude_Ligne'] = f"{row_similarity:.1f}%"
            row_similarities.append(row_similarity)
            
            detailed_results.append(row_data)
        
        # Créer le DataFrame des résultats
        results_df = pd.DataFrame(detailed_results)
        
        # 3. Statistiques globales
        global_similarity = np.mean(row_similarities) if row_similarities else 0.0
        
        self.results = {
            'column_similarities': column_similarities,
            'row_similarities': row_similarities,
            'global_similarity': global_similarity,
            'detailed_df': results_df,
            'total_common_columns': len(self.common_columns),
            'total_rows_compared': max_rows
        }
        
        return self.results
    
    def print_summary(self):
        """Affiche un résumé des résultats"""
        if not self.results:
            print("Aucun résultat disponible. Lancez d'abord la comparaison.")
            return
        
        print("\n" + "="*60)
        print("RÉSUMÉ DE LA COMPARAISON")
        print("="*60)
        
        print(f"Colonnes communes identifiées: {self.results['total_common_columns']}")
        print(f"Lignes comparées: {self.results['total_rows_compared']}")
        print(f"Similitude globale: {self.results['global_similarity']:.2f}%")
        
        print("\n--- SIMILITUDE PAR COLONNE ---")
        for col, sim in self.results['column_similarities'].items():
            col2 = self.column_mapping.get(col, col)
            print(f"  {col} ↔ {col2}: {sim:.2f}%")
        
        print(f"\n--- DISTRIBUTION DES SIMILITUDES DE LIGNES ---")
        row_sims = self.results['row_similarities']
        if row_sims:
            print(f"  Minimum: {min(row_sims):.2f}%")
            print(f"  Maximum: {max(row_sims):.2f}%")
            print(f"  Moyenne: {np.mean(row_sims):.2f}%")
            print(f"  Médiane: {np.median(row_sims):.2f}%")
    
    def save_results(self, output_prefix="comparaison_excel"):
        """Sauvegarde les résultats dans des fichiers Excel"""
        if not self.results:
            print("Aucun résultat à sauvegarder.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Fichier de comparaison détaillée
        detailed_file = f"{output_prefix}_detaille_{timestamp}.xlsx"
        self.results['detailed_df'].to_excel(detailed_file, index=False)
        print(f"Comparaison détaillée sauvegardée: {detailed_file}")
        
        # 2. Fichier de résumé
        summary_data = []
        
        # Informations générales
        summary_data.append({
            'Métrique': 'Fichier 1',
            'Valeur': os.path.basename(self.file1_path),
            'Détail': f"{self.df1.shape[0]} lignes, {self.df1.shape[1]} colonnes"
        })
        summary_data.append({
            'Métrique': 'Fichier 2', 
            'Valeur': os.path.basename(self.file2_path),
            'Détail': f"{self.df2.shape[0]} lignes, {self.df2.shape[1]} colonnes"
        })
        summary_data.append({
            'Métrique': 'Colonnes communes',
            'Valeur': self.results['total_common_columns'],
            'Détail': ', '.join(self.common_columns)
        })
        summary_data.append({
            'Métrique': 'Similitude globale',
            'Valeur': f"{self.results['global_similarity']:.2f}%",
            'Détail': f"Basée sur {self.results['total_rows_compared']} lignes"
        })
        
        # Similitudes par colonne
        for col, sim in self.results['column_similarities'].items():
            col2 = self.column_mapping.get(col, col)
            summary_data.append({
                'Métrique': f'Similitude colonne',
                'Valeur': f"{sim:.2f}%",
                'Détail': f"'{col}' ↔ '{col2}'"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = f"{output_prefix}_resume_{timestamp}.xlsx"
        summary_df.to_excel(summary_file, index=False)
        print(f"Résumé sauvegardé: {summary_file}")
        
        return detailed_file, summary_file
    
    def run_complete_analysis(self):
        """Lance l'analyse complète"""
        print("COMPARATEUR EXCEL AVANCÉ")
        print("=" * 40)
        
        # 1. Charger les fichiers
        if not self.load_files():
            return False
        
        # 2. Identifier les colonnes similaires
        if not self.find_similar_columns():
            print("Aucune colonne commune trouvée!")
            return False
        
        # 3. Générer la comparaison détaillée
        self.generate_detailed_comparison()
        
        # 4. Afficher le résumé
        self.print_summary()
        
        # 5. Sauvegarder les résultats
        self.save_results()
        
        return True

# Fonction principale pour utilisation directe
def compare_excel_files(file1, file2, similarity_threshold=80):
    """
    Compare deux fichiers Excel
    
    Args:
        file1: Chemin vers le premier fichier
        file2: Chemin vers le second fichier
        similarity_threshold: Seuil de similitude pour les colonnes (0-100)
    """
    comparator = ExcelComparator(file1, file2, similarity_threshold)
    return comparator.run_complete_analysis()

# Exemple d'utilisation
if __name__ == "__main__":
    # Modifier ces chemins selon vos fichiers
    fichier1 = r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\test_gemini.xlsx"
    fichier2 = r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\Data_matching\LePoint\data_propre_ext_LP-167_Acc_Risque.xlsx"
    
    # Vérifier si les fichiers existent
    if os.path.exists(fichier1) and os.path.exists(fichier2):
        print("Lancement de la comparaison...")
        compare_excel_files(fichier1, fichier2, similarity_threshold=70)
    else:
        print("Fichiers de test non trouvés. Veuillez modifier les chemins dans le script.")
        print(f"Recherche de: {fichier1}")
        print(f"Recherche de: {fichier2}")
        
        # Exemple avec des fichiers personnalisés
        print("\nPour utiliser vos propres fichiers:")
        print("comparator = ExcelComparator('votre_fichier1.xlsx', 'votre_fichier2.xlsx')")
        print("comparator.run_complete_analysis()")