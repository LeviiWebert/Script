import pandas as pd
import os

def match_excel_files(file1, file2, output_file):
    try:
        # Vérifier que les fichiers existent
        if not os.path.exists(file1):
            raise FileNotFoundError(f"Le fichier {file1} n'existe pas.")
        if not os.path.exists(file2):
            raise FileNotFoundError(f"Le fichier {file2} n'existe pas.")

        # Read the Excel files
        print("Lecture des fichiers Excel...")
        df1 = pd.read_excel(file1)
        df2 = pd.read_excel(file2)
        
        print(f"Fichier 1 - Lignes: {len(df1)}, Colonnes: {list(df1.columns)}")
        print(f"Fichier 2 - Lignes: {len(df2)}, Colonnes: {list(df2.columns)}")

        # Trouver les colonnes communes
        common_columns = list(set(df1.columns) & set(df2.columns))
        
        if not common_columns:
            raise ValueError("Aucune colonne commune trouvée entre les deux fichiers.")
        
        print(f"Colonnes communes trouvées: {common_columns}")
        
        # Utiliser la première colonne commune pour le merge
        # ou demander à l'utilisateur de choisir si plusieurs colonnes communes
        if len(common_columns) == 1:
            merge_column = common_columns[0]
        else:
            print("Plusieurs colonnes communes trouvées:")
            for i, col in enumerate(common_columns, 1):
                print(f"{i}. {col}")
            choice = input("Choisissez le numéro de la colonne pour la correspondance: ")
            try:
                merge_column = common_columns[int(choice) - 1]
            except (ValueError, IndexError):
                print("Choix invalide, utilisation de la première colonne commune.")
                merge_column = common_columns[0]
        
        print(f"Colonne utilisée pour la correspondance: {merge_column}")
            
        # Merge the DataFrames on the selected column
        merged_df = pd.merge(df1, df2, on=merge_column, how='outer', indicator=True, suffixes=('_file1', '_file2'))

        # Save the merged DataFrame to a new Excel file
        merged_df.to_excel(output_file, index=False)
        print(f"Données fusionnées sauvegardées dans {output_file}")
        print(f"Nombre total de lignes dans le fichier fusionné: {len(merged_df)}")
        
        # Afficher un résumé des correspondances
        summary = merged_df['_merge'].value_counts()
        print("\nRésumé des correspondances:")
        print(f"- Présent dans les deux fichiers: {summary.get('both', 0)}")
        print(f"- Uniquement dans le fichier 1: {summary.get('left_only', 0)}")
        print(f"- Uniquement dans le fichier 2: {summary.get('right_only', 0)}")
        
    except Exception as e:
        print(f"Erreur lors du traitement: {e}")
        return False
    
    return True

def main():
    print("=== Fusion de fichiers Excel ===")
    file1 = input('Saisissez le chemin du premier fichier Excel : ').strip()
    file2 = input('Saisissez le chemin du deuxième fichier Excel : ').strip()
    output_file = input('Nom du fichier de sortie (par défaut: matched_output.xlsx): ').strip()
    
    if not output_file:
        output_file = 'matched_output.xlsx'
    
    success = match_excel_files(file1, file2, output_file)
    
    if success:
        print("Traitement terminé avec succès!")
    else:
        print("Le traitement a échoué.")

if __name__ == "__main__":
    main()