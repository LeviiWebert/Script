import pandas as pd
from fuzzywuzzy import fuzz
import os

def calcul_score_fuzzy(df, col1, col2):
    """
    Calculate fuzzy matching scores between two columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the columns to compare.
    col1 (str): Name of the first column.
    col2 (str): Name of the second column.

    Returns:
    pd.Series: Series containing the fuzzy matching scores.
    """
    return df.apply(lambda row: fuzz.ratio(str(row[col1]), str(row[col2])), axis=1)

def traiter_fichier_excel(chemin_fichier, nom_colonne_sortie="score_fuzzy"):
    """
    Traite un fichier Excel pour calculer les scores de similarité fuzzy
    entre les colonnes 'Nom' et 'NomScanSante'.
    
    Parameters:
    chemin_fichier (str): Chemin vers le fichier Excel
    nom_colonne_sortie (str): Nom de la colonne de score à ajouter
    
    Returns:
    pd.DataFrame: DataFrame avec la colonne de score ajoutée
    """
    try:
        # Lire le fichier Excel
        print(f"Lecture du fichier: {chemin_fichier}")
        df = pd.read_excel(chemin_fichier)
        
        # Vérifier que les colonnes existent
        colonnes_requises = ['Nom', 'NomScanSante']
        colonnes_manquantes = [col for col in colonnes_requises if col not in df.columns]
        
        if colonnes_manquantes:
            raise ValueError(f"Colonnes manquantes dans le fichier: {colonnes_manquantes}")
        
        print(f"Fichier lu avec succès. Nombre de lignes: {len(df)}")
        print(f"Colonnes disponibles: {list(df.columns)}")
        
        # Calculer les scores fuzzy
        print("Calcul des scores de similarité fuzzy...")
        df[nom_colonne_sortie] = calcul_score_fuzzy(df, 'Nom', 'NomScanSante')
        
        # Statistiques des scores
        score_moyen = df[nom_colonne_sortie].mean()
        score_min = df[nom_colonne_sortie].min()
        score_max = df[nom_colonne_sortie].max()
        
        print(f"Scores calculés:")
        print(f"  - Score moyen: {score_moyen:.2f}")
        print(f"  - Score minimum: {score_min}")
        print(f"  - Score maximum: {score_max}")
        
        return df
        
    except Exception as e:
        print(f"Erreur lors du traitement: {str(e)}")
        return None

def sauvegarder_resultat(df, chemin_original, suffixe="_avec_scores"):
    """
    Sauvegarde le DataFrame avec les scores dans un nouveau fichier Excel.
    
    Parameters:
    df (pd.DataFrame): DataFrame à sauvegarder
    chemin_original (str): Chemin du fichier original
    suffixe (str): Suffixe à ajouter au nom du fichier
    """
    try:
        # Créer le nom du fichier de sortie
        base_nom = os.path.splitext(chemin_original)[0]
        extension = os.path.splitext(chemin_original)[1]
        chemin_sortie = f"{base_nom}{suffixe}{extension}"
        
        # Sauvegarder
        df.to_excel(chemin_sortie, index=False)
        print(f"Fichier sauvegardé: {chemin_sortie}")
        
        return chemin_sortie
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {str(e)}")
        return None

def main():
    """
    Fonction principale pour traiter le fichier Excel
    """
    # Demander le chemin du fichier à l'utilisateur
    chemin_fichier = input("Entrez le chemin complet du fichier Excel: ").strip().strip('"')
    
    # Vérifier que le fichier existe
    if not os.path.exists(chemin_fichier):
        print(f"Erreur: Le fichier {chemin_fichier} n'existe pas.")
        return
    
    # Traiter le fichier
    df_resultat = traiter_fichier_excel(chemin_fichier)
    
    if df_resultat is not None:
        # Afficher quelques exemples de résultats
        print("\nAperçu des résultats:")
        print(df_resultat[['Nom', 'NomScanSante', 'score_fuzzy']].head(10))
        
        # Sauvegarder le résultat
        chemin_sortie = sauvegarder_resultat(df_resultat, chemin_fichier)
        
        if chemin_sortie:
            print(f"\nTraitement terminé avec succès!")
            print(f"Le fichier avec les scores a été sauvegardé dans: {chemin_sortie}")
        else:
            print("\nErreur lors de la sauvegarde.")
    else:
        print("\nErreur lors du traitement du fichier.")

if __name__ == "__main__":
    main()
