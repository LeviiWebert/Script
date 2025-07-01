import pandas as pd
from fuzzywuzzy import fuzz
import os

def trouver_meilleur_match_interne(df, colonne, seuil_score=80):
    """
    Pour chaque ligne, trouve la meilleure correspondance dans la même colonne
    et retourne le score et l'objet correspondant si le score > seuil.
    VERSION OPTIMISÉE pour éviter les doublons et accélérer le traitement.

    Parameters:
    df (pd.DataFrame): DataFrame contenant la colonne à analyser.
    colonne (str): Nom de la colonne à analyser.
    seuil_score (int): Score minimum pour considérer une correspondance (défaut: 80).

    Returns:
    tuple: (scores, matches) - Series contenant les scores et les correspondances
    """
    scores = [None] * len(df)
    matches = [None] * len(df)
    
    # Créer une liste des valeurs uniques pour éviter les calculs redondants
    valeurs_uniques = df[colonne].drop_duplicates().tolist()
    print(f"Analyse de {len(df)} lignes ({len(valeurs_uniques)} valeurs uniques)...")
    
    # Cache pour éviter de recalculer les mêmes comparaisons
    cache_comparaisons = {}
    
    for i, row in df.iterrows():
        if i % 500 == 0:  # Affichage du progrès moins fréquent
            print(f"Traitement ligne {i+1}/{len(df)}")
            
        valeur_actuelle = str(row[colonne]).strip()
        
        # Vérifier si on a déjà calculé pour cette valeur
        if valeur_actuelle in cache_comparaisons:
            meilleur_score, meilleur_match = cache_comparaisons[valeur_actuelle]
        else:
            meilleur_score = 0
            meilleur_match = ""
            
            # Comparer seulement avec les valeurs uniques (optimisation majeure)
            for valeur_autre in valeurs_uniques:
                valeur_autre = str(valeur_autre).strip()
                if valeur_actuelle != valeur_autre:  # Ne pas se comparer avec soi-même
                    score = fuzz.ratio(valeur_actuelle, valeur_autre)
                    
                    if score > meilleur_score:
                        meilleur_score = score
                        meilleur_match = valeur_autre
            
            # Mettre en cache le résultat
            cache_comparaisons[valeur_actuelle] = (meilleur_score, meilleur_match)
        
        # Ajouter seulement si le score dépasse le seuil
        if meilleur_score > seuil_score:
            scores[i] = meilleur_score
            matches[i] = meilleur_match
    
    print(f"Comparaisons effectuées: {len(cache_comparaisons)} au lieu de {len(df) * len(df)}")
    return pd.Series(scores), pd.Series(matches)

def traiter_fichier_excel(chemin_fichier, nom_colonne="Nom_Hopital", seuil_score=80):
    """
    Traite un fichier Excel pour détecter les variantes d'établissements
    dans une même colonne (abréviations, variations, etc.).
    
    Parameters:
    chemin_fichier (str): Chemin vers le fichier Excel
    nom_colonne (str): Nom de la colonne à analyser
    seuil_score (int): Score minimum pour considérer une correspondance
    
    Returns:
    pd.DataFrame: DataFrame avec les colonnes de correspondances ajoutées
    """
    try:
        # Lire le fichier Excel
        print(f"Lecture du fichier: {chemin_fichier}")
        df = pd.read_excel(chemin_fichier)
        
        # Vérifier que la colonne existe
        if nom_colonne not in df.columns:
            print(f"Colonnes disponibles: {list(df.columns)}")
            raise ValueError(f"Colonne '{nom_colonne}' non trouvée dans le fichier.")
        
        print(f"Fichier lu avec succès. Nombre de lignes: {len(df)}")
        print(f"Analyse de la colonne: '{nom_colonne}'")
        
        # Trouver les meilleures correspondances
        print(f"Recherche de variantes avec un score > {seuil_score}...")
        scores, matches = trouver_meilleur_match_interne(df, nom_colonne, seuil_score)
        
        # Ajouter les résultats au DataFrame
        df['meilleur_score_variante'] = scores
        df['variante_detectee'] = matches
        
        # Compter les correspondances trouvées
        correspondances_trouvees = scores.notna().sum()
        total_lignes = len(df)
        
        print(f"\nRésultats:")
        print(f"  - Lignes analysées: {total_lignes}")
        print(f"  - Variantes détectées: {correspondances_trouvees}")
        print(f"  - Pourcentage avec variantes: {(correspondances_trouvees/total_lignes)*100:.1f}%")
        
        if correspondances_trouvees > 0:
            score_moyen = scores.dropna().mean()
            score_min = scores.dropna().min()
            score_max = scores.dropna().max()
            
            print(f"  - Score moyen des variantes: {score_moyen:.2f}")
            print(f"  - Score minimum: {score_min}")
            print(f"  - Score maximum: {score_max}")
        
        return df
        
    except Exception as e:
        print(f"Erreur lors du traitement: {str(e)}")
        return None

def sauvegarder_resultat(df, chemin_original, suffixe="_variantes_detectees"):
    """
    Sauvegarde le DataFrame avec les variantes détectées dans un nouveau fichier Excel.
    
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
    Fonction principale pour détecter les variantes d'établissements
    """
    # Demander le chemin du fichier à l'utilisateur
    chemin_fichier = input("Entrez le chemin complet du fichier Excel: ").strip().strip('"')
    
    # Vérifier que le fichier existe
    if not os.path.exists(chemin_fichier):
        print(f"Erreur: Le fichier {chemin_fichier} n'existe pas.")
        return
    
    # Demander le nom de la colonne à analyser
    nom_colonne = input("Entrez le nom de la colonne à analyser (défaut: Nom_Hopital): ").strip()
    if not nom_colonne:
        nom_colonne = "Nom_Hopital"
    
    # Demander le seuil de score
    try:
        seuil_input = input("Entrez le seuil de score minimum (défaut: 80): ").strip()
        seuil_score = int(seuil_input) if seuil_input else 80
    except ValueError:
        seuil_score = 80
        print("Seuil invalide, utilisation de la valeur par défaut: 80")
    
    # Traiter le fichier
    df_resultat = traiter_fichier_excel(chemin_fichier, nom_colonne, seuil_score)
    
    if df_resultat is not None:
        # Afficher quelques exemples de résultats
        variantes_detectees = df_resultat[df_resultat['meilleur_score_variante'].notna()]
        
        if len(variantes_detectees) > 0:
            print(f"\nAperçu des variantes détectées (premières 10):")
            print(variantes_detectees[[nom_colonne, 'meilleur_score_variante', 'variante_detectee']].head(10).to_string(index=False))
            
            # Sauvegarder le résultat
            chemin_sortie = sauvegarder_resultat(df_resultat, chemin_fichier)
            
            if chemin_sortie:
                print(f"\nTraitement terminé avec succès!")
                print(f"Le fichier avec les variantes détectées a été sauvegardé dans: {chemin_sortie}")
            else:
                print("\nErreur lors de la sauvegarde.")
        else:
            print(f"\nAucune variante détectée avec un score > {seuil_score}")
            print("Vous pouvez essayer avec un seuil plus bas.")
    else:
        print("\nErreur lors du traitement du fichier.")

if __name__ == "__main__":
    main()
