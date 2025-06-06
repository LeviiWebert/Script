#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
import os
import sys

def extract_significant(name, stopwords):
    """
    Extrait les mots “significatifs” d’une chaîne `name` en éliminant
    tous les tokens dont la forme uppercase correspond à une stopword.
    """
    if pd.isna(name):
        return ""
    # On récupère tous les mots (lettres latines + accents)
    # par exemple, dans "CHU - Hôpital Jeanne-de-Flandre,Lille(59)" 
    # on récupère ["CHU", "Hôpital", "Jeanne", "de", "Flandre", "Lille"]
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", name)
    # On filtre : on ne garde que ceux qui ne sont pas dans stopwords (majuscules)
    significant = [w for w in words if w.upper() not in stopwords]
    # On recompose la chaîne (séparateur espace)
    return " ".join(significant)

def main():
    # 1) Chemin vers le fichier d’entrée et sortie
    input_path = "C:/Users/LeviWEBERT/OneDrive - ALBUS PARTNERS/Bureau/Scan Medecine/TABLEAU à TRAIté/data_propre_ext_LP-167_Acc_Risque.xlsx"
    output_path = "C:/Users/LeviWEBERT/OneDrive - ALBUS PARTNERS/Bureau/Scan Medecine/TABLEAU à TRAIté/data_propre_ext_LP-167_Acc_Risque_with_mots_significatifs.xlsx"


    if not os.path.isfile(input_path):
        print(f"❌ Le fichier d’entrée n’existe pas : {input_path}", file=sys.stderr)
        sys.exit(1)

    # 2) Lecture du fichier Excel
    print("🔄 Lecture du fichier Excel……")
    try:
        df = pd.read_excel(input_path, dtype=str)
    except Exception as e:
        print(f"❌ Impossible de lire le fichier Excel : {e}", file=sys.stderr)
        sys.exit(1)

    # 3) Définition des mots génériques à retirer (en majuscules pour la comparaison)
    stopwords = {
        "CHU",       # Centre Hospitalier Universitaire
        "CH",        # Centre Hospitalier
        "GBU",       # Groupe Hospitalier Universitaire (voire variante)
        "HÔPITAL",   # Hôpital (avec accent)
        "HOPITAL",   # Hôpital (sans accent)
        "CLINIQUE",  # Clinique
        "CHI"
    }

    # 4) Vérifier que la colonne “Hopital” existe
    if "Nom hopital" not in df.columns:
        print("❌ La colonne “Nom hopital” n’a pas été trouvée dans le fichier.")
        sys.exit(1)

    # 5) Application de la fonction d’extraction sur chaque ligne
    print("🔄 Extraction des mots significatifs pour chaque hôpital……")
    df["Mots significatif"] = df["Nom hopital"].apply(lambda x: extract_significant(x, stopwords))

    # 6) Aperçu rapide (les 10 premières lignes)
    print("\n📋 Aperçu de la colonne “Nom hopital” vs “Mots significatif” :\n")
    print(df[["Nom hopital", "Mots significatif"]].head(10).to_string(index=False))

    # 7) Sauvegarde du nouveau fichier Excel
    try:
        df.to_excel(output_path, index=False)
    except Exception as e:
        print(f"❌ Impossible d’écrire le fichier de sortie : {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n✅ Le fichier modifié a été enregistré ici : {output_path}")

if __name__ == "__main__":
    main()
