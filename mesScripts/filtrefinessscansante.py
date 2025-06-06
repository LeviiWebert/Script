#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filtre un fichier FI-NESS en ne gardant que les lignes dont les valeurs FI-NESS figurent "
                    "dans les colonnes correspondantes du fichier de finessscansante."
    )
    parser.add_argument(
        "--finess", "-f",
        required=True,
        help="Chemin vers le fichier FI-NESS (CSV ou XLSX)."
    )
    parser.add_argument(
        "--finesscansante", "-fs",
        required=True,
        help="Chemin vers le fichier de finessScansante (XLSX)."
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Chemin de sortie pour le fichier filtré (sera au format XLSX)."
    )
    return parser.parse_args()

def load_file(path):
    if not os.path.isfile(path):
        print(f"Erreur : le fichier ",path," spécifié n'existe pas : {path}", file=sys.stderr)
        sys.exit(1)
    try:
        df = pd.read_excel(path, dtype=str)
        return df
    except Exception as e:
        print(f"Erreur lors de la lecture du XLSX ",path," : {e}", file=sys.stderr)
        sys.exit(1)
    

def equalizeFiness(df_finess,df_scansante):
    # Nettoyer les colonnes
    df_finess.columns = df_finess.columns.str.strip()
    df_scansante.columns = df_scansante.columns.str.strip()

    # Récupérer FINESS
    if "FINESS" in df_finess.columns:
        finess1 = df_finess["FINESS"].copy()
    else:
        print("FINESS non trouvée dans df_finess")
        sys.exit(1)

    if "FINESS" in df_scansante.columns:
        finess2 = df_scansante["FINESS"].copy()
    else:
        print("FINESS non trouvée dans df_scansante")
        sys.exit(1)

    # Normalisation : forcer 9 chiffres (adapter si besoin)
    finess1 = finess1.str.zfill(9)
    finess2 = finess2.str.zfill(9)

    # Remettre dans les dataframes
    df_finess["FINESS"] = finess1
    df_scansante["FINESS"] = finess2
    return df_finess,df_scansante

def main():
    args = parse_args()
    # 1. Chargement des données
    print("🔄 Chargement du fichier FI-NESS…")
    df_finess = load_file(args.finess)
    print(f"  ➜ FI-NESS chargé : {df_finess.shape[0]} lignes × {df_finess.shape[1]} colonnes")

    print("🔄 Chargement du fichier Finess_Scansante")
    df_scansante = load_file(args.finesscansante)
    print(f"  ➜ FScansante chargé : {df_scansante.shape[0]} lignes × {df_scansante.shape[1]} colonnes")

    print("Mise en forme des colonnes FINESS")
    df_finess,df_scansante=equalizeFiness(df_finess,df_scansante)
    print("Merge des deux fichiers\n")
    print("Premier merge sur le finess établissement")
    df_f = df_finess.merge(df_scansante,on="FINESS",how="inner")
    print("Premier merge sur le finess juridique")
    df_j = df_finess.merge(df_scansante,on="FINESSJ",how="inner")

    #df_filtered=df_f.merge(df_j,on="Nom_y",how="left")
    print("concaténation des deux fusion")
    df_filtered = pd.concat([df_f, df_j], ignore_index=True)
    
    print(f"  ➜ Résultat après merge : {df_filtered.shape[0]} lignes × {df_filtered.shape[1]} colonnes")


    # 5. Enregistrement
    folder = os.path.dirname(args.output)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    
    try:
        df_filtered.to_excel(args.output, index=False)
    except Exception as e:
        print(f"Erreur lors de l’enregistrement du fichier filtré : {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n📂 Fichier filtré enregistré ici : {args.output}")

if __name__ == "__main__":
    main()
