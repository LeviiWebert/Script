#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filtre un fichier FI-NESS en ne gardant que les lignes dont les valeurs figurent "
                    "dans les colonnes correspondantes du fichier de nomenclature."
    )
    parser.add_argument(
        "--finess", "-f",
        required=True,
        help="Chemin vers le fichier FI-NESS (CSV ou XLSX)."
    )
    parser.add_argument(
        "--nomenclature", "-n",
        required=True,
        help="Chemin vers le fichier de nomenclature (XLSX)."
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Chemin de sortie pour le fichier filtré (sera au format XLSX)."
    )
    return parser.parse_args()

def load_finess(path):
    """
    Charge le fichier FI-NESS, en détectant automatiquement s'il s'agit d'un CSV ou d'un XLSX.
    """
    if not os.path.isfile(path):
        print(f"Erreur : le fichier FI-NESS spécifié n'existe pas : {path}", file=sys.stderr)
        sys.exit(1)

    _, ext = os.path.splitext(path.lower())
    if ext == ".csv":
        try:
            df = pd.read_csv(path, dtype=str)
        except Exception as e:
            print(f"Erreur lors de la lecture du CSV FI-NESS : {e}", file=sys.stderr)
            sys.exit(1)
    elif ext in (".xls", ".xlsx"):
        try:
            df = pd.read_excel(path, dtype=str)
        except Exception as e:
            print(f"Erreur lors de la lecture du XLSX FI-NESS : {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Extension non prise en charge pour FI-NESS : {ext}. Attendu .csv/.xls/.xlsx", file=sys.stderr)
        sys.exit(1)
    return df

def load_nomenclature(path):
    """
    Charge le fichier de nomenclature au format XLSX.
    """
    if not os.path.isfile(path):
        print(f"Erreur : le fichier Nomenclature spécifié n'existe pas : {path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_excel(path, dtype=str)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier Nomenclature (XLSX) : {e}", file=sys.stderr)
        sys.exit(1)
    return df

def main():
    args = parse_args()

    # 1. Chargement des données
    print("🔄 Chargement du fichier FI-NESS…")
    df_finess = load_finess(args.finess)
    print(f"  ➜ FI-NESS chargé : {df_finess.shape[0]} lignes × {df_finess.shape[1]} colonnes")

    print("🔄 Chargement du fichier Nomenclature…")
    df_nomen = load_nomenclature(args.nomenclature)
    print(f"  ➜ Nomenclature chargé : {df_nomen.shape[0]} lignes × {df_nomen.shape[1]} colonnes")

    # 2. Identification des colonnes communes
    common_cols = [col for col in df_nomen.columns if col in df_finess.columns]
    if not common_cols:
        print("⚠️  Aucune colonne commune trouvée entre FI-NESS et Nomenclature.")
        print("   Vérifiez l’orthographe des en-têtes ou le format des fichiers.")
        sys.exit(1)
    print(f"🔎 Colonnes communes trouvées pour le filtrage :\n   • " + "\n   • ".join(common_cols))

    # 3. Filtrage
    print("🔄 Application du filtrage…")
    df_filtered = df_finess.copy()
    for col in common_cols:
        # On convertit en chaîne, on élimine les NaN, puis on récupère la liste des valeurs valides dans Nomenclature
        valid_values = df_nomen[col].dropna().astype(str).unique().tolist()
        before_count = df_filtered.shape[0]
        df_filtered = df_filtered[df_filtered[col].astype(str).isin(valid_values)]
        after_count = df_filtered.shape[0]
        print(f"   • Colonne « {col} » : {before_count} ➔ {after_count} (après avoir gardé les valeurs présentes dans la nomenclature)")

    # 4. Résultat
    print()
    print("✅ Filtrage terminé.")
    print(f"   Taille avant filtrage : {df_finess.shape[0]} lignes × {df_finess.shape[1]} colonnes")
    print(f"   Taille après filtrage : {df_filtered.shape[0]} lignes × {df_filtered.shape[1]} colonnes")
    if df_filtered.shape[0] == 0:
        print("⚠️  AUCUNE ligne n’a été conservée après filtrage. Vérifiez les valeurs de vos colonnes.")
    else:
        print("\nExtrait des 5 premières lignes après filtrage :")
        print(df_filtered.head(5).to_string(index=False))

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
