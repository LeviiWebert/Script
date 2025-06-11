
"""
Il faut installer pandas : pip install pandas

-f chemin vers le fichiers finess
-fs chemin vers le finess scansante
-o chemin complet vers l'output(il faut que ce soit un chemin vers un fichier excel en .xlsx et pas juste un dossiers)

exemple :

python filtrefinessscansante.py -f "C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\Data-FINESS_Modele.xlsx" -fs "C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\FinessScansante.xlsx" -o "C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\Filtered_Finess.xlsx"
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd
import logging

# --- Configuration du logger ---
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger()

# --- Fonctions ---
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
        log.error(f"Erreur : le fichier {path} spécifié n'existe pas.")
        sys.exit(1)
    try:
        df = pd.read_excel(path, dtype=str)
        return df
    except Exception as e:
        log.error(f"Erreur lors de la lecture du fichier {path} : {e}")
        sys.exit(1)

def equalizeFiness(df_finess, df_scansante):
    # Nettoyer les colonnes
    df_finess.columns = df_finess.columns.str.strip()
    df_scansante.columns = df_scansante.columns.str.strip()

    # Récupérer FINESS
    if "FINESS" in df_finess.columns:
        finess1 = df_finess["FINESS"].copy()
    else:
        log.error("FINESS non trouvée dans df_finess")
        sys.exit(1)

    if "FINESS" in df_scansante.columns:
        finess2 = df_scansante["FINESS"].copy()
    else:
        log.error("FINESS non trouvée dans df_scansante")
        sys.exit(1)

    # Normalisation : forcer 9 chiffres
    finess1 = finess1.str.zfill(9)
    finess2 = finess2.str.zfill(9)

    # Remettre dans les dataframes
    df_finess["FINESS"] = finess1
    df_scansante["FINESS"] = finess2
    return df_finess, df_scansante

# --- Main ---
def main():
    args = parse_args()

    # 1. Chargement des données
    log.info("🔄 Chargement du fichier FI-NESS…")
    df_finess = load_file(args.finess)
    log.info(f"  ➜ FI-NESS chargé : {df_finess.shape[0]} lignes × {df_finess.shape[1]} colonnes")

    log.info("🔄 Chargement du fichier Finess_Scansante")
    df_scansante = load_file(args.finesscansante)
    log.info(f"  ➜ FScansante chargé : {df_scansante.shape[0]} lignes × {df_scansante.shape[1]} colonnes")

    # 2. Mise en forme des colonnes FINESS
    log.info("Mise en forme des colonnes FINESS")
    df_finess, df_scansante = equalizeFiness(df_finess, df_scansante)

    # 3. Merge
    log.info("Merge des deux fichiers")
    df_filtered = df_scansante.merge(df_finess, on="FINESS", how="left")
    log.info(f"  ➜ Résultat après merge : {df_filtered.shape[0]} lignes × {df_filtered.shape[1]} colonnes")

    # 4. Vérifier unicité FINESS
    nb_total = df_filtered.shape[0]
    nb_unique = df_filtered["FINESS"].nunique()
    log.info(f"✅ FINESS uniques dans df_filtered : {nb_unique} sur {nb_total} lignes")

    # 5. Garde seulement les FINESS uniques
    df_resultat = df_filtered[df_filtered["FINESS"].duplicated(keep=False) == False]
    log.info(f"✅ Résultat : {df_resultat.shape[0]} lignes uniques")

    # 6. Exclure FINESS déjà utilisés
    finess_matched = df_resultat["FINESS"].unique()
    df_scansante_remaining = df_scansante[~df_scansante["FINESS"].isin(finess_matched)]
    log.info(f"✅ Scansante restant : {df_scansante_remaining.shape[0]} lignes")

    # 7. Faire le match avec FINESSJ pour le scansante restant
    df_scansante2 = df_scansante_remaining.merge(df_finess,on="FINESSJ",how="left")
    log.info(f"Scansante récupérer du 2eme merge : {df_scansante2.shape[0]}")



    # 8. Créer df_final = concaténer les deux merges
    df_final = pd.concat([df_resultat, df_scansante2], ignore_index=True)
    log.info(f"✅ df_final total : {df_final.shape[0]} lignes (merge1 + merge2)")


    # 9. Enregistrement
    folder = os.path.dirname(args.output)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    try:
        df_final.to_excel(args.output, index=False)
    except Exception as e:
        log.error(f"Erreur lors de l’enregistrement du fichier filtré : {e}")
        sys.exit(1)

    log.info(f"\n📂 Fichier filtré enregistré ici : {args.output}")

if __name__ == "__main__":
    main()
