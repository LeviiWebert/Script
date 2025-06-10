#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script complet pour faire correspondre, entre deux fichiers Excel :
  - “Table A” (votre table point) contenant :
      • “Nom hopital” (ou “Nom clinique”)
      • “Ville”
      • “Département” (présent mais non utilisé pour le match)
      • “Mots significment :
  1) La “Ville” (après normalisation)atifs” (optionnel, on peut le générer si absent)
  - “Table B” (FINESS) contenant :
      • “Nom établissement” (ou un équivalent)
      • “Ville” au format “XXXXX NOM_VILLE” (code postal + nom)
      • “Département” (présent mais non utilisé pour le match)
      • “Code FINESS”

Dans cette version, on matche sur deux clés seule
  2) Les tokens (mots) extraits de “Mots significatifs” : au lieu d’exiger une égalité exacte,
     on considère qu’il y a match si **tous** les mots de “Mots significatifs” de A
     sont **contenus** dans le “Nom établissement” de B (après normalisation et découpage en mots).
     
Le script :
  - Génère (ou recalcule) “Mots significatifs” dans la table A.
  - Normalise la colonne “Ville” de A (MAJ + trim).
  - Dans la table B, on normalise la colonne “Ville” (on retire le code postal + CEDEX) et
    on construit pour chaque ligne une liste de tokens à partir de “Nom établissement”.
  - Pour chaque ligne de A, on filtre B par “Ville” = “Ville A”, puis on sélectionne parmi ces candidats
    ceux pour lesquels la liste des mots de “Mots significatifs” est un sous-ensemble (inclus) de la liste
    des tokens du “Nom établissement” en B.
  - Selon le nombre de correspondances (0, 1, ≥2), on définit le “Statut final” et on crée “Code FINESS final”.
  - On écrit un fichier Excel avec les colonnes :
      Nom hopital, Ville, Département, Mots significatifs, Code FINESS final, Statut final, ...
  
Usage :
  1. Adaptez en début de script les variables :
       PATH_TABLE_A   → chemin du fichier Excel “table point”
       PATH_TABLE_B   → chemin du fichier Excel FINESS
       OUTPUT_PATH    → chemin du fichier Excel de sortie
  2. Installez pandas & openpyxl : `pip install pandas openpyxl`
  3. Exécutez : `python match_finess_partial.py`

"""

import pandas as pd
import re
import os
import sys


# ──────────────────────────────────────────────────────────────────────────────
#                               RÉGLAGES À ADAPTER
# ──────────────────────────────────────────────────────────────────────────────

# Chemin vers votre "table point" (table A)
PATH_TABLE_A = r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\data_propre_ext_LP-167_Acc_Risque.xlsx"

# Chemin vers votre fichier FINESS (table B)
PATH_TABLE_B = r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\Filtered_FINESS.xlsx"

# Chemin de sortie pour le fichier final
OUTPUT_PATH = r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\résultat_matches_finess.xlsx"

# Nom exact des colonnes concernées dans chaque table
COLA_NOM_HOPITAL     = "Nom hopital"         # dans table A
COLA_NOM_CLINIQUE    = "Nom clinique"        # alternative pour table A
COLA_VILLE           = "Ville"               # dans table A
COLA_MOTS_SIG        = "Mots significatifs"  # dans table A (sera créé si absent)

COLB_NOM_ETAB        = "Nom"   # dans table B
COLB_VILLE           = "Ville"               # dans table B (format "XXXXX NOM_VILLE")
COLB_CODE_FINESS     = "Code FINESS"         # dans table B

# Liste des mots génériques à retirer (en MAJUSCULE pour comparaison)
STOPWORDS = {
    "GRAND",
    "CHU", "CHI", "CH", "GBU",
    "HÔPITAL", "HOPITAL", "CLINIQUE",
    "HCL", "GH"
}

# ──────────────────────────────────────────────────────────────────────────────
#                          FIN DES RÉGLAGES À ADAPTER
# ──────────────────────────────────────────────────────────────────────────────


def extract_significant(name: str) -> str:
    """
    Extrait les mots "significatifs" d'une chaîne `name` en supprimant
    tout token qui est dans STOPWORDS. Retourne une chaîne formée des
    mots restants, séparés par un espace.
    """
    if pd.isna(name):
        return ""
    s = re.sub(r"[-/_,()]", " ", str(name))
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", s)
    sig = [t for t in tokens if t.upper() not in STOPWORDS]
    return " ".join(sig)


def normalize_finess_city(name: str) -> str:
    """
    Normalise la ville du fichier FINESS de la forme "XXXXX NOM_VILLE"
    (code postal + nom), en supprimant les 5 chiffres initiaux et le
    suffixe "CEDEX", puis en mettant en majuscules et en retirant
    les espaces multiples.
    """
    if pd.isna(name):
        return ""
    s = str(name).strip().upper()
    s = re.sub(r"^\d{5}\s*", "", s)        # supprime "XXXXX "
    s = re.sub(r"\s+CEDEX$", "", s)         # supprime " CEDEX" en fin
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(name: str) -> list:
    """
    Découpe une chaîne en liste de tokens (mots), en ne gardant que
    les caractères alphabétiques (lettres + accents).
    Le résultat est renvoyé en majuscules.
    Les interjections (< = 3 lettres) sont filtrées.
    """
    if pd.isna(name) or str(name).strip() == "":
        return []
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", str(name).upper())
    # Ne garder que les tokens de longueur > 3 (mots entiers, pas interjections)
    return [t for t in tokens if len(t) > 3]


def main():
    # 1) Vérification que les fichiers existent
    for path in (PATH_TABLE_A, PATH_TABLE_B):
        if not os.path.isfile(path):
            print(f"❌ Fichier introuvable : {path}", file=sys.stderr)
            sys.exit(1)

    # 2) Charger la table A (table point)
    try:
        dfA = pd.read_excel(PATH_TABLE_A, dtype=str)
    except Exception as e:
        print(f"❌ Erreur lecture table A : {e}", file=sys.stderr)
        sys.exit(1)

    # 3) Charger la table B (FINESS)
    try:
        dfB = pd.read_excel(PATH_TABLE_B, dtype=str)
    except Exception as e:
        print(f"❌ Erreur lecture table B : {e}", file=sys.stderr)
        sys.exit(1)

    # 4) Identifier la colonne "Nom hopital" de table A (ou "Nom clinique")
    if COLA_NOM_HOPITAL in dfA.columns:
        col_nomA = COLA_NOM_HOPITAL
    elif COLA_NOM_CLINIQUE in dfA.columns:
        col_nomA = COLA_NOM_CLINIQUE
    else:
        print(f"❌ Ni '{COLA_NOM_HOPITAL}' ni '{COLA_NOM_CLINIQUE}' n'a été trouvée dans la table A.", file=sys.stderr)
        sys.exit(1)

    # 5) Vérifier colonnes "Ville" dans A
    if COLA_VILLE not in dfA.columns:
        print(f"❌ Colonne '{col}' manquante dans la table A.", file=sys.stderr)
        sys.exit(1)

    # 6) Vérifier colonnes "Nom établissement", "Ville", "Département", "Code FINESS" dans B
    for col in (COLB_NOM_ETAB, COLB_VILLE, COLB_CODE_FINESS):
        if col not in dfB.columns:
            print(f"❌ Colonne '{col}' manquante dans la table B (FINESS).", file=sys.stderr)
            sys.exit(1)

    # 7) Créer ou mettre à jour la colonne "Mots significatifs" dans A
    if COLA_MOTS_SIG not in dfA.columns:
        dfA[COLA_MOTS_SIG] = dfA[col_nomA].apply(extract_significant)
        print(f"ℹ️ Colonne '{COLA_MOTS_SIG}' créée dans la table A.")
    else:
        dfA[COLA_MOTS_SIG] = dfA[col_nomA].apply(extract_significant)
        print(f"ℹ️ Colonne '{COLA_MOTS_SIG}' existante mise à jour dans la table A.")

    # 8) Normaliser "Ville" (A) et créer tokens_A (liste de mots de "Mots significatifs")
    dfA[COLA_VILLE] = dfA[COLA_VILLE].astype(str).str.strip().str.upper()
    dfA[COLA_MOTS_SIG] = dfA[COLA_MOTS_SIG].astype(str).str.strip()
    dfA["tokens_A"] = dfA[COLA_MOTS_SIG].apply(lambda x: tokenize(x))

    # 9) Dans B, normaliser "Ville", créer tokens_B à partir de "Nom établissement"
    dfB_temp = dfB.copy()
    dfB_temp[COLB_VILLE] = dfB_temp[COLB_VILLE].apply(normalize_finess_city)
    dfB_temp["tokens_B"] = dfB_temp[COLB_NOM_ETAB].apply(lambda x: tokenize(x))

    # 10) Construire un index par ville pour B, pour accélérer les recherches
    #     Cela crée un dictionnaire : ville -> DataFrame restreint à cette ville
    b_by_city = {
        ville: sub_df.reset_index(drop=True)
        for ville, sub_df in dfB_temp.groupby(COLB_VILLE, sort=False)
    }

    # 11) Pour chaque ligne de A, chercher les correspondances dans B (même ville ET tokens_A ⊆ tokens_B)
    codes_fin = []
    statuts = []

    for idx, rowA in dfA.iterrows():
        villeA = rowA[COLA_VILLE]
        tokensA = rowA["tokens_A"]
        # Récupérer le DataFrame B restreint à la même ville
        candidates = b_by_city.get(villeA, pd.DataFrame())
        if candidates.empty or not tokensA:
            # Aucun candidat ou pas de mots significatifs : 0 match
            codes_fin.append("")
            statuts.append("0 - pas bon")
            continue

        # Parcourir les candidats et vérifier le critère "tokensA ⊆ tokensB"
        matched_codes = []
        for _, rowB in candidates.iterrows():
            tokensB = rowB["tokens_B"]
            # Vérifier que tous les mots de tokensA soient dans tokensB
            if any(tok in tokensB for tok in tokensA):
                matched_codes.append(str(rowB[COLB_CODE_FINESS]).strip())

        if len(matched_codes) == 0:
            codes_fin.append("")
            statuts.append("0 - pas bon")
        elif len(matched_codes) == 1:
            codes_fin.append(matched_codes[0])
            statuts.append("1 - réussi")
        else:
            # Plusieurs codes => ambiguïté
            unique_codes = sorted(set(matched_codes))
            codes_fin.append(";".join(unique_codes))
            statuts.append(f"{len(unique_codes)} - ambigu")

    # 12) Ajouter les colonnes "Code FINESS final" et "Statut final" à dfA
    dfA["Code FINESS final"] = codes_fin
    dfA["Statut final"] = statuts

    # 13) Préparer le DataFrame de sortie :
    cols_to_keep = [
        col_nomA,
        COLA_VILLE,
        COLA_MOTS_SIG,
        "Code FINESS final",
        "Statut final"
    ]
    autres_cols = [c for c in dfA.columns if c not in cols_to_keep + ["tokens_A"]]
    final_df = dfA[cols_to_keep + autres_cols]

    # 14) Enregistrer le résultat dans un fichier Excel
    try:
        final_df.to_excel(OUTPUT_PATH, index=False)
        print(f"✅ Résultat enregistré dans : {OUTPUT_PATH}")
    except Exception as e:
        print(f"❌ Impossible d’écrire le fichier de sortie : {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
