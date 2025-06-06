#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script optimisé pour faire correspondre et assigner le Code FINESS entre :
  - Table A (Data) : data_propre_ext_LP-167_Acc_Risque.xlsx
      • Colonnes essentielles : “Nom hopital” (ou “Nom clinique”), “Ville”, “Département”, “Mots significatifs”
  - Table B (DF) : Filtered_FINESS.xlsx
      • Colonnes essentielles : “Nom” (prioritaire), “Nom2” (secondaire), “Ville” (code postal + nom), “Code FINESS”

Spécificités intégrées :
  - Matching par département ET ville
    • Data : “Département” (deux chiffres), “Ville” (nom simple)
    • DF   : “Ville” (code postal + nom) → on extrait “Dept” = 2 premiers chiffres + “City_norm” = nom
  - Extraction de tokens > 3 lettres et filtrage STOPWORDS pour Data
  - Pour DF, on conserve aussi les abréviations hospitalières (CH, CHU, CHI, HCL, GHL) dans les tokens
  - Si “Mots significatifs” vide, fallback sur abréviations hospitalières dans Data
  - Matching d’abord “tous les tokens” puis “au moins un token” sur “Nom”, puis désambiguïsation via “Nom2”
  - Si plusieurs correspondances persistantes → “PLUSIEURS CAS”, si aucune → “0 - pas bon”, si une → “1 - réussi”
  - Mode DEBUG pour afficher le cheminement de comparaison des tokens (activation via DEBUG = True)

Usage :
  1. Ajuster en début de script les chemins PATH_TABLE_A, PATH_TABLE_B, OUTPUT_PATH
  2. pip install pandas openpyxl
  3. python match_finess_optimise_dept_city.py
"""

import pandas as pd
import re
import os
import sys

# ──────────────────────────────────────────────────────────────────────────────
#                               RÉGLAGES À ADAPTER
# ──────────────────────────────────────────────────────────────────────────────

PATH_TABLE_A = r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\data_propre_ext_LP-167_Acc_Risque.xlsx"
PATH_TABLE_B = r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\Filtered_FINESS.xlsx"
OUTPUT_PATH  = r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\résultat_matches_finess.xlsx"

# Colonnes Data (Table A)
COLA_NOM_HOPITAL  = "Nom hopital"
COLA_NOM_CLINIQUE = "Nom clinique"
COLA_VILLE        = "Ville"
COLA_DEPT         = "Département"
COLA_MOTS_SIG     = "Mots significatifs"

# Colonnes DF (Table B)
COLB_NOM          = "Nom"
COLB_NOM2         = "Nom2"
COLB_VILLE        = "Ville"
COLB_CODE_FINESS  = "Code FINESS"

# STOPWORDS et abréviations
STOPWORDS = {
    "DE","DU","DES","UN","UNE","LE","LA","LES","AU","AUX","ET","EN","L",
    "GRAND","HÔPITAL","HOPITAL","CLINIQUE","MATERNITÉ","MATERNITE","HCL","GHL"
}
HOSP_ABBREV = {"CHU","CHI","CH","HCL","GHL"}

# Activer/désactiver les logs de DEBUG
DEBUG = True

# ──────────────────────────────────────────────────────────────────────────────
#                      FONCTIONS UTILITAIRES (définies AVANT usage)
# ──────────────────────────────────────────────────────────────────────────────

def extract_significant(text: str) -> str:
    """
    Extrait les mots 'significatifs' d'une chaîne (Data) :
    - MAJUSCULE, ponctuation/apostrophes/tirets → espaces
    - tokens alphabétiques (> 3 lettres, hors STOPWORDS)
    """
    if pd.isna(text):
        return ""
    s = re.sub(r"[’'\-–_/(),]", " ", str(text).upper())
    raw_tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", s)
    filtered = [t for t in raw_tokens if len(t) > 3 and t not in STOPWORDS]
    return " ".join(filtered)

def tokenize_data(text: str) -> list:
    """
    Extrait les tokens alphabétiques (> 3 lettres, hors STOPWORDS) en MAJUSCULE (Data).
    """
    if pd.isna(text) or not str(text).strip():
        return []
    s = re.sub(r"[’'\-–_/(),]", " ", str(text).upper())
    raw = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", s)
    return [t for t in raw if len(t) > 3 and t not in STOPWORDS]

def tokenize_df(text: str) -> list:
    """
    Extrait les tokens alphabétiques pour DF :
    - On garde tokens > 3 lettres hors STOPWORDS, et on conserve aussi abréviations hospitalières (CH, CHU, CHI, HCL, GHL).
    """
    if pd.isna(text) or not str(text).strip():
        return []
    s = re.sub(r"[’'\-–_/(),]", " ", str(text).upper())
    raw = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", s)
    tokens = []
    for t in raw:
        if (len(t) > 3 and t not in STOPWORDS) or (t in HOSP_ABBREV):
            tokens.append(t)
    return tokens

def extract_fallback_abbrev(name: str) -> list:
    """
    Si tokens_A vide, extrait abréviations hospitalières (CHU, CHI, CH, HCL, GHL) du nom Data.
    """
    if pd.isna(name):
        return []
    upper = str(name).upper()
    return [abbr for abbr in HOSP_ABBREV if re.search(rf"\b{abbr}\b", upper)]

def normalize_data_city(v: str) -> str:
    """
    Normalise la ville dans Data (nom simple) :
    - MAJUSCULE, trim, tirets/apostrophes → espaces, SAINT → ST, double espaces → un seul.
    """
    if pd.isna(v):
        return ""
    s = str(v).strip().upper()
    s = re.sub(r"[’'\-–]", " ", s)
    s = re.sub(r"\bSAINT\b", "ST", s)
    return re.sub(r"\s+", " ", s).strip()

def normalize_df_city(v: str) -> str:
    """
    Normalise la ville dans DF (format 'XXXXX NOM' ou 'XXXXX NOM CEDEX') :
    - MAJUSCULE, trim, supprimer code postal, CEDEX, tirets/apostrophes → espaces, SAINT → ST, double espaces → un seul.
    """
    if pd.isna(v):
        return ""
    s = str(v).strip().upper()
    s = re.sub(r"^\d{5}\s*", "", s)    # supprime le code postal
    s = re.sub(r"\s+CEDEX$", "", s)    # supprime CEDEX
    s = re.sub(r"[’'\-–]", " ", s)
    s = re.sub(r"\bSAINT\b", "ST", s)
    return re.sub(r"\s+", " ", s).strip()

def try_match_on_column(candidates: pd.DataFrame, column: str, tokens_req: list, debug_prefix="") -> list:
    """
    Parmi candidats (même département + même ville), renvoie d'abord la liste
    des Code FINESS où tous tokens_req ∈ tokenize_df(rowB[column]). Si aucun,
    renvoie ceux où au moins un token_req ∈ tokenize_df(rowB[column]). Logs si DEBUG.
    """
    matched_all, matched_any = [], []

    for idxB, rowB in candidates.iterrows():
        nom_b = rowB.get(column, "")
        tokensB = tokenize_df(nom_b)
        if DEBUG:
            print(f"{debug_prefix}>> Candidat [{column}] idx {idxB}: '{nom_b}' → tokensB = {tokensB}")

        # Étape 'all'
        if tokens_req and all(tok in tokensB for tok in tokens_req):
            matched_all.append(str(rowB[COLB_CODE_FINESS]).strip())
            if DEBUG:
                print(f"{debug_prefix}   → ALL match: tokens_req {tokens_req} sont tous dans {tokensB}")

    if matched_all:
        return matched_all

    # 'any' si rien trouvé en 'all'
    for idxB, rowB in candidates.iterrows():
        nom_b = rowB.get(column, "")
        tokensB = tokenize_df(nom_b)
        if any(tok in tokensB for tok in tokens_req):
            matched_any.append(str(rowB[COLB_CODE_FINESS]).strip())
            if DEBUG:
                print(f"{debug_prefix}   → ANY match: au moins un de {tokens_req} est dans {tokensB}")

    return matched_any

def match_row(rowA, dfB_indexed, idxA=None):
    """
    Pour une ligne Data, retourne (code_finess, statut) selon :
    1) tokens_A = tokenize_data("Mots significatifs") ou fallback abrév si vide.
    2) Filtrer DF sur même département ET même ville.
    3) Matching sur "Nom" (tous tokens puis au moins un), puis si plusieurs, "Nom2".
    4) Statut : 0/1/PLUSIEURS CAS.
    """
    prefix = f"[Data idx {idxA}] " if idxA is not None else ""

    deptA = str(rowA[COLA_DEPT]).zfill(2)             # deux chiffres
    cityA = normalize_data_city(rowA[COLA_VILLE])     # nom simple
    tokensA = tokenize_data(rowA[COLA_MOTS_SIG])

    if DEBUG:
        print(f"{prefix}Département Data: '{rowA[COLA_DEPT]}' → '{deptA}'")
        print(f"{prefix}Ville Data: '{rowA[COLA_VILLE]}' → '{cityA}'")
        print(f"{prefix}Mots significatifs Data: '{rowA[COLA_MOTS_SIG]}' → tokensA = {tokensA}")

    if not tokensA:
        source = rowA.get(COLA_NOM_HOPITAL, "") or rowA.get(COLA_NOM_CLINIQUE, "")
        tokensA = extract_fallback_abbrev(source)
        if DEBUG:
            print(f"{prefix}Fallback abbréviations sur '{source}' → tokensA = {tokensA}")

    candidats = dfB_indexed.get(deptA)
    if candidats is None or candidats.empty:
        if DEBUG:
            print(f"{prefix}Aucun candidat DF pour département '{deptA}'")
        return "", "0 - pas bon"

    # Filtrer par ville normalisée également
    filt = candidats["City_norm"] == cityA
    candidats = candidats[filt].reset_index(drop=True)

    if DEBUG:
        print(f"{prefix}Candidats restants pour département '{deptA}' ET ville '{cityA}': {len(candidats)}")

    if candidats.empty:
        if DEBUG:
            print(f"{prefix}Aucun candidat après filtre ville → 0 - pas bon")
        return "", "0 - pas bon"

    # 1) Matching sur "Nom"
    if DEBUG:
        print(f"{prefix}Tentative match sur colonne '{COLB_NOM}' avec tokensA = {tokensA}")
    codes = try_match_on_column(candidats, COLB_NOM, tokensA, debug_prefix=prefix)

    # 2) Si plusieurs et "Nom2" existe, retenter
    if len(codes) > 1 and COLB_NOM2 in candidats.columns:
        if DEBUG:
            print(f"{prefix}Plusieurs codes trouvés ({codes}) sur '{COLB_NOM}', réessaie sur '{COLB_NOM2}'")
        codes2 = try_match_on_column(candidats, COLB_NOM2, tokensA, debug_prefix=prefix + "  [Nom2] ")
        if len(codes2) == 1:
            if DEBUG:
                print(f"{prefix}Désambiguïsation réussie sur '{COLB_NOM2}': code unique = {codes2[0]}")
            return codes2[0], "1 - réussi"
        elif len(codes2) > 1:
            if DEBUG:
                print(f"{prefix}Toujours plusieurs ({codes2}) sur '{COLB_NOM2}' → PLUSIEURS CAS")
            return "PLUSIEURS CAS", "PLUSIEURS CAS"
        else:
            if DEBUG:
                print(f"{prefix}Aucun code trouvé sur '{COLB_NOM2}', on garde codes initiaux = {codes}")

    # 3) Interpréter codes trouvés sur "Nom"
    if not codes:
        if DEBUG:
            print(f"{prefix}Aucun code trouvé → 0 - pas bon")
        return "", "0 - pas bon"
    elif len(codes) == 1:
        if DEBUG:
            print(f"{prefix}Correspondance unique trouvée: {codes[0]}")
        return codes[0], "1 - réussi"
    else:
        if DEBUG:
            print(f"{prefix}Plusieurs codes trouvés ({codes}) → PLUSIEURS CAS")
        return "PLUSIEURS CAS", "PLUSIEURS CAS"

def main():
    # Vérifier que les fichiers existent
    for p in (PATH_TABLE_A, PATH_TABLE_B):
        if not os.path.isfile(p):
            print(f"❌ Fichier introuvable : {p}", file=sys.stderr)
            return

    # Charger Data et DF
    dfA = pd.read_excel(PATH_TABLE_A, dtype=str)
    dfB = pd.read_excel(PATH_TABLE_B, dtype=str)

    # Identifier colonne “Nom hopital” OU “Nom clinique”
    if COLA_NOM_HOPITAL in dfA.columns:
        nomA_col = COLA_NOM_HOPITAL
    elif COLA_NOM_CLINIQUE in dfA.columns:
        nomA_col = COLA_NOM_CLINIQUE
    else:
        print("❌ Ni 'Nom hopital' ni 'Nom clinique' introuvés dans Data.", file=sys.stderr)
        return

    # Vérifier colonnes nécessaires
    for col in (COLA_DEPT, COLA_VILLE, COLA_MOTS_SIG):
        if col not in dfA.columns:
            print(f"❌ Colonne '{col}' absente dans Data.", file=sys.stderr)
            return
    for col in (COLB_NOM, COLB_VILLE, COLB_CODE_FINESS):
        if col not in dfB.columns:
            print(f"❌ Colonne '{col}' absente dans DF.", file=sys.stderr)
            return

    # 1) Générer/recalculer "Mots significatifs" dans Data
    dfA[COLA_MOTS_SIG] = dfA[nomA_col].apply(lambda x: extract_significant(x))
    if DEBUG:
        print("==> Colonne 'Mots significatifs' générée dans Data")

    # 2) Normaliser "Département" (2 chiffres) et "Ville" (nom simple) dans Data
    dfA[COLA_DEPT] = dfA[COLA_DEPT].astype(str).str.strip().str.zfill(2)
    dfA["City_norm_A"] = dfA[COLA_VILLE].apply(normalize_data_city)
    if DEBUG:
        print("==> Colonnes 'Département' et 'City_norm_A' créées dans Data")

    # 3) Préparer DF : extraire code département + normaliser ville
    dfB["Dept"]      = dfB[COLB_VILLE].astype(str).str.strip().str[:2].str.zfill(2)
    dfB["City_norm"] = dfB[COLB_VILLE].apply(normalize_df_city)
    if DEBUG:
        print("==> Colonnes 'Dept' et 'City_norm' créées dans DF")

    # Grouper DF par département
    dfB_grouped = dfB.groupby("Dept")
    dfB_indexed = {dept: sub.reset_index(drop=True) for dept, sub in dfB_grouped}
    if DEBUG:
        print("==> DF indexé par 'Dept'")

    # 4) Parcourir chaque ligne de Data, faire correspondance
    codes_fin, statuts = [], []
    for idxA, rowA in dfA.iterrows():
        if DEBUG:
            print(f"\n--> Traitement de la ligne {idxA} de Data")
        code, statut = match_row(rowA, dfB_indexed, idxA)
        codes_fin.append(code)
        statuts.append(statut)

    dfA["Code FINESS final"] = codes_fin
    dfA["Statut final"]      = statuts

    # 5) Préparer et sauvegarder le résultat
    cols_to_keep = [nomA_col, COLA_DEPT, "City_norm_A", COLA_MOTS_SIG, "Code FINESS final", "Statut final"]
    other_cols  = [c for c in dfA.columns if c not in cols_to_keep]
    final_df    = dfA[cols_to_keep + other_cols]

    final_df.to_excel(OUTPUT_PATH, index=False)
    print(f"\n✅ Résultat enregistré dans : {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
