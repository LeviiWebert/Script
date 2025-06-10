import pandas as pd
import logging
from fuzzywuzzy import fuzz
import os
import sys
# --- Configuration du logger ---
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger()

# 1. Chargement des fichiers
df_finess = pd.read_excel(
    r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\Data-FINESS_Modele.xlsx",
    dtype=str,
    engine="openpyxl"
)
df_finessj = pd.read_excel(
    r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\FinessJR2.xlsx",
    dtype=str,
    engine="openpyxl"
)
df_scansante = pd.read_excel(
    r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\FinessScansante.xlsx",
    dtype=str,
    engine="openpyxl"
)

# 2. Normalisation à 9 chiffres (on garde les 9 derniers caractères puis on complète à gauche si nécessaire)
def normalize_code(series):
    return (series
            .fillna("")
            .astype(str)
            .str.strip()
            .str[-9:]
            .str.zfill(9))

df_finess["FINESS_norm"] = normalize_code(df_finess["FINESS"])
df_finess["FINESSJ_norm"] = normalize_code(df_finess["FINESSJ"])
df_scansante["FINESS_norm"] = normalize_code(df_scansante["FINESS"])

# 3. On ne garde, dans FinessScansante, que le code normalisé et le nom à injecter
df_scansante_sub = (
    df_scansante[["FINESS_norm", "Nom"]]
    .rename(columns={
        "FINESS_norm": "FINESS_scansante",
        "Nom": "Nom_scansante"
    })
)
# 7. DEUXIÈME MERGE sur FINESSJ_norm
merge2 = df_finessj.merge(
    df_scansante_sub,
    left_on="FINESSJ",
    right_on="FINESSJ",
    how="inner"
)
merge2.to_excel()
print("match sur finessj uniquement, généré ene excel")
# 4. PREMIER MERGE sur FINESS_norm
merge1 = df_finess.merge(
    df_scansante_sub,
    left_on="FINESS_norm",
    right_on="FINESS_scansante",
    how="inner"
)
merge1.to_excel(r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\OnlyFiness.xlsx", index=False)
# 5. On repère les lignes de df_finess déjà appariées
matched_idx = merge1.index.drop_duplicates()

# 6. On filtre Data-FINESS_Modele pour n’avoir que les lignes qui n’ont pas matché sur FINESS
df_finess_remaining = df_finess.drop(index=matched_idx)



# Calculer un score de similarité (0–100) pour chaque paire de noms
"""merge2["sim_score"] = merge2.apply(
    lambda row: fuzz.token_set_ratio(row["Nom_scansante"], row["Nom"]),
    axis=1
)

# Ne garder que les lignes où la similarité est suffisante (ex. ≥ 80)
SEUIL = 80
merge2 = merge2[merge2["sim_score"] >= SEUIL]"""
# 8. On concatène les deux résultats pour avoir toutes les lignes appariées
df_final = pd.concat([merge1, merge2], ignore_index=True)
# 9. Aperçu du résultat final
folder = os.path.dirname(r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\Filt_Fine_SCSant.xlsx")
if folder and not os.path.isdir(folder):
    os.makedirs(folder, exist_ok=True)
try:
    df_final.to_excel(r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\Filt_Fine_SCSant.xlsx", index=False)
    print("Filt_Fine_SCSant.xlsx a été générer")
except Exception as e:
    log.error(f"Erreur lors de l’enregistrement du fichier filtré : {e}")
    sys.exit(1)
