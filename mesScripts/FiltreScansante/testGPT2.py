# import pandas as pd
# from fuzzywuzzy import fuzz

# # 1) Chargement du fichier
# df = pd.read_excel( r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\R3R4R5restant.xlsx", dtype=str)

# # 2) Extraction du département depuis FINESSJ (2 premiers caractères)
# df['Dept'] = df['FINESSJ'].str[:2]

# # 3) Calcul des scores fuzzy entre NomScanSante et Nom / Nom2
# def compute_best_score(row):
#     scan = str(row['NomScanSante']).upper()
#     scores = []
#     if pd.notna(row.get('Nom')):
#         scores.append(fuzz.ratio(scan, str(row['Nom']).upper()))
#     if pd.notna(row.get('Nom2')):
#         scores.append(fuzz.ratio(scan, str(row['Nom2']).upper()))
#     return max(scores) if scores else 0

# df['ScoreFuzzy'] = df.apply(compute_best_score, axis=1)

# # 4) Sélection de la meilleure correspondance par département
# df_best_per_dept = (
#     df
#     .sort_values(by='ScoreFuzzy', ascending=False)
#     .drop_duplicates(subset='Dept', keep='first')
#     .reset_index(drop=True)
# )

# # 5) Sauvegarde du résultat
# output_path =  r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\R6_ALgo.xlsx"
# df_best_per_dept.to_excel(output_path, index=False)


# print(f"Fichier généré : {output_path}")
import pandas as pd
from fuzzywuzzy import fuzz

# 1) Chargement du fichier
df = pd.read_excel( r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\R3R4R5restant.xlsx", dtype=str)

# 2) Extraction du département depuis FINESSJ (2 premiers caractères)
df['Dept'] = df['FINESSJ'].str[:2]

# 3) Fonction de calcul du meilleur score fuzzy pour une ligne
def compute_best_score(row):
    scan = str(row['NomScanSante']).upper()
    scores = []
    if pd.notna(row.get('Nom')):
        scores.append(fuzz.ratio(scan, str(row['Nom']).upper()))
    if pd.notna(row.get('Nom2')):
        scores.append(fuzz.ratio(scan, str(row['Nom2']).upper()))
    return max(scores) if scores else 0

# 4) Pour chaque département, calculer le score et garder la meilleure ligne
best_matches = []
for dept, group in df.groupby('Dept'):
    group = group.copy()
    group['ScoreFuzzy'] = group.apply(compute_best_score, axis=1)
    # Sélection de la ligne avec le score maximum
    best_row = group.loc[group['ScoreFuzzy'].idxmax()]
    best_matches.append(best_row)

df_best_per_dept = pd.DataFrame(best_matches).reset_index(drop=True)

# 5) Sauvegarde et affichage
output_path =  r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\R6_ALgo2.xlsx"
df_best_per_dept.to_excel(output_path, index=False)


print(f"✅ Fichier généré : {output_path}")
