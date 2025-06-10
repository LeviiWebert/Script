import pandas as pd
import re
import os
import sys

# Chemin vers votre "table point" (table A)
PATH_MATCH_FJ= r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\MatchFinessJOnlyR2.xlsx"
PATH_OUTPUT=r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\MatchR1R2bis.xlsx"
COL_NOM_SS="NomScanSante"
COL_NOM_GOUV="Nom"


STOPWORDS = {
    "INST","EHPAD","USLD",
    "SSIAD","HAD","ANTENNE",
    "CSADA","IFSI","IFAS","MCO",
    "UNITE","CSAPA","CEGIDD"
}

CH = "CENTRE HOSPITALIER"

df_match = pd.read_excel(PATH_MATCH_FJ)
df_final = []
i=1
for idx, row in df_match.iterrows():
    nom = str(row[COL_NOM_GOUV]).upper()
    if CH in nom or " CH " in nom:
        if not any(word in nom for word in STOPWORDS):
            df_final.append(row)
            print(f"Ligne {i} ajoutée")
            i += 1


df_final = pd.DataFrame(df_final)

df_final.to_excel(PATH_OUTPUT,index=False)

print("Excel de matching généré")