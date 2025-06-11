import pandas as pd
import re

# Chemin vers votre "table point" (table A)
#PATH_MATCH_FJ= r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\Result_Algo_R2.xlsx"
PATH_MATCH_FJ= r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\RestantR3R4àTraité.xlsx"
PATH_OUTPUT=r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\Match_Algo_R4.xlsx"
COL_NOM_SS="NomScanSante"
COL_NOM_GOUV="Nom"
CH = "CENTRE HOSPITALIER"
STOPWORDS = {
    "INST","EHPAD","USLD",
    "SSIAD","HAD","ANTENNE",
    "CSADA","IFSI","IFAS",
    "UNITE","CSAPA","CEGIDD",
    "CLICK"
}
# R5
STOPWORDS |= {
    "CLAT","CV","CMPP",
    "FAM","HDJ","MAS",
    "PSY","SMUR","SSR",
    "EHPAA", "HAD", "SSIAD", "USLD",
    "SSR", "SAMU", "SMUR", "CMP", "CMPP", "CSAPA", "CLIC", "CMRR", "FAM", "MAS",
    "CS", "CMS", "ES", "EPS", "EPSM"

}
REPLACE = {
    "-"," DE "," DU ",
    "D'"," DES "
}


df_match = pd.read_excel(PATH_MATCH_FJ)
df_final = []
i=1
for idx, row in df_match.iterrows():
    nom = str(row[COL_NOM_GOUV]).upper()
    scans = str(row[COL_NOM_SS]).upper()
    for mot in REPLACE:
        nom = nom.replace(mot, " ")
        scans = scans.replace(mot, " ")
    mots_nom = nom.split()
    if not any(word in nom for word in STOPWORDS):
        # comparaison des mots entre eux en tant qu'entier
        if all(re.search(rf"\b{re.escape(mot)}\b", scans) for mot in mots_nom):
            df_final.append(row)
            print(f"Ligne {i} ajoutée")
            i += 1
        if CH in nom:
            df_final.append(row)
            print(f"Ligne {i} ajoutée")
            i += 1


df_final = pd.DataFrame(df_final)

df_final.to_excel(PATH_OUTPUT,index=False)

print("Excel de matching généré")