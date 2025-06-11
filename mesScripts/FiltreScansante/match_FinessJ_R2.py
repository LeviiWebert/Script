import pandas as pd

PATH_DATA_MODELE=r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\Data-FINESS_Modele.xlsx"
PATH_FINESS_SCSA=r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\FinessScansante.xlsx"
OUTPUT=r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\Result_Algo_R2.xlsx"


df_modele=pd.read_excel(
    PATH_DATA_MODELE,
    dtype=str,
    engine="openpyxl"
    )
df_scsa=pd.read_excel(
    PATH_FINESS_SCSA,
    dtype=str,
    engine="openpyxl"
    )

resultat=df_modele.merge(
    df_scsa,
    left_on="FINESSJ",
    right_on="FINESSJ",
    how="inner"
    )
ligne=resultat.count(axis=0)[1]
print(f"Merge effectué : on a {ligne} lignes obtenue")
resultat.to_excel(OUTPUT)
print(f"R2 appliqué : excel généré à {OUTPUT}")