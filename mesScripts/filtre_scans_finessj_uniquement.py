import pandas as pd
import logging
from fuzzywuzzy import fuzz
import os
import sys

df_finessj = pd.read_excel(
    r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\FinessJR2.xlsx",
    dtype=str,
    engine="openpyxl"
)
df_finess = pd.read_excel(
    r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\Data-FINESS_Modele.xlsx",
    dtype=str,
    engine="openpyxl"
)


# 7. DEUXIÈME MERGE sur FINESSJ_norm
merge2 = df_finessj.merge(
    df_finess,
    left_on="FINESSJ",
    right_on="FINESSJ",
    how="inner"
)
merge2.to_excel(r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\TABLEAU à TRAIté\MatchFinessJOnlyR2.xlsx", index=False)
print("match sur finessj uniquement généré")
