import pandas as pd

# Charger les deux fichiers
df1 = pd.read_excel("JPG-to-excel_syT.xlsx")
df2 = pd.read_excel("ext_LP-167_Acc_Risque.xlsx")
# 1. Alignement des colonnes (join='outer' pour prendre l'union)
df1, df2 = df1.align(df2, join='outer', axis=1)

# 2. Alignement des index (idem ; souvent utile si vos feuilles n'ont ni même nombre de lignes ni mêmes labels)
df1, df2 = df1.align(df2, join='outer', axis=0)

# 3. Comparaison
diff = df1.compare(df2, keep_shape=True, keep_equal=False)

# 4. Sauvegarde du résultat
diff.to_excel("différences.xlsx")