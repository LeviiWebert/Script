import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

# Dictionnaire des régions avec leurs codes
regions = {
    "84": "AUVERGNE-RHÔNE-ALPES",
    "27": "BOURGOGNE-FRANCHE-COMTÉ",
    "53": "BRETAGNE",
    "24": "CENTRE-VAL DE LOIRE",
    "94": "CORSE",
    "44": "GRAND EST",
    "01": "GUADELOUPE",
    "03": "GUYANE",
    "32": "HAUTS-DE-FRANCE",
    "11": "ILE-DE-FRANCE",
    "04": "LA RÉUNION",
    "02": "MARTINIQUE",
    "06": "MAYOTTE",
    "28": "NORMANDIE",
    "75": "NOUVELLE-AQUITAINE",
    "76": "OCCITANIE",
    "52": "PAYS DE LA LOIRE",
    "93": "PROVENCE-ALPES-CÔTE D'AZUR"
}

def nettoyer_texte(texte):
    """Nettoie le texte en supprimant les espaces multiples et les retours à la ligne"""
    if texte is None:
        return ""
    return re.sub(r'\s+', '', texte.replace('%',''))

def recuperer_donnees_region(code_region, url):
    """Récupère les données pour une région donnée"""
    url = url[0] + code_region + url[1]
    try:
        print(f"Récupération des données pour {regions[code_region]}...")
        print(f"URL: {url}")
        
        # Headers pour simuler un navigateur
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Récupérer les titres depuis les tableaux avec class 'systitleandfootercontainer'
        titres = []
        title_tables = soup.find_all('table', class_='systitleandfootercontainer')
        for title_table in title_tables:
            title_text = title_table.get_text().strip()
            if title_text:
                titres.append(title_text)
        
        # Chercher TOUS les tableaux avec class 'table'
        all_tables = soup.find_all('table', class_='table')
        
        if not all_tables:
            print(f"  Aucun tableau trouvé pour {regions[code_region]}")
            return pd.DataFrame()
        
        print(f"  {len(all_tables)} tableau(x) trouvé(s) pour {regions[code_region]}")
        
        # Traiter tous les tableaux trouvés
        all_dataframes = []
        
        for table_idx, table in enumerate(all_tables):
            print(f"    Traitement du tableau {table_idx + 1}/{len(all_tables)}")
            
            # Extraire les en-têtes
            headers_row = table.find('thead')
            
            if headers_row:
                headers = []
                for th in headers_row.find_all('th'):
                    header_text = th.get_text().replace('\n', ' ').replace('\r', ' ').strip()
                    headers.append(header_text)
            else:
                # Essayer de prendre la première ligne comme en-têtes
                first_row = table.find('tr')
                if first_row:
                    headers = []
                    for cell in first_row.find_all(['th', 'td']):
                        header_text = cell.get_text().replace('\n', ' ').replace('\r', ' ').strip()
                        headers.append(header_text)
                else:
                    print(f"    Pas d'en-têtes trouvés pour le tableau {table_idx + 1}")
                    continue
            
            # Extraire les données en gérant les rowspan
            tbody = table.find('tbody')
            if tbody:
                rows = tbody.find_all('tr')
            else:
                # Si pas de tbody, prendre toutes les lignes sauf la première (en-têtes)
                all_rows = table.find_all('tr')
                rows = all_rows[1:] if len(all_rows) > 1 else []
            
            rows_data = []
            rowspan_data = {}  # Pour stocker les données des cellules avec rowspan
            
            for row_idx, tr in enumerate(rows):
                row = []
                col_idx = 0
                
                # D'abord, ajouter les valeurs des rowspan précédents
                for col_pos in sorted(rowspan_data.keys()):
                    if rowspan_data[col_pos]['remaining'] > 0:
                        if col_pos <= col_idx:
                            row.insert(col_pos, rowspan_data[col_pos]['value'])
                            col_idx += 1
                        rowspan_data[col_pos]['remaining'] -= 1
                
                # Traiter les cellules de la ligne actuelle
                for td in tr.find_all(['td', 'th']):
                    # Vérifier s'il y a un rowspan
                    rowspan = int(td.get('rowspan', 1))
                    
                    # Ajuster col_idx si des cellules rowspan occupent cette position
                    while col_idx in rowspan_data and rowspan_data[col_idx]['remaining'] > 0:
                        col_idx += 1
                    
                    # Nettoyer le texte de la cellule
                    if col_idx > 3:  # Colonnes numériques (après les 4 premières)
                        cell_text = nettoyer_texte(td.get_text())
                    else:
                        cell_text = str(td.get_text()).strip()
                    
                    # Ajouter la cellule à la position appropriée
                    while len(row) <= col_idx:
                        row.append("")
                    row[col_idx] = cell_text
                    
                    # Si la cellule a un rowspan > 1, l'enregistrer
                    if rowspan > 1:
                        rowspan_data[col_idx] = {
                            'value': cell_text,
                            'remaining': rowspan - 1
                        }
                    
                    col_idx += 1
                
                # Nettoyer les rowspan expirés
                rowspan_data = {k: v for k, v in rowspan_data.items() if v['remaining'] > 0}
                
                # Ajouter la ligne si elle n'est pas vide
                if row and any(cell.strip() for cell in row):
                    rows_data.append(row)
            
            if rows_data:
                # Ajuster la longueur des colonnes
                max_cols_data = max(len(row) for row in rows_data)
                
                # Ajuster les headers si nécessaire
                if len(headers) > max_cols_data:
                    headers = headers[:max_cols_data]
                elif len(headers) < max_cols_data:
                    for i in range(len(headers), max_cols_data):
                        headers.append(f"Colonne_{i+1}")
                
                # Ajuster chaque ligne pour avoir le même nombre de colonnes
                for row in rows_data:
                    while len(row) < len(headers):
                        row.append("")
                    if len(row) > len(headers):
                        row[:] = row[:len(headers)]
                
                # Créer le DataFrame pour ce tableau
                df_table = pd.DataFrame(rows_data, columns=headers)
                
                # Déterminer le titre pour ce tableau spécifique
                titre_tableau = ""
                if titres and table_idx < len(titres):
                    titre_tableau = titres[table_idx]
                elif titres:
                    titre_tableau = titres[0]  # Utiliser le premier titre si pas assez de titres
                else:
                    titre_tableau = "Aucun titre trouvé"
                
                # Ajouter des métadonnées seulement si elles n'existent pas déjà
                metadata_columns = [
                    ('Région', regions[code_region]),
                    ('Code_Région', code_region),
                    ('Numéro_Tableau', table_idx + 1),
                    ('Titre_Section', titre_tableau)
                ]
                
                # Insérer les colonnes de métadonnées en partant de la position 0
                insert_position = 0
                for col_name, col_value in metadata_columns:
                    if col_name not in df_table.columns:
                        df_table.insert(insert_position, col_name, col_value)
                        insert_position += 1
                    else:
                        # Si la colonne existe déjà, mettre à jour sa valeur
                        df_table[col_name] = col_value
                
                all_dataframes.append(df_table)
                print(f"    {len(df_table)} lignes récupérées pour le tableau {table_idx + 1}")
            else:
                print(f"    Aucune donnée dans le tableau {table_idx + 1}")
        
        # Combiner tous les DataFrames
        if all_dataframes:
            df_final = pd.concat(all_dataframes, ignore_index=True)
            print(f"  Total: {len(df_final)} lignes récupérées pour {regions[code_region]}")
            if titres:
                print(f"  Titres trouvés: {' | '.join(titres)}")
            return df_final
        else:
            print(f"  Aucune donnée exploitable trouvée pour {regions[code_region]}")
            return pd.DataFrame()
        
    except requests.exceptions.RequestException as e:
        print(f"  Erreur de requête pour {regions[code_region]}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"  Erreur lors du traitement de {regions[code_region]}: {e}")
        import traceback
        print(f"  Détails: {traceback.format_exc()}")
        return pd.DataFrame()

def main():
    """Fonction principale"""
    print("Début de la récupération des données SSR...")
    
    # Commentaire sur les URLs PSY et SSR désactivées
    # ["https://www.scansante.fr/applications/cartographie-activite-PSY/submit?snatnav=&annee=2024&tgeo=reg&codegeo=","&base=bpri&nat_pec=99&form_act=99&type_diag=1&cat_diag=", "PSY"],
    # ["https://www.scansante.fr/applications/cartographie-activite-SSR/submit?snatnav=&annee=2024&tgeo=reg&codegeo=","&SePP=0&SeTailEt=0&catmaj=&gn=&gpmedeco=", "SSR"],
    
    
    urls = [
        ["https://www.scansante.fr/applications/cartographie-activite-PSY/submit?snatnav=&annee=2024&tgeo=reg&codegeo=","&base=bpri&nat_pec=99&form_act=99&type_diag=1&cat_diag=", "PSY"],
        ["https://www.scansante.fr/applications/cartographie-activite-SSR/submit?snatnav=&annee=2024&tgeo=reg&codegeo=","&SePP=0&SeTailEt=0&catmaj=&gn=&gpmedeco=", "SSR"],
        ["https://www.scansante.fr/applications/analyse-croisee-consommation-production-SSR/submit?snatnav=&mbout=&type_fin=finP&annee=2024&tgeo=reg_ts&codegeo=","&CM=&GN=&type_hosp=HC", "Consommation_Production_SSR"],
        ["https://www.scansante.fr/applications/analyse-croisee-consommation-production-Psy/submit?snatnav=&mbout=&annee=2024&tgeo=reg_ts&codegeo=","&type_hosp=TP&DP=", "Consommation_Production_PSY"]
    ]
    # Traiter chaque URL séparément
    for url_info in urls:
        url_parts = url_info[:2]  # [début_url, fin_url]
        nom_fichier = f"donnees_{url_info[2]}_regions_2024.xlsx"
        
        print(f"\n=== Traitement de {url_info[2]} ===")
        
        try:
            with pd.ExcelWriter(nom_fichier, engine='openpyxl') as writer:
                df_complet = pd.DataFrame()
                regions_traitees = 0
                
                # Parcourir toutes les régions pour cette URL
                for code_region, nom_region in regions.items():
                    print(f"\nTraitement de {nom_region} pour {url_info[2]}...")
                    
                    try:
                        # Récupérer les données de la région pour cette URL
                        df_region_data = recuperer_donnees_region(code_region, url_parts)
                        
                        if not df_region_data.empty:
                            # Séparer les données par tableau pour cette région
                            tableaux_region = {}
                            for num_tableau in df_region_data['Numéro_Tableau'].unique():
                                df_tableau = df_region_data[df_region_data['Numéro_Tableau'] == num_tableau].copy()
                                tableaux_region[num_tableau] = df_tableau
                            
                            # Créer une feuille combinée pour cette région avec tous ses tableaux
                            nom_feuille = nom_region.replace('Ô', 'O').replace('É', 'E').replace('-', '_').replace(' ', '_')[:31]
                            
                            try:
                                # Créer un DataFrame consolidé pour la région avec séparation des tableaux
                                df_region_final = pd.DataFrame()
                                
                                for num_tableau in sorted(tableaux_region.keys()):
                                    df_tableau = tableaux_region[num_tableau]
                                    
                                    if not df_tableau.empty:
                                        titre_tableau = df_tableau['Titre_Section'].iloc[0]
                                        
                                        # Si c'est le premier tableau de la région, créer la structure avec toutes les colonnes possibles
                                        if df_region_final.empty:
                                            all_columns = list(df_tableau.columns)
                                            df_region_final = pd.DataFrame(columns=all_columns)
                                        
                                        # Assurer que df_region_final a toutes les colonnes de df_tableau
                                        for col in df_tableau.columns:
                                            if col not in df_region_final.columns:
                                                df_region_final[col] = ""
                                        
                                        # Assurer que df_tableau a toutes les colonnes de df_region_final
                                        for col in df_region_final.columns:
                                            if col not in df_tableau.columns:
                                                df_tableau[col] = ""
                                        
                                        # Réorganiser les colonnes de df_tableau pour correspondre à df_region_final
                                        df_tableau = df_tableau.reindex(columns=df_region_final.columns)
                                        
                                        # Créer une ligne de titre
                                        ligne_titre = pd.DataFrame([{col: '' for col in df_region_final.columns}])
                                        if 'Titre_Section' in ligne_titre.columns:
                                            ligne_titre.loc[0, 'Titre_Section'] = f"=== {titre_tableau} ==="
                                        else:
                                            # Si pas de colonne Titre_Section, utiliser la première colonne
                                            ligne_titre.iloc[0, 0] = f"=== {titre_tableau} ==="
                                        
                                        # Ajouter la ligne de titre puis les données du tableau
                                        df_region_final = pd.concat([df_region_final, ligne_titre, df_tableau], ignore_index=True)
                                        
                                        # Ajouter une ligne vide entre les tableaux (sauf pour le dernier)
                                        if num_tableau != max(tableaux_region.keys()):
                                            ligne_vide = pd.DataFrame([{col: '' for col in df_region_final.columns}])
                                            df_region_final = pd.concat([df_region_final, ligne_vide], ignore_index=True)
                                
                                # Écrire dans Excel avec gestion d'erreur
                                if not df_region_final.empty:
                                    df_region_final.to_excel(writer, sheet_name=nom_feuille, index=False)
                                    print(f"    Feuille '{nom_feuille}' créée avec {len(tableaux_region)} tableau(x) et {len(df_region_final)} lignes totales")
                                
                            except Exception as e:
                                print(f"    Erreur lors de la création de la feuille {nom_feuille}: {e}")
                                # Créer une feuille d'erreur avec les données brutes
                                try:
                                    df_region_data.to_excel(writer, sheet_name=f"{nom_feuille}_ERROR", index=False)
                                    print(f"    Feuille d'erreur créée: {nom_feuille}_ERROR")
                                except:
                                    pass
                            
                            # Ajouter au DataFrame complet pour la feuille consolidée
                            df_complet = pd.concat([df_complet, df_region_data], ignore_index=True)
                            regions_traitees += 1
                        else:
                            print(f"    Aucune donnée pour {nom_region}")
                            
                    except Exception as e:
                        print(f"    Erreur critique pour {nom_region}: {e}")
                        
                    # Pause entre les requêtes pour ne pas surcharger le serveur
                    time.sleep(3)
                
                # Créer une feuille avec toutes les données consolidées
                if not df_complet.empty:
                    try:
                        # Organiser la feuille consolidée par région et tableau
                        df_consolide_final = pd.DataFrame()
                        
                        for code_region in df_complet['Code_Région'].unique():
                            df_region = df_complet[df_complet['Code_Région'] == code_region]
                            nom_region = df_region['Région'].iloc[0]
                            
                            # Ajouter un en-tête de région
                            if not df_region.empty:
                                # Assurer que df_consolide_final a les bonnes colonnes
                                if df_consolide_final.empty:
                                    df_consolide_final = pd.DataFrame(columns=df_region.columns)
                                
                                # Assurer la cohérence des colonnes
                                for col in df_region.columns:
                                    if col not in df_consolide_final.columns:
                                        df_consolide_final[col] = ""
                                
                                ligne_region = pd.DataFrame([{col: '' for col in df_consolide_final.columns}])
                                if 'Région' in ligne_region.columns:
                                    ligne_region.loc[0, 'Région'] = f"=== RÉGION: {nom_region} ==="
                                else:
                                    ligne_region.iloc[0, 0] = f"=== RÉGION: {nom_region} ==="
                                
                                df_consolide_final = pd.concat([df_consolide_final, ligne_region], ignore_index=True)
                                
                                # Ajouter tous les tableaux de cette région
                                for num_tableau in sorted(df_region['Numéro_Tableau'].unique()):
                                    df_tableau = df_region[df_region['Numéro_Tableau'] == num_tableau].copy()
                                    
                                    if not df_tableau.empty:
                                        titre_tableau = df_tableau['Titre_Section'].iloc[0]
                                        
                                        # Réorganiser les colonnes pour correspondre
                                        df_tableau = df_tableau.reindex(columns=df_consolide_final.columns, fill_value="")
                                        
                                        # Ligne de titre du tableau
                                        ligne_titre = pd.DataFrame([{col: '' for col in df_consolide_final.columns}])
                                        if 'Titre_Section' in ligne_titre.columns:
                                            ligne_titre.loc[0, 'Titre_Section'] = f"--- {titre_tableau} ---"
                                        else:
                                            ligne_titre.iloc[0, 1] = f"--- {titre_tableau} ---"
                                        
                                        df_consolide_final = pd.concat([df_consolide_final, ligne_titre, df_tableau], ignore_index=True)
                                
                                # Ligne vide entre régions
                                ligne_vide = pd.DataFrame([{col: '' for col in df_consolide_final.columns}])
                                df_consolide_final = pd.concat([df_consolide_final, ligne_vide], ignore_index=True)
                        
                        # Nettoyer les valeurs NaN avant d'écrire
                        df_consolide_final = df_consolide_final.fillna('')
                        
                        df_consolide_final.to_excel(writer, sheet_name='TOUTES_REGIONS', index=False)
                        print(f"\nDonnées consolidées pour {url_info[2]}: {len(df_complet)} lignes de données organisées par région et tableau")
                        
                    except Exception as e:
                        print(f"Erreur lors de la création de la feuille consolidée: {e}")
                        # Essayer de sauvegarder les données brutes en cas d'erreur
                        try:
                            df_complet.fillna('').to_excel(writer, sheet_name='DONNEES_BRUTES', index=False)
                            print("Feuille de données brutes créée en cas d'erreur")
                        except:
                            pass
                else:
                    # Créer une feuille vide si aucune donnée
                    try:
                        pd.DataFrame({"Message": ["Aucune donnée trouvée"]}).to_excel(writer, sheet_name='Aucune_donnee', index=False)
                        print(f"Aucune donnée trouvée pour {url_info[2]}")
                    except Exception as e:
                        print(f"Erreur lors de la création de la feuille vide: {e}")
        
        except Exception as e:
            print(f"Erreur critique lors de la création du fichier {nom_fichier}: {e}")
            # Essayer de créer un fichier de récupération
            try:
                nom_fichier_backup = f"BACKUP_{url_info[2]}_regions_2024.xlsx"
                with pd.ExcelWriter(nom_fichier_backup, engine='openpyxl') as backup_writer:
                    pd.DataFrame({"Erreur": [f"Échec de création du fichier principal: {e}"]}).to_excel(backup_writer, sheet_name='Erreur', index=False)
                print(f"Fichier de récupération créé: {nom_fichier_backup}")
            except:
                print("Impossible de créer un fichier de récupération")
        
        print(f"Fichier traité: {nom_fichier} ({regions_traitees} régions traitées)")
    
    print(f"\nTerminé ! Tous les fichiers ont été créés.")

if __name__ == "__main__":
    main()