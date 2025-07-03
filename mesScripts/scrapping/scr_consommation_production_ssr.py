import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os

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
    return re.sub(r'\s+', ' ', texte.strip())

def nettoyer_donnee_numerique(texte):
    """Nettoie spécifiquement les données numériques en supprimant les espaces"""
    if texte is None:
        return ""
    
    texte_nettoye = texte.strip()
    
    # Vérifier si c'est un nombre (avec ou sans décimales, pourcentages, etc.)
    # Pattern pour détecter les nombres : chiffres, espaces, virgules, points, %, €, etc.
    pattern_nombre = r'^[\s\d\.,€%\-+()]*$'
    
    if re.match(pattern_nombre, texte_nettoye):
        # C'est probablement un nombre, supprimer tous les espaces
        return re.sub(r'\s+', '', texte_nettoye)
    else:
        # C'est du texte, nettoyer normalement
        return re.sub(r'\s+', ' ', texte_nettoye)

def nettoyer_nom_fichier(nom):
    """Nettoie le nom pour qu'il soit valide comme nom de fichier"""
    # Supprimer ou remplacer les caractères invalides pour les noms de fichiers
    nom_propre = re.sub(r'[<>:"/\\|?*]', '_', nom)
    nom_propre = nom_propre.replace('(', '').replace(')', '')
    return nom_propre[:50]  # Limiter la longueur

def extraire_nom_departement(texte):
    """Extrait le nom du département à partir du texte de l'en-tête"""
    if not texte:
        return "Inconnu"
    
    # Chercher le pattern "Département: NOM_DEPARTEMENT"
    match = re.search(r'Département\s*:\s*(.+?)(?:\s*\(|$)', texte, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Si pas trouvé, prendre le texte complet nettoyé
    return nettoyer_nom_fichier(texte)

def recuperer_donnees_region(code_region):
    """Récupère les données pour une région donnée et retourne un dictionnaire des départements"""
    url = f"https://www.scansante.fr/applications/analyse-croisee-consommation-production-SSR/submit?snatnav=&mbout=&type_fin=finP&annee=2024&tgeo=reg_ts&codegeo={code_region}&CM=&GN=&type_hosp=HC"
    
    try:
        print(f"Récupération des données pour {regions[code_region]}...")
        
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
        
        # Chercher tous les tableaux
        tables = soup.find_all('table', class_='table')
        
        if not tables:
            print(f"  Aucun tableau trouvé pour {regions[code_region]}")
            return {}
        
        print(f"  {len(tables)} tableau(x) trouvé(s) pour {regions[code_region]}")
        
        departements_data = {}
        
        for i, table in enumerate(tables):
            print(f"    Traitement du tableau {i+1}/{len(tables)}")
            
            # Chercher l'en-tête du département (généralement dans un h3 ou h4 avant le tableau)
            nom_departement = f"Departement_{i+1}"
            
            # Chercher les éléments précédents pour trouver le nom du département
            previous_elements = table.find_all_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'div', 'p'], limit=10)
            for elem in previous_elements:
                text = elem.get_text().strip()
                if 'département' in text.lower() or 'dept' in text.lower():
                    nom_departement = extraire_nom_departement(text)
                    break
            
            # Extraire les en-têtes
            headers_row = table.find('thead')
            
            if headers_row:
                headers = []
                for th in headers_row.find_all('th'):
                    header_text = nettoyer_texte(th.get_text())
                    headers.append(header_text)
            else:
                # Si pas d'en-tête, essayer la première ligne
                first_row = table.find('tr')
                if first_row:
                    headers = [nettoyer_texte(td.get_text()) for td in first_row.find_all(['th', 'td'])]
                else:
                    print(f"    Impossible de trouver les en-têtes pour le tableau {i+1}")
                    continue
            
            # Extraire les données
            tbody = table.find('tbody')
            if not tbody:
                # Si pas de tbody, prendre toutes les lignes sauf la première (en-tête)
                rows = table.find_all('tr')[1:]
            else:
                rows = tbody.find_all('tr')
            
            rows_data = []
            for tr in rows:
                row = []
                for j, td in enumerate(tr.find_all(['td', 'th'])):
                    # Utiliser le nettoyage numérique pour toutes les colonnes sauf les premières (noms, etc.)
                    if j == 0:  # Première colonne (généralement nom/libellé)
                        cell_text = nettoyer_texte(td.get_text())
                    else:  # Autres colonnes (potentiellement numériques)
                        cell_text = nettoyer_donnee_numerique(td.get_text())
                    row.append(cell_text)
                
                if row and any(cell.strip() for cell in row):  # Ajouter seulement si la ligne n'est pas vide
                    rows_data.append(row)
            
            if rows_data:
                # Ajuster la longueur des lignes si nécessaire
                max_cols = max(len(row) for row in rows_data) if rows_data else len(headers)
                if len(headers) < max_cols:
                    headers.extend([f"Colonne_{j+1}" for j in range(len(headers), max_cols)])
                
                for row in rows_data:
                    while len(row) < len(headers):
                        row.append("")
                
                # Créer le DataFrame avec des colonnes temporaires pour éviter les conflits
                temp_headers = [f"temp_col_{idx}" for idx in range(len(headers))]
                df = pd.DataFrame(rows_data, columns=temp_headers)
                
                # Renommer les colonnes avec les vrais noms après création
                column_mapping = {f"temp_col_{idx}": header for idx, header in enumerate(headers)}
                df = df.rename(columns=column_mapping)
                
                # Nettoyer toutes les colonnes numériques du DataFrame
                for col in df.columns:
                    df[col] = df[col].apply(lambda x: nettoyer_donnee_numerique(str(x)) if pd.notna(x) else x)
                
                # Ajouter des métadonnées de manière sécurisée
                metadata_columns = {
                    'Région': regions[code_region],
                    'Code_Région': code_region,
                    'Tableau_Numéro': i+1
                }
                
                # Insérer les colonnes de métadonnées seulement si elles n'existent pas déjà
                insert_position = 0
                for meta_col, meta_value in metadata_columns.items():
                    if meta_col not in df.columns:
                        df.insert(insert_position, meta_col, meta_value)
                        insert_position += 1
                    else:
                        # Si la colonne existe déjà, la mettre à jour
                        df[meta_col] = meta_value
                
                # Utiliser un nom de département unique pour éviter les conflits
                nom_departement_unique = f"{nom_departement}_tableau_{i+1}_{code_region}"
                departements_data[nom_departement_unique] = df
                print(f"    {len(df)} lignes récupérées pour {nom_departement_unique}")
        
        return departements_data
        
    except requests.exceptions.RequestException as e:
        print(f"  Erreur de requête pour {regions[code_region]}: {e}")
        return {}
    except Exception as e:
        print(f"  Erreur lors du traitement de {regions[code_region]}: {e}")
        import traceback
        print(f"  Détails de l'erreur: {traceback.format_exc()}")
        return {}

def main():
    """Fonction principale"""
    print("Début de la récupération des données consommation-production SSR...")
    
    # Créer un dossier pour les fichiers de sortie
    dossier_sortie = "donnees_consommation_production_ssr_2024"
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)
    
    # Dictionnaire pour stocker toutes les données par département
    tous_departements = {}
    
    # Parcourir toutes les régions
    for code_region, nom_region in regions.items():
        departements_region = recuperer_donnees_region(code_region)
        
        # Ajouter les départements au dictionnaire global
        for nom_dept, df_dept in departements_region.items():
            # Extraire le vrai nom du département (sans suffixe unique)
            nom_dept_clean = nom_dept.split('_tableau_')[0]
            
            if nom_dept_clean not in tous_departements:
                tous_departements[nom_dept_clean] = []
            tous_departements[nom_dept_clean].append(df_dept)
        
        # Pause entre les requêtes
        time.sleep(2)
    
    print(f"\nNombre de départements collectés: {len(tous_departements)}")
    
    # Créer un fichier Excel par département
    if tous_departements:
        print(f"\nCréation des fichiers Excel par département...")
        for nom_dept, list_df in tous_departements.items():
            if not list_df:  # Vérifier que la liste n'est pas vide
                continue
                
            nom_fichier_propre = nettoyer_nom_fichier(nom_dept)
            nom_fichier = os.path.join(dossier_sortie, f"{nom_fichier_propre}_consommation_production_ssr_2024.xlsx")
            
            try:
                with pd.ExcelWriter(nom_fichier, engine='openpyxl') as writer:
                    if len(list_df) == 1:
                        # Un seul DataFrame
                        list_df[0].to_excel(writer, sheet_name='Données', index=False)
                    else:
                        # Plusieurs DataFrames - créer une feuille par tableau
                        df_complet = pd.DataFrame()
                        for i, df in enumerate(list_df):
                            if not df.empty:  # Vérifier que le DataFrame n'est pas vide
                                nom_feuille = f"Tableau_{i+1}"
                                df.to_excel(writer, sheet_name=nom_feuille, index=False)
                                df_complet = pd.concat([df_complet, df], ignore_index=True)
                        
                        # Ajouter une feuille consolidée seulement si on a des données
                        if not df_complet.empty:
                            df_complet.to_excel(writer, sheet_name='TOUTES_DONNEES', index=False)
                
                print(f"  Fichier créé: {nom_fichier}")
                
            except Exception as e:
                print(f"  Erreur lors de la création du fichier pour {nom_dept}: {e}")
        
        # Créer également un fichier consolidé avec tous les départements
        print(f"\nCréation du fichier consolidé...")
        nom_fichier_consolide = os.path.join(dossier_sortie, "TOUS_DEPARTEMENTS_consommation_production_ssr_2024.xlsx")
        
        try:
            with pd.ExcelWriter(nom_fichier_consolide, engine='openpyxl') as writer:
                df_global = pd.DataFrame()
                has_data = False
                
                for nom_dept, list_df in tous_departements.items():
                    if not list_df:  # Vérifier que la liste n'est pas vide
                        continue
                        
                    nom_feuille_dept = nettoyer_nom_fichier(nom_dept)[:31]  # Limite Excel
                    df_dept_complet = pd.concat([df for df in list_df if not df.empty], ignore_index=True)
                    
                    if not df_dept_complet.empty:
                        df_dept_complet.to_excel(writer, sheet_name=nom_feuille_dept, index=False)
                        df_global = pd.concat([df_global, df_dept_complet], ignore_index=True)
                        has_data = True
                
                # Feuille avec toutes les données seulement si on a des données
                if not df_global.empty and has_data:
                    df_global.to_excel(writer, sheet_name='TOUTES_DONNEES', index=False)
                elif not has_data:
                    # Créer une feuille vide pour éviter l'erreur "At least one sheet must be visible"
                    pd.DataFrame({"Message": ["Aucune donnée collectée"]}).to_excel(writer, sheet_name='Aucune_donnee', index=False)
            
            print(f"  Fichier consolidé créé: {nom_fichier_consolide}")
            
        except Exception as e:
            print(f"  Erreur lors de la création du fichier consolidé: {e}")
            import traceback
            print(f"  Détails de l'erreur: {traceback.format_exc()}")
    else:
        print("Aucune donnée collectée pour créer les fichiers Excel.")
    
    print(f"\nTerminé ! {len(tous_departements)} départements traités")
    print(f"Fichiers créés dans le dossier: {dossier_sortie}")

if __name__ == "__main__":
    main()
