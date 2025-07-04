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

def recuperer_donnees_region(code_region,url):
    """Récupère les données pour une région donnée"""
    url = url[0] + code_region + url[1]
    try:
        print(f"Récupération des données pour {regions[code_region]}...")
        
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
        
        # Chercher le tableau
        table = soup.find_all('table', class_='table',limit=2)
        
        if not table:
            print(f"  Aucun tableau trouvé pour {regions[code_region]}")
            return pd.DataFrame()
        if len(table) < 2:
            print(f"  Moins de 2 tableaux trouvés pour {regions[code_region]}, vérifiez la structure HTML")
            return pd.DataFrame()
        else:
            table = table[1]  # Prendre le deuxième tableau qui contient les données
        # Extraire les en-têtes
        headers_row = table.find('thead')
        
        if headers_row:
            headers = []
            for th in headers_row.find_all('th'):
                # Remplacer les retours à la ligne par des espaces dans les en-têtes
                header_text = th.get_text().replace('\n', ' ').replace('\r', ' ')
                headers.append(header_text)
        else:
            print(f"  En-têtes non trouvés pour {regions[code_region]}")
            return pd.DataFrame()
        
        # Extraire les données
        tbody = table.find('tbody')
        if not tbody:
            print(f"  Corps du tableau non trouvé pour {regions[code_region]}")
            return pd.DataFrame()
        
        rows_data = []
        for tr in tbody.find_all('tr'):
            row = []
            col = 0
            for td in tr.find_all('td'):
                col += 1
                if col>4:
                    cell_text = nettoyer_texte(td.get_text())
                else:
                    cell_text = str(td.get_text())
                row.append(cell_text)
    
            
            if row:  # Ajouter seulement si la ligne n'est pas vide
                rows_data.append(row)
        
        if not rows_data:
            print(f"  Aucune donnée trouvée pour {regions[code_region]}")
            return pd.DataFrame()
        
        # Créer le DataFrame
        df = pd.DataFrame(rows_data, columns=headers)
        
        # Ajouter une colonne avec le nom de la région
        df.insert(0, 'Région', code_region)
        
        # Ajouter une colonne avec les titres extraits (concaténés)
        titre_complet = " | ".join(titres) if titres else "Aucun titre trouvé"
        df.insert(1, 'Titre_Section', titre_complet)
        
        print(f"  {len(df)} lignes récupérées pour {regions[code_region]}")
        if titres:
            print(f"  Titres trouvés: {titre_complet}")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"  Erreur de requête pour {regions[code_region]}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"  Erreur lors du traitement de {regions[code_region]}: {e}")
        return pd.DataFrame()

def main():
    """Fonction principale"""
    print("Début de la récupération des données SSR...")
    """["https://www.scansante.fr/applications/cartographie-activite-PSY/submit?snatnav=&annee=2024&tgeo=reg&codegeo=","&base=bpri&nat_pec=99&form_act=99&type_diag=1&cat_diag=", "PSY"],
        ["https://www.scansante.fr/applications/cartographie-activite-SSR/submit?snatnav=&annee=2024&tgeo=reg&codegeo=","&SePP=0&SeTailEt=0&catmaj=&gn=&gpmedeco=", "SSR"],"""
    urls = [
        
        ["https://www.scansante.fr/applications/analyse-croisee-consommation-production-SSR/submit?snatnav=&mbout=&type_fin=finP&annee=2024&tgeo=reg_ts&codegeo=","&CM=&GN=&type_hosp=HC", "Consommation_Production_SSR"],
        ["https://www.scansante.fr/applications/analyse-croisee-consommation-production-Psy/submit?snatnav=&mbout=&annee=2024&tgeo=reg_ts&codegeo=","&type_hosp=TP&DP=", "Consommation_Production_PSY"]
    ]
    
    # Traiter chaque URL séparément
    for url_info in urls:
        url_parts = url_info[:2]  # [début_url, fin_url]
        nom_fichier = f"donnees_{url_info[2]}_regions_2024.xlsx"
        
        print(f"\n=== Traitement de {url_info[2]} ===")
        
        with pd.ExcelWriter(nom_fichier, engine='openpyxl') as writer:
            df_complet = pd.DataFrame()
            regions_traitees = 0
            
            # Parcourir toutes les régions pour cette URL
            for code_region, nom_region in regions.items():
                print(f"\nTraitement de {nom_region} pour {url_info[2]}...")
                
                # Récupérer les données de la région pour cette URL
                df_region = recuperer_donnees_region(code_region, url_parts)
                
                if not df_region.empty:
                    # Créer une feuille par région
                    nom_feuille = nom_region.replace('Ô', 'O').replace('É', 'E').replace('-', '_').replace(' ', '_')[:31]  # Limite Excel: 31 caractères
                    df_region.to_excel(writer, sheet_name=nom_feuille, index=False)
                    
                    # Ajouter au DataFrame complet
                    df_complet = pd.concat([df_complet, df_region], ignore_index=True)
                    regions_traitees += 1
                    
                # Pause entre les requêtes pour ne pas surcharger le serveur
                time.sleep(1)
            
            # Créer une feuille avec toutes les données consolidées
            if not df_complet.empty:
                df_complet.to_excel(writer, sheet_name='TOUTES_REGIONS', index=False)
                print(f"\nDonnées consolidées pour {url_info[2]}: {len(df_complet)} lignes au total")
            else:
                # Créer une feuille vide si aucune donnée
                pd.DataFrame({"Message": ["Aucune donnée trouvée"]}).to_excel(writer, sheet_name='Aucune_donnee', index=False)
        
        print(f"Fichier créé: {nom_fichier} ({regions_traitees} régions traitées)")
    
    print(f"\nTerminé ! Tous les fichiers ont été créés.")

if __name__ == "__main__":
    main()