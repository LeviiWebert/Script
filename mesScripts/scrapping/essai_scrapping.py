import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import re
from urllib.parse import urlparse

def extract_portfolio_data(url):
    """
    Extrait les données Entrée, Employés et Pays d'une page portfolio GIMV
    """
    try:
        # Ajouter des headers pour éviter d'être bloqué
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Chercher la div avec class "tns-inner" ou directement les champs
        data = {
            'url': url,
            'entree': None,
            'employes': None,
            'pays': None,
            'company_name': None
        }
        
        # Extraire le nom de la compagnie depuis l'URL
        path_parts = url.split('/')
        if len(path_parts) > 0:
            data['company_name'] = path_parts[-1]
        
        # Chercher tous les champs de portfolio
        portfolio_fields = soup.find_all('div', class_=re.compile(r'field.*field--portfolio.*'))
        
        for field in portfolio_fields:
            # Chercher le titre (h4)
            title_elem = field.find('h4', class_='h4')
            if not title_elem:
                continue
                
            title = title_elem.get_text(strip=True).lower()
            
            if 'entrée' in title or 'entry' in title:
                # Chercher la date dans un élément time ou directement dans le texte
                time_elem = field.find('time')
                if time_elem:
                    data['entree'] = time_elem.get_text(strip=True)
                else:
                    # Chercher après le h4
                    wrapper = field.find('div', class_='field--portfolio-case--wrapper')
                    if wrapper:
                        text = wrapper.get_text(strip=True)
                        # Extraire l'année/date après "Entrée"
                        match = re.search(r'Entrée\s*(\d{4})', text)
                        if match:
                            data['entree'] = match.group(1)
                            
            elif 'employés' in title or 'employees' in title or 'staff' in title:
                wrapper = field.find('div', class_='field--portfolio-case--wrapper')
                if wrapper:
                    text = wrapper.get_text(strip=True)
                    # Extraire le nombre après "Employés"
                    match = re.search(r'Employés\s*(\d+)', text)
                    if match:
                        data['employes'] = match.group(1)
                    else:
                        # Si pas de match, prendre le dernier nombre trouvé
                        numbers = re.findall(r'\d+', text)
                        if numbers:
                            data['employes'] = numbers[-1]
                            
            elif 'pays' in title or 'country' in title or 'region' in title:
                wrapper = field.find('div', class_='field--portfolio-case--wrapper')
                if wrapper:
                    text = wrapper.get_text(strip=True)
                    # Extraire le pays après "Pays"
                    match = re.search(r'Pays\s+(.+)', text)
                    if match:
                        data['pays'] = match.group(1).strip()
                    else:
                        # Chercher le pays dans le texte complet, en évitant de remplacer par vide
                        clean_text = text.strip()
                        if clean_text and clean_text != 'Pays':
                            # Enlever seulement le mot "Pays" au début s'il est suivi d'un espace
                            if clean_text.startswith('Pays '):
                                data['pays'] = clean_text[5:].strip()
                            elif clean_text != 'Pays':
                                data['pays'] = clean_text
        
        # Méthode alternative: chercher directement dans le HTML
        if not any([data['entree'], data['employes'], data['pays']]):
            # Chercher avec des sélecteurs CSS plus spécifiques
            
            # Entrée
            entry_field = soup.find('div', class_=re.compile(r'.*portfolio-entry-date.*'))
            if entry_field:
                time_elem = entry_field.find('time')
                if time_elem:
                    data['entree'] = time_elem.get_text(strip=True)
            
            # Employés
            staff_field = soup.find('div', class_=re.compile(r'.*portfolio-staff.*'))
            if staff_field:
                text = staff_field.get_text(strip=True)
                numbers = re.findall(r'\d+', text)
                if numbers:
                    data['employes'] = numbers[-1]
            
            # Pays
            region_field = soup.find('div', class_=re.compile(r'.*portfolio-region.*'))
            if region_field:
                text = region_field.get_text(strip=True)
                # Nettoyer le texte pour extraire seulement le pays, sans remplacer par vide
                if text and text != 'Pays':
                    if text.startswith('Pays '):
                        country = text[5:].strip()
                    elif text != 'Pays':
                        country = text
                    else:
                        country = None
                    
                    if country:
                        data['pays'] = country
        
        return data
        
    except requests.RequestException as e:
        print(f"Erreur lors de la requête pour {url}: {e}")
        return {'url': url, 'entree': None, 'employes': None, 'pays': None, 'company_name': None, 'error': str(e)}
    except Exception as e:
        print(f"Erreur lors du traitement de {url}: {e}")
        return {'url': url, 'entree': None, 'employes': None, 'pays': None, 'company_name': None, 'error': str(e)}

def main():
    # Liste des URLs
    urls = [
        "https://www.gimv.com/fr/portefeuille/sustainable-cities/acceo",
        "https://www.gimv.com/fr/portefeuille/agrobiothers",
        "https://www.gimv.com/fr/portefeuille/alpine",
        "https://www.gimv.com/fr/portefeuille/alro-group",
        "https://www.gimv.com/fr/portefeuille/alt-technologies",
        "https://www.gimv.com/fr/portefeuille/ambulantis",
        "https://www.gimv.com/fr/portefeuille/ame",
        "https://www.gimv.com/fr/portefeuille/apraxon",
        "https://www.gimv.com/fr/portefeuille/arplas-systems",
        "https://www.gimv.com/fr/portefeuille/arseus-medical",
        "https://www.gimv.com/fr/portefeuille/baas",
        "https://www.gimv.com/fr/portefeuille/babyshop-group",
        "https://www.gimv.com/fr/portefeuille/bioconnection",
        "https://www.gimv.com/fr/portefeuille/biotalys",
        "https://www.gimv.com/fr/portefeuille/blendwell-food-group",
        "https://www.gimv.com/fr/portefeuille/sustainable-cities/castelein-sealants",
        "https://www.gimv.com/fr/portefeuille/cegeka",
        "https://www.gimv.com/fr/portefeuille/citymesh",
        "https://www.gimv.com/fr/portefeuille/complement-therapeutics",
        "https://www.gimv.com/fr/portefeuille/la-comtoise",
        "https://www.gimv.com/fr/portefeuille/curana",
        "https://www.gimv.com/fr/portefeuille/egruppe",
        "https://www.gimv.com/fr/portefeuille/ers",
        "https://www.gimv.com/fr/portefeuille/fire1",
        "https://www.gimv.com/fr/portefeuille/france-thermes",
        "https://www.gimv.com/fr/portefeuille/fronnt",
        "https://www.gimv.com/fr/portefeuille/la-croissanterie",
        "https://www.gimv.com/fr/portefeuille/gsdi",
        "https://www.gimv.com/fr/portefeuille/ilc",
        "https://www.gimv.com/fr/portefeuille/imcheck-therapeutics",
        "https://www.gimv.com/fr/portefeuille/immunos",
        "https://www.gimv.com/fr/portefeuille/istar-medical",
        "https://www.gimv.com/fr/portefeuille/itineris",
        "https://www.gimv.com/fr/portefeuille/joolz",
        "https://www.gimv.com/fr/portefeuille/kinaset-therapeutics",
        "https://www.gimv.com/fr/portefeuille/kivu-bioscience",
        "https://www.gimv.com/fr/portefeuille/laser-2000",
        "https://www.gimv.com/fr/portefeuille/les-psy-reunis",
        "https://www.gimv.com/fr/portefeuille/medi-markt",
        "https://www.gimv.com/fr/portefeuille/lupine",
        "https://www.gimv.com/fr/portefeuille/mediar-therapeutics",
        "https://www.gimv.com/fr/portefeuille/mvz-holding",
        "https://www.gimv.com/fr/portefeuille/olyn",
        "https://www.gimv.com/fr/portfolio/life-sciences/onera-health",
        "https://www.gimv.com/fr/portefeuille/onward",
        "https://www.gimv.com/fr/portefeuille/paleo",
        "https://www.gimv.com/fr/portefeuille/picot",
        "https://www.gimv.com/fr/portefeuille/precirixr",
        "https://www.gimv.com/fr/portefeuille/projective-group",
        "https://www.gimv.com/fr/portefeuille/robojob",
        "https://www.gimv.com/fr/portefeuille/sgh-healthcaring",
        "https://www.gimv.com/fr/portefeuille/smart-battery-solutions",
        "https://www.gimv.com/fr/portefeuille/smg-sportplatzmaschinenbau",
        "https://www.gimv.com/fr/portefeuille/sofatutor",
        "https://www.gimv.com/fr/portefeuille/spice-factory",
        "https://www.gimv.com/fr/portefeuille/spineart",
        "https://www.gimv.com/fr/portefeuille/sustainable-cities/techinfra",
        "https://www.gimv.com/fr/portefeuille/televic",
        "https://www.gimv.com/fr/portefeuille/tibbloc",
        "https://www.gimv.com/fr/portefeuille/topas-therapeutics",
        "https://www.gimv.com/fr/portefeuille/variass",
        "https://www.gimv.com/fr/portefeuille/variotech",
        "https://www.gimv.com/fr/portefeuille/verkley",
        "https://www.gimv.com/fr/portefeuille/grandeco",
        "https://www.gimv.com/fr/portefeuille/wdm-deutenberg-group",
        "https://www.gimv.com/fr/portefeuille/witec"
    ]
    
    results = []
    
    print(f"Début de l'extraction pour {len(urls)} URLs...")
    
    for i, url in enumerate(urls, 1):
        print(f"Traitement {i}/{len(urls)}: {url}")
        
        data = extract_portfolio_data(url)
        results.append(data)
        
        # Afficher les résultats pour cette URL
        if data.get('entree') or data.get('employes') or data.get('pays'):
            print(f"  ✓ Entrée: {data.get('entree', 'N/A')}")
            print(f"  ✓ Employés: {data.get('employes', 'N/A')}")
            print(f"  ✓ Pays: {data.get('pays', 'N/A')}")
        else:
            print(f"  ✗ Aucune donnée trouvée")
        
        # Pause pour éviter de surcharger le serveur
        time.sleep(1)
    
    # Sauvegarder les résultats dans un fichier Excel
    df = pd.DataFrame(results)
    
    # Réorganiser les colonnes
    column_order = ['company_name', 'url', 'entree', 'employes', 'pays']
    if 'error' in df.columns:
        column_order.append('error')
    
    df = df[column_order]
    
    # Sauvegarder en Excel
    excel_filename = 'gimv_portfolio_data.xlsx'
    df.to_excel(excel_filename, index=False, engine='openpyxl')
    
    print(f"\nExtraction terminée! Résultats sauvegardés dans '{excel_filename}'")
    
    # Afficher un résumé
    successful = sum(1 for r in results if r.get('entree') or r.get('employes') or r.get('pays'))
    print(f"Données extraites avec succès: {successful}/{len(results)} pages")
    
    return results

if __name__ == "__main__":
    results = main()
    
    # Afficher quelques statistiques
    print("\n--- RÉSUMÉ DES DONNÉES EXTRAITES ---")
    countries = {}
    entry_years = {}
    
    for result in results:
        if result.get('pays'):
            countries[result['pays']] = countries.get(result['pays'], 0) + 1
        if result.get('entree'):
            entry_years[result['entree']] = entry_years.get(result['entree'], 0) + 1
    
    if countries:
        print(f"\nPays les plus fréquents:")
        for country, count in sorted(countries.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {country}: {count}")
    
    if entry_years:
        print(f"\nAnnées d'entrée les plus fréquentes:")
        for year, count in sorted(entry_years.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {year}: {count}")