import json
import requests
from bs4 import BeautifulSoup
import time
import urllib.parse
import pandas as pd
import os
import sys

# Try to import selenium, if not available, provide installation instructions
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("ERREUR: Le module 'selenium' n'est pas installé.")
    print("Pour installer les dépendances nécessaires, exécutez:")
    print("pip install selenium beautifulsoup4 pandas openpyxl requests")
    print("\nVous devrez également télécharger ChromeDriver depuis:")
    print("https://chromedriver.chromium.org/")
    print("Et l'ajouter à votre PATH système.")
    sys.exit(1)

def load_coordinates_data():
    """Load coordinates data from JSON file"""
    json_path = r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\Script\chefs_lieux_departements_coords.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_driver():
    """Setup Chrome driver with options"""
    if not SELENIUM_AVAILABLE:
        return None
        
    try:
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    except Exception as e:
        print(f"Erreur lors de l'initialisation du driver Chrome: {e}")
        print("Assurez-vous que ChromeDriver est installé et dans votre PATH.")
        print("Téléchargez ChromeDriver depuis: https://chromedriver.chromium.org/")
        return None

def scrape_emeis_for_location(driver, chef_lieu, latitude, longitude, numero_dept, nom_dept):
    """Scrape Emeis clinics for a specific location"""
    try:
        # Construct URL
        base_url = "https://www.emeis-cliniques.fr/resultats-recherche"
        params = {
            'combine': chef_lieu,
            'field_lat_long_distance[latitude]': str(latitude),
            'field_lat_long_distance[longitude]': str(longitude),
            'field_lat_long_distance[search_distance]': '25',
            'field_metier_site_tid': 'All'
        }
        
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        print(f"Scraping {chef_lieu} ({numero_dept}) - {nom_dept}")
        print(f"URL: {url}")
        
        driver.get(url)
        time.sleep(3)
        
        # Wait for page to load and find the table length selector
        try:
            wait = WebDriverWait(driver, 10)
            select_element = wait.until(EC.presence_of_element_located((By.NAME, "table_recherche_length")))
            
            # Set table length to 100
            select = Select(select_element)
            select.select_by_value("100")
            time.sleep(2)
            
        except Exception as e:
            print(f"Could not find or set table length selector: {e}")
        
        # Wait for table to load
        try:
            table = wait.until(EC.presence_of_element_located((By.ID, "table_recherche")))
        except Exception as e:
            print(f"Table not found for {chef_lieu}: {e}")
            return []
        
        # Extract data from table
        clinics = []
        tbody = table.find_element(By.TAG_NAME, "tbody")
        rows = tbody.find_elements(By.TAG_NAME, "tr")
        
        for row in rows:
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) >= 5:
                    # Extract image URL
                    img_cell = cells[0]
                    img_element = img_cell.find_element(By.TAG_NAME, "img")
                    image_url = img_element.get_attribute("src") if img_element else ""
                    
                    # Extract clinic name and URL
                    name_cell = cells[1]
                    link_element = name_cell.find_element(By.TAG_NAME, "a")
                    clinic_name = link_element.text.strip()
                    clinic_url = link_element.get_attribute("href")
                    
                    # Extract location
                    location = cells[2].text.strip()
                    
                    # Extract type of stay
                    type_sejour = cells[3].text.strip()
                    
                    # Extract phone number
                    telephone = cells[4].text.strip()
                    
                    clinic_data = {
                        "Nom de la clinique": clinic_name,
                        "URL de la clinique": clinic_url,
                        "Localité": location,
                        "Type de séjour": type_sejour,
                        "Téléphone": telephone,
                        "URL de l'image": image_url,
                        "Ville de recherche": chef_lieu,
                        "Numéro département": numero_dept,
                        "Nom département": nom_dept,
                        "Latitude recherche": latitude,
                        "Longitude recherche": longitude
                    }
                    
                    clinics.append(clinic_data)
                    
            except Exception as e:
                print(f"Error extracting row data: {e}")
                continue
        
        print(f"Found {len(clinics)} clinics for {chef_lieu}")
        return clinics
        
    except Exception as e:
        print(f"Error scraping {chef_lieu}: {e}")
        return []

def main():
    """Main function to scrape all locations"""
    if not SELENIUM_AVAILABLE:
        print("Impossible de continuer sans Selenium. Installez les dépendances d'abord.")
        return
        
    # Load coordinates data
    coordinates_data = load_coordinates_data()
    
    # Setup driver
    driver = setup_driver()
    if not driver:
        print("Impossible d'initialiser le driver. Arrêt du script.")
        return
    
    all_clinics = []
    
    try:
        for location in coordinates_data:
            chef_lieu = location["chef_lieu"]
            latitude = location["latitude"]
            longitude = location["longitude"]
            numero_dept = location["numéro"]
            nom_dept = location["département"]
            
            # Scrape clinics for this location
            clinics = scrape_emeis_for_location(
                driver, chef_lieu, latitude, longitude, numero_dept, nom_dept
            )
            
            all_clinics.extend(clinics)
            
            # Add delay between requests
            time.sleep(2)
    
    finally:
        if driver:
            driver.quit()
    
    # Save results to Excel file
    if all_clinics:
        df = pd.DataFrame(all_clinics)
        output_file = r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\emeis_cliniques_results.xlsx"
        
        # Create Excel file with formatting
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Cliniques Emeis', index=False)
            
            # Get the worksheet to format
            worksheet = writer.sheets['Cliniques Emeis']
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"\nScraping completed!")
        print(f"Total clinics found: {len(all_clinics)}")
        print(f"Results saved to: {output_file}")
    else:
        print("No clinics found!")

if __name__ == "__main__":
    main()
