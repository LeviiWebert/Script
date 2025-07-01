import pandas as pd
import re
from pathlib import Path
from datetime import datetime

class HospitalExtractorToExcel:
    def __init__(self, input_file="resultats_analyse_gemini.txt", output_file="hopitaux_classements.xlsx"):
        """
        Initialise l'extracteur d'hôpitaux vers Excel
        
        Args:
            input_file (str): Fichier de résultats Gemini
            output_file (str): Fichier Excel de sortie
        """
        self.input_file = input_file
        self.output_file = output_file
        self.data = {}  # Structure: {classement: [(rang, hopital_original, hopital_sans_rang), ...]}
        
    def read_gemini_results(self):
        """
        Lit et parse le fichier de résultats Gemini
        """
        try:
            with open(self.input_file, 'r', encoding='utf-8') as file:
                content = file.read()
            
            print(f"✅ Lecture du fichier {self.input_file} réussie")
            return content
        except FileNotFoundError:
            print(f"❌ Fichier {self.input_file} non trouvé")
            return None
        except Exception as e:
            print(f"❌ Erreur lors de la lecture: {e}")
            return None
    
    def extract_hospitals_from_content(self, content):
        """
        Extrait les hôpitaux du contenu avec des regex
        
        Args:
            content (str): Contenu du fichier de résultats
        """
        # Pattern pour identifier les sections de dossier et fichier
        dossier_pattern = r'DOSSIER:\s*(.+?)(?=\n|$)'
        fichier_pattern = r'FICHIER:\s*(.+?)(?=\n|$)'
        
        # Pattern pour identifier les hôpitaux avec leur rang
        # Cherche après "HÔPITAUX IDENTIFIÉS:" ou "HOPITAUX IDENTIFIÉS:"
        hospital_section_pattern = r'H[ÔO]PITAUX IDENTIFI[ÉE]S:\s*(.*?)(?=FICHIER:|DOSSIER:|===|$)'
        
        # Pattern pour extraire rang et nom d'hôpital
        # Formats possibles: [1]Nom, 1.Nom, 1 - Nom, etc.
        hospital_line_pattern = r'^[\s]*(?:\[?(\d+)\]?[\.\-\s]*)?(.+?)(?:\n|$)'
        
        # Division du contenu par sections de dossier
        sections = re.split(r'(?=DOSSIER:|FICHIER:)', content)
        
        current_classement = "Classement_Inconnu"
        current_dossier = "Dossier_Inconnu"
        
        for section in sections:
            # Recherche du nom du dossier
            dossier_match = re.search(dossier_pattern, section)
            if dossier_match:
                current_dossier = dossier_match.group(1).strip()
                print(f"📁 Dossier détecté: {current_dossier}")
            
            # Recherche du nom du fichier/classement
            fichier_match = re.search(fichier_pattern, section)
            if fichier_match:
                current_classement = fichier_match.group(1).strip()
                print(f"📂 Traitement du classement: {current_classement}")
            
            # Recherche des sections d'hôpitaux
            hospital_sections = re.findall(hospital_section_pattern, section, re.DOTALL | re.IGNORECASE)
            
            for hospital_section in hospital_sections:
                hospitals = self.parse_hospital_section(hospital_section, current_classement, current_dossier)
                if hospitals:
                    if current_classement not in self.data:
                        self.data[current_classement] = []
                    self.data[current_classement].extend(hospitals)
    
    def parse_hospital_section(self, hospital_section, classement, dossier):
        """
        Parse une section d'hôpitaux pour extraire rang et nom
        
        Args:
            hospital_section (str): Section contenant les hôpitaux
            classement (str): Nom du classement
            dossier (str): Nom du dossier
            
        Returns:
            list: Liste de tuples (rang, nom_hopital_original, nom_hopital_sans_rang_normalise, dossier)
        """
        hospitals = []
        lines = hospital_section.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('-') or line.startswith('TYPE:') or line.startswith('STATUT:'):
                continue
            
            # Nettoyage de la ligne
            line = line.replace('- ', '').strip()            # Tentative d'extraction du rang et du nom
            rang = None
            nom_hopital_original = line  # Garde le libellé original complet
            nom_hopital_sans_rang = line  # Libellé sans le rang
            
            # Pattern 1: [1]Nom ou (1)Nom - Format avec crochets/parenthèses
            match1 = re.match(r'[\[\(](\d+)[\]\)]\s*(.+)', line)
            if match1:
                rang = int(match1.group(1))
                nom_hopital_sans_rang = match1.group(2).strip()
            else:
                # Pattern 2: 1. Nom ou 1 - Nom ou 1 Nom - Format avec points/tirets
                match2 = re.match(r'(\d+)[\.\-\s]+(.+)', line)
                if match2:
                    rang = int(match2.group(1))
                    nom_hopital_sans_rang = match2.group(2).strip()
                else:
                    # Pattern 3: [1er] [2e] [10e] etc. - Format avec crochets et suffixes
                    match3 = re.match(r'\[(\d{1,2})(er|e|ème)\]\s*(.+)', line)
                    if match3:
                        rang = int(match3.group(1))
                        nom_hopital_sans_rang = match3.group(3).strip()
                    else:                        # Pattern 4: 1er 2e 10e etc. - Format avec suffixes simples
                        match4 = re.match(r'(\d{1,2})(er|e|ème)\s+(.+)', line)
                        if match4:
                            rang = int(match4.group(1))
                            nom_hopital_sans_rang = match4.group(3).strip()
                        else:
                            # Pattern 5: 1ᵉʳ 2ᵉ 10ᵉ etc. - Format avec exposants Unicode
                            match5 = re.match(r'(\d{1,2})[ᵉᵉʳᵈᵗʰ]+\s+(.+)', line)
                            if match5:
                                rang = int(match5.group(1))
                                nom_hopital_sans_rang = match5.group(2).strip()
                            else:
                                # Pattern 6: 1erCHU 2eCH 10eCH etc. - Format collé sans espace
                                match6 = re.match(r'(\d{1,2})(er|e|ème)([A-Z].+)', line)
                                if match6:
                                    rang = int(match6.group(1))
                                    nom_hopital_sans_rang = match6.group(3).strip()
                                else:                                    # Pattern 7: Nom seul, on attribue un rang automatique
                                    if nom_hopital_original and len(nom_hopital_original) > 2:
                                        rang = len(hospitals) + 1
            
            # Nettoyage des crochets inutiles dans le libellé sans rang
            # Supprime les crochets qui entourent le nom d'hôpital : [CHU xxx] devient CHU xxx
            if nom_hopital_sans_rang.startswith('[') and nom_hopital_sans_rang.endswith(']'):
                nom_hopital_sans_rang = nom_hopital_sans_rang[1:-1].strip()
            
            # Normalisation du nom d'hôpital (remplacement des abréviations)
            nom_hopital_sans_rang_normalise = self.normalize_hospital_name(nom_hopital_sans_rang)
                                    
            # Ajout de l'hôpital à la liste
            if nom_hopital_original and len(nom_hopital_original) > 2:  # Éviter les noms trop courts
                hospitals.append((rang if rang else len(hospitals) + 1, nom_hopital_original, nom_hopital_sans_rang_normalise, dossier))
                print(f"    🏥 Extrait: Rang {rang if rang else 'Auto'} - Original: '{nom_hopital_original}' | Normalisé: '{nom_hopital_sans_rang_normalise}' | Dossier: '{dossier}'")
        
        return hospitals
    
    def normalize_hospital_name(self, name):
        """
        Normalise le nom de l'hôpital en remplaçant les abréviations courantes
        
        Args:
            name (str): Nom d'hôpital à normaliser
            
        Returns:
            str: Nom normalisé avec abréviations remplacées
        """
        if not name:
            return name
        
        # Suppression des astérisques qui peuvent apparaître
        name_clean = name.replace('*', '').strip()
        
        # Dictionnaire des abréviations courantes -> forme complète
        abbreviations = {
            r'\bHôp\.\s*': 'Hôpital ',
            r'\bCli\.\s*': 'Clinique ',
            r'\bPolycli\.\s*': 'Polyclinique ',
            r'\bInstit\.\s*': 'Institut ',
            r'\bcancéro\.\s*': 'cancérologie ',
            r'\bcardio\.\s*': 'cardiologie ',
            r'\bProtest\.\s*': 'Protestante ',
            r'\bHôp\b(?!\.)': 'Hôpital',  # Hôp sans point
            r'\bCli\b(?!\.)': 'Clinique',  # Cli sans point
            r'\bPolycli\b(?!\.)': 'Polyclinique',  # Polycli sans point
            r'\bInstit\b(?!\.)': 'Institut',  # Instit sans point
            r'\binf\b(?!\.)': 'infirmerie',  # Inf sans point
            r'\bP\b(?!\.)': 'Pierre'  # P sans point
        }
        
        # Application des remplacements
        normalized_name = name_clean
        
        for abbrev_pattern, full_form in abbreviations.items():
            new_name = re.sub(abbrev_pattern, full_form, normalized_name, flags=re.IGNORECASE)
            if new_name != normalized_name:
                # Nettoyage du pattern pour l'affichage
                clean_pattern = abbrev_pattern.replace(r'\b', '').replace(r'(?!\.)', '').replace(r'\s*', '')
                print(f"    🔄 Abréviation détectée: '{clean_pattern}' → '{full_form.strip()}'")
                normalized_name = new_name
        
        # Nettoyage final des espaces multiples
        normalized_name = re.sub(r'\s+', ' ', normalized_name).strip()
        
        return normalized_name
    
    def extract_hospital_details(self, full_name):
        """
        Extrait le nom de l'hôpital, la ville et le département du libellé complet
        
        Args:
            full_name (str): Nom complet de l'hôpital avec ville et département
            
        Returns:
            tuple: (nom_hopital, ville, departement)
        """
        if not full_name:
            return "", "", ""
        
        # Pattern pour extraire ville et département: "Nom, Ville (XX)" ou "Nom, Ville-Autre (XX)"
        pattern = r'^(.+?),\s*(.+?)\s*\((\d{2,3})\)\s*$'
        match = re.match(pattern, full_name.strip())
        
        if match:
            nom_hopital = match.group(1).strip()
            ville = match.group(2).strip()
            departement = match.group(3).strip()
            
            # Nettoyage du nom d'hôpital (suppression des virgules résiduelles)
            nom_hopital = nom_hopital.rstrip(',').strip()
            
            print(f"      📍 Détails extraits: Hôpital='{nom_hopital}' | Ville='{ville}' | Dept='{departement}'")
            return nom_hopital, ville, departement
        else:
            # Si le pattern ne correspond pas, on garde le nom complet comme nom d'hôpital
            print(f"      ⚠️  Format non reconnu pour: '{full_name}'")
            return full_name, "", ""

    def create_excel_file(self):
        """
        Crée le fichier Excel avec les données extraites
        """
        if not self.data:
            print("❌ Aucune donnée à exporter")
            return
          # Création d'un dictionnaire pour le DataFrame principal
        excel_data = []
        
        for classement, hospitals in self.data.items():
            for rang, nom_hopital_original, nom_hopital_sans_rang, dossier in hospitals:
                # Extraction des détails géographiques
                nom_hopital, ville, departement = self.extract_hospital_details(nom_hopital_sans_rang)
                
                excel_data.append({
                    'Classement': classement,
                    'Dossier': dossier,
                    'Rang': rang,
                    'Libelle_Original': nom_hopital_original,
                    'Libelle_Normalisé_Sans_Rang': nom_hopital_sans_rang,
                    'Nom_Hopital': nom_hopital,
                    'Ville': ville,
                    'Departement': departement,
                    'Date_Extraction': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # Création du DataFrame principal
        df_principal = pd.DataFrame(excel_data)
        
        # Création d'un résumé par classement
        resume_data = []
        for classement, hospitals in self.data.items():
            resume_data.append({
                'Classement': classement,
                'Nombre_Hopitaux': len(hospitals),
                'Premier_Rang': min([h[0] for h in hospitals]) if hospitals else 0,
                'Dernier_Rang': max([h[0] for h in hospitals]) if hospitals else 0
            })
        
        df_resume = pd.DataFrame(resume_data)
        
        # Écriture dans Excel avec plusieurs feuilles
        try:
            with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
                # Feuille principale avec tous les hôpitaux
                df_principal.to_excel(writer, sheet_name='Tous_Hopitaux', index=False)
                
                # Feuille de résumé
                df_resume.to_excel(writer, sheet_name='Resume_Classements', index=False)
                  # Feuilles séparées par classement
                for classement, hospitals in self.data.items():
                    if hospitals:
                        classement_data = []
                        for rang, nom_original, nom_sans_rang, dossier in hospitals:
                            nom_hopital, ville, departement = self.extract_hospital_details(nom_sans_rang)
                            classement_data.append({
                                'Rang': rang, 
                                'Libelle_Original': nom_original, 
                                'Libelle_Sans_Rang': nom_sans_rang,
                                'Nom_Hopital': nom_hopital,
                                'Ville': ville,
                                'Departement': departement,
                                'Dossier': dossier
                            })
                        
                        df_classement = pd.DataFrame(classement_data)
                        
                        # Nom de feuille sécurisé (Excel a des limites)
                        sheet_name = re.sub(r'[^\w\s-]', '', classement)[:30]
                        df_classement.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"✅ Fichier Excel créé: {self.output_file}")
            print(f"📊 Total d'hôpitaux extraits: {len(excel_data)}")
            print(f"📂 Nombre de classements: {len(self.data)}")
            
        except Exception as e:
            print(f"❌ Erreur lors de la création du fichier Excel: {e}")
    
    def process(self):
        """
        Lance le processus complet d'extraction
        """
        print("🚀 Début de l'extraction des hôpitaux...")
        
        # Lecture du fichier de résultats
        content = self.read_gemini_results()
        if not content:
            return
        
        # Extraction des hôpitaux
        print("🔍 Extraction des hôpitaux en cours...")
        self.extract_hospitals_from_content(content)
        
        # Création du fichier Excel
        print("📝 Création du fichier Excel...")
        self.create_excel_file()
        
        print("✅ Extraction terminée!")
        
        # Affichage du résumé
        self.print_summary()
    
    def print_summary(self):
        """
        Affiche un résumé des données extraites
        """
        print("\n" + "="*60)
        print("RÉSUMÉ DE L'EXTRACTION")
        print("="*60)
        
        total_hospitals = 0
        for classement, hospitals in self.data.items():
            print(f"📂 {classement}: {len(hospitals)} hôpitaux")
            total_hospitals += len(hospitals)
        
        print(f"\n🏥 Total général: {total_hospitals} hôpitaux")
        print(f"📄 Fichier Excel: {self.output_file}")

def main():
    """Fonction principale"""
    # Configuration
    INPUT_FILE = "resultats_analyse_gemini.txt"
    OUTPUT_FILE = f"hopitaux_classements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    # Vérification de l'existence du fichier d'entrée
    if not Path(INPUT_FILE).exists():
        print(f"❌ Fichier {INPUT_FILE} non trouvé dans le répertoire courant")
        print("💡 Assurez-vous d'avoir lancé le script askgemini.py d'abord")
        return
    
    try:
        # Création et lancement de l'extracteur
        extractor = HospitalExtractorToExcel(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE
        )
        
        extractor.process()
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")

if __name__ == "__main__":
    main()
