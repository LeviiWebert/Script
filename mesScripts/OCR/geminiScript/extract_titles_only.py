import re
from pathlib import Path
from datetime import datetime

class TitleExtractor:
    def __init__(self, input_file="resultats_analyse_gemini.txt", output_file="titres_extraits.txt"):
        """
        Initialise l'extracteur de titres
        
        Args:
            input_file (str): Fichier de résultats Gemini
            output_file (str): Fichier de sortie pour les titres
        """
        self.input_file = input_file
        self.output_file = output_file
        self.titles_data = []
        
    def read_gemini_results(self):
        """
        Lit le fichier de résultats Gemini
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
    
    def extract_titles_from_content(self, content):
        """
        Extrait les titres du contenu avec leurs informations d'origine
        
        Args:
            content (str): Contenu du fichier de résultats
        """
        # Pattern pour identifier les sections complètes
        section_pattern = r'FICHIER:\s*(.+?)\s*\nCHEMIN:\s*(.+?)\s*\nRÉPONSE GEMINI:.*?TITRES EXTRAITS:\s*(.*?)(?=FICHIER:|DOSSIER:|={40}|$)'
        
        # Recherche de toutes les sections
        sections = re.findall(section_pattern, content, re.DOTALL | re.IGNORECASE)
        
        print(f"🔍 {len(sections)} sections trouvées")
        
        for fichier, chemin, titres_section in sections:
            fichier = fichier.strip()
            chemin = chemin.strip()
            
            print(f"📂 Traitement: {fichier}")
            
            # Extraction des titres de la section
            titres = self.parse_titles_section(titres_section, fichier, chemin)
            if titres:
                self.titles_data.extend(titres)
    
    def parse_titles_section(self, titles_section, fichier, chemin):
        """
        Parse une section de titres pour extraire chaque titre
        
        Args:
            titles_section (str): Section contenant les titres
            fichier (str): Nom du fichier image
            chemin (str): Chemin complet du fichier
            
        Returns:
            list: Liste de dictionnaires contenant les données des titres
        """
        titles = []
        lines = titles_section.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Nettoyage de la ligne (suppression des tirets et espaces)
            if line.startswith('-'):
                line = line[1:].strip()
            
            # Vérification que ce n'est pas une ligne de métadonnées ou de séparation
            if (line.startswith('TYPE:') or 
                line.startswith('STATUT:') or 
                line.startswith('RÉPONSE:') or
                line.startswith('FICHIER:') or
                line.startswith('CHEMIN:') or
                line.startswith('----') or  # Lignes de séparation
                line.startswith('====') or  # Lignes de séparation
                line.count('-') > 10 or     # Lignes principalement composées de tirets
                len(line) < 3):
                continue
            
            # Ajout du titre à la liste
            if line:
                titles.append({
                    'fichier': fichier,
                    'chemin': chemin,
                    'titre': line
                })
                print(f"  📝 Titre extrait: '{line}'")
        
        return titles
    
    def save_titles_to_file(self):
        """
        Sauvegarde les titres extraits dans un fichier texte
        """
        if not self.titles_data:
            print("❌ Aucun titre à sauvegarder")
            return
        
        try:
            with open(self.output_file, 'w', encoding='utf-8') as file:
                # En-tête du fichier
                file.write("=" * 80 + "\n")
                file.write("EXTRACTION DES TITRES - RÉSULTATS GEMINI\n")
                file.write(f"Date d'extraction: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                file.write(f"Fichier source: {self.input_file}\n")
                file.write(f"Total de titres extraits: {len(self.titles_data)}\n")
                file.write("=" * 80 + "\n\n")
                
                # Groupement par fichier pour un affichage ordonné
                current_fichier = None
                for title_data in self.titles_data:
                    fichier = title_data['fichier']
                    chemin = title_data['chemin']
                    titre = title_data['titre']
                    
                    # Nouvelle section pour chaque fichier
                    if fichier != current_fichier:
                        if current_fichier is not None:
                            file.write("\n" + "-" * 60 + "\n\n")
                        
                        file.write(f"FICHIER: {fichier}\n")
                        file.write(f"CHEMIN: {chemin}\n")
                        file.write("TITRES:\n")
                        current_fichier = fichier
                    
                    # Ajout du titre
                    file.write(f"- {titre}\n")
                
                # Statistiques finales
                file.write("\n" + "=" * 80 + "\n")
                file.write("STATISTIQUES:\n")
                
                # Nombre de titres par fichier
                fichiers_count = {}
                for title_data in self.titles_data:
                    fichier = title_data['fichier']
                    fichiers_count[fichier] = fichiers_count.get(fichier, 0) + 1
                
                file.write(f"Nombre de fichiers traités: {len(fichiers_count)}\n")
                file.write(f"Répartition par fichier:\n")
                for fichier, count in fichiers_count.items():
                    file.write(f"  - {fichier}: {count} titre(s)\n")
            
            print(f"✅ Fichier de titres créé: {self.output_file}")
            print(f"📝 Total de titres extraits: {len(self.titles_data)}")
            print(f"📂 Nombre de fichiers traités: {len(set(t['fichier'] for t in self.titles_data))}")
            
        except Exception as e:
            print(f"❌ Erreur lors de la création du fichier: {e}")
    
    def process(self):
        """
        Lance le processus complet d'extraction des titres
        """
        print("🚀 Début de l'extraction des titres...")
        
        # Lecture du fichier de résultats
        content = self.read_gemini_results()
        if not content:
            return
        
        # Extraction des titres
        print("🔍 Extraction des titres en cours...")
        self.extract_titles_from_content(content)
        
        # Sauvegarde des titres
        print("💾 Sauvegarde des titres...")
        self.save_titles_to_file()
        
        print("✅ Extraction des titres terminée!")
    
    def print_summary(self):
        """
        Affiche un résumé des titres extraits
        """
        if not self.titles_data:
            print("❌ Aucun titre extrait")
            return
            
        print("\n" + "="*60)
        print("RÉSUMÉ DE L'EXTRACTION DES TITRES")
        print("="*60)
        
        # Groupement par fichier
        fichiers_count = {}
        for title_data in self.titles_data:
            fichier = title_data['fichier']
            fichiers_count[fichier] = fichiers_count.get(fichier, 0) + 1
        
        for fichier, count in fichiers_count.items():
            print(f"📂 {fichier}: {count} titre(s)")
        
        print(f"\n📝 Total général: {len(self.titles_data)} titres")
        print(f"📄 Fichier de sortie: {self.output_file}")

def main():
    """Fonction principale"""
    # Configuration
    INPUT_FILE = "resultats_analyse_gemini.txt"
    OUTPUT_FILE = f"titres_extraits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Vérification de l'existence du fichier d'entrée
    if not Path(INPUT_FILE).exists():
        print(f"❌ Fichier {INPUT_FILE} non trouvé dans le répertoire courant")
        print("💡 Assurez-vous d'avoir lancé le script askgemini.py d'abord")
        return
    
    try:
        # Création et lancement de l'extracteur
        extractor = TitleExtractor(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE
        )
        
        extractor.process()
        extractor.print_summary()
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")

if __name__ == "__main__":
    main()
