import os
import re
from pathlib import Path
import shutil
from datetime import datetime

class FolderRenamer:
    def __init__(self, results_file, images_folder="images", backup=True):
        """
        Initialise le renommeur de dossiers
        
        Args:
            results_file (str): Chemin vers le fichier de résultats Gemini
            images_folder (str): Dossier contenant les sous-dossiers à renommer
            backup (bool): Créer une sauvegarde avant renommage
        """
        self.results_file = Path(results_file)
        self.images_folder = Path(images_folder)
        self.backup = backup
        self.folder_titles = {}
        self.rename_log = []
        
    def clean_title_for_filename(self, title):
        """
        Nettoie un titre pour qu'il soit utilisable comme nom de dossier
        
        Args:
            title (str): Titre original
            
        Returns:
            str: Titre nettoyé pour nom de dossier
        """
        # Suppression des caractères interdits dans les noms de fichiers/dossiers
        forbidden_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        cleaned = title.strip()
        
        for char in forbidden_chars:
            cleaned = cleaned.replace(char, '')
        
        # Remplacement des espaces multiples par un seul
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Limitation de la longueur (max 100 caractères pour éviter les problèmes)
        if len(cleaned) > 100:
            cleaned = cleaned[:97] + "..."
        
        # Suppression des points en fin de nom
        cleaned = cleaned.rstrip('.')
        
        return cleaned.strip()
    
    def parse_results_file(self):
        """
        Parse le fichier de résultats pour extraire les titres par dossier
        
        Returns:
            dict: Dictionnaire {nom_dossier_original: [liste_titres]}
        """
        if not self.results_file.exists():
            print(f"❌ Le fichier de résultats {self.results_file} n'existe pas")
            return {}
        
        print(f"📖 Lecture du fichier de résultats: {self.results_file}")
        
        try:
            with open(self.results_file, 'r', encoding='utf-8') as file:
                content = file.read()
        except Exception as e:
            print(f"❌ Erreur lors de la lecture du fichier: {e}")
            return {}
          # Pattern pour identifier les sections avec séparateurs
        section_pattern = r'-{40,}\s*\n\s*FICHIER:\s*(.+?)\s*\n\s*CHEMIN:\s*(.+?)\s*\n\s*RÉPONSE GEMINI:\s*\n\s*TYPE:\s*Image originale - Page de classement\s*\n\s*TITRES EXTRAITS:\s*\n((?:\s*-\s*.+?\n)*?)\s*-{40,}'
          # Extraire le nom du dossier depuis le chemin
        def extract_folder_from_path(chemin_complet):
            # Extraire le nom du dossier depuis le chemin complet
            # Format: ...images\NOMDOSSIER\fichier.jpg
            path_parts = chemin_complet.split('\\')
            for i, part in enumerate(path_parts):
                if part == 'images' and i + 1 < len(path_parts):
                    return path_parts[i + 1]
            return None
        
        # Trouver toutes les sections
        section_matches = re.finditer(section_pattern, content, re.DOTALL)
        
        for section_match in section_matches:
            file_name = section_match.group(1).strip()
            chemin_complet = section_match.group(2).strip()
            titles_block = section_match.group(3)
            
            # Extraire le nom du dossier depuis le chemin
            folder_name = extract_folder_from_path(chemin_complet)
            
            if not folder_name:
                print(f"⚠️  Impossible d'extraire le nom du dossier pour {file_name}")
                continue
            
            # Extraction des titres
            title_lines = re.findall(r'-\s*(.+)', titles_block)
            titles = [title.strip() for title in title_lines if title.strip()]
            
            if titles:
                # Ajouter les titres au dossier (ou les combiner s'il existe déjà)
                if folder_name in self.folder_titles:
                    # Ajouter les nouveaux titres sans doublons
                    for title in titles:
                        if title not in self.folder_titles[folder_name]:
                            self.folder_titles[folder_name].append(title)
                else:
                    self.folder_titles[folder_name] = titles
                
                print(f"✅ Dossier '{folder_name}' -> {len(titles)} titre(s) trouvé(s) dans {file_name}")
            else:
                print(f"⚠️  Aucun titre trouvé pour {file_name} dans le dossier '{folder_name}'")
        
        # Nettoyage final : suppression des doublons dans chaque dossier
        for folder_name in self.folder_titles:
            original_count = len(self.folder_titles[folder_name])
            unique_titles = []
            for title in self.folder_titles[folder_name]:
                if title not in unique_titles:
                    unique_titles.append(title)
            self.folder_titles[folder_name] = unique_titles
            
            if len(unique_titles) != original_count:
                print(f"🧹 Dossier '{folder_name}' -> {original_count - len(unique_titles)} doublon(s) supprimé(s)")
        
        return self.folder_titles
    
    def generate_new_folder_name(self, original_name, titles):
        """
        Génère un nouveau nom de dossier basé sur les titres
        
        Args:
            original_name (str): Nom original du dossier
            titles (list): Liste des titres extraits
            
        Returns:
            str: Nouveau nom de dossier
        """
        if not titles:
            return original_name
        
        # Si un seul titre, l'utiliser directement
        if len(titles) == 1:
            new_name = self.clean_title_for_filename(titles[0])
        else:
            # Plusieurs titres : prendre le premier et ajouter un indicateur
            main_title = self.clean_title_for_filename(titles[0])
            new_name = f"{main_title} (+{len(titles)-1} autres)"
        
        # Vérification que le nom n'est pas vide
        if not new_name or new_name.isspace():
            return original_name
        
        return new_name
    
    def create_backup(self):
        """Crée une sauvegarde du dossier images"""
        if not self.backup:
            return
        
        backup_name = f"images_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.images_folder.parent / backup_name
        
        try:
            print(f"💾 Création de la sauvegarde: {backup_path}")
            shutil.copytree(self.images_folder, backup_path)
            print("✅ Sauvegarde créée avec succès")
        except Exception as e:
            print(f"❌ Erreur lors de la création de la sauvegarde: {e}")
            raise
    
    def rename_folders(self):
        """
        Copie les dossiers avec les nouveaux noms (garde les anciens)
        
        Returns:
            bool: True si la copie s'est bien passée
        """
        if not self.folder_titles:
            print("❌ Aucun titre trouvé, pas de copie à effectuer")
            return False
        
        if not self.images_folder.exists():
            print(f"❌ Le dossier {self.images_folder} n'existe pas")
            return False
        
        # Création de la sauvegarde si demandée
        if self.backup:
            self.create_backup()
        
        success_count = 0
        error_count = 0
        
        print(f"\n🔄 Début de la copie des dossiers avec nouveaux noms...")
        
        for original_name, titles in self.folder_titles.items():
            original_path = self.images_folder / original_name
            
            if not original_path.exists():
                print(f"⚠️  Le dossier '{original_name}' n'existe pas, ignoré")
                continue
            
            # Génération du nouveau nom
            new_name = self.generate_new_folder_name(original_name, titles)
            new_path = self.images_folder / new_name
            
            # Vérification que le nouveau nom est différent
            if original_name == new_name:
                print(f"➡️  '{original_name}' -> Pas de changement nécessaire")
                continue
            
            # Vérification que le nouveau dossier n'existe pas déjà
            if new_path.exists():
                # Ajout d'un suffixe numérique
                counter = 1
                while new_path.exists():
                    new_name_with_suffix = f"{new_name} ({counter})"
                    new_path = self.images_folder / new_name_with_suffix
                    counter += 1
                new_name = new_name_with_suffix
            
            # Copie du dossier (au lieu de renommage)
            try:
                shutil.copytree(original_path, new_path)
                print(f"✅ '{original_name}' -> COPIÉ vers '{new_name}' (original conservé)")
                
                # Log de la copie
                self.rename_log.append({
                    'original': original_name,
                    'new': new_name,
                    'titles': titles,
                    'status': 'success',
                    'action': 'copied'
                })
                success_count += 1
                
            except Exception as e:
                print(f"❌ Erreur lors de la copie de '{original_name}': {e}")
                self.rename_log.append({
                    'original': original_name,
                    'new': new_name,
                    'titles': titles,
                    'status': 'error',
                    'error': str(e),
                    'action': 'copy_failed'
                })
                error_count += 1
        
        # Résumé
        print(f"\n📊 Résumé de la copie:")
        print(f"   ✅ Succès: {success_count}")
        print(f"   ❌ Erreurs: {error_count}")
        print(f"   📁 Total traité: {len(self.folder_titles)}")
        print(f"   📂 Dossiers originaux conservés")
        
        return error_count == 0
    
    def save_rename_log(self, log_file="copy_log.txt"):
        """
        Sauvegarde un log détaillé des copies de dossiers
        
        Args:
            log_file (str): Nom du fichier de log
        """
        try:
            with open(log_file, 'w', encoding='utf-8') as file:
                file.write(f"=== LOG DE COPIE DES DOSSIERS ===\n")
                file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                file.write(f"Fichier source: {self.results_file}\n")
                file.write(f"Dossier images: {self.images_folder}\n")
                file.write(f"Action: Copie (dossiers originaux conservés)\n")
                file.write("="*60 + "\n\n")
                
                for entry in self.rename_log:
                    file.write(f"ORIGINAL: {entry['original']} (conservé)\n")
                    file.write(f"COPIE: {entry['new']}\n")
                    file.write(f"STATUT: {entry['status']}\n")
                    
                    if entry['status'] == 'error':
                        file.write(f"ERREUR: {entry['error']}\n")
                    
                    file.write("TITRES TROUVÉS:\n")
                    for title in entry['titles']:
                        file.write(f"  - {title}\n")
                    file.write("-" * 40 + "\n\n")
            
            print(f"📝 Log sauvegardé dans: {log_file}")
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde du log: {e}")
    
    def preview_changes(self):
        """
        Affiche un aperçu des changements sans les appliquer
        
        Returns:
            dict: Dictionnaire des changements prévus
        """
        print("\n👀 APERÇU DES CHANGEMENTS:")
        print("=" * 60)
        
        changes = {}
        
        for original_name, titles in self.folder_titles.items():
            new_name = self.generate_new_folder_name(original_name, titles)
            changes[original_name] = {
                'new_name': new_name,
                'titles': titles,
                'will_change': original_name != new_name
            }
            
            if original_name != new_name:
                print(f"📁 '{original_name}'")
                print(f"   -> '{new_name}'")
                print(f"   Basé sur: {', '.join(titles[:2])}")
                if len(titles) > 2:
                    print(f"   (+{len(titles)-2} autres titres)")
            else:
                print(f"📁 '{original_name}' -> Pas de changement")
            print()
        
        return changes

def main():
    """Fonction principale"""
    # Configuration
    RESULTS_FILE = "resultats_analyse_gemini.txt"  # Fichier de résultats Gemini
    IMAGES_FOLDER = r"C:\Users\LeviWEBERT\OneDrive - ALBUS PARTNERS\Bureau\Scan Medecine\tri_image_script\images"  # Dossier contenant les sous-dossiers à renommer
    CREATE_BACKUP = True  # Créer une sauvegarde avant renommage
    PREVIEW_ONLY = True  # True pour aperçu seulement, False pour renommer
    
    try:
        # Création du renommeur
        renamer = FolderRenamer(
            results_file=RESULTS_FILE,
            images_folder=IMAGES_FOLDER,
            backup=CREATE_BACKUP
        )
        
        # Parse des résultats
        print("🔍 Analyse du fichier de résultats...")
        titles_found = renamer.parse_results_file()
        
        if not titles_found:
            print("❌ Aucun titre trouvé dans le fichier de résultats")
            return
        
        print(f"✅ {len(titles_found)} dossier(s) avec titres trouvés")
        
        # Aperçu des changements
        changes = renamer.preview_changes()
        
        if PREVIEW_ONLY:
            print("\n👁️ Mode aperçu activé - Aucune copie effectuée")
            return
        
        # Confirmation utilisateur
        will_change = sum(1 for c in changes.values() if c['will_change'])
        if will_change == 0:
            print("ℹ️ Aucun dossier ne nécessite de copie")
            return
        
        print(f"\n⚠️ {will_change} dossier(s) seront copiés avec nouveaux noms")
        print("📂 Les dossiers originaux seront conservés")
        confirmation = input("Continuer ? (o/N): ").lower().strip()
        
        if confirmation not in ['o', 'oui', 'y', 'yes']:
            print("❌ Copie annulée")
            return
        
        # Copie effective
        success = renamer.rename_folders()
        
        # Sauvegarde du log
        renamer.save_rename_log()
        
        if success:
            print("✅ Copie terminée avec succès!")
            print("📁 Vous avez maintenant les dossiers originaux ET les nouveaux")
        else:
            print("⚠️ Copie terminée avec des erreurs")
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")

if __name__ == "__main__":
    main()