"""
Vérificateur de dépendances pour le système OCR médical.
Vérifie que toutes les dépendances nécessaires sont installées.
"""

import sys
import importlib
from typing import List, Tuple, Dict

def check_dependencies() -> Dict[str, bool]:
    """
    Vérifie toutes les dépendances nécessaires.
    
    Returns:
        Dictionnaire avec l'état de chaque dépendance
    """
    dependencies = {
        # Dépendances obligatoires
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'opencv-python': 'cv2',
        'pytesseract': 'pytesseract',
        'PyMuPDF': 'fitz',
        
        # Dépendances Python standard (normalement présentes)
        'os': 'os',
        'logging': 'logging',
        're': 're',
        'datetime': 'datetime',
        'typing': 'typing'
    }
    
    results = {}
    
    print("🔍 VÉRIFICATION DES DÉPENDANCES")
    print("=" * 40)
    
    for package_name, import_name in dependencies.items():
        try:
            importlib.import_module(import_name)
            results[package_name] = True
            print(f"✅ {package_name:<15} : OK")
        except ImportError:
            results[package_name] = False
            print(f"❌ {package_name:<15} : MANQUANT")
    
    return results

def get_installation_commands(missing_packages: List[str]) -> List[str]:
    """
    Génère les commandes d'installation pour les packages manquants.
    
    Args:
        missing_packages: Liste des packages manquants
        
    Returns:
        Liste des commandes pip install
    """
    # Mapping des noms d'import vers les noms de packages pip
    pip_names = {
        'cv2': 'opencv-python',
        'fitz': 'PyMuPDF',
        'pytesseract': 'pytesseract',
        'pandas': 'pandas',
        'numpy': 'numpy'
    }
    
    commands = []
    for package in missing_packages:
        pip_name = pip_names.get(package, package)
        commands.append(f"pip install {pip_name}")
    
    return commands

def check_tesseract_installation() -> bool:
    """
    Vérifie spécifiquement l'installation de Tesseract OCR.
    
    Returns:
        True si Tesseract est disponible
    """
    import os
    
    # Chemins typiques pour Tesseract
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract"
    ]
    
    print("\n🔍 VÉRIFICATION TESSERACT OCR")
    print("=" * 35)
    
    # Vérifier les chemins communs
    for path in common_paths:
        if os.path.exists(path):
            print(f"✅ Tesseract trouvé: {path}")
            return True
    
    # Vérifier via la variable PATH
    try:
        import subprocess
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Tesseract trouvé dans PATH")
            return True
    except:
        pass
    
    print("❌ Tesseract non trouvé")
    print("💡 Installez Tesseract depuis: https://github.com/UB-Mannheim/tesseract/wiki")
    return False

def check_python_version() -> bool:
    """
    Vérifie la version de Python.
    
    Returns:
        True si la version est compatible
    """
    print("\n🐍 VÉRIFICATION VERSION PYTHON")
    print("=" * 32)
    
    version = sys.version_info
    print(f"Version Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✅ Version Python compatible")
        return True
    else:
        print("❌ Version Python trop ancienne (requis: Python 3.8+)")
        return False

def main():
    """Point d'entrée principal."""
    print("🏥 VÉRIFICATEUR DE DÉPENDANCES - OCR MÉDICAL")
    print("=" * 50)
    
    # Vérifier Python
    python_ok = check_python_version()
    
    # Vérifier les dépendances Python
    deps_results = check_dependencies()
    missing_deps = [pkg for pkg, status in deps_results.items() if not status]
    
    # Vérifier Tesseract
    tesseract_ok = check_tesseract_installation()
    
    # Résumé
    print("\n📊 RÉSUMÉ")
    print("=" * 12)
    
    total_deps = len(deps_results)
    missing_count = len(missing_deps)
    success_count = total_deps - missing_count
    
    print(f"Python version: {'✅ OK' if python_ok else '❌ KO'}")
    print(f"Dépendances Python: {success_count}/{total_deps} installées")
    print(f"Tesseract OCR: {'✅ OK' if tesseract_ok else '❌ KO'}")
    
    # Instructions d'installation
    if missing_deps:
        print(f"\n🔧 PACKAGES MANQUANTS ({len(missing_deps)})")
        print("=" * 25)
        
        commands = get_installation_commands(missing_deps)
        for cmd in commands:
            print(f"  {cmd}")
        
        print("\n💡 INSTALLATION RAPIDE:")
        print("  pip install -r requirements.txt")
    
    if not tesseract_ok:
        print("\n🔧 INSTALLATION TESSERACT")
        print("=" * 25)
        print("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  Linux: sudo apt-get install tesseract-ocr tesseract-ocr-fra")
        print("  macOS: brew install tesseract")
    
    # Statut final
    all_ok = python_ok and not missing_deps and tesseract_ok
    
    print(f"\n🎯 STATUT FINAL: {'✅ PRÊT' if all_ok else '❌ CONFIGURATION INCOMPLÈTE'}")
    
    if all_ok:
        print("🚀 Vous pouvez maintenant utiliser:")
        print("   python main.py scan/1.pdf")
    else:
        print("⚠️  Installez les dépendances manquantes avant de continuer")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    exit(main())
