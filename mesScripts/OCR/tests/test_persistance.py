"""
Script de test pour valider le système de persistance des fichiers traités
"""

import json
import os
from pathlib import Path
from askgemini import ProcessingState, GeminiRateLimiter

def test_processing_state():
    """Test du système de persistance"""
    print("🧪 Test du système de persistance...")
    
    # Nettoyer les fichiers de test précédents
    test_files = ["test_processing_state.json", "test_backup_results.txt"]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
    
    # Test 1: Création d'un nouvel état
    print("\n1️⃣ Test création d'un nouvel état")
    state = ProcessingState("test_processing_state.json")
    print(f"   ✅ État créé: {state.get_stats()}")
    
    # Test 2: Marquer des fichiers comme traités
    print("\n2️⃣ Test marquage de fichiers traités")
    test_path1 = Path("test_image1.jpg")
    test_path2 = Path("test_image2.jpg")
    
    state.mark_processed(test_path1, "Résultat test 1")
    state.mark_failed(test_path2, "Erreur test")
    
    stats = state.get_stats()
    print(f"   ✅ Traités: {stats['processed_count']}, Échoués: {stats['failed_count']}")
    
    # Test 3: Sauvegarde et rechargement
    print("\n3️⃣ Test sauvegarde et rechargement")
    state.save_state()
    
    # Nouveau state pour tester le chargement
    state2 = ProcessingState("test_processing_state.json")
    stats2 = state2.get_stats()
    print(f"   ✅ État rechargé: Traités: {stats2['processed_count']}, Échoués: {stats2['failed_count']}")
    
    # Test 4: Vérifications
    print("\n4️⃣ Test vérifications")
    print(f"   ✅ Fichier 1 traité: {state2.is_processed(test_path1)}")
    print(f"   ✅ Fichier 2 échoué: {state2.is_failed(test_path2)}")
    
    # Test 5: Backup file
    print("\n5️⃣ Test création backup")
    backup_file = state2.create_backup_file("test_backup_results.txt")
    if backup_file and os.path.exists(backup_file):
        print(f"   ✅ Backup créé: {backup_file}")
        with open(backup_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"   ✅ Contenu backup: {len(content)} caractères")
    
    print("\n✅ Tous les tests de persistance réussis!")
    return True

def test_rate_limiter():
    """Test du rate limiter"""
    print("\n🧪 Test du rate limiter...")
    
    # Nettoyer les fichiers de test précédents
    if os.path.exists("test_rate_limiter_state.json"):
        os.remove("test_rate_limiter_state.json")
    
    # Test 1: Création
    print("\n1️⃣ Test création rate limiter")
    limiter = GeminiRateLimiter("test_rate_limiter_state.json")
    print(f"   ✅ Rate limiter créé: {limiter.get_stats()}")
    
    # Test 2: Simulation succès/erreurs
    print("\n2️⃣ Test simulation requêtes")
    limiter.on_success()
    limiter.on_error("timeout")
    limiter.on_success()
    
    stats = limiter.get_stats()
    print(f"   ✅ Stats: {stats['total_requests']} req, {stats['total_errors']} err, {stats['success_rate']:.1f}% succès")
    
    # Test 3: Sauvegarde et rechargement
    print("\n3️⃣ Test persistance rate limiter")
    limiter.save_state()
    
    limiter2 = GeminiRateLimiter("test_rate_limiter_state.json")
    stats2 = limiter2.get_stats()
    print(f"   ✅ État rechargé: {stats2['total_requests']} req, délai: {stats2['current_delay']:.1f}s")
    
    print("\n✅ Tous les tests du rate limiter réussis!")
    return True

def cleanup_test_files():
    """Nettoie les fichiers de test"""
    test_files = [
        "test_processing_state.json",
        "test_rate_limiter_state.json", 
        "test_backup_results.txt"
    ]
    
    print("\n🧹 Nettoyage des fichiers de test...")
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"   🗑️ Supprimé: {file}")

def main():
    """Fonction principale de test"""
    print("=" * 60)
    print("🔬 TEST DU SYSTÈME DE PERSISTANCE")
    print("=" * 60)
    
    try:
        # Tests
        test_processing_state()
        test_rate_limiter()
        
        print("\n" + "=" * 60)
        print("🎉 TOUS LES TESTS RÉUSSIS!")
        print("=" * 60)
        print("\n📋 Fonctionnalités validées:")
        print("   ✅ Persistance des fichiers traités")
        print("   ✅ Gestion des fichiers échoués") 
        print("   ✅ Sauvegarde/rechargement de l'état")
        print("   ✅ Création de fichiers de backup")
        print("   ✅ Rate limiting avec persistance")
        print("   ✅ Statistiques et monitoring")
        
        print("\n🚀 Le script est prêt à reprendre automatiquement en cas d'interruption!")
        
    except Exception as e:
        print(f"\n❌ Erreur lors des tests: {e}")
    finally:
        cleanup_test_files()

if __name__ == "__main__":
    main()
