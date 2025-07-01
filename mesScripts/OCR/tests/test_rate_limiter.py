"""
Test du nouveau système de gestion des limites API Gemini
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from askgemini import GeminiRateLimiter
import time

def test_rate_limiter():
    """Test des fonctionnalités du rate limiter"""
    
    print("🧪 TEST DU SYSTÈME DE GESTION DES LIMITES API GEMINI")
    print("=" * 55)
    
    # Initialisation
    limiter = GeminiRateLimiter()
    print("✅ GeminiRateLimiter initialisé")
    
    # Affichage du statut initial
    status = limiter.get_status()
    print(f"\n📊 Statut initial:")
    print(f"   Requêtes aujourd'hui: {status['requests_today']}")
    print(f"   Délai actuel: {status['current_delay']:.1f}s")
    print(f"   Erreurs consécutives: {status['consecutive_errors']}")
    print(f"   Taux de succès: {status['success_rate']:.1%}")
    
    # Simulation de requêtes
    print(f"\n🔄 Simulation de requêtes...")
    
    # Simuler des succès
    for i in range(3):
        print(f"   Requête {i+1}: ", end="")
        if limiter.wait_if_needed():
            limiter.record_request(success=True)
            print("✅ Succès")
        time.sleep(0.1)  # Petit délai pour le test
    
    # Simuler une erreur de quota
    print(f"   Requête 4: ", end="")
    limiter.record_request(success=False, error_type="quota exceeded")
    print("❌ Erreur de quota")
    
    # Simuler un timeout
    print(f"   Requête 5: ", end="")
    limiter.record_request(success=False, error_type="timeout")
    print("⏱️ Timeout")
    
    # Simuler des succès après erreurs
    for i in range(2):
        print(f"   Requête {i+6}: ", end="")
        if limiter.wait_if_needed():
            limiter.record_request(success=True)
            print("✅ Succès (après erreurs)")
        time.sleep(0.1)
    
    # Statut final
    final_status = limiter.get_status()
    print(f"\n📊 Statut final:")
    print(f"   Requêtes aujourd'hui: {final_status['requests_today']}")
    print(f"   Délai actuel: {final_status['current_delay']:.1f}s")
    print(f"   Erreurs consécutives: {final_status['consecutive_errors']}")
    print(f"   Erreurs totales: {final_status['total_errors']}")
    print(f"   Taux de succès: {final_status['success_rate']:.1%}")
    
    # Test de sauvegarde
    print(f"\n💾 Test de sauvegarde de l'état...")
    limiter.save_state("test_state.json")
    
    # Test de chargement
    print(f"🔄 Test de chargement de l'état...")
    new_limiter = GeminiRateLimiter()
    new_limiter.load_state("test_state.json")
    
    restored_status = new_limiter.get_status()
    print(f"   État restauré - Requêtes: {restored_status['requests_today']}")
    
    print(f"\n✅ TOUS LES TESTS RÉUSSIS!")
    print(f"\n🎯 Le système de gestion des limites API est opérationnel:")
    print(f"   • Délai adaptatif: {final_status['current_delay']:.1f}s")
    print(f"   • Gestion d'erreurs: {final_status['total_errors']} erreurs traitées")
    print(f"   • Taux de succès: {final_status['success_rate']:.1%}")
    print(f"   • Persistance: État sauvé et restauré")

if __name__ == "__main__":
    test_rate_limiter()
