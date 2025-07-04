import pandas as pd
from geopy.geocoders import Nominatim
from time import sleep
import json

# Liste des régions et chefs-lieux
# departements = [
#     (1, "Ain", "Bourg-en-Bresse"),
#     (2, "Aisne", "Laon"),
#     (3, "Allier", "Moulins"),
#     (4, "Alpes-de-Haute-Provence", "Digne-les-Bains"),
#     (5, "Hautes-Alpes", "Gap"),
#     (6, "Alpes-Maritimes", "Nice"),
#     (7, "Ardèche", "Privas"),
#     (8, "Ardennes", "Charleville-Mézières"),
#     (9, "Ariège", "Foix"),
#     (10, "Aube", "Troyes"),
#     (11, "Aude", "Carcassonne"),
#     (12, "Aveyron", "Rodez"),
#     (13, "Bouches-du-Rhône", "Marseille"),
#     (14, "Calvados", "Caen"),
#     (15, "Cantal", "Aurillac"),
#     (16, "Charente", "Angoulême"),
#     (17, "Charente-Maritime", "La Rochelle"),
#     (18, "Cher", "Bourges"),
#     (19, "Corrèze", "Tulle"),
#     ("2A", "Corse-du-Sud", "Ajaccio"),
#     ("2B", "Haute-Corse", "Bastia"),
#     (21, "Côte-d'Or", "Dijon"),
#     (22, "Côtes-d'Armor", "Saint-Brieuc"),
#     (23, "Creuse", "Guéret"),
#     (24, "Dordogne", "Périgueux"),
#     (25, "Doubs", "Besançon"),
#     (26, "Drôme", "Valence"),
#     (27, "Eure", "Évreux"),
#     (28, "Eure-et-Loir", "Chartres"),
#     (29, "Finistère", "Quimper"),
#     (30, "Gard", "Nîmes"),
#     (31, "Haute-Garonne", "Toulouse"),
#     (32, "Gers", "Auch"),
#     (33, "Gironde", "Bordeaux"),
#     (34, "Hérault", "Montpellier"),
#     (35, "Ille-et-Vilaine", "Rennes"),
#     (36, "Indre", "Châteauroux"),
#     (37, "Indre-et-Loire", "Tours"),
#     (38, "Isère", "Grenoble"),
#     (39, "Jura", "Lons-le-Saunier"),
#     (40, "Landes", "Mont-de-Marsan"),
#     (41, "Loir-et-Cher", "Blois"),
#     (42, "Loire", "Saint-Étienne"),
#     (43, "Haute-Loire", "Le Puy-en-Velay"),
#     (44, "Loire-Atlantique", "Nantes"),
#     (45, "Loiret", "Orléans"),
#     (46, "Lot", "Cahors"),
#     (47, "Lot-et-Garonne", "Agen"),
#     (48, "Lozère", "Mende"),
#     (49, "Maine-et-Loire", "Angers"),
#     (50, "Manche", "Saint-Lô"),
#     (51, "Marne", "Châlons-en-Champagne"),
#     (52, "Haute-Marne", "Chaumont"),
#     (53, "Mayenne", "Laval"),
#     (54, "Meurthe-et-Moselle", "Nancy"),
#     (55, "Meuse", "Bar-le-Duc"),
#     (56, "Morbihan", "Vannes"),
#     (57, "Moselle", "Metz"),
#     (58, "Nièvre", "Nevers"),
#     (59, "Nord", "Lille"),
#     (60, "Oise", "Beauvais"),
#     (61, "Orne", "Alençon"),
#     (62, "Pas-de-Calais", "Arras"),
#     (63, "Puy-de-Dôme", "Clermont-Ferrand"),
#     (64, "Pyrénées-Atlantiques", "Pau"),
#     (65, "Hautes-Pyrénées", "Tarbes"),
#     (66, "Pyrénées-Orientales", "Perpignan"),
#     (67, "Bas-Rhin", "Strasbourg"),
#     (68, "Haut-Rhin", "Colmar"),
#     (69, "Rhône", "Lyon"),
#     (70, "Haute-Saône", "Vesoul"),
#     (71, "Saône-et-Loire", "Mâcon"),
#     (72, "Sarthe", "Le Mans"),
#     (73, "Savoie", "Chambéry"),
#     (74, "Haute-Savoie", "Annecy"),
#     (75, "Paris", "Paris"),
#     (76, "Seine-Maritime", "Rouen"),
#     (77, "Seine-et-Marne", "Melun"),
#     (78, "Yvelines", "Versailles"),
#     (79, "Deux-Sèvres", "Niort"),
#     (80, "Somme", "Amiens"),
#     (81, "Tarn", "Albi"),
#     (82, "Tarn-et-Garonne", "Montauban"),
#     (83, "Var", "Toulon"),
#     (84, "Vaucluse", "Avignon"),
#     (85, "Vendée", "La Roche-sur-Yon"),
#     (86, "Vienne", "Poitiers"),
#     (87, "Haute-Vienne", "Limoges"),
#     (88, "Vosges", "Épinal"),
#     (89, "Yonne", "Auxerre"),
#     (90, "Territoire de Belfort", "Belfort"),
#     (91, "Essonne", "Évry-Courcouronnes"),
#     (92, "Hauts-de-Seine", "Nanterre"),
#     (93, "Seine-Saint-Denis", "Bobigny"),
#     (94, "Val-de-Marne", "Créteil"),
#     (95, "Val-d'Oise", "Cergy"),
#     (971, "Guadeloupe", "Basse-Terre"),
#     (972, "Martinique", "Fort-de-France"),
#     (973, "Guyane", "Cayenne"),
#     (974, "La Réunion", "Saint-Denis"),
#     (976, "Mayotte", "Mamoudzou")
# ]
regions = [
    (1, "Hauts-de-France", "Montigny-en-Gohelle"),
    (2, "Hauts-de-France", "Saint-Omer"),
    (3, "Hauts-de-France", "Loos"),
    (4, "Hauts-de-France", "Berlairmont"),
    (5, "Hauts-de-France", "Le Cateau-Cambrésis"),
    (6, "Hauts-de-France", "Fontaine-au-Pire"),
    (7, "Hauts-de-France", "Péronne"),
    (8, "Hauts-de-France", "Noyon"),
    (9, "Hauts-de-France", "Creil"),
    (10, "Hauts-de-France", "Beauvais"),
    (11, "Hauts-de-France", "Beaurevoir"),
    (12, "Hauts-de-France", "Saint-Quentin"),
    (13, "Hauts-de-France", "Tergnier"),
    (14, "Hauts-de-France", "Hirson"),
    (15, "Hauts-de-France", "Soissons"),
    (16, "Hauts-de-France", "Brasles"),
    (17, "Hauts-de-France", "Fère-en-Tardenois"),
    (18, "Provence-Alpes-Côte d’Azur", "Rognac"),
    (19, "Provence-Alpes-Côte d’Azur", "Vitrolles"),
    (20, "Provence-Alpes-Côte d’Azur", "Cabriès"),
    (21, "Provence-Alpes-Côte d’Azur", "Marignane"),
    (22, "Provence-Alpes-Côte d’Azur", "Miramas"),
    (23, "Provence-Alpes-Côte d’Azur", "Fos-sur-Mer"),
    (24, "Provence-Alpes-Côte d’Azur", "Marseille"),
    (25, "Île-de-France", "Châtenay-Malabry"),
    (26, "Île-de-France", "Neuilly-sur-Seine"),
    (27, "Île-de-France", "Bonnières-sur-Seine"),
    (28, "Île-de-France", "Saint-Mammès"),
    (29, "Île-de-France", "Saint-Rémy-lès-Chevreuse"),
    (30, "Île-de-France", "Montereau-Fault-Yonne"),
    (31, "Île-de-France", "Buchelay"),
    (32, "Île-de-France", "Paris"),
    (33, "Île-de-France", "Clamart"),
    (34, "Pays de la Loire", "Nantes"),
    (35, "Pays de la Loire", "Saint-Sébastien-sur-Loire"),
    (36, "Pays de la Loire", "Guérande"),
    (37, "Pays de la Loire", "Rezé"),
    (38, "Pays de la Loire", "Batz-sur-Mer"),
    (39, "Pays de la Loire", "Saint-Barthélemy-d’Anjou"),
    (40, "Pays de la Loire", "Le Mans"),
    (41, "Nouvelle-Aquitaine", "Angoulême"),
    (42, "Nouvelle-Aquitaine", "L’Isle-d’Espagnac"),
    (43, "Nouvelle-Aquitaine", "La Rochelle"),
    (44, "Nouvelle-Aquitaine", "Royan"),
    (45, "Nouvelle-Aquitaine", "Saintes"),
    (46, "Nouvelle-Aquitaine", "Bordeaux"),
    (47, "Nouvelle-Aquitaine", "Talence"),
    (48, "Nouvelle-Aquitaine", "Mérignac"),
    (49, "Nouvelle-Aquitaine", "Biganos"),
    (50, "Nouvelle-Aquitaine", "Agen"),
    (51, "Nouvelle-Aquitaine", "Niort"),
    (52, "Nouvelle-Aquitaine", "Limoges"),
    (53, "Occitanie", "Carcassonne"),
    (54, "Occitanie", "Nîmes"),
    (55, "Occitanie", "Parignargues"),
    (56, "Occitanie", "Toulouse"),
    (57, "Occitanie", "Villeneuve-de-Rivière"),
    (58, "Occitanie", "Montpellier"),
    (59, "Occitanie", "Cahors"),
    (60, "Occitanie", "Tarbes"),
    (61, "Occitanie", "Espira-de-l’Agly"),
    (62, "Occitanie", "Albi"),
    (63, "Grand Est", "Charleville-Mézières"),
    (64, "Grand Est", "Villers-Semeuse"),
    (65, "Grand Est", "Vouziers"),
    (66, "Grand Est", "Signy-l’Abbaye"),
    (67, "Grand Est", "Revin"),
    (68, "Grand Est", "Troyes"),
    (69, "Grand Est", "Romilly-sur-Seine"),
    (70, "Grand Est", "Villers-Allerand"),
    (71, "Grand Est", "Reims"),
    (72, "Grand Est", "Nancy"),
    (73, "Grand Est", "Schiltigheim"),
    (74, "Grand Est", "Heimsbrunn")
]

# Configuration du géolocalisateur
geolocator = Nominatim(user_agent="ville_locator_v2")
results = []

def get_coordinates(ville, region=None):
    """
    Essaie plusieurs stratégies pour obtenir les coordonnées
    """
    strategies = [
        f"{ville}, France",
        f"{ville}, {region}, France" if region else None,
        ville,
        f"{ville} France"
    ]
    
    # Supprimer les None
    strategies = [s for s in strategies if s]
    
    for strategy in strategies:
        try:
            print(f"  Tentative: '{strategy}'")
            location = geolocator.geocode(strategy, timeout=15)
            if location:
                print(f"  ✅ Trouvé: {location.latitude}, {location.longitude}")
                return location.latitude, location.longitude
            else:
                print(f"  ❌ Aucun résultat")
        except Exception as e:
            print(f"  ⚠️ Erreur: {e}")
            continue
    
    return None, None

print("🚀 Début de la géolocalisation des chefs-lieux...")
print("=" * 60)

for i, (num, reg, locality) in enumerate(regions, 1):
    print(f"\n[{i}/{len(regions)}] 📍 {locality} ({reg})")
    
    # Obtenir les coordonnées avec plusieurs stratégies
    lat, lon = get_coordinates(locality, reg)
    
    results.append({
        "numéro": num,
        "régions": reg,
        "ville": locality,
        "latitude": lat,
        "longitude": lon,
        "status": "trouvé" if lat else "non_trouvé"
    })
    
    # Pause pour éviter les limites de l'API
    sleep(1.2)

print("\n" + "=" * 60)
print("📊 RÉSULTATS:")

# Statistiques
trouvees = sum(1 for r in results if r["latitude"] is not None)
print(f"✅ Coordonnées trouvées: {trouvees}/{len(results)}")
print(f"❌ Coordonnées manquantes: {len(results) - trouvees}")

# Afficher les villes non trouvées
non_trouvees = [r for r in results if r["latitude"] is None]
if non_trouvees:
    print(f"\n⚠️ Villes non localisées:")
    for locality in non_trouvees:
        print(f"  - {locality['ville']} ({locality['régions']})")

# Sauvegarde en JSON
output_file = "chefs_lieux_departements_coords.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n💾 Fichier JSON sauvegardé: {output_file}")

# Sauvegarde en CSV pour Excel
df = pd.DataFrame(results)
csv_file = "chefs_lieux_departements_coords.csv"
df.to_csv(csv_file, index=False, encoding="utf-8")
print(f"📊 Fichier CSV sauvegardé: {csv_file}")

print("✅ Traitement terminé avec succès !")
