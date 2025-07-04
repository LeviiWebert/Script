import matplotlib.pyplot as plt
from geopy.distance import distance
from geopy.geocoders import Nominatim
import numpy as np
import time

# Liste des villes choisies
cities = [
    "Lille", "Dunkerque", "Amiens", "Reims", "Metz", "Strasbourg",
    "Paris", "Chartres", "Orléans", "Troyes",
    "Caen", "Rennes", "Saint-Brieuc", "Brest", "Lorient", "Vannes",
    "Nantes", "Angers", "Tours", "Poitiers", "Limoges", "Clermont-Ferrand",
    "Dijon", "Besançon",
    "La Rochelle", "Bordeaux", "Mont-de-Marsan", "Pau", "Bayonne",
    "Toulouse", "Montpellier", "Marseille", "Avignon", "Lyon", "Grenoble"
]

# Initialisation du géocodeur
geolocator = Nominatim(user_agent="carte_couverture_france")

# Récupération des coordonnées GPS
city_coords = []
for city in cities:
    try:
        location = geolocator.geocode(city + ", France")
        if location:
            city_coords.append((location.latitude, location.longitude, city))
            print(f"✔ Coordonnées trouvées pour {city}")
        else:
            print(f"❌ Coordonnées non trouvées pour {city}")
    except Exception as e:
        print(f"Erreur pour {city} : {e}")
    time.sleep(1)  # Pause pour éviter le blocage par le serveur

# Rayon du cercle en km
radius_km = 100

# Fonction pour tracer un cercle géographique
def plot_circle(lat, lon, radius_km, **kwargs):
    angles = np.linspace(0, 2 * np.pi, 100)
    circle_lats = []
    circle_lons = []
    for angle in angles:
        point = distance(kilometers=radius_km).destination((lat, lon), np.degrees(angle))
        circle_lats.append(point.latitude)
        circle_lons.append(point.longitude)
    plt.plot(circle_lons, circle_lats, **kwargs)

# Tracer la carte
plt.figure(figsize=(10, 12))
for lat, lon, name in city_coords:
    plot_circle(lat, lon, radius_km, color='skyblue', alpha=0.4)
    plt.plot(lon, lat, 'ro')
    plt.text(lon, lat, name, fontsize=8, ha='right')

plt.title("Couverture de la France par des cercles de 100 km de rayon")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()
