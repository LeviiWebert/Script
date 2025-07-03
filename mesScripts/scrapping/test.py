import pandas as pd
from geopy.geocoders import Nominatim
from time import sleep
import json
geolocator = Nominatim(user_agent="test_locator")
location = geolocator.geocode({"city": "Nice", "country": "France"}, exactly_one=True)
print(location.latitude, location.longitude)