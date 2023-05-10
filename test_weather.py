import pandas as pd
from langchain.chains.api import open_meteo_docs
from langchain.chains import APIChain
from langchain.llms import OpenAI
import geocoder
import streamlit as st
import pydeck as pdk
import yaml
import requests
import streamlit.config as config
import urllib

config.set_option('server.live_save', True)

with open("creds.yaml", "r") as f:
    config = yaml.safe_load(f)

openai_api_key = config['OPENAI_API_KEY']
openweather_api_key = config['OPENWEATHER_API_KEY']
# print(openweather_api_key)
st.title("Dynamic Weather Map")

def get_location(location):
    location_dict = {}
    for l in location:
        g = geocoder.osm(l)
        if g.ok:
            location_dict[l] = g.latlng
        else:
            continue
    return location_dict

# Define a function to make the API call and retrieve weather data
def get_weather_map_data(lat, lon, api_key):
    palette = "-65:821692;-55:821692;-45:821692;-40:821692;-30:8257db;-20:208cec;-10:20c4e8;0:23dddd;10:c2ff28;20:fff028;25:ffc228;30:fc8014"
    # http://maps.openweathermap.org/maps/2.0/weather/TA2/2/40.8358846/14.2487679?appid=4d0b923a8608e4306e8a7709350409e0&fill_bound=true&opacity=0.6&palette=-65:821692;-55:821692;-45:821692;-40:821692;-30:8257db;-20:208cec;-10:20c4e8;0:23dddd;10:c2ff28;20:fff028;25:ffc228;30:fc8014
    palette_encoded = urllib.parse.quote(palette)
    url = f"http://api.openweathermap.org/data/2.5/weather?q=94040,US&APPID={api_key}"
    response = requests.get(url)
    print(url)
    print(response)
    return response.json()

# Add a text input widget for the user to enter their location
location = st.text_input("Enter your location")

# If the user has entered a location, convert it to latitude and longitude coordinates
if location:
    try:
        lat, lon = get_location(location)
        st.write(f"Latitude: {lat}, Longitude: {lon}")
        weather_map_data = get_weather_map_data(lat, lon, openweather_api_key)
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/satellite-streets-v11",
            initial_view_state=pdk.ViewState(
                latitude=lat,
                longitude=lon,
                zoom=3,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    "ImageLayer",
                    data=[{"url": weather_map_data}],
                    bounds=[lat-1, lon-1, lat+1, lon+1],
                ),
            ],
        ))
    except:
        st.write("Error: Could not find location.")


