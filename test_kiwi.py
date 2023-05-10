import pandas as pd
from langchain.chains.api import open_meteo_docs
from langchain.chains import APIChain
from langchain.llms import OpenAI
import folium
import streamlit as st
from streamlit_folium import folium_static
import yaml
import requests

st.title("Dynamic Weather Map")

st.header("Kiwi.com Widget")

html_code = '''

<div id="widget-holder"></div>
<script 
data-affilid="anakosyourmyth" 
data-to="naxos_gr" 
data-results-only="true" 
src="https://widgets.kiwi.com/scripts/widget-search-iframe.js">
</script>
</script>
'''

st.components.v1.html(html_code,height=1000, scrolling=True)

# Obtain your OpenWeatherMap API key
# api_key = "your_openweathermap_api_key"

# # Set the coordinates and zoom level for the initial map view
# center = [40.7128, -74.0060]
# zoom = 5

# # Fetch weather data from OpenWeatherMap
# weather_url = f"https://tile.openweathermap.org/map/temp_new/{zoom}/{center[0]}/{center[1]}.png?appid={api_key}"
# weather_layer = {
#     "type": "TileLayer",
#     "url": weather_url,
#     "attribution": "Weather data © OpenWeatherMap",
# }

# # Render the map with the weather layer
# map = st_leaflet_map(center=center, zoom=zoom, layers=[weather_layer])


with open("creds.yaml", "r") as f:
    config = yaml.safe_load(f)

openai_api_key = config['OPENAI_API_KEY']

def load_LLM(query):
    """Logic for loading the chain you want to use should go here."""
    # Make sure your openai_api_key is set as an environment variable
    llm = OpenAI(temperature=0,openai_api_key=openai_api_key,max_tokens=150,model_name="text-davinci-003")
    chain_new = APIChain.from_llm_and_api_docs(llm, open_meteo_docs.OPEN_METEO_DOCS, verbose=True)
    results = chain_new.run(query)
    return results

# User input
query = st.text_area("Enter the location and timeframe (e.g., Rhodes, Greece tomorrow)",height=300)

if query:
    forecast = load_LLM(query)
    st.write(forecast)