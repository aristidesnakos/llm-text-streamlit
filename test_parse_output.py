import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import folium
import streamlit as st
from streamlit_folium import folium_static
import yaml
import streamlit.config as config
import tiktoken
import geocoder
import json

config.set_option('server.live_save', True)

with open("creds.yaml", "r") as f:
    config = yaml.safe_load(f)

openai_api_key = config['OPENAI_API_KEY']

encoding = tiktoken.get_encoding("p50k_base")
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

template = """
    Your purpose is to output a travel itinerary in a JSON format.
    Use the following variables to generate the recommendation: {location}, {activities}, {duration}
    
    The recommendation should be a list of dictionaries, where each dictionary has the following keys:

    - location: The name of the {location}
    - summary: A short summary of the {location}
    - duration: The {duration} of the trip in days
    - activities: A list of {activities} to do in the {location}
    - itinerary: A dictionary of the itinerary, where each key is a day of the trip, and each value is a dictionary with key "places" and a list of places per day as the values. 

    An example of the dictionary with key places under the key itinerary: "places": ["Lindos Acropolis", "Elli Beach", "Old Town"]

    Remember the answer needs to be in JSON format.
"""

prompt = PromptTemplate(
    input_variables=["location","activities","duration"],
    template=template,
)

def get_places(itinerary_):
    places_dict = {}
    for day, details in itinerary_.items():
        for place in details["places"]:
            try:
                location = geocoder.osm(place)
                places_dict[place] = (location.lat, location.lng)
            except Exception as e:
                print(e," for place: ", place)
                continue
    return places_dict

def get_map(coordinates):
    center_lat = sum(coord[0] for coord in coordinates.values()) / len(coordinates)
    center_lng = sum(coord[1] for coord in coordinates.values()) / len(coordinates)

    # Create a folium Map instance
    map = folium.Map(location=[center_lat, center_lng], zoom_start=16)

    # Add markers for each place
    for place, coord in coordinates.items():
        folium.Marker(location=coord, popup=place).add_to(map)

    # Display the map
    return map

def load_LLM(location,filters,duration):
    """Logic for loading the chain you want to use should go here."""
    # Make sure your openai_api_key is set as an environment variable
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    chain = LLMChain(llm=llm, prompt=prompt)
    # inputs = {"location": location, "filters": filters, "duration": duration}
    results = chain.predict(location=location, activities=filters, duration=duration, max_length=1000, num_return_sequences=1)
    response_data = json.loads(results)
    itinerary = response_data[0]["itinerary"]
    places_dict = get_places(itinerary)
    tokencount = num_tokens_from_string(results, encoding_name="p50k_base")
    return results, tokencount, places_dict

# Streamlit app
st.title("Trip Planner")

activities = ["coffee", "museum", "beach", "temple", "dinner", "shopping","church","restaurant","bar","mountain","walk"]

# User input
location = st.text_input("Enter the location (e.g., Rhodes, Greece):")
filters = st.multiselect("Select activities:", activities)
duration = st.select_slider("Enter the duration of the trip in days:", options=[1,2,3,4,5], value=1)

if location and filters:
    # m = generate_map(location, filters, duration)
    # folium_static(m)

    itinerary, tokens, places = load_LLM(location, filters, duration)
    # st.write(itinerary)
    st.write("I used the following number of tokens: ", tokens)
    # st.write(places)
    folium_static(get_map(places))
