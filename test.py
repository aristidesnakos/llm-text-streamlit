import pandas as pd
from langchain import PromptTemplate
from langchain.llms import OpenAI
import openai
from langchain.chains import LLMChain
import folium
import streamlit as st
from streamlit_folium import folium_static
import yaml

with open("creds.yaml", "r") as f:
    config = yaml.safe_load(f)

openai_api_key = config['OPENAI_API_KEY']

template = """
    Your purpose is to provide a travel itinerary.
    Below you find instructions for your purpose.

    - Make a daily plan for a trip in the {location} for {duration} days.
    - Daily plan needs to entail a places based on user's preferred {filters}.
    - These places are filtered from a pandas dataframe with the following columns: {filters}, {location}.
    - Mention these places in the daily itineraries.
    
    Here are some examples of daily itineraries:

    - Day 1: 
        Start your morning by having a coffee in Monastiraki. 
        Proceed to visit the Acropolis, the Parthenon, and the Acropolis Museum.
        Then in the afternoon, have lunch at Thissio and go for a walk at the National Garden.
        In the evening, go for dinner at Plaka.
    - Day 1: 
        Rent a car and visit the Temple of Poseidon at Sounion.
        Have lunch at Vouliagmeni and go for a swim at Varkiza. 
        Then go for dinner at Glyfada.
    - Day 3: 
        In the morning you will take the metro to Piraeaus for your ferry to Spetses. 
        You can rent a bike and go around the island.

    Here are examples of summaries for different locations:
    - Amorgos: Amorgos is a paradise for explorers, divers, and hikers. Its small bays, mountain paths and all-white churches, make it the ideal destination for an alternative Cycladic holiday. You should visit Ammos AMORGOS, a beach bar with a great view of the sea and the sunset. 
    - Catania: Catania in Sicily has a long history in a picturesque scene. With Aetna on its backdrop it offers a lot to lovers of Nature. Moreover, it's on the water and has beautiful beaches with turquoise waters. Visit Villa Bellini, a beautiful park in the heart of the city.
    - Samothrace: Samothrace in the north east Aegean is famed for its wild mountain, Saos, that towers on the island and attracts travelers who prefer alternative types of tourism beyond the standard Greek island beach vibes (which if you wish, you can also indulge in here). Mount Saos is the highest mountain in the Aegean and its gorges, forests and waterfalls make for great exploring and action. Its peak, called Fegari (Moon), is at an impressive and panoramic height of 1.611 metres (5,285ft) and according to Homer, is where Poseidon watched the Achaeans, led by King Agamemnon, besiege Troy.
    - Epidaurus: Ancient Epidaurus is a must for lovers of culture. Not just one of the most significant archaeological sites in Greece, but also a place that has managed to become a travel destination by preserving its ancient identity, keeping its theatre in operation, and hosting dozens of shows each year.
    - Paros: Paros has a healthy art scene with numerous galleries and public spaces given over to exhibitions. Paros Park hosts a series of summer events, such as concerts in the amphitheatre and films under the stars at Cine Enastron. Also in summer, a week never seems to pass without some celebration or feast day. 
    - Samos: Experience Samos and its vibrant nightlife, a North Aegean Island in Greece. An array of bars caters to diverse tastes. From sophisticated artistic soirees to lively waterfront dance parties and themed nights, Samos offers a unique blend of ambiance and entertainment. In Iera Odos, an establishment exuding class, you’ll find erudite crowds drawn to its intellectual ambiance, artistic events, and live music. 
    - Mykonos: Mykonos is perhaps the epicenter of nightlife in Greece and for a good reason. Savor a selection of Mykonos’s finest Greek wines during a wine-tasting tour. Visit the island’s most famous beaches, including Super Paradise, Paradise, and Paraga.

    Please start with a summary for the {location} followed by the daily plan.

    Summary for {location}:

    Day 1 : Daily itinerary
    Day N : Daily itinerary
    Where N is the duration of the trip.
"""

prompt = PromptTemplate(
    input_variables=["location","filters","duration"],
    template=template,
)

data=pd.read_csv('places_greece.csv')

def generate_map(location, filters, duration=5):
    # Filter the data based on the location
    filtered_data = data[data["place_address"].str.contains(location, case=False)]

    # Filter the data based on the has_type values
    filtered_data.loc[:, "has_type"] = filtered_data["has_type"].fillna("").astype(str)
    filtered_data = filtered_data[filtered_data["has_type"].apply(lambda x: any(f in x.split(', ') for f in filters))]

    # Sort the filtered data based on the rating
    sorted_data = filtered_data.sort_values(by="rating", ascending=False)

    # Select the top-rated places based on the duration of the trip (e.g., 5 places per day)
    selected_places = sorted_data.head(duration * 2)

    # Get the latitude and longitude of the first result to center the map
    center_lat, center_lng = selected_places.iloc[0]["latitude"], selected_places.iloc[0]["longitude"]

    # Create a map centered on the location
    m = folium.Map(location=[center_lat, center_lng], zoom_start=12)

    # Add markers for the selected places
    for _, row in selected_places.iterrows():
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=f"{row['place_name']} ({row['rating']})",
        ).add_to(m)

    return m

def extract_unique_filters(data):
    all_filters = set()
    for types in data["has_type"].dropna().astype(str):
        all_filters.update(types.split(', '))
    return sorted(list(all_filters))

def load_LLM(location,filters,duration):
    """Logic for loading the chain you want to use should go here."""
    # Make sure your openai_api_key is set as an environment variable
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    chain = LLMChain(llm=llm, prompt=prompt)
    # inputs = {"location": location, "filters": filters, "duration": duration}
    results = chain.predict(location=location, filters=filters, duration=duration, max_length=1000, num_return_sequences=1)
    return results

# Streamlit app
st.title("Trip Planner")

# User input
location = st.text_input("Enter the location (e.g., Rhodes, Greece):")
unique_filters = extract_unique_filters(data)
filters = st.multiselect("Select filters:", unique_filters)
duration = st.number_input("Enter the duration of the trip in days:", min_value=1, value=5)

if location and filters:
    m = generate_map(location, filters, duration)
    folium_static(m)

    itinerary = load_LLM(location, filters, duration)
    st.write(itinerary)