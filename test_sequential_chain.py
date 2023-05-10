import pandas as pd
from langchain import PromptTemplate
from langchain.llms import OpenAI, LLMChain
from langchain.chains import SequentialChain
import folium
import streamlit as st
from streamlit_folium import folium_static
import yaml

with open("creds.yaml", "r") as f:
    config = yaml.safe_load(f)

openai_api_key = config['OPENAI_API_KEY']
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

template_places = """
    You are knowledgeable about all places in Greece, such as beaches, mountains, restaurants, bars, museums, and hotels.
    You need to find places to visit in the given {location} for {duration} days.

    Follow these instructions:
    - Daily plan needs to entail places based on user's preferred {filters}.
    - No more than 2 restaurants per day should be recommended.
    - There should be at least 1 place to visit per day.
    - Make a list of these places that can be used for querying.

    These are the recommended places for your trip:
"""

prompt_places = PromptTemplate(input_variables=["location", 'duration','filters'], template=template_places)
places_chain = LLMChain(llm=llm, prompt=prompt_places, output_key="places")

template_itinerary = """
    You are a travel planner.
    
    The recommended places for your trip are:
    {places}
    
    Here are examples of daily itineraries:

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
    - Samothrace: Samothrace in the north east Aegean is famed for its wild mountain, Saos, that towers on the island and attracts travelers who prefer alternative types of tourism beyond the standard Greek island beach vibes (which if you wish, you can also indulge in here). Mount Saos is the highest mountain in the Aegean and its gorges, forests and waterfalls make for great exploring and action. Its peak, called Fegari (Moon), is at an impressive and panoramic height of 1.611 metres (5,285ft).

    Please start with a summary, followed by the daily itinerary.

    Daily Itinerary:
"""

prompt_itinerary = PromptTemplate(input_variables=["places"], template=template_itinerary)
itinerary_chain = LLMChain(llm=llm, prompt=prompt_itinerary, output_key="itinerary")

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
    overall_chain = SequentialChain(
        chains=[places_chain, itinerary_chain],
        input_variables=["location","filters","duration"],
        # Here we return multiple variables
        output_variables=["places", "itinerary"],
        verbose=True)
    # inputs = {"location": location, "filters": filters, "duration": duration}
    results = overall_chain({"location":location, "filters": filters, "duration": duration})
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

    itinerary = load_LLM(location, filters, duration)['itinerary']
    st.write(itinerary)