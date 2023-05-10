import streamlit as st
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import openai
import pinecone
import yaml

# Load the YAML file into a dictionary
with open("creds.yaml", "r") as f:
    config = yaml.safe_load(f)

openai_api_key = config['OPENAI_API_KEY']
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
pinecone.init(
    api_key=config['PINECONE_API_KEY'],  # find at app.pinecone.io
    environment=config['PINECONE_API_ENV']  # next to api key in console
)

index_name = "yourmyth-mvp"
index = pinecone.Index(index_name)
pinecone_docs = Pinecone.from_existing_index(index_name,embedding=embeddings)

template = """
    Below is a user input for a desired travel recommendation.

    Your goal is to:

    - Make a set of travel recommendations for the country the user prefers.
    - Recommendations needs to match a user's criteria.
    
    Here are some examples of criteria:

    - Outdoors: Swimming in the sea, walking along the beach, going to natural hot springs, hiking a mountain, island hopping, snorkeling, scuba diving, surfing, sailing, kayaking, canoeing, fishing, camping, horseback riding, biking, skiing, snowboarding, snowshoeing, snowmobiling, dog sledding, ice skating, ice fishing, ice climbing, rock climbing, bungee jumping, skydiving, paragliding, hang gliding, zip lining, canyon
    - Cultural: Art galleries, museums, churches, amphitheatre, ancient ruins, temples, castles, palaces
    - Nightlife: Bars, clubs, pubs, live music, dancing
    - Food: Burger, pasta, pizza, greek salad, seafood 

    Here are some examples of locations depending on the country:

    - Italy: Palermo, Catania, Taormina, Syracuse, Agrigento, Ragusa, Cefalù, Aeolian Islands, Lipari, Stromboli, Panarea, Vulcano, Salina, Filicudi, Alicudi, Favignana, Pantelleria, Lampedusa, Ustica, San Vito Lo Capo, Scopello, Trapani, Marsala, Erice, Selinunte, Segesta, Monreale, Noto, Modica, Piazza Armerina, Enna, Messina, Milazzo, Tindari, Etna, Alcantara Gorges, Nebrodi Mountains, Naples
    - Greece: Amorgos, Rhodes, Santorini, Agathonisi, Chalki, Leros, Ikaria, Samothrace, Thasos, Agios Nikolaos, Skiathos, Zakynthos, Corfu, Nafplio, Spetses, Kilkis, Prespes, Lefkada, Volos, Mani, Elafonisos, Kythera
    - Spain: Barcelona, Malaga, Bilbao, Tenerife, Valencia, Madrid, Ibiza, Majorca, Menorca, Lanzarote, La Palma, La Gomera, El Hierro

    Here are examples of recommendations:
    - Amorgos: Amorgos is a paradise for explorers, divers, and hikers. Its small bays, mountain paths and all-white churches, make it the ideal destination for an alternative Cycladic holiday. You should visit Ammos AMORGOS, a beach bar with a great view of the sea and the sunset. 
    - Catania: Catania in Sicily has a long history in a picturesque scene. With Aetna on its backdrop it offers a lot to lovers of Nature. Moreover, it's on the water and has beautiful beaches with turquoise waters. Visit Villa Bellini, a beautiful park in the heart of the city.
    - Samothrace: Samothrace in the north east Aegean is famed for its wild mountain, Saos, that towers on the island and attracts travelers who prefer alternative types of tourism beyond the standard Greek island beach vibes (which if you wish, you can also indulge in here). Mount Saos is the highest mountain in the Aegean and its gorges, forests and waterfalls make for great exploring and action. Its peak, called Fegari (Moon), is at an impressive and panoramic height of 1.611 metres (5,285ft) and according to Homer, is where Poseidon watched the Achaeans, led by King Agamemnon, besiege Troy.
    - Epidaurus: Ancient Epidaurus is a must for lovers of culture. Not just one of the most significant archaeological sites in Greece, but also a place that has managed to become a travel destination by preserving its ancient identity, keeping its theatre in operation, and hosting dozens of shows each year.
    - Paros: Paros has a healthy art scene with numerous galleries and public spaces given over to exhibitions. Paros Park hosts a series of summer events, such as concerts in the amphitheatre and films under the stars at Cine Enastron. Also in summer, a week never seems to pass without some celebration or feast day. 
    - Samos: Experience Samos and its vibrant nightlife, a North Aegean Island in Greece. An array of bars caters to diverse tastes. From sophisticated artistic soirees to lively waterfront dance parties and themed nights, Samos offers a unique blend of ambiance and entertainment. In Iera Odos, an establishment exuding class, you’ll find erudite crowds drawn to its intellectual ambiance, artistic events, and live music. 
    - Mykonos: Mykonos is perhaps the epicenter of nightlife in Greece and for a good reason. Savor a selection of Mykonos’s finest Greek wines during a wine-tasting tour. Visit the island’s most famous beaches, including Super Paradise, Paradise, and Paraga.

    Please start with a summary of the user's input for {country} followed by recommendations.
    COUNTRY:{country}
    
    YOUR {recommendations} RESPONSE:
"""

prompt = PromptTemplate(
    input_variables=["country","recommendations"],
    template=template,
)

def load_LLM(query, docs):
    """Logic for loading the chain you want to use should go here."""
    # Make sure your openai_api_key is set as an environment variable
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    # vectored_query = gpt3_embedding(query)
    # docs = docs.similarity_search(query, k=5, include_metadata=True)
    res = openai.Embedding.create(input=[query],engine=embed_model)

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts (including the questions)
    res = index.query(xq, top_k=5, include_metadata=True)
    results = chain.run(input_documents=docs, question=query)
    return results

st.set_page_config(page_title="YourMyth", page_icon=":robot:")
st.header("YourMyth")

col1, col2 = st.columns(2)

with col1:
    st.markdown("Want to travel, but don't know where? \n\n This tool \
                will help you find your next vacation given your preferred activities. This tool is powered by \
                [LangChain](https://langchain.com/) and [OpenAI](https://openai.com) and made by \
                [@AristidesNakos](https://twitter.com/aristidesnakos). \n\n View Source Code on [Github](https://github.com/aristidesnakos/globalize-text-streamlit/blob/main/main.py)")

st.markdown("## Enter Your Preferred Activities.")

col1, col2 = st.columns(2)
with col1:
    option_country = st.selectbox(
        'Where would you like recommendations for?',
        ('Denmark', 'Spain', 'Italy', 'Greece'))
    
# with col2:
#     option_country = st.selectbox(
#         'What is your country?',
#         ('Greece', 'Italy','Spain'))

def get_text():
    input_text = st.text_area(label="User Input", label_visibility='collapsed', placeholder="Your activities", key="user_input")
    return input_text

# def get_text():
#     input_text = "I want to go to "+option_country+" because I want to "+option_activities+"."
#     return input_text

user_input = get_text()

def gpt3_embedding(content,engine='text-embedding-ada-002'):
    content = content.encode('ASCII', errors='ignore').decode()
    response = openai.Embedding.create(engine=engine, input=content)
    vector = response['data'][0]['embedding']
    return vector

def complete(prompt):
    # query text-davinci-003
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()

if len(user_input.split(" ")) > 700:
    st.write("Please enter a shorter request.")
    st.stop()

st.markdown("### Recommendation:")

if user_input:
    if not openai_api_key:
        st.warning('Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', icon="⚠️")
        st.stop()

    modified_user_input = prompt.format(country=option_country,recommendations=user_input)

    itinerary = load_LLM(modified_user_input, pinecone_docs)

    st.write(itinerary)