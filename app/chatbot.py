import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import matplotlib.pyplot as plt
import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud

from utils import get_most_similar_response
from utils import generate_word_frequency_bar_graph
from utils import generate_wordcloud

# Download NLTK data (if not already downloaded)
nltk.download('punkt')

import os
current_directory = os.getcwd()

df = pd.read_csv(f"{current_directory}/app/df.csv", delimiter=";")
df = df.drop_duplicates(subset='response')

st.title("RandoBot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask about the Human Body, or type 'trivia'"):
    # Display user message in chat message container
    st.chat_message("user").markdown(f"You: {prompt}")

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": f"You: {prompt}"})

    responses = ""
    if prompt == "trivia":
        trivia = df.iloc[random.randint(0,103)] 
        responses = ["Here's a trivia: " + trivia["user_chat"] + " Answer: " + trivia["response"]]
    else:
        responses = get_most_similar_response(df, prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        for response in responses:
            st.markdown(f"Rando: {response}")

    # Add assistant response to chat history
    for response in responses:
        st.session_state.messages.append({"role": "assistant", "content": f"Rando: {response}"})

# Create a new tab for word frequency analysis
if st.button("Chat Analytics"):
    
    # Get all user messages and concatenate them
    user_input = [message["content"] for message in st.session_state.messages if message["role"] == "user"]
    concatenated_user_input = " ".join(user_input) \
        .replace("You","") \
        .replace(":","") \
        .strip()

    chatbot_response = [message["content"] for message in st.session_state.messages if message["role"] == "assistant"]
    concatenated_chatbot_response = " ".join(chatbot_response).replace("Rando", "") \
       .replace("Surely", "") \
       .replace("I'm sorry, I don't have the answer to that trivia question.", "") \
       .replace("Hmm, I'm not sure about that one.", "") \
       .replace("I don't have that trivia in my database.", "") \
       .replace("I guess it's ","") \
       .replace("Here's a trivia","") \
       .replace("Answer","") \
       .replace(":","") \
       .replace(",","") \
       .strip()

    if concatenated_user_input!="" and concatenated_chatbot_response!="":
        generate_word_frequency_bar_graph(concatenated_user_input, "User")
        generate_wordcloud(concatenated_user_input, "User")
        generate_word_frequency_bar_graph(concatenated_chatbot_response, "Chatbot")
        generate_wordcloud(concatenated_chatbot_response, "Chatbot")

    else:
        st.chat_message("assistant").markdown(f"Rando: Analytics is only be available when chatbot is used")
