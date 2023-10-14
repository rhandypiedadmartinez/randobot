import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

def get_most_similar_response(df, query, top_k=1):
    # Step 1: Prepare Data
    vectorizer = TfidfVectorizer()
    all_data = list(df['user_chat']) + [query]

    # Step 2: TF-IDF Vectorization
    tfidf_matrix = vectorizer.fit_transform(all_data)

    # Step 3: Compute Similarity
    document_vectors = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1]
    similarity_scores = cosine_similarity(query_vector, document_vectors)

    # Step 4: Sort and Pick Top k Responses
    sorted_indexes = similarity_scores.argsort()[0][-top_k:]
    
    # Fetch the corresponding responses from the DataFrame
    most_similar_responses = df.iloc[sorted_indexes]['response'].values
    return most_similar_responses

st.title("RandoBot")

import os
current_directory = os.getcwd()

df = pd.read_csv(f"{current_directory}/app/df.csv", delimiter=";")
df = df.drop_duplicates(subset='response')

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