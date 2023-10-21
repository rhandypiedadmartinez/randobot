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

# Download NLTK data (if not already downloaded)
nltk.download('punkt')

import os
current_directory = os.getcwd()

df = pd.read_csv(f"{current_directory}/app/df.csv", delimiter=";")
df = df.drop_duplicates(subset='response')

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

    highest_score = similarity_scores[0][sorted_indexes[0]]
    print(highest_score)
    if highest_score<=0.3:
        sorry_response = ["I'm sorry, I don't have the answer to that trivia question.","Hmm, I'm not sure about that one.","I don't have that trivia in my database."]
        return [sorry_response[random.randint(0,2)]]
    elif highest_score<=0.7:
        return ["I guess it's "] + most_similar_responses    
    return ["Surely, "] + most_similar_responses

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



# Function to generate word frequency bar graph
def generate_word_frequency_bar_graph(str_input, role):
    if str_input=="":
        return

    words = word_tokenize(str_input)
    word_counts = Counter(words)
    word_list, frequency_list = zip(*word_counts.most_common())

    # Create a bar graph
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(word_list, frequency_list)
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{role} Word Frequency Bar Graph')
    ax.set_xticklabels(word_list, rotation=45, fontsize=8)

    # Show the plot
    st.pyplot(fig)

# Function to generate WordCloud
def generate_wordcloud(str_input, role):
    try:
        st.write(str_input)
        wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(str_input)

        # Plot the WordCloud image
        st.image(wordcloud.to_image(), caption=f"{role} WordCloud")
    except:
        pass

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
       .replace(":","") \
       .strip()

    if concatenated_user_input!="" and concatenated_chatbot_response!="":
        generate_word_frequency_bar_graph(concatenated_user_input, "User")
        generate_wordcloud(concatenated_user_input, "User")

        generate_word_frequency_bar_graph(concatenated_chatbot_response, "Chatbot")
        generate_wordcloud(concatenated_chatbot_response, "Chatbot")

    else:
        st.chat_message("assistant").markdown(f"Rando: Analytics is only be available when chatbot is used")
