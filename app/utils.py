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
import spacy
import re
import en_core_web_sm
import spacy

# Load the "en_core_web_sm" model
nlp = spacy.load("en_core_web_sm")

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


def tokenize(text):
	doc = nlp(text)
	return doc

def remove_stop_words(doc):
	filtered_tokens = [token.text for token in doc if not token.is_stop]
	filtered_text = ' '.join(filtered_tokens)
	return (' '.join(filtered_tokens))

def remove_special_keys(filtered_text):
	pattern = r'[^a-zA-z0-9\s(\)\[\]\{\}]'
	cleaned_text = re.sub(pattern, '', filtered_text)
	return cleaned_text.strip(' ')

# For data cleaning
def treat(text):
	return remove_special_keys(remove_stop_words(tokenize(text)))

# Function to generate word frequency bar graph
def generate_word_frequency_bar_graph(str_input, role):
    if str_input=="":
        return

    str_input = treat(str_input)
    
    words = word_tokenize(str_input)
    word_counts = Counter(words)
    word_list, frequency_list = zip(*word_counts.most_common())

    # Create a bar graph
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(word_list[:10], frequency_list[:10])
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{role} Word Frequency Bar Graph')
    ax.set_xticklabels(word_list, rotation=45, fontsize=8)

    # Show the plot
    st.pyplot(fig)

# Function to generate WordCloud
def generate_wordcloud(str_input, role):
	try:
		str_input = treat(str_input)
		wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(str_input)
		# Plot the WordCloud image
		st.image(wordcloud.to_image(), caption=f"{role} WordCloud")
	except:
		pass
