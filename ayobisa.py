import pandas as pd
import numpy as np
import re
import nltk
import streamlit as st
import pickle
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif, SelectPercentile
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from indoNLP.preprocessing import replace_slang
nltk.download('punkt')

# Function to clean text
def clean_text(text):
    text = re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.lower()
    return text

# Function to normalize text
def normalize_text(text):
    return replace_slang(text)

# Initialize stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Function to apply stemming
def stemming(text_tokens):
    return ' '.join([stemmer.stem(word) for word in text_tokens])

# Streamlit app
st.title('Aplikasi Analisis Sentimen Komentar Masyarakat Terhadap Penutupan TikTok Shop')

# Input text area for user
input_text = st.text_area("Masukkan Ulasan")

if st.button("Proses"):
    # Load the best model, vectorizer, and selector
    try:
        with open('best_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
            tfidf_vectorizer = pickle.load(vectorizer_file)
        with open('selector.pkl', 'rb') as selector_file:
            selector = pickle.load(selector_file)
    except FileNotFoundError:
        st.error("Model file, vectorizer, or selector not found. Please ensure 'best_model.pkl', 'tfidf_vectorizer.pkl', and 'selector.pkl' are in the same directory.")
        st.stop()

    # Process input text
    cleaned_text = clean_text(input_text)
    normalized_text = normalize_text(cleaned_text)
    tokenized_text = word_tokenize(normalized_text)
    stemmed_text = stemming(tokenized_text)
    
    input_tfidf = tfidf_vectorizer.transform([stemmed_text])
    input_selected = selector.transform(input_tfidf)
    
    prediction = model.predict(input_selected)
    
    sentiment = "Positif" if prediction[0] == 1 else "Negatif"
    st.write(f"Hasil prediksi sentimen: {sentiment}")
else:
    st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
