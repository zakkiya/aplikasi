import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from indoNLP.preprocessing import replace_slang

# Load the best model, vectorizer, and selector
try:
    with open('best_model1.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vectorizer1.pkl', 'rb') as vectorizer_file:
        tfidf_vectorizer = pickle.load(vectorizer_file)
    with open('selector1.pkl', 'rb') as selector_file:
        selector = pickle.load(selector_file)
except FileNotFoundError:
    st.error("Model file, vectorizer, atau selector tidak ditemukan. Pastikan 'best_model1.pkl', 'tfidf_vectorizer1.pkl', dan 'selector1.pkl' ada di direktori yang sama.")
    st.stop()

# Function to perform case folding
def case_folding(text):
    return text.lower()

# Function to clean text
def clean_text(text):
    text = re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    text = re.sub(r"\d+", "", text)
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

# Apply custom CSS for background and button colors
st.markdown(
    """
    <style>
    .stApp {
        background-color: #8FBC8F;  /* Background color */
    }
    .stButton>button {
        color: white;
        background-color: #6B8E23; 
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;  /* Darker green */
    }
    .justified-text {
        text-align: justify;
    }
    .highlight {
        background-color: #f0e68c;  /* Light yellow */
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h2 class="centered-title">APLIKASI ANALISIS SENTIMEN KOMENTAR YOUTUBE TERHADAP PENUTUPAN TIKTOKSHOP MENGGUNAKAN NAÏVE BAYES DENGAN SELEKSI FITUR INFORMATION GAIN</h2>', unsafe_allow_html=True)


st.markdown(
    """
    <div class="justified-text">
    TikTok merupakan platform yang menggabungkan media sosial dengan E-Commerce atau sering disebut Social Commerce. TikTokShop adalah fitur E-Commerce dari TikTok yang memungkinkan penjual, merek, dan konten kreator untuk mempromosikan dan menjual melalui konten video, live streaming, dan fitur belanja langsung. Penutupan TikTokShop pada tanggal 4 Oktober 2023 disebabkan oleh peraturan dagang yang berlaku di Indonesia bahwa Social Commerce dilarang melakukan transaksi jual beli. Penutupan tersebut menimbulkan banyak komentar dari masyarakat baik pro dan kontra. Aplikasi ini dibuat dengan tujuan untuk mengklasifikasikan komentar baru.
    </div>
    """,
    unsafe_allow_html=True
)
st.write("") 
st.write("") 
input_text = st.text_area("Masukkan komentar di sini")

if st.button("Cek Hasil"):
    # Case folding
    folded_text = case_folding(input_text)
    st.write(f'<div class="highlight">Teks setelah case folding: {folded_text}</div>', unsafe_allow_html=True)


    # Clean text
    cleaned_text = clean_text(folded_text)
    st.markdown(f'<div class="highlight">Teks setelah pembersihan: {cleaned_text}</div>', unsafe_allow_html=True)

    # Normalize text
    normalized_text = normalize_text(cleaned_text)
    st.markdown(f'<div class="highlight">Teks setelah normalisasi: {normalized_text}</div>', unsafe_allow_html=True)

    # Tokenize text
    tokenized_text = word_tokenize(normalized_text)
    st.markdown(f'<div class="highlight">Teks setelah tokenisasi: {tokenized_text}</div>', unsafe_allow_html=True)

    # Stem text
    stemmed_text = stemming(tokenized_text)
    st.markdown(f'<div class="highlight">Teks setelah stemming: {stemmed_text}</div>', unsafe_allow_html=True)
    
    # Predict sentiment
    input_tfidf = tfidf_vectorizer.transform([stemmed_text])
    input_selected = selector.transform(input_tfidf)
    prediction = model.predict(input_selected)
    sentiment = "Positif" if prediction[0] == 'positif' else "Negatif"
    st.markdown(f'<div class="highlight">Hasil prediksi sentimen: {sentiment}</div>', unsafe_allow_html=True)
else:
    st.write("Masukkan komentar dan klik tombol 'Cek Hasil' untuk melihat hasil.")
