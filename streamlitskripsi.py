import pickle, pandas as pd, numpy as np, re, string, nltk, streamlit as st, matplotlib.pyplot as plt, seaborn as sns, pickle
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif, SelectPercentile
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from imblearn.over_sampling import SMOTE
from indoNLP.preprocessing import replace_slang
nltk.download('punkt')

# Load data
url = "https://raw.githubusercontent.com/zakkiya/datamining/gh-pages/pelabelan4kfixtanpanetral.csv"
dp = pd.read_csv(url)

# Drop missing values
dp = dp.dropna()

# Function to clean text
def clean_text(text):
    text = re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.lower()
    return text

# Function to normalize text
def normalize_text(text):
    return replace_slang(text)

# Apply functions to clean and normalize text
dp['text_clean'] = dp['text_stemindo'].apply(clean_text)
dp['text_normalized'] = dp['text_clean'].apply(normalize_text)

# Tokenize text
dp['text_tokenized'] = dp['text_normalized'].apply(word_tokenize)

# Initialize stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Function to apply stemming
def stemming(text_tokens):
    return ' '.join([stemmer.stem(word) for word in text_tokens])

# Apply stemming
dp['text_stemmed'] = dp['text_tokenized'].apply(stemming)

# Separate texts and labels
texts = dp['text_stemmed'].values
labels = dp['polarity'].values

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(texts)

# SMOTE for balancing
smote = SMOTE(k_neighbors=3, random_state=42)
X_res, y_res = smote.fit_resample(X_tfidf, labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Feature selection
threshold_percent = 60
selector = SelectPercentile(mutual_info_classif, percentile=threshold_percent)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Load the best model 
with open('best model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Predictions
y_pred = model.predict(X_test_selected)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Streamlit app
st.title('Aplikasi Analisis Sentimen Komentar Masyarakat Terhadap Penutupan TikTok Shop')

# Input text area for user
input_text = st.text_area("Masukkan Ulasan")

if st.button("Proses"):
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