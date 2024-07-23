import pandas as pd
import numpy as np
import re
import nltk
import pickle
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, SelectPercentile
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from imblearn.over_sampling import SMOTE
from indoNLP.preprocessing import replace_slang
import matplotlib.pyplot as plt
import seaborn as sns

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

# Initialize stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Function to apply stemming
def stemming(text_tokens):
    return ' '.join([stemmer.stem(word) for word in text_tokens])

# Apply functions to clean, normalize, tokenize, and stem text
dp['text_clean'] = dp['text_stemindo'].apply(clean_text)
dp['text_normalized'] = dp['text_clean'].apply(normalize_text)
dp['text_tokenized'] = dp['text_normalized'].apply(word_tokenize)
dp['text_stemmed'] = dp['text_tokenized'].apply(stemming)

# Separate texts and labels
texts = dp['text_stemmed'].values
labels = dp['polarity'].values

# Check class distribution
print("Distribusi kelas sebelum SMOTE:")
print(pd.Series(labels).value_counts())

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(texts)

# SMOTE for balancing
smote = SMOTE(k_neighbors=3, random_state=42)
X_res, y_res = smote.fit_resample(X_tfidf, labels)

# Check class distribution after SMOTE
print("Distribusi kelas setelah SMOTE:")
print(pd.Series(y_res).value_counts())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Feature selection
threshold_percent = 60
selector = SelectPercentile(mutual_info_classif, percentile=threshold_percent)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_selected, y_train)

# Predict on the test set
y_pred = model.predict(X_test_selected)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model, vectorizer, and selector
with open('best_model1.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('tfidf_vectorizer1.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)
with open('selector1.pkl', 'wb') as selector_file:
    pickle.dump(selector, selector_file)
