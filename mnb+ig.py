import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca data
dp = pd.read_csv("https://raw.githubusercontent.com/zakkiya/datamining/gh-pages/pelabelan4kfixtanpanetral.csv")
print(dp)

# Memisahkan teks dan label
texts = dp['text_stemindo'].values
labels = dp['polarity'].values

# Membangun TF-IDF feature vector dari seluruh teks
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(texts)

# Memisahkan data menjadi data latih dan data uji
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42)
print("Jumlah data train:", X_train_tfidf.shape[0])
print("Jumlah data test:", X_test_tfidf.shape[0])

# Menentukan jumlah fitur yang akan dipilih (5% dari total fitur)
total_features = X_train_tfidf.shape[1]
k_best = int(total_features * 0.95)
print("Jumlah fitur terbaik yang dipilih:", k_best)

# Seleksi fitur menggunakan Information Gain
selector = SelectKBest(mutual_info_classif, k=k_best)
X_train_tfidf = selector.fit_transform(X_train_tfidf, y_train)
X_test_tfidf = selector.transform(X_test_tfidf)

# Membuat dan melatih model Multinomial Naive Bayes sambil mengukur waktu
start_train_pred = time.time()
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Memprediksi kelas pada data uji
y_pred = model.predict(X_test_tfidf)
end_train_pred = time.time()

# Menghitung metrik evaluasi
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Menampilkan hasil evaluasi
print("Akurasi:", accuracy)
print("Presisi:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(conf_matrix)

# Menghitung dan menampilkan waktu pelatihan dan prediksi
train_pred_time = end_train_pred - start_train_pred
print("Waktu Pelatihan dan Prediksi Naive Bayes:", train_pred_time, "detik")

# Menampilkan confusion matrix dalam bentuk tabel warna
plt.figure(figsize=(5, 2))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
