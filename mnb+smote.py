import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca data
dp = pd.read_csv(r"E:\bismillah skripsi zaza\codeskripsi\pelabelan4kfixtanpanetral.csv")

# Memisahkan teks dan label
texts = dp['text_stemindo'].values
labels = dp['polarity'].values

# Membangun TF-IDF feature vector dari seluruh teks
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(texts)

# Menggunakan SMOTE untuk menangani ketidakseimbangan kelas pada data latih yang sudah diubah menjadi TF-IDF vectors
smote = SMOTE(k_neighbors=3, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, labels)

# Pembagian data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print("Jumlah data train setelah SMOTE:", X_train.shape[0])
print("Jumlah data test setelah SMOTE:", X_test.shape[0])

# Membuat dan melatih model Multinomial Naive Bayes sambil mengukur waktu
start_train_pred = time.time()
model = MultinomialNB()
model.fit(X_train, y_train)

# Memprediksi kelas pada data uji
y_pred = model.predict(X_test)
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
print("Waktu Pelatihan dan Prediksi Naive Bayes setelah SMOTE:", train_pred_time, "detik")