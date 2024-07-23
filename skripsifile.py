import numpy as np
import pandas as pd

dp = pd.read_csv("https://raw.githubusercontent.com/zakkiya/datamining/gh-pages/pelabelan4kfixtanpanetral.csv")
print(dp)

dp = dp.dropna()
dp.isnull().sum()
print(dp)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif, SelectPercentile
from imblearn.over_sampling import SMOTE
import time
import matplotlib.pyplot as plt
import seaborn as sns

# # Memisahkan teks dan label
texts = dp['text_stemindo'].values
labels = dp['polarity'].values

# # Membangun TF-IDF feature vector dari seluruh teks
# start_tfidf = time.time()
# tfidf_vectorizer = TfidfVectorizer()
# X_tfidf = tfidf_vectorizer.fit_transform(texts)
# end_tfidf = time.time()

# # Memisahkan data menjadi data latih dan data uji berdasarkan hasil TF-IDF
# X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42)
# print("Jumlah data train:", X_train_tfidf.shape[0])
# print("Jumlah data test:", X_test_tfidf.shape[0])

# # Membuat dan melatih model Multinomial Naive Bayes sambil mengukur waktu
# start_nb = time.time()
# model = MultinomialNB()
# model.fit(X_train_tfidf, y_train)
# end_nb = time.time()

# # Memprediksi kelas pada data uji
# start_pred = time.time()
# y_pred = model.predict(X_test_tfidf)
# end_pred = time.time()

# # Menghitung metrik evaluasi
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='weighted')
# recall = recall_score(y_test, y_pred, average='weighted')
# f1 = f1_score(y_test, y_pred, average='weighted')
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Menampilkan hasil evaluasi
# print("Akurasi:", accuracy)
# print("Presisi:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)
# print("Classification Report:\n", classification_report(y_test, y_pred))
# print("Confusion Matrix:")
# print(conf_matrix)

# # Menampilkan confusion matrix dengan warna
# plt.figure(figsize=(4, 3))
# sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.show()

# # Menghitung dan menampilkan waktu setiap proses
# tfidf_time = end_tfidf - start_tfidf
# nb_time = end_nb - start_nb
# pred_time = end_pred - start_pred

# print("Waktu TF-IDF:", tfidf_time, "detik")
# print("Waktu Pelatihan Naive Bayes:", nb_time, "detik")
# print("Waktu Prediksi:", pred_time, "detik")

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 #mnb+smote
# Memisahkan teks dan label
texts = dp['text_stemindo'].values
labels = dp['polarity'].values
# # Mengukur waktu untuk TF-IDF vectorization
# start_tfidf = time.time()
# tfidf_vectorizer = TfidfVectorizer()
# X_tfidf = tfidf_vectorizer.fit_transform(texts)
# end_tfidf = time.time()
# tfidf_time = end_tfidf - start_tfidf
# print("Waktu TF-IDF:", tfidf_time, "detik")

# # Mengukur waktu untuk SMOTE
# start_smote = time.time()
# smote = SMOTE(k_neighbors=3, random_state=42)
# X_res, y_res = smote.fit_resample(X_tfidf, labels)
# end_smote = time.time()
# smote_time = end_smote - start_smote
# print("Waktu SMOTE:", smote_time, "detik")

# # Mengukur waktu untuk pembagian data
# start_split = time.time()
# X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
# end_split = time.time()
# split_time = end_split - start_split
# print("Waktu Pembagian Data:", split_time, "detik")
# print("Jumlah data train:", X_train.shape[0])
# print("Jumlah data test:", X_test.shape[0])

# # Mengukur waktu untuk pelatihan Naive Bayes
# start_nb = time.time()
# model = MultinomialNB()
# model.fit(X_train, y_train)
# end_nb = time.time()
# nb_time = end_nb - start_nb
# print("Waktu Pelatihan Naive Bayes:", nb_time, "detik")

# # Mengukur waktu untuk prediksi
# start_pred = time.time()
# y_pred = model.predict(X_test)
# end_pred = time.time()
# pred_time = end_pred - start_pred
# print("Waktu Prediksi:", pred_time, "detik")

# # Menghitung metrik evaluasi
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='weighted')
# recall = recall_score(y_test, y_pred, average='weighted')
# f1 = f1_score(y_test, y_pred, average='weighted')
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Menampilkan hasil evaluasi
# print("Akurasi:", accuracy)
# print("Presisi:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)
# print("Classification Report:\n", classification_report(y_test, y_pred))
# print("Confusion Matrix:")
# print(conf_matrix)

# # Menampilkan confusion matrix dengan warna
# plt.figure(figsize=(4, 3))
# sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.show()


#mnb+smote+ig
# Tokenisasi dan stemming sudah dilakukan sebelumnya dalam 'text_stemindo'

# # Mengukur waktu untuk TF-IDF vectorization
# start_tfidf = time.time()
tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 2))
X_tfidf = tfidf_vectorizer.fit_transform(texts)
# end_tfidf = time.time()
# tfidf_time = end_tfidf - start_tfidf
# print("Waktu TF-IDF:", tfidf_time, "detik")

# Mengukur waktu untuk SMOTE
# start_smote = time.time()
smote = SMOTE(k_neighbors=3, random_state=42)
X_res, y_res = smote.fit_resample(X_tfidf, labels)
# end_smote = time.time()
# smote_time = end_smote - start_smote
# print("Waktu SMOTE:", smote_time, "detik")

# Mengukur waktu untuk pembagian data
# start_split = time.time()
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
# end_split = time.time()
# split_time = end_split - start_split
# print("Waktu Pembagian Data:", split_time, "detik")
print("Jumlah data train:", X_train.shape[0])
print("Jumlah data test:", X_test.shape[0])

# Mengukur waktu untuk seleksi fitur
start_fs = time.time()
threshold_percent = 5
selector = SelectPercentile(mutual_info_classif, percentile=threshold_percent)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
end_fs = time.time()
fs_time = end_fs - start_fs
print("Waktu Seleksi Fitur:", fs_time, "detik")

# Menampilkan jumlah fitur yang dipilih
selected_features_indices = selector.get_support(indices=True)
selected_feature_names = tfidf_vectorizer.get_feature_names_out()[selected_features_indices]
print(f"Jumlah fitur yang dipilih pada threshold {threshold_percent}%:", len(selected_feature_names))
print(f"Fitur yang dipilih:", selected_feature_names)

# Mengukur waktu untuk pelatihan Naive Bayes
start_nb = time.time()
model = MultinomialNB()
model.fit(X_train_selected, y_train)
end_nb = time.time()
nb_time = end_nb - start_nb
print("Waktu Pelatihan Naive Bayes:", nb_time, "detik")

# Mengukur waktu untuk prediksi
# start_pred = time.time()
y_pred = model.predict(X_test_selected)
# end_pred = time.time()
# pred_time = end_pred - start_pred
# print("Waktu Prediksi:", pred_time, "detik")

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

# # Menampilkan confusion matrix dengan warna
# plt.figure(figsize=(4, 3))
# sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.show()


 