import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import Word2Vec
import os
import re
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


print("Loading dataset...")


# ---------------------------REPLACE THE FILE PATH WITH YOUR OWN FILE PATH---------------------------
df = pd.read_csv('E:\\ECE\\CS410\\Spam_detector\\spam_ham_dataset.csv')
# df = pd.read_csv('/Users/diwakar/Code/CS410_Group38_SpamDetector/spam_ham_dataset.csv')
# ---------------------------REPLACE THE FILE PATH WITH YOUR OWN FILE PATH---------------------------



print(f"Dataset loaded: {df.shape[0]} rows")

stop_words = set(stopwords.words('english')) # set of stopwords

# preprocess
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    words = text.split()
    words = [word for word in words if word not in stop_words] 
    return ' '.join(words)

# apply the preprocess_text function to the 'text' column
df['processed_text'] = df['content'].apply(preprocess_text)

# Print sample of original vs processed text only once
print("Sample of original and processed text:")
print(df[['content', 'processed_text']].head())

# ---- TF-IDF FEATURE EXTRACTION ----
print("Extracting features using TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features if needed
X_tfidf = tfidf_vectorizer.fit_transform(df['processed_text'])
X_tfidf = X_tfidf.toarray()
print(f"TF-IDF extraction complete. Matrix shape: {X_tfidf.shape}")
print(f"Sample TF-IDF features (first row):\n{X_tfidf[0][:10]}")  # Print a sample of features from first row

# ---- WORD2VEC FEATURE EXTRACTION ----
print("Extracting features using Word2Vec...")
tokenized_text = df['processed_text'].apply(lambda x: x.split())
word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

# Function to get average Word2Vec embeddings
def get_average_word2vec(tokens_list, model, vector_size):
    word_vectors = [model.wv[word] for word in tokens_list if word in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)

X_word2vec = np.array([get_average_word2vec(tokens, word2vec_model, 100) for tokens in tokenized_text])
print(f"Word2Vec extraction complete. Matrix shape: {X_word2vec.shape}")
print(f"Sample Word2Vec features (first row):\n{X_word2vec[0][:10]}")  # Print a sample of features from first row

# ---- CBOW FEATURE EXTRACTION ----
print("Extracting features using CBOW...")
cbow_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4, sg=0)
X_cbow = np.array([get_average_word2vec(tokens, cbow_model, 100) for tokens in tokenized_text])
print(f"CBOW extraction complete. Matrix shape: {X_cbow.shape}")
print(f"Sample CBOW features (first row):\n{X_cbow[0][:10]}")  # Print a sample of features from first row

# ---- SVM MODEL TRAINING AND EVALUATION USING TF-IDF ----
print("Splitting data and training SVM using TF-IDF features...")
y = df['label']  # Target labels (spam=1, ham=0)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print(f"Evaluation results using TF-IDF features:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ---- SVM MODEL TRAINING AND EVALUATION USING WORD2VEC ----
print("Splitting data and training SVM using Word2Vec features...")
X_train, X_test, y_train, y_test = train_test_split(X_word2vec, y, test_size=0.2, random_state=42)

svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print(f"Evaluation results using Word2Vec features:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ---- SVM MODEL TRAINING AND EVALUATION USING CBOW ----
print("Splitting data and training SVM using CBOW features...")
X_train, X_test, y_train, y_test = train_test_split(X_cbow, y, test_size=0.2, random_state=42)

svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print(f"Evaluation results using CBOW features:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")






