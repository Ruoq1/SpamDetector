import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from gensim.models import Word2Vec
import os
import re
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
nltk.download('stopwords')
from nltk.corpus import stopwords

print("Loading dataset...")

df = pd.read_csv('data/spam_ham_dataset.csv')

print(f"Dataset loaded: {df.shape[0]} rows")

class TextProcessor:
    def __init__(self, custom_stopwords=None):
        self.punctuations = r'''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'''
        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'\d+', '', text)  # Remove numbers
        clean_text = ''.join([char if char not in self.punctuations else ' ' for char in text])  # Remove punctuation
        words = clean_text.split()
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 1]
        return ' '.join(filtered_words)

# Set custom stopwords
custom_stopwords = {"enron", "hou", "subject", "ect", "com", "http", "www", "cc", "forwarded", "pm", "am"}
processor = TextProcessor(custom_stopwords=custom_stopwords)

# Apply the preprocess_text function to the 'text' column
df['processed_text'] = df['content'].apply(processor.preprocess_text)

print("Sample of original and processed text:")
print(df[['content', 'processed_text']].head())

# ---- TF-IDF FEATURE EXTRACTION ----
print("Extracting features using TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(df['processed_text']).toarray()
print(f"TF-IDF extraction complete. Matrix shape: {X_tfidf.shape}")

# ---- WORD2VEC FEATURE EXTRACTION ----
print("Extracting features using Word2Vec...")
tokenized_text = df['processed_text'].apply(lambda x: x.split())
word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

def get_average_word2vec(tokens_list, model, vector_size):
    word_vectors = [model.wv[word] for word in tokens_list if word in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)

X_word2vec = np.array([get_average_word2vec(tokens, word2vec_model, 100) for tokens in tokenized_text])
print(f"Word2Vec extraction complete. Matrix shape: {X_word2vec.shape}")

# ---- CBOW FEATURE EXTRACTION ----
print("Extracting features using CBOW...")
cbow_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4, sg=0)
X_cbow = np.array([get_average_word2vec(tokens, cbow_model, 100) for tokens in tokenized_text])
print(f"CBOW extraction complete. Matrix shape: {X_cbow.shape}")

# ---- MODEL TRAINING AND CONFUSION MATRIX ----
# def train_and_evaluate(X, y, feature_type):
#     """
#     Train and evaluate an SVM model using the specified features and plot confusion matrix.
#     """
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     svm_model = svm.SVC(kernel='linear')
#     svm_model.fit(X_train, y_train)
#     y_pred = svm_model.predict(X_test)

#     # Evaluate performance
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)

#     print(f"Evaluation results using {feature_type} features:")
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")

#     # Confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], annot_kws={"size": 10})
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.title(f"Confusion Matrix ({feature_type})")
#     plt.show()
def train_and_evaluate(X, y, feature_type):
    """
    Train and evaluate an SVM model using the specified features and output results in text format.
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train SVM model
    svm_model = svm.SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Output evaluation results
    print(f"\nEvaluation results using {feature_type} features:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Classification breakdown
    tp = ((y_pred == 1) & (y_test == 1)).sum()  # True Positives
    tn = ((y_pred == 0) & (y_test == 0)).sum()  # True Negatives
    fp = ((y_pred == 1) & (y_test == 0)).sum()  # False Positives
    fn = ((y_pred == 0) & (y_test == 1)).sum()  # False Negatives
    
    print("\nClassification Breakdown:")
    print(f"True Positives (Spam correctly identified): {tp}")
    print(f"True Negatives (Ham correctly identified): {tn}")
    print(f"False Positives (Ham misclassified as Spam): {fp}")
    print(f"False Negatives (Spam misclassified as Ham): {fn}")
# Target labels
y = df['label']

print("Training and evaluating using TF-IDF features...")
train_and_evaluate(X_tfidf, y, feature_type="TF-IDF")

print("Training and evaluating using Word2Vec features...")
train_and_evaluate(X_word2vec, y, feature_type="Word2Vec")

print("Training and evaluating using CBOW features...")
train_and_evaluate(X_cbow, y, feature_type="CBOW")
