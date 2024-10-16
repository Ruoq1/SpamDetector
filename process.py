import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import os
import re
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords



df = pd.read_csv('/Users/diwakar/Code/CS410_Group38_SpamDetector/spam_ham_dataset.csv')
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

print(df[['content', 'processed_text']].head())






