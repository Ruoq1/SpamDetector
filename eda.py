import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords')

# Load dataset
stop_words = set(stopwords.words('english'))
df = pd.read_csv('data/spam_ham_dataset.csv')

print(f"Dataset loaded: {df.shape[0]} rows")

class TextProcessor:
    def __init__(self, custom_stopwords=None):
        # Initialize punctuation and stopwords
        self.punctuations = r'''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'''
        self.stop_words = set(stopwords.words('english'))
        
        # Add custom stopwords
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
    
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        clean_text = ''.join([char if char not in self.punctuations else ' ' for char in text])
        
        # Tokenize and remove stopwords
        words = clean_text.split()
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 1]
        
        return ' '.join(filtered_words)

# Set custom stopwords
custom_stopwords = {"enron", "hou", "subject", "ect", "com", "http", "www", "cc", "forwarded", "pm", "am", "company", "information", "please", "statements", "business", "time", "new", "report", "corp", "energy", "trading", "deal", "power"}
processor = TextProcessor(custom_stopwords=custom_stopwords)

# Apply preprocessing function
df['processed_text'] = df['content'].apply(processor.preprocess_text)

# Word frequency analysis
def get_word_frequency(texts, num_words=20):
    all_words = []
    for text in texts:
        words = re.findall(r'\w+', text.lower())
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    return word_counts.most_common(num_words)

# Retrieve spam and ham texts
spam_texts = df[df['label'] == 1]['processed_text']
ham_texts = df[df['label'] == 0]['processed_text']

spam_common_words = get_word_frequency(spam_texts)
ham_common_words = get_word_frequency(ham_texts)

# Convert to DataFrame for plotting
spam_df = pd.DataFrame(spam_common_words, columns=['Word', 'Frequency'])
ham_df = pd.DataFrame(ham_common_words, columns=['Word', 'Frequency'])

# Plot word frequency bar charts
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
spam_df.plot(kind='barh', x='Word', y='Frequency', ax=axes[0], color='orange', legend=False)
axes[0].set_title('Top 20 Words in Spam Emails')
axes[0].invert_yaxis()

ham_df.plot(kind='barh', x='Word', y='Frequency', ax=axes[1], color='blue', legend=False)
axes[1].set_title('Top 20 Words in Ham Emails')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('spam_ham_plot.png')

# ---- Cosine Similarity Analysis ----
print("Calculating cosine similarity between documents...")

# Vectorize text using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])

# Calculate cosine similarity
sample_size = 1000
sample_indices = pd.Series(df.index[:sample_size].tolist() + df.index[-sample_size:].tolist())

selected_tfidf_matrix = tfidf_matrix[sample_indices]

cosine_sim = cosine_similarity(selected_tfidf_matrix, selected_tfidf_matrix)

# Convert similarity matrix to DataFrame for easier viewing
cosine_sim_df = pd.DataFrame(cosine_sim, index=sample_indices, columns=sample_indices)

# Display a sample of the similarity matrix
print("Cosine Similarity Matrix (sample):")
print(cosine_sim_df.iloc[:5, :5])  # Show only the first 5 rows and columns

# Visualize similarity distribution (e.g., average similarity between spam and between ham emails)
spam_similarity = cosine_sim_df[df['label'] == 1][df['label'] == 1].values
ham_similarity = cosine_sim_df[df['label'] == 0][df['label'] == 0].values

# Calculate average similarity
avg_spam_similarity = spam_similarity.mean()
avg_ham_similarity = ham_similarity.mean()

print(f"Average Cosine Similarity between Spam Emails: {avg_spam_similarity:.4f}")
print(f"Average Cosine Similarity between Ham Emails: {avg_ham_similarity:.4f}")
