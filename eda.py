import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords')

# Load datasets
stop_words = set(stopwords.words('english'))
df = pd.read_csv('data/spam_ham_dataset.csv')
print(f"Enron dataset loaded: {df.shape[0]} rows")
df_new = pd.read_csv('data/spam_assassin.csv')
print(f"SpamAssassin dataset loaded: {df_new.shape[0]} rows")

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
df_new['processed_text'] = df_new['text'].apply(processor.preprocess_text)

# --------------- Word frequency analysis ---------------

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
spam_texts_new = df_new[df_new['target'] == 1]['processed_text']
ham_texts_new = df_new[df_new['target'] == 0]['processed_text']

spam_common_words = get_word_frequency(spam_texts)
ham_common_words = get_word_frequency(ham_texts)
spam_common_words_new = get_word_frequency(spam_texts_new)
ham_common_words_new = get_word_frequency(ham_texts_new)

# Convert to DataFrame for plotting
spam_df = pd.DataFrame(spam_common_words, columns=['Word', 'Frequency'])
ham_df = pd.DataFrame(ham_common_words, columns=['Word', 'Frequency'])
spam_df_new = pd.DataFrame(spam_common_words_new, columns=['Word', 'Frequency'])
ham_df_new = pd.DataFrame(ham_common_words_new, columns=['Word', 'Frequency'])

# Plot word frequency bar charts
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
spam_df.plot(kind='barh', x='Word', y='Frequency', ax=axes[0], color='orange', legend=False)
axes[0].set_title('Top 20 Words in Spam Emails from Enron Dataset')
axes[0].invert_yaxis()

ham_df.plot(kind='barh', x='Word', y='Frequency', ax=axes[1], color='blue', legend=False)
axes[1].set_title('Top 20 Words in Ham Emails from Enron Dataset')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('word_freq_plot_enron.png')

fig_new, axes_new = plt.subplots(1, 2, figsize=(14, 6))
spam_df_new.plot(kind='barh', x='Word', y='Frequency', ax=axes_new[0], color='red', legend=False)
axes_new[0].set_title('Top 20 Words in Spam Emails from SpamAssassin Dataset')
axes_new[0].invert_yaxis()

ham_df_new.plot(kind='barh', x='Word', y='Frequency', ax=axes_new[1], color='green', legend=False)
axes_new[1].set_title('Top 20 Words in Ham Emails from SpamAssassin Dataset')
axes_new[1].invert_yaxis()

plt.tight_layout()
plt.savefig('word_freq_plot_assassin.png')

# --------------- Cosine Similarity Analysis ---------------

print("Calculating Cosine Similarity of Enron Dataset...")

# Vectorize text using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])

# Calculate cosine similarity, sample 2000 rows to avoid out of memory
sample_size = 1000
sample_indices = pd.Series(df.index[:sample_size].tolist() + df.index[-sample_size:].tolist())
selected_tfidf_matrix = tfidf_matrix[sample_indices]

cosine_sim = cosine_similarity(selected_tfidf_matrix, selected_tfidf_matrix)

cosine_sim_df = pd.DataFrame(cosine_sim, index=sample_indices, columns=sample_indices)

# Display a sample of the similarity matrix
print("Cosine Similarity Matrix of Enron Dataset (sample):")
print(cosine_sim_df.iloc[:5, :5])  # Show only the first 5 rows and columns

# Visualize similarity distribution
spam_similarity = cosine_sim_df[df['label'] == 1][df['label'] == 1].values
ham_similarity = cosine_sim_df[df['label'] == 0][df['label'] == 0].values

# Calculate average similarity
avg_spam_similarity = spam_similarity.mean() if spam_similarity.size > 0 else 0
avg_ham_similarity = ham_similarity.mean() if ham_similarity.size > 0 else 0

print(f"Average Cosine Similarity of Enron Dataset between Spam Emails: {avg_spam_similarity:.4f}")
print(f"Average Cosine Similarity of Enron Dataset between Ham Emails: {avg_ham_similarity:.4f}")

print("Calculating Cosine Similarity of SpamAssassin Dataset...")

# Vectorize text using TF-IDF
tfidf_matrix_new = tfidf_vectorizer.fit_transform(df_new['processed_text'])

# Calculate cosine similarity
sample_size_new = 1000
sample_indices_new = pd.Series(df_new.index[:sample_size_new].tolist() + df_new.index[-sample_size_new:].tolist())
selected_tfidf_matrix_new = tfidf_matrix_new[sample_indices_new]

cosine_sim_new = cosine_similarity(selected_tfidf_matrix_new, selected_tfidf_matrix_new)

cosine_sim_new_df = pd.DataFrame(cosine_sim_new, index=sample_indices_new, columns=sample_indices_new)

# cosine_sim_new = cosine_similarity(tfidf_matrix_new, tfidf_matrix_new)

# cosine_sim_new_df = pd.DataFrame(cosine_sim_new, index=df_new.index, columns=df_new.index)

# Display a sample of the similarity matrix
print("Cosine Similarity Matrix of SpamAssassin Dataset (sample):")
print(cosine_sim_new_df.iloc[:5, :5])  # Show only the first 5 rows and columns

# Visualize similarity distribution
spam_similarity_new = cosine_sim_new_df[df_new['target'] == 1][df_new['target'] == 1].values
ham_similarity_new = cosine_sim_new_df[df_new['target'] == 0][df_new['target'] == 0].values

# Calculate average similarity
avg_spam_similarity_new = spam_similarity_new.mean() if spam_similarity_new.size > 0 else 0
avg_ham_similarity_new = ham_similarity_new.mean() if ham_similarity_new.size > 0 else 0

print(f"Average Cosine Similarity of SpamAssassin Dataset between Spam Emails: {avg_spam_similarity_new:.4f}")
print(f"Average Cosine Similarity of SpamAssassin Dataset between Ham Emails: {avg_ham_similarity_new:.4f}")