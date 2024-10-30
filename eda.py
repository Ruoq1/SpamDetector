import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# 假设 df 是你的数据集
stop_words = set(stopwords.words('english'))
df = pd.read_csv('E:\\ECE\\CS410\\Spam_detector\\spam_ham_dataset.csv')

class TextProcessor:
    def __init__(self, custom_stopwords=None):
        # 初始化标点符号和停用词
        self.punctuations = r'''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'''
        self.stop_words = set(stopwords.words('english'))
        
        # 添加自定义忽略词
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
    
    def preprocess_text(self, text):
        # 转换为小写
        text = text.lower()
        
        # 移除 HTML 标签
        text = re.sub(r'<.*?>', '', text)
        
        # 去除数字
        text = re.sub(r'\d+', '', text)
        
        # 去除标点符号
        clean_text = ''.join([char if char not in self.punctuations else ' ' for char in text])
        
        # 分词并去除停用词、自定义忽略词和单个字母
        words = clean_text.split()
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 1]
        
        return ' '.join(filtered_words)

# 设置自定义忽略词，例如 "enron" 和 "hou"
custom_stopwords = {"enron", "hou", "subject", "ect", "com", "http", "www", "cc", "forwarded", "pm", "am"}
processor = TextProcessor(custom_stopwords=custom_stopwords)

# 应用预处理函数
df['processed_text'] = df['content'].apply(processor.preprocess_text)

# 计算词频函数
def get_word_frequency(texts, num_words=20):
    all_words = []
    for text in texts:
        words = re.findall(r'\w+', text.lower())
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    return word_counts.most_common(num_words)

# 分别获取 spam 和 ham 的前 20 个常见词
spam_texts = df[df['label'] == 1]['processed_text']
ham_texts = df[df['label'] == 0]['processed_text']

spam_common_words = get_word_frequency(spam_texts)
ham_common_words = get_word_frequency(ham_texts)

# 转换为 DataFrame 便于绘图
spam_df = pd.DataFrame(spam_common_words, columns=['Word', 'Frequency'])
ham_df = pd.DataFrame(ham_common_words, columns=['Word', 'Frequency'])

# 绘制条形图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
spam_df.plot(kind='barh', x='Word', y='Frequency', ax=axes[0], color='orange', legend=False)
axes[0].set_title('Top 20 Words in Spam Emails')
axes[0].invert_yaxis()

ham_df.plot(kind='barh', x='Word', y='Frequency', ax=axes[1], color='blue', legend=False)
axes[1].set_title('Top 20 Words in Ham Emails')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()
