
# pip install pandas scikit-learn nltk datasets

from datasets import load_dataset
dataset = load_dataset("adsabs/SciX_UAT_keywords")

print(dataset)
print(dataset['train'][0])

from collections import Counter
import matplotlib.pyplot as plt

keywords = [kw for sample in dataset['train'] for kw in sample['verified_uat_labels']]
class_distribution = Counter(keywords)
"""
plt.figure(figsize=(80,10))
plt.bar(class_distribution.keys(), class_distribution.values())
plt.xticks(rotation=90)
plt.xlabel('Keywords')

plt.ylabel('Count(Freq)')
plt.title('Class Distribution')
plt.show()
"""

# the labels are to much condensed, we will choose the top 50 most frequents labels 
# Étape 1 : créer une liste avec tous les labels
all_labels = []
for example in dataset["train"]:
    all_labels.extend(example["verified_uat_labels"])

# Étape 2 : compter les occurrences de chaque label
keyword_counts = Counter(all_labels)

# Top 50 labels
top_n = 50
most_common = keyword_counts.most_common(top_n)
labels, counts = zip(*most_common)

plt.figure(figsize=(14, 6))
plt.bar(labels, counts)
plt.xticks(rotation=90)
plt.title(f'Top {top_n} Most Frequent Labels')
plt.xlabel('Label')
plt.ylabel('Count (Freq)')
plt.tight_layout()
plt.show()


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def preprocess_text(text):
  stop_words = set(stopwords.words('english'))
  tokens = word_tokenize(text.lower())
  filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
  return " ".join(filtered_tokens)

processed_texts = [preprocess_text(sample['title']) for sample in dataset['train']]

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(processed_texts)

print(X.shape)

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
y = mlb.fit_transform([sample['verified_uat_labels'] for sample in dataset['train']])

print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(processed_texts[:5])