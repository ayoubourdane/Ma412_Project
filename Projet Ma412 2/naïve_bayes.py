

#pip install datasets

from datasets import load_dataset
dataset = load_dataset("adsabs/SciX_UAT_keywords")

print(dataset)
print(dataset['train'][0])

from collections import Counter
import matplotlib.pyplot as plt

keywords = [kw for sample in dataset['train'] for kw in sample['verified_uat_labels']]
class_distribution = Counter(keywords)

plt.figure(figsize=(80,10))
plt.bar(class_distribution.keys(), class_distribution.values())
plt.xticks(rotation=90)
plt.xlabel('Keywords')

plt.ylabel('Count(Freq)')
plt.title('Class Distribution')
plt.show()

#pip install nltk

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

import numpy as np

label_frequencies = np.sum(y_train, axis=0)
total_samples = y_train.shape[0]
always_present_labels = np.where(label_frequencies == total_samples)[0]

print("Always Present Labels:")
print([mlb.classes_[i]for i in always_present_labels])

y_train = np.delete(y_train, always_present_labels, axis=1)
y_test = np.delete(y_test, always_present_labels, axis=1)

mlb.classes_ = np.delete(mlb.classes_, always_present_labels)

from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

nb_model = OneVsRestClassifier(MultinomialNB())
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='micro'))
print("Recall:", recall_score(y_test, y_pred, average='micro'))
print("F1 Score:", f1_score(y_test, y_pred, average='micro'))

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))