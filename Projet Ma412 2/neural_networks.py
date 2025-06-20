
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

#pip install tensorflow

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

input_shape = X_train.shape[1]
num_classes = y_train.shape[1]

model = Sequential([
    Dense(512, activation='relu', input_shape=(input_shape,)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train_dense = X_train.toarray()
y_train_dense = y_train

X_test_dense = X_test.toarray()
y_test_dense = y_test

history = model.fit(
    X_train_dense, y_train_dense,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_dense, y_test_dense)
)

loss,accuracy = model.evaluate(X_test_dense, y_test_dense)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

predictions = model.predict(X_test_dense)
predicted_labels = mlb.inverse_transform(predictions > 0.5)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix
import matplotlib.pyplot as plt

print("Accuracy:", accuracy_score(y_test, predictions > 0.5))
print("Precision:", precision_score(y_test, predictions > 0.5, average='micro'))
print("Recall:", recall_score(y_test, predictions > 0.5, average='micro'))
print("F1 Score:", f1_score(y_test, predictions > 0.5, average='micro'))

num_labels = y_test.shape[1]
cms = multilabel_confusion_matrix(y_test, predictions > 0.5)

for i in range(num_labels):
    print(f"Confusion Matrix for Label {mlb.classes_[i]}:")
    print(cms[i])

disp = ConfusionMatrixDisplay(confusion_matrix=cms[i], display_labels=[f"Not {mlb.classes_[i]}",mlb.classes_[i]])
disp.plot(cmap=plt.cm.Blues,values_format="d")
plt.show()