# Multi-label Text Classification for Scientific Keywords

This project implements multi-label text classification to predict scientific keywords (UAT labels) from scientific paper titles using various machine learning approaches.

## 📋 Project Overview

The project uses the SciX UAT keywords dataset to train models that can automatically assign relevant scientific keywords to research paper titles. We implement and compare three different approaches:

- **Decision Trees** with Multi-output Classification
- **Naive Bayes** with One-vs-Rest Strategy
- **Neural Networks** with Deep Learning

## 🗂️ Project Structure

```
Projet Ma412 2/
├── data_preprocessing.py    # Data loading and preprocessing pipeline
├── decision_tree.py        # Decision tree classifier implementation
├── naïve_bayes.py          # Naive Bayes classifier implementation
├── neural_networks.py      # Deep learning model implementation
├── requirements.txt        # Setup instructions
└── README.md              # This file
```

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.7+
- Virtual environment (recommended)

### Installation Steps

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv project
   source project/bin/activate  # On Windows: project\Scripts\activate
   ```

2. **Install required packages:**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install pandas scikit-learn nltk datasets tensorflow
   ```

3. **Download NLTK data (automatically handled in the scripts):**
   - punkt tokenizer
   - stopwords corpus
   - punkt_tab

## 📊 Dataset

The project uses the `adsabs/SciX_UAT_keywords` dataset from Hugging Face, which contains:
- Scientific paper titles
- Verified UAT (Unified Astronomy Thesaurus) labels
- Multi-label classification setup

## 🔧 Data Preprocessing

The `data_preprocessing.py` script handles:

- **Text Preprocessing:**
  - Tokenization using NLTK
  - Lowercasing
  - Stopword removal
  - Alphanumeric filtering

- **Feature Engineering:**
  - TF-IDF vectorization (max 5000 features)
  - Multi-label binarization

- **Data Analysis:**
  - Class distribution visualization
  - Top 50 most frequent labels analysis

## 🤖 Models Implemented

### 1. Decision Tree Classifier (`decision_tree.py`)
- Uses `MultiOutputClassifier` wrapper
- Handles multi-label classification directly
- Provides interpretable results

**Key Features:**
- Random state for reproducibility
- Multi-output classification strategy
- Comprehensive evaluation metrics

### 2. Naive Bayes Classifier (`naïve_bayes.py`)
- Implements `MultinomialNB` with `OneVsRestClassifier`
- Handles label filtering (removes always-present labels)
- Optimized for text classification

**Key Features:**
- Always-present label detection and removal
- One-vs-Rest multi-label strategy
- Probabilistic predictions

### 3. Neural Network (`neural_networks.py`)
- Deep learning approach using TensorFlow/Keras
- Multi-layer perceptron with dropout regularization
- Sigmoid activation for multi-label output

**Architecture:**
- Input layer: 5000 features (TF-IDF)
- Hidden layer 1: 512 neurons + ReLU + Dropout(0.5)
- Hidden layer 2: 256 neurons + ReLU + Dropout(0.5)
- Output layer: Sigmoid activation for multi-label classification

## 📈 Evaluation Metrics

All models are evaluated using:
- **Accuracy Score**
- **Precision** (micro-averaged)
- **Recall** (micro-averaged)
- **F1 Score** (micro-averaged)
- **Classification Report**
- **Confusion Matrix** (Neural Network only)

## 🚀 Usage

### Running Individual Models

1. **Data Preprocessing:**
   ```bash
   python data_preprocessing.py
   ```

2. **Decision Tree:**
   ```bash
   python decision_tree.py
   ```

3. **Naive Bayes:**
   ```bash
   python naïve_bayes.py
   ```

4. **Neural Network:**
   ```bash
   python neural_networks.py
   ```

### Expected Output

Each script will output:
- Dataset information
- Feature dimensions
- Model performance metrics
- Visualizations (where applicable)

## 📋 Requirements

- `pandas` - Data manipulation
- `scikit-learn` - Machine learning algorithms
- `nltk` - Natural language processing
- `datasets` - Hugging Face datasets
- `tensorflow` - Deep learning framework
- `matplotlib` - Data visualization
- `numpy` - Numerical computations

## 🔍 Key Features

- **Multi-label Classification:** Handles multiple keywords per document
- **Text Preprocessing:** Comprehensive NLP pipeline
- **Model Comparison:** Three different ML approaches
- **Evaluation:** Comprehensive metrics and visualizations
- **Reproducibility:** Fixed random seeds for consistent results

## 📝 Notes

- The dataset is automatically downloaded from Hugging Face
- NLTK downloads are handled automatically in the scripts
- Large visualizations are commented out by default to prevent display issues
- The neural network model uses binary crossentropy loss for multi-label classification

## 🚨 Common Issues

1. **Memory Issues:** Large TF-IDF matrices may cause memory problems on limited systems
2. **Download Times:** Initial dataset download may take time depending on connection
3. **NLTK Downloads:** First run requires internet connection for NLTK data

## 🤝 Contributing

Feel free to contribute by:
- Adding new models or approaches
- Improving preprocessing techniques
- Enhancing evaluation metrics
- Optimizing hyperparameters

## 📄 License

This project is for educational purposes as part of the Ma412 course.