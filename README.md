# Sentiment Analysis for Legal Notices

## Project Overview

This project implements an end-to-end Natural Language Processing (NLP) pipeline for sentiment classification of legal notices. The system performs text preprocessing, feature extraction, model training, evaluation, hyperparameter tuning, experiment tracking, and version control integration.

---

## Project Structure

```text
sentiment-lab-67077/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── preprocess.py
│   ├── features.py
│   └── evaluate.py
│
├── results/
│
├── config.json
├── requirements.txt
└── README.md
```

---

## Features

* Text preprocessing and cleaning
* Tokenization
* Stopword removal
* Lemmatization/Stemming
* Bag of Words (BoW)
* TF-IDF Vectorization
* Logistic Regression Classification
* Multinomial Naive Bayes Classification
* Model Evaluation
* Hyperparameter Tuning
* MLflow Experiment Tracking
* Git & GitHub Integration

---

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* Matplotlib
* Seaborn
* MLflow
* Git
* GitHub

---

## Dataset Processing

The dataset contains legal notices with sentiment labels.

### Preprocessing Steps

1. Convert text to lowercase
2. Remove punctuation
3. Remove special characters
4. Tokenize text
5. Remove stopwords
6. Apply lemmatization/stemming
7. Save processed dataset

---

## Feature Engineering

### Bag of Words (BoW)

Converts text into numerical vectors based on word frequency.

### TF-IDF

Assigns importance scores to words based on frequency and rarity across documents.

---

## Machine Learning Models

### Logistic Regression

A linear classification algorithm used for sentiment prediction.

### Multinomial Naive Bayes

A probabilistic classifier commonly used in text classification tasks.

---

## Evaluation Metrics

The following metrics are used:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Training Time
* Inference Time

---

## Hyperparameter Tuning

Logistic Regression is tuned using different values of:

```python
C = [0.1, 5.0, 10.0]
```

The best configuration is selected based on model performance.

---

## MLflow Tracking

MLflow is used to:

* Track experiments
* Log parameters
* Store metrics
* Save trained models

Run MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://localhost:5000
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd sentiment-lab-67077
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1: Preprocess Data

```bash
python src/preprocess.py
```

### Step 2: Generate Features

```bash
python src/features.py
```

### Step 3: Train and Evaluate Models

```bash
python src/evaluate.py
```

### Step 4: Launch MLflow

```bash
mlflow ui
```

---

## Results

The project compares:

* Logistic Regression + BoW
* Logistic Regression + TF-IDF
* Naive Bayes + BoW
* Naive Bayes + TF-IDF

Performance metrics are stored in the results directory and MLflow logs.

---

## Version Control

Git commands used:

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <repository-url>
git push -u origin main
```

---

## Conclusion

This project demonstrates a complete NLP workflow for sentiment analysis of legal notices, covering preprocessing, feature extraction, machine learning, evaluation, experiment tracking, and repository management.
