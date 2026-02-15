# Sports vs Politics Text Classification

## Overview

This project implements a machine learning based text classifier that predicts whether a news article belongs to the **Sports** category or the **Politics** category.

The objective of this project is to:
- Perform binary text classification.
- Use TF-IDF feature representation.
- Compare at least three machine learning techniques.
- Evaluate performance using quantitative metrics.

---

## Dataset

The dataset was obtained from **Kaggle – News Category Dataset**.

Only the following two categories were selected:
- SPORTS
- POLITICS

Each data sample consists of:
- Headline
- Short description

These were combined to form the final input text.

### Dataset Balancing

Initially, the dataset was imbalanced:

- Sports: 5077 samples  
- Politics: 35602 samples  

To avoid bias, the dataset was balanced by selecting 5077 samples from each class.

Final dataset size:

- Total samples: 10,154  
- Sports: 5077  
- Politics: 5077  

---

## Feature Engineering

Text was converted into numerical form using:

### TF-IDF (Term Frequency – Inverse Document Frequency)

- English stopwords removed
- Unigrams and bigrams used
- High-dimensional sparse representation

TF-IDF helps give more importance to meaningful words and reduce the impact of common words.

---

## Machine Learning Models Used

Three different classifiers were implemented and compared:

1. **Naive Bayes**
2. **Logistic Regression**
3. **Support Vector Machine (SVM)**

All models were trained using:

- 80% training data
- 20% testing data

---

## Results

| Model | Accuracy |
|--------|----------|
| Naive Bayes | 97.24% |
| Logistic Regression | 97.14% |
| Support Vector Machine | 97.68% |

### Observations

- All models performed very well.
- SVM achieved the highest accuracy.
- Naive Bayes performed surprisingly strong despite its independence assumption.
- The dataset appears well-separated in feature space.

---

## How to Run the Project

### 1. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
