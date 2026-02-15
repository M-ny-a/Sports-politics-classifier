import json
import random
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


sports_texts = []
politics_texts = []


#reading dataset 
with open("News_Category_Dataset_v3.json", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        category = data["category"]
        text = (data["headline"] + " " + data["short_description"]).lower()

        if category == "SPORTS":
            sports_texts.append(text)
        elif category == "POLITICS":
            politics_texts.append(text)

# Balance dataset
min_samples = min(len(sports_texts), len(politics_texts))
sports_texts = sports_texts[:min_samples]
politics_texts = politics_texts[:min_samples]

print("Balanced samples per class:", min_samples)



# Prepare data

X = sports_texts + politics_texts
y = [1] * min_samples + [0] * min_samples   # 1 = Sports, 0 = Politics

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# TF-IDF Vectorization

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


##Naive Bayes

nb = MultinomialNB()
nb.fit(X_train_vec, y_train)

nb_pred = nb.predict(X_test_vec)

print("\n--- Naive Bayes ---")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))



## Logistic Regression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_vec, y_train)

lr_pred = lr.predict(X_test_vec)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

## SVM
svm = LinearSVC()
svm.fit(X_train_vec, y_train)

svm_pred = svm.predict(X_test_vec)

print("\n--- Support Vector Machine ---")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))
