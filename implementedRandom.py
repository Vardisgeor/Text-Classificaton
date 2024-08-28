import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from glob import glob
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def read_text_files(folder_path, n=100):
    files = glob(os.path.join(folder_path, '*.txt'))
    files = files[:n]
    texts = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            texts.append(text)
    return texts

# Hyperparameters
m = 1000  # Number of most frequent words to include
n = 10   # Number of most frequent words to exclude
k = 10   # Number of most rare words to exclude
n_texts = 1000

train_folder = 'train'
neg_folder = 'neg'
pos_folder = 'pos'

neg_texts = read_text_files(os.path.join('/Users/vardisgeorgilas/Desktop/aclImdb/train', neg_folder), n_texts)
neg_texts_test = read_text_files(os.path.join('/Users/vardisgeorgilas/Desktop/aclImdb/test', neg_folder), n_texts)
pos_texts = read_text_files(os.path.join('/Users/vardisgeorgilas/Desktop/aclImdb/train', pos_folder), n_texts)
pos_texts_test = read_text_files(os.path.join('/Users/vardisgeorgilas/Desktop/aclImdb/test', pos_folder), n_texts)


all_texts = neg_texts + pos_texts
all_texts_test = neg_texts_test + pos_texts_test
all_labels = [0] * len(neg_texts) + [1] * len(pos_texts)

X_train, X_test, y_train, y_test = train_test_split(all_texts, all_labels, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train a Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# learning curve

train_sizes, train_scores, test_scores = learning_curve(random_forest, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training Accuracy', color='blue')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
plt.plot(train_sizes, test_mean, label='Validation Accuracy', color='orange')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='orange', alpha=0.2)
plt.title('Learning Curve')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Plot precision, recall, and F1-score curves

precisions = []
accs = []
recalls = []
f1s = []
data = [300,500,700,900,1100,1300]

for i in data:
    neg_texts = read_text_files(os.path.join('/Users/vardisgeorgilas/Desktop/aclImdb/train', neg_folder), i)
    neg_texts_test = read_text_files(os.path.join('/Users/vardisgeorgilas/Desktop/aclImdb/test', neg_folder), i)
    pos_texts = read_text_files(os.path.join('/Users/vardisgeorgilas/Desktop/aclImdb/train', pos_folder), i)
    pos_texts_test = read_text_files(os.path.join('/Users/vardisgeorgilas/Desktop/aclImdb/test', pos_folder), i)
    
    all_texts = neg_texts + pos_texts
    all_texts_test = neg_texts_test + pos_texts_test
    all_labels = [0] * len(neg_texts) + [1] * len(pos_texts)

    X_train, X_test, y_train, y_test = train_test_split(all_texts, all_labels, test_size=0.2, random_state=42)
    
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    random_forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    random_forest.fit(X_train, y_train)

    pred = random_forest.predict(X_test)

    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    precisions.append(precision)
    accs.append(acc)
    recalls.append(recall)
    f1s.append(f1)
print(accs,precisions, recalls, f1s)


plt.plot(data, accs, label='Accuracy')
plt.plot(data, precisions, label='Precision')
plt.plot(data, recalls, label='Recall')
plt.plot(data, f1s, label='F1 Score')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.legend()
plt.show()
        
    
    