from sklearn.feature_extraction.text import TfidfVectorizer
import os
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_score, recall_score, f1_score

def load_reviews(directory, n):
    reviews = []
    labels = []
    for label in ['pos', 'neg']:
        current_dir = os.path.join(directory, label)
        for filename in os.listdir(current_dir)[:n]:
            with open(os.path.join(current_dir, filename), 'r', encoding='utf-8') as file:
                review_text = file.read()
                reviews.append(review_text)
                labels.append(1 if label == 'pos' else -1)
    return reviews, labels

def vectorize_data(train_data, test_data, max_features, stop_words='english', n=0, k=0):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    X_train = vectorizer.fit_transform(train_data).toarray()
    X_test = vectorizer.transform(test_data).toarray()

    # exclude the top n and bottom k words
    sorted = np.argsort(vectorizer.idf_)
    excluded = np.concatenate([sorted[:n], sorted[-k:]])

    X_train_filtered = np.delete(X_train, excluded, axis=1)
    X_test_filtered = np.delete(X_test, excluded, axis=1)

    return X_train_filtered, X_test_filtered

if __name__ == "__main__":
    
    M = 50
    n_texts = 1000
    m = 5000
    n = 10
    k = 10

    train_data, train_labels = load_reviews('/Users/vardisgeorgilas/Desktop/aclImdb/train', n=n_texts)
    test_data, test_labels = load_reviews('/Users/vardisgeorgilas/Desktop/aclImdb/test', n=n_texts)

    X_train_filtered, X_test_filtered = vectorize_data(
        train_data, test_data, m, n=n, k=k
    )

    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    base_classifier = DecisionTreeClassifier(max_depth=1)
    clf = AdaBoostClassifier(base_classifier, n_estimators=M, random_state=42)
    clf.fit(X_train_filtered, y_train)
    y_pred = clf.predict(X_test_filtered)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    ############################################### Diagrams #############################################
    
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train_filtered, y_train, cv=5, scoring='accuracy', n_jobs=-1)


    # Plot learning curve for accuracy
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Accuracy')
    plt.plot(train_sizes, test_mean, label='Cross-validation Accuracy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve - Accuracy')
    plt.legend()
    plt.show()

    # curves for precision, recall, and F1-score
    precision_scores = []
    recall_scores = []
    accs = []
    f1_scores = []
    
    train_sizes = [300,500,700,900,1100,1300]

    for i in train_sizes:
        
        train_data, train_labels = load_reviews('/Users/vardisgeorgilas/Desktop/aclImdb/train', i)
        test_data, test_labels = load_reviews('/Users/vardisgeorgilas/Desktop/aclImdb/test', i)
        
        X_train_filtered, X_test_filtered = vectorize_data(train_data, test_data, m, n=n, k=k)

        y_train = np.array(train_labels)
        y_test = np.array(test_labels)
        
        clf = AdaBoostClassifier(base_classifier, n_estimators=50, random_state=42)
        clf.fit(X_train_filtered, y_train)
        y_pred = clf.predict(X_test_filtered)
      

        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        accs.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

    # Plot precision, recall, and F1-score curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, accs, label='Accuracy')
    plt.plot(train_sizes, precision_scores, label='Precision')
    plt.plot(train_sizes, recall_scores, label='Recall')
    plt.plot(train_sizes, f1_scores, label='F1-score')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title('Learning Curves - Precision, Recall, F1-score')
    plt.legend()
    plt.show()
    
