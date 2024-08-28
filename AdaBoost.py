import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Adaboost:
    def __init__(self, M, n, k):
        self.M = M
        self.alphas = []
        self.H = []
        self.vectorizer = None
        self.n = n
        self.k = k

    def get_params(self, deep=True):
        return {'M': self.M, 'n': self.n, 'k': self.k}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def fit(self, X, y):
        self.alphas = []
        w = np.ones(len(y)) / len(y)
        self.vectorizer = CountVectorizer(max_features=7000, stop_words='english')
        X = self.vectorizer.fit_transform(X)

        for m in range(self.M):
            l = DecisionTreeClassifier(max_depth=1)
            l.fit(X, y, sample_weight=w)
            pred = l.predict(X)
            
            self.H.append(l)
            
            error = np.sum(w * (pred != y))
            
            if error >= 0.5:
                break
    
            for j in range(1, len(y)):
                if pred[j] == y[j]:
                    w[j] *= error / (1 - error)
                    
            w = [float(i) / sum(w) for i in w]  # normalize weights
            
            epsilon = 1e-10  # avoid dividing by zero
            z = 0.5 * np.log((1 - error + epsilon) / (error + epsilon))
            self.alphas.append(z)

        # exclude words
        exclude = np.argsort(self.vectorizer.vocabulary_.values())[:self.n] + \
                                     np.argsort(self.vectorizer.vocabulary_.values())[-self.k:]
        X = np.delete(X.toarray(), exclude, axis=1)
        self.vectorizer.vocabulary_ = {word: idx for idx, word in enumerate(self.vectorizer.get_feature_names_out())}

    def predict(self, X):
        X = self.vectorizer.transform(X)
        preds = np.zeros((X.shape[0], self.M))
        for m in range(self.M):
            y_pred = self.H[m].predict(X) * self.alphas[m]
            preds[:, m] = y_pred

        p = np.sign(preds.sum(axis=1)).astype(int)
        return p

def load_reviewss(directory, label, n):
    reviews = []
    labels = []
    filenames = os.listdir(os.path.join(directory, label))[:n]
    for filename in filenames:
        with open(os.path.join(directory, label, filename), 'r', encoding='utf-8') as file:
            review_text = file.read()
            reviews.append(review_text)
            labels.append(1 if label == 'pos' else -1)
    return reviews, np.array(labels, dtype=int) 

if __name__ == "__main__":
    
    M = 50
    n_texts = 1000  # number of texts to load
    n = 10  # number of most common words to exclude
    k = 10  # number of least common words to exclude

    train_pos_reviews, train_pos_labels = load_reviewss('/Users/vardisgeorgilas/Desktop/aclImdb/train', 'pos', n_texts)
    train_neg_reviews, train_neg_labels = load_reviewss('/Users/vardisgeorgilas/Desktop/aclImdb/train', 'neg', n_texts)
    test_pos_reviews, test_pos_labels = load_reviewss('/Users/vardisgeorgilas/Desktop/aclImdb/test', 'pos', n_texts)
    test_neg_reviews, test_neg_labels = load_reviewss('/Users/vardisgeorgilas/Desktop/aclImdb/test', 'neg', n_texts)

    X_train = train_pos_reviews + train_neg_reviews
    X_test = test_pos_reviews + test_neg_reviews

    y_train = np.concatenate([train_pos_labels, train_neg_labels])
    y_test = np.concatenate([test_pos_labels, test_neg_labels])

    clf = Adaboost(M, n, k)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    print("Accuracy:", acc)

    ############################################### Diagrams #############################################

    
    #learning curve
    
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Accuracy')
    plt.plot(train_sizes, test_mean, label='Validation Accuracy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()
    
    #accuracy, recall, precision and f1 curves
    
    precisions = []
    accs = []
    recalls = []
    f1s = []
    data = [500,700,900,1100,1300,1500]
    for i in data:
        train_pos_reviews, train_pos_labels = load_reviewss('/Users/vardisgeorgilas/Desktop/aclImdb/train', 'pos', i)
        train_neg_reviews, train_neg_labels = load_reviewss('/Users/vardisgeorgilas/Desktop/aclImdb/train', 'neg', i)
        test_pos_reviews, test_pos_labels = load_reviewss('/Users/vardisgeorgilas/Desktop/aclImdb/test', 'pos', i)
        test_neg_reviews, test_neg_labels = load_reviewss('/Users/vardisgeorgilas/Desktop/aclImdb/test', 'neg', i)
        
        X_train = train_pos_reviews + train_neg_reviews
        X_test = test_pos_reviews + test_neg_reviews

        y_train = np.concatenate([train_pos_labels, train_neg_labels])
        y_test = np.concatenate([test_pos_labels, test_neg_labels])
        
        clf = Adaboost(M,n,k)
        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)

        acc = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        precisions.append(precision)
        accs.append(acc)
        recalls.append(recall)
        f1s.append(f1)
    
    plt.plot(data, accs, label='Accuracy')
    plt.plot(data, precisions, label='Precision')
    plt.plot(data, recalls, label='Recall')
    plt.plot(data, f1s, label='F1 Score')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend()
    plt.show()
        
    
    
    