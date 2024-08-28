import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def read_text_files(folder_path, n=100):
    files = glob(os.path.join(folder_path, '*.txt'))
    files = files[:n]
    texts = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            texts.append(text)
    return texts


class Node:
    def __init__(self, checking_feature=None, is_leaf=False, category=None):
        self.checking_feature = checking_feature
        self.left_child = None
        self.right_child = None
        self.is_leaf = is_leaf
        self.category = category
        

class ID3:
    def __init__(self, features, max_depth=None):
        self.tree = None
        self.features = features
        self.max_depth = max_depth

    def fit(self, x, y, depth=10 ):
        most_common = np.argmax(np.bincount(np.array(y).flatten()))
        self.tree = self.create_tree(x, y, features=np.arange(len(self.features)), category=most_common, depth=depth)
        return self.tree

    def create_tree(self, x_train, y_train, features, category, depth):

        if x_train.shape[0] == 0:
            return Node(checking_feature=None, is_leaf=True, category=category)  

        if np.all(np.array(y_train).flatten() == 0):
            return Node(checking_feature=None, is_leaf=True, category=0)
        elif np.all(np.array(y_train).flatten() == 1):
            return Node(checking_feature=None, is_leaf=True, category=1)

        if len(features) == 0 or (self.max_depth is not None and depth == self.max_depth):
            return Node(checking_feature=None, is_leaf=True, category=np.argmax(np.bincount(y_train.flatten())))

        igs = list()
        for feat_index in features.flatten():
            igs.append(self.calculate_ig(np.array(y_train).flatten(), [example[feat_index] for example in x_train]))

        max_ig_idx = np.argmax(np.array(igs).flatten())
        m = np.argmax(np.bincount(np.array(y_train).flatten())) 

        root = Node(checking_feature=max_ig_idx)

        x_train_0 = x_train[x_train[:, max_ig_idx] == 0, :]
        y_train_0 = y_train[x_train[:, max_ig_idx] == 0].flatten()

        x_train_1 = x_train[x_train[:, max_ig_idx] == 1, :]
        y_train_1 = y_train[x_train[:, max_ig_idx] == 1].flatten()

        new_features_indices = np.delete(features.flatten(), max_ig_idx)  # remove the current feature

        root.left_child = self.create_tree(x_train=x_train_1, y_train=y_train_1, features=new_features_indices,
                                           category=m, depth=depth + 1) 

        root.right_child = self.create_tree(x_train=x_train_0, y_train=y_train_0, features=new_features_indices,
                                            category=m, depth=depth + 1) 

        return root


    @staticmethod
    def calculate_ig(classes_vector, feature):
        classes = set(classes_vector)

        HC = 0
        for c in classes:
            PC = np.sum(classes_vector == c) / len(classes_vector)  
            HC += - PC * np.log2(PC) if PC != 0 else 0  

        if isinstance(feature, list):
            feature_values = set(feature)
        else:  
            feature_values = set(feature.data)

        HC_feature = 0

        for value in feature_values:
            
            if isinstance(feature, list):
                pf = feature.count(value) / len(feature)  # count occurrences of value
                indices = [i for i in range(len(feature)) if feature[i] == value]  
            else:  
                pf = np.sum(feature.data == value) / len(feature) 
                indices = np.where(feature.data == value)[0]  

            classes_of_feat = [classes_vector[i] for i in indices]  
            for c in classes:
                
                pcf = np.sum(classes_of_feat == c) / len(classes_of_feat) if len(classes_of_feat) != 0 else 0  # given X=x, count C
                if pcf != 0:

                    temp_H = - pf * pcf * np.log2(pcf)
                    HC_feature += temp_H

        ig = HC - HC_feature
        return ig
    
    def predict(self, x):
        predicted_classes = list()

        for unlabeled in x: 
            tmp = self.tree 
            while not tmp.is_leaf:
                if unlabeled.flatten()[tmp.checking_feature] == 1:
                    tmp = tmp.left_child
                else:
                    tmp = tmp.right_child
            
            predicted_classes.append(tmp.category)
        return np.array(predicted_classes)
    

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        for _ in range(self.n_estimators):

            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_subset, y_subset = X[indices, :].toarray(), np.asarray(y)[indices].ravel()

            id3_tree = ID3(features=np.arange(X_subset.shape[1]), max_depth=self.max_depth)
            id3_tree.fit(X_subset, y_subset, depth=10) 

            self.trees.append(id3_tree)

    def predict(self, X):
       
        predictions = np.array([tree.predict(X.toarray()) for tree in self.trees])
        np.set_printoptions(threshold=np.inf)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
    
    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,          
            'random_state': self.random_state
        }

# Hyperparameters
m = 1000  # Number of most frequent words to include #1000
n = 400 # Number of most frequent words to exclude
k = 400  # Number of most rare words to exclude

train_folder = 'train'
neg_folder = 'neg'
pos_folder = 'pos'

neg_texts = read_text_files(os.path.join('/Users/vardisgeorgilas/Desktop/aclImdb/train', neg_folder), 600)
pos_texts = read_text_files(os.path.join('/Users/vardisgeorgilas/Desktop/aclImdb/train', pos_folder), 600) 
neg_texts_test = read_text_files(os.path.join('/Users/vardisgeorgilas/Desktop/aclImdb/test', neg_folder), 600)
pos_texts_test = read_text_files(os.path.join('/Users/vardisgeorgilas/Desktop/aclImdb/test', pos_folder), 600) 


# Combine positive and negative texts
all_texts = neg_texts + pos_texts
all_labels = [0] * len(neg_texts) + [1] * len(pos_texts)
all_texts_test = neg_texts_test + pos_texts_test
all_labels_test = [0] * len(neg_texts_test) + [1] * len(pos_texts_test)

vectorizer = CountVectorizer(max_features=m)
all_texts = vectorizer.fit_transform(all_texts)
all_texts_test = vectorizer.transform(all_texts_test)

feature_selector = SelectKBest(chi2, k=m - n - k)
all_texts = feature_selector.fit_transform(all_texts, all_labels)
all_texts_test = feature_selector.transform(all_texts_test)

all_texts, all_labels = shuffle(all_texts, all_labels, random_state=42)
all_texts_test, all_labels_test = shuffle(all_texts_test, all_labels_test, random_state=42)


random_forest = RandomForest(n_estimators= 25, max_depth=5, random_state=42)
random_forest.fit(all_texts, all_labels)

y_pred = random_forest.predict(all_texts_test)
accuracy = accuracy_score(all_labels_test, y_pred)
 
precision = precision_score(all_labels_test, y_pred)
recall = recall_score(all_labels_test, y_pred)
f1 = f1_score(all_labels_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")

# learning curve

train_sizes, train_scores, test_scores = learning_curve(random_forest, all_texts, all_labels, cv=5, scoring='accuracy', n_jobs=-1)

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


# curves for precision, recall, and F1-score

# precision_scores = []
# recall_scores = []
# accs = []
# f1_scores = []

# train_sizes = [100,200,300,500,700,900]

# for i in train_sizes:
    
#     neg_texts = read_text_files(os.path.join('/Users/vardisgeorgilas/Desktop/aclImdb/train', neg_folder), i)
#     pos_texts = read_text_files(os.path.join('/Users/vardisgeorgilas/Desktop/aclImdb/train', pos_folder), i) 
#     neg_texts_test = read_text_files(os.path.join('/Users/vardisgeorgilas/Desktop/aclImdb/test', neg_folder), i)
#     pos_texts_test = read_text_files(os.path.join('/Users/vardisgeorgilas/Desktop/aclImdb/test', pos_folder), i) 

#     # Combine positive and negative texts
#     all_texts = neg_texts + pos_texts
#     all_labels = [0] * len(neg_texts) + [1] * len(pos_texts)
#     all_texts_test = neg_texts_test + pos_texts_test
#     all_labels_test = [0] * len(neg_texts_test) + [1] * len(pos_texts_test)

#     vectorizer = CountVectorizer(max_features=m)
#     all_texts = vectorizer.fit_transform(all_texts)
#     all_texts_test = vectorizer.transform(all_texts_test)

#     feature_selector = SelectKBest(chi2, k=m - n - k)
#     all_texts = feature_selector.fit_transform(all_texts, all_labels)
#     all_texts_test = feature_selector.transform(all_texts_test)

#     all_texts, all_labels = shuffle(all_texts, all_labels, random_state=42)
#     all_texts_test, all_labels_test = shuffle(all_texts_test, all_labels_test, random_state=42)


#     random_forest = RandomForest(n_estimators= 10, max_depth=5, random_state=42)
#     random_forest.fit(all_texts, all_labels)

#     y_pred = random_forest.predict(all_texts_test)
    

#     precision_scores.append(precision_score(all_labels_test, y_pred))
#     recall_scores.append(recall_score(all_labels_test, y_pred))
#     accs.append(accuracy_score(all_labels_test, y_pred))
#     f1_scores.append(f1_score(all_labels_test, y_pred))

# # Plot precision, recall, and F1-score curves
# plt.figure(figsize=(10, 6))
# plt.plot(train_sizes, accs, label='Accuracy')
# plt.plot(train_sizes, precision_scores, label='Precision')
# plt.plot(train_sizes, recall_scores, label='Recall')
# plt.plot(train_sizes, f1_scores, label='F1-score')
# plt.xlabel('Training Examples')
# plt.ylabel('Score')
# plt.title('Learning Curves - Precision, Recall, F1-score')
# plt.legend()
# plt.show()

