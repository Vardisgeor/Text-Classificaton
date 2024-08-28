import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from IPython.display import Image
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras import backend as K



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

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_score(y_true, y_pred):
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2 * ((precision_val * recall_val) / (precision_val + recall_val + K.epsilon()))


if __name__ == "__main__":
    
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data()
    
    word_index = tf.keras.datasets.imdb.get_word_index()
    index2word = dict((i + 3, word) for (word, i) in word_index.items())
    index2word[0] = '[pad]'
    index2word[1] = '[bos]'
    index2word[2] = '[oov]'
    train_data = np.array([' '.join([index2word[idx] for idx in text]) for text in train_data])
    test_data = np.array([' '.join([index2word[idx] for idx in text]) for text in test_data])
    
    binary_vectorizer = CountVectorizer(binary=True,min_df=100)
    x_train_b = binary_vectorizer.fit_transform(train_data)
    x_test_b = binary_vectorizer.transform(test_data)


    vocab_size = len(binary_vectorizer.vocabulary_) 

    
    x_train_b = x_train_b.toarray()
    x_test_b = x_test_b.toarray()
    
    imdb_mlp = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(vocab_size,)), 
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(units=1, activation='sigmoid') 
    ])


    imdb_mlp.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['binary_accuracy', precision, recall, f1_score])

    
    imdb_mlp_history = imdb_mlp.fit(x=x_train_b, y=np.array(train_labels),
                                    epochs=20, verbose=1, batch_size=32,
                                    validation_data=(x_test_b, np.array(test_labels)))

    print(imdb_mlp.evaluate(x_test_b, np.array(test_labels)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(imdb_mlp_history.history['binary_accuracy'], label='Training Accuracy')
    plt.plot(imdb_mlp_history.history['val_binary_accuracy'], label='Validation Accuracy')
    plt.title('IMDB Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(imdb_mlp_history.history['precision'], label='Training Precision')
    plt.plot(imdb_mlp_history.history['val_precision'], label='Validation Precision')
    plt.plot(imdb_mlp_history.history['recall'], label='Training Recall')
    plt.plot(imdb_mlp_history.history['val_recall'], label='Validation Recall')
    plt.plot(imdb_mlp_history.history['f1_score'], label='Training F1 Score')
    plt.plot(imdb_mlp_history.history['val_f1_score'], label='Validation F1 Score')
    plt.title('IMDB Model Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()

    plt.show()
    
    
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()
    x_train_mnist = x_train_mnist.astype("float32") / 255
    x_test_mnist = x_test_mnist.astype("float32") / 255
    
    def get_mnist_mlp(num_classes): 
        inp = tf.keras.layers.Input(shape=(28, 28), name='inputs')
        x = tf.keras.layers.Flatten(name='flatten_inputs')(inp) 
        x = tf.keras.layers.Dense(units=512, activation='relu', name='hidden_1')(x)
        x = tf.keras.layers.Dense(units=256, activation='relu', name='hidden_2')(x)
        x = tf.keras.layers.Dense(units=num_classes, activation='softmax',
                                    name='classifier')(x)
        return tf.keras.models.Model(inputs=inp, outputs=x, name='mnist_mlp')

    mnist_mlp = get_mnist_mlp(len(np.unique(y_train_mnist)))

    mnist_mlp.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy'], optimizer=tf.keras.optimizers.Adam())
    y_train_mnist = tf.keras.utils.to_categorical(y_train_mnist, num_classes=10,
                                                dtype="int32")
    y_test_mnist = tf.keras.utils.to_categorical(y_test_mnist, num_classes=10,
                                                dtype="int32")
    y_train_mnist[0]
    
    mnist_mlp_history = mnist_mlp.fit(x_train_mnist, y_train_mnist, batch_size=32,
                                  epochs=10, validation_split=0.2)
    
    def plot(his, kind):
        train = his.history[kind]
        val = his.history['val_' + kind]
        epochs = range(1, len(train)+1)
        plt.figure(figsize=(12,9))
        plt.plot(epochs, train, 'b', label='Training ' + kind)
        plt.plot(epochs, val, 'orange', label='Validation ' + kind)
        plt.title('Training and validation ' + kind)
        plt.xlabel('Epochs')
        plt.ylabel(kind)
        plt.legend()
        plt.show()
        
    plot(mnist_mlp_history, 'loss')
