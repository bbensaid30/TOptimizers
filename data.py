import os
os.chdir("/home/bbensaid/Documents/Anabase/NN_shaman")

from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import preprocessing
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from utils import convert_sparse_matrix_to_sparse_tensor, vectorize_sequences

def Runge(type,N_train,N_test):
    x_train = np.linspace(-1,1,num=N_train).reshape(N_train,1)
    y_train = 1/(1+25*x_train**2)

    x_test = np.linspace(-1,1,num=N_test).reshape(N_test,1)
    y_test = 1/(1+25*x_test**2)

    if(type=='float64'):
        return tf.convert_to_tensor(x_train,dtype=tf.float64), tf.convert_to_tensor(y_train,dtype=tf.float64), tf.convert_to_tensor(x_test,dtype=tf.float64), tf.convert_to_tensor(y_test,dtype=tf.float64)
    else:
        return tf.convert_to_tensor(x_train,dtype=tf.float32), tf.convert_to_tensor(y_train,dtype=tf.float32), tf.convert_to_tensor(x_test,dtype=tf.float32), tf.convert_to_tensor(y_test,dtype=tf.float32)


def MNIST_flatten(type):
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    num_pixels = x_train.shape[1] * x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], num_pixels)
    x_test = x_test.reshape(x_test.shape[0], num_pixels)

    x_train = x_train.astype(type)
    x_test = x_test.astype(type)
    x_train = x_train/255
    x_test = x_test/255

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    if(type=='float64'):
        return tf.convert_to_tensor(x_train,dtype=tf.float64), tf.convert_to_tensor(y_train,dtype=tf.float64), tf.convert_to_tensor(x_test,dtype=tf.float64), tf.convert_to_tensor(y_test,dtype=tf.float64)
    else:
        return tf.convert_to_tensor(x_train,dtype=tf.float32), tf.convert_to_tensor(y_train,dtype=tf.float32), tf.convert_to_tensor(x_test,dtype=tf.float32), tf.convert_to_tensor(y_test,dtype=tf.float32)

def MNIST(type):
    ((trainData, trainLabels), (testData, testLabels)) = datasets.mnist.load_data()

    trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
    testData = testData.reshape((testData.shape[0], 28, 28, 1))

    trainData = trainData.astype(type)/255.0
    testData = testData.astype(type)/255.0

    trainLabels = to_categorical(trainLabels, 10)
    testLabels = to_categorical(testLabels, 10)

    return trainData,trainLabels,testData,testLabels

def FASHION_MNIST_flatten(type):
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

    num_pixels = x_train.shape[1] * x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], num_pixels)
    x_test = x_test.reshape(x_test.shape[0], num_pixels)

    x_train = x_train.astype(type)
    x_test = x_test.astype(type)
    x_train = x_train/255
    x_test = x_test/255

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    if(type=='float64'):
        return tf.convert_to_tensor(x_train,dtype=tf.float64), tf.convert_to_tensor(y_train,dtype=tf.float64), tf.convert_to_tensor(x_test,dtype=tf.float64), tf.convert_to_tensor(y_test,dtype=tf.float64)
    else:
        return tf.convert_to_tensor(x_train,dtype=tf.float32), tf.convert_to_tensor(y_train,dtype=tf.float32), tf.convert_to_tensor(x_test,dtype=tf.float32), tf.convert_to_tensor(y_test,dtype=tf.float32)

def FASHION_MNIST(type):
    ((trainData, trainLabels), (testData, testLabels)) = datasets.mnist.load_data()

    trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
    testData = testData.reshape((testData.shape[0], 28, 28, 1))

    trainData = trainData.astype(type)/255.0
    testData = testData.astype(type)/255.0

    trainLabels = to_categorical(trainLabels, 10)
    testLabels = to_categorical(testLabels, 10)

    return trainData,trainLabels,testData,testLabels

def IMDB_onehot():
    dimension=10000
    (train_data, train_labels), (test_data, test_labels) = datasets.imdb.load_data(num_words=dimension)

    # Our vectorized training data
    x_train = vectorize_sequences(train_data,dimension)
    # Our vectorized test data
    x_test = vectorize_sequences(test_data,dimension)

    # Our vectorized labels
    y_train = np.asarray(train_labels).astype('uint8'); y_train = y_train.reshape(-1,1)
    y_test = np.asarray(test_labels).astype('uint8'); y_test = y_test.reshape(-1,1)

    return x_train, y_train, x_test, y_test

def IMDB_embedding():
    max_features = 5000  # Only consider the top 10k words
    maxlen = 500  # Only consider the first 100 words of each movie review

    (x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=max_features)
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

    y_train = np.asarray(y_train).astype('uint8'); y_train = y_train.reshape(-1,1)
    y_test = np.asarray(y_test).astype('uint8'); y_test = y_test.reshape(-1,1)

    return x_train, y_train, x_test, y_test

def IMDB_IDF():
    dimension=3000

    word_to_index = datasets.imdb.get_word_index()
    index_to_word = [None] * (max(word_to_index.values()) + 1)
    for w, i in word_to_index.items():
        index_to_word[i] = w

    (X_train, y_train), (X_test, y_test) = datasets.imdb.load_data(num_words=dimension)
    X_train = [' '.join(index_to_word[i] for i in X_train[i] if i < len(index_to_word)) for i in range(X_train.shape[0])]
    X_test = [' '.join(index_to_word[i] for i in X_test[i] if i < len(index_to_word)) for i in range(X_test.shape[0])]

    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    x_train = vectorizer.fit_transform(X_train); x_train = x_train.astype('float32')
    x_test = vectorizer.fit_transform(X_test); x_test = x_test.astype('float32')
    x_train, x_test = convert_sparse_matrix_to_sparse_tensor(x_train, x_test)

    y_train = y_train.astype('uint8'); y_train = y_train.reshape(-1,1)
    y_test = y_test.astype('uint8'); y_test = y_test.reshape(-1,1)

    return x_train, y_train, x_test, y_test

def HIGGS():
    resolution="low"
    x_train = pd.read_hdf('Data/higgs_train_'+resolution+'.h5')
    x_test = pd.read_hdf('Data/higgs_test_'+resolution+'.h5')
    y_train = pd.read_hdf('Data/higgs_train_output.h5', key='y_train')
    y_test = pd.read_hdf('Data/higgs_test_output.h5', key='y_test')

    x_train.reset_index(inplace=True); x_train.index
    x_test.reset_index(inplace=True); x_test.index
    y_train.reset_index(inplace=True); y_train.index
    y_test.reset_index(inplace=True); y_test.index

    x_train = x_train.drop(['index'], axis=1); x_test = x_test.drop(['index'], axis=1)
    y_train = y_train.drop(['index'], axis=1); y_test = y_test.drop(['index'], axis=1)

    nb_data = 500000
    x_train = x_train[0:nb_data]; y_train = y_train[0:nb_data]

    return tf.convert_to_tensor(x_train), tf.convert_to_tensor(y_train,dtype=tf.float32), tf.convert_to_tensor(x_test), tf.convert_to_tensor(y_test,dtype=tf.float32)

def Speed():
    x_train = pd.read_csv('Data/speed_inputs_train.csv', header=None)
    x_test = pd.read_csv('Data/speed_inputs_test.csv', header=None)
    y_train = pd.read_csv('Data/speed_outputs_train.csv', header=None)
    #y_train = pd.read_csv('Data/residuals_v1.csv', header=None)
    y_test = pd.read_csv('Data/speed_outputs_test.csv', header=None)

    #nb_data = 1000
    #x_train = x_train[0:nb_data]; y_train = y_train[0:nb_data]

    return tf.convert_to_tensor(x_train, dtype=tf.float32), tf.convert_to_tensor(y_train,dtype=tf.float32), tf.convert_to_tensor(x_test, dtype=tf.float32), tf.convert_to_tensor(y_test,dtype=tf.float32)