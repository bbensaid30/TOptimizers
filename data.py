import os
os.chdir("/home/bbensaid/Documents/Anabase/NN")

from keras import datasets
from keras.utils import to_categorical
from keras import preprocessing

import numpy as np
import tensorflow as tf
import pandas as pd
from keras import backend as K

import zipfile
import requests
import io
from PIL import Image

# tensor processing
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from scipy.io import loadmat


def Runge(N_train,N_test,typef="float32"):
    x_train = np.linspace(-1,1,num=N_train).reshape(N_train,1)
    y_train = 1/(1+25*x_train**2)

    x_test = np.linspace(-1,1,num=N_test).reshape(N_test,1)
    y_test = 1/(1+25*x_test**2)

    if(typef=='float64'):
        return tf.convert_to_tensor(x_train,dtype=tf.float64), tf.convert_to_tensor(y_train,dtype=tf.float64), tf.convert_to_tensor(x_test,dtype=tf.float64), tf.convert_to_tensor(y_test,dtype=tf.float64)
    else:
        return tf.convert_to_tensor(x_train,dtype=tf.float32), tf.convert_to_tensor(y_train,dtype=tf.float32), tf.convert_to_tensor(x_test,dtype=tf.float32), tf.convert_to_tensor(y_test,dtype=tf.float32)


def MNIST_flatten(typef="float32"):
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    num_pixels = x_train.shape[1] * x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], num_pixels)
    x_test = x_test.reshape(x_test.shape[0], num_pixels)

    x_train = x_train.astype(typef)
    x_test = x_test.astype(typef)
    x_train = x_train/255
    x_test = x_test/255

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    if(typef=='float64'):
        return tf.convert_to_tensor(x_train,dtype=tf.float64), tf.convert_to_tensor(y_train,dtype=tf.float64), tf.convert_to_tensor(x_test,dtype=tf.float64), tf.convert_to_tensor(y_test,dtype=tf.float64)
    else:
        return tf.convert_to_tensor(x_train,dtype=tf.float32), tf.convert_to_tensor(y_train,dtype=tf.float32), tf.convert_to_tensor(x_test,dtype=tf.float32), tf.convert_to_tensor(y_test,dtype=tf.float32)

def MNIST(typef="float32"):
    ((trainData, trainLabels), (testData, testLabels)) = datasets.mnist.load_data()

    trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
    testData = testData.reshape((testData.shape[0], 28, 28, 1))

    trainData = trainData.astype(typef)/255.0
    testData = testData.astype(typef)/255.0

    trainLabels = to_categorical(trainLabels, 10)
    testLabels = to_categorical(testLabels, 10)

    return trainData,trainLabels,testData,testLabels

#balanced: 112800 vs 18800 (-1 in the csv file)
def EMNIST(typef="float32", flatten=False):
    train = pd.read_csv("Data/emnist-balanced-train.csv",delimiter = ',')
    test = pd.read_csv("Data/emnist-balanced-test.csv", delimiter = ',')
    mapp = pd.read_csv("Data/emnist-balanced-mapping.txt", delimiter = ' ', \
                   index_col=0, header=None).squeeze("columns")
    
    train_x = train.iloc[:,1:]
    train_y = train.iloc[:,0]
    del train
    test_x = test.iloc[:,1:]
    test_y = test.iloc[:,0]
    del test

    HEIGHT = 28
    WIDTH = 28

    def rotate(image):
        image = image.reshape([HEIGHT, WIDTH])
        image = np.fliplr(image)
        image = np.rot90(image)
        return image
    
    train_x = np.asarray(train_x)
    train_x = np.apply_along_axis(rotate, 1, train_x)
    test_x = np.asarray(test_x)
    test_x = np.apply_along_axis(rotate, 1, test_x)

    train_x = train_x.astype(typef)
    train_x /= 255
    test_x = test_x.astype(typef)
    test_x /= 255

    num_classes = train_y.nunique()
    train_y = to_categorical(train_y, num_classes)
    test_y = to_categorical(test_y, num_classes)

    if(flatten):
        train_x = train_x.reshape(-1, HEIGHT*WIDTH)
        test_x = test_x.reshape(-1, HEIGHT*WIDTH)
    else:
        train_x = train_x.reshape(-1, HEIGHT, WIDTH)
        test_x = test_x.reshape(-1, HEIGHT, WIDTH)

    return train_x, train_y, test_x, test_y

def FASHION_MNIST_flatten(typef="float32"):
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

    num_pixels = x_train.shape[1] * x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], num_pixels)
    x_test = x_test.reshape(x_test.shape[0], num_pixels)

    x_train = x_train.astype(typef)
    x_test = x_test.astype(typef)
    x_train = x_train/255
    x_test = x_test/255

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    if(typef=='float64'):
        return tf.convert_to_tensor(x_train,dtype=tf.float64), tf.convert_to_tensor(y_train,dtype=tf.float64), tf.convert_to_tensor(x_test,dtype=tf.float64), tf.convert_to_tensor(y_test,dtype=tf.float64)
    else:
        return tf.convert_to_tensor(x_train,dtype=tf.float32), tf.convert_to_tensor(y_train,dtype=tf.float32), tf.convert_to_tensor(x_test,dtype=tf.float32), tf.convert_to_tensor(y_test,dtype=tf.float32)

def FASHION_MNIST(typef="float32"):
    ((trainData, trainLabels), (testData, testLabels)) = datasets.mnist.load_data()

    trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
    testData = testData.reshape((testData.shape[0], 28, 28, 1))

    trainData = trainData.astype(typef)/255.0
    testData = testData.astype(typef)/255.0

    trainLabels = to_categorical(trainLabels, 10)
    testLabels = to_categorical(testLabels, 10)

    return trainData,trainLabels,testData,testLabels

def CIFAR10(typef="float32", flatten=False):
    ((trainData, trainLabels), (testData, testLabels)) = datasets.cifar10.load_data()

    #trainData = tf.image.rgb_to_grayscale(trainData).numpy()
    #testData = tf.image.rgb_to_grayscale(testData).numpy()

    trainData = trainData.reshape((trainData.shape[0], 32, 32, 3))
    testData = testData.reshape((testData.shape[0], 32, 32, 3))

    trainData = trainData.astype(typef)/255.0
    testData = testData.astype(typef)/255.0

    trainLabels = to_categorical(trainLabels, 10)
    testLabels = to_categorical(testLabels, 10)

    return trainData,trainLabels,testData,testLabels

#27000 total
#80/20: 21600/5400
#50/50: 13500/13500
def EUROSAT_RGB(typef="float32"):
    url = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"

    # download zip
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    # get file names
    txtfiles = []
    for file in z.namelist():
        txtfiles.append(file)
    # keep only those containing ".jpg"
    txtfiles = [x for x in txtfiles if ".jpg" in x]

    # read images to numpy array
    XImages = np.zeros([len(txtfiles), 64, 64, 3])
    i = 0
    for pic in txtfiles:
        XImages[i] = np.asarray(Image.open(z.open(pic))).astype('uint8')/255
        i += 1

    del r # clear memory
    del z 

    # Get labels in numpy array as strings
    labs = np.empty(len(txtfiles), dtype = 'S20')
    i = 0
    for label in txtfiles:
        labs[i] = label.split('/')[1]
        i += 1
    
    # change them to integers in alphabetical order
    label_names, yLabels = np.unique(labs, return_inverse=True)
    label_Dict = dict(zip(np.unique(yLabels), label_names))
    np.array(np.unique(yLabels, return_counts=True)).T

    # find the smallest class
    smallest_class = np.argmin(np.bincount(yLabels))

    # number of classes
    num_classes = len(np.array(np.unique(yLabels)))
    # observations in smallest class
    smallest_class_obs = np.where(yLabels == smallest_class)[0]
    # Get 2000 observations from each class
    indBal = np.empty(0, dtype=int)
    for i in range(num_classes):
        indTemp = shuffle(np.where(yLabels == i)[0], random_state=42)[0:smallest_class_obs.shape[0]]
        indBal = np.concatenate([indBal, indTemp])
    # shuffle the balanced index
    indBal = shuffle(indBal, random_state = 42)

    # first line uses balanced labels
    # second line uses original imbalanced labels
    # X_train, X_test, y_train, y_test = train_test_split(XBal, yBal, stratify = yBal, train_size = 0.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(XImages, yLabels, stratify = yLabels, train_size = 0.5, random_state=42)

    #np.array(np.unique(y_train, return_counts=True)).T

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return X_train, y_train, X_test, y_test

#73257/26032
def SVHN(typef="float32", flatten=False):
    # Load the data
    train_raw = loadmat('/home/bbensaid/Documents/Anabase/NN/Data/train_32x32.mat')
    test_raw = loadmat('/home/bbensaid/Documents/Anabase/NN/Data/test_32x32.mat')

    # Load images and labels
    train_images = np.array(train_raw['X'])
    test_images = np.array(test_raw['X'])
    train_labels = train_raw['y']
    test_labels = test_raw['y']

    # Fix the axes of the images
    train_images = np.moveaxis(train_images, -1, 0)
    test_images = np.moveaxis(test_images, -1, 0)

    train_images = train_images.astype(typef)/255.0
    test_images = test_images.astype(typef)/255.0
    train_labels = train_labels.astype(typef)
    test_labels = test_labels.astype(typef)

    if(flatten):
        train_images = train_images.reshape(train_images.shape[0],3072)
        test_images = test_images.reshape(test_images.shape[0],3072)

    # One-hot encoding of train and test labels
    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.fit_transform(test_labels)

    return train_images, train_labels, test_images, test_labels


def load(f):
    return np.load(f)['arr_0']

#60000/10000
def KMNIST(typef):
    img_rows, img_cols = 28, 28

    x_train = load('Data/kmnist-train-imgs.npz')
    x_test = load('Data/kmnist-test-imgs.npz')
    y_train = load('Data/kmnist-train-labels.npz')
    y_test = load('Data/kmnist-test-labels.npz')

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    x_train = x_train.astype(typef)
    x_test = x_test.astype(typef)
    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test

#232365/38547
def K49(typef):
    img_rows, img_cols = 28, 28

    x_train = load('Data/k49-train-imgs.npz')
    x_test = load('Data/k49-test-imgs.npz')
    y_train = load('Data/k49-train-labels.npz')
    y_test = load('Data/k49-test-labels.npz')

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    y_train = to_categorical(y_train, 49)
    y_test = to_categorical(y_test, 49)

    x_train = x_train.astype(typef)
    x_test = x_test.astype(typef)
    x_train /= 255
    x_test /= 255

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

    #nb_data = 500000
    #x_train = x_train[0:nb_data]; y_train = y_train[0:nb_data]

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