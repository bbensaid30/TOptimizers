import numpy as np
import pandas as pd

#Tensorflow: un individu par ligne

#fichiers avec header obligatoire
def read_data(x_train_file, y_train_file, x_test_file, y_test_file):
    x_train = pd.read_csv(x_train_file,header=0,float_precision='round_trip')
    y_train = pd.read_csv(y_train_file,header=0,float_precision='round_trip')
    x_test = pd.read_csv(x_test_file,header=0,float_precision='round_trip')
    y_test = pd.read_csv(y_test_file,header=0,float_precision='round_trip')
    return x_train,y_train,x_test,y_test

def poly_data(typef):
    x = np.array([[0.0],[1.0]])
    y = np.array([[0.0],[0.0]])
    return x.astype(typef),y.astype(typef)

def ex1(typef):
    x = np.array([[1.0],[2.0]])
    y = np.array([[0.0],[1.0]])
    return x.astype(typef),y.astype(typef)

def ex5(typef):
    x = np.array([[1.0],[2.0],[2.0]])
    y = np.array([[0.0],[1.0],[-3.0]])
    return x.astype(typef),y.astype(typef)