import os
os.chdir("/home/bbensaid/Documents/Anabase/NN_shaman/Data") 

import seaborn as sns
from sklearn import preprocessing

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.api.types import is_string_dtype 
from pandas.api.types import is_numeric_dtype
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer

import tensorflow as tf

#------------------------------------------ Basic data reading and computing  ----------------------------------------------------------------------------

def minPhysics(numbers=["01","03","05","06"]):
    minsPhysics=[[],[],[],[],[]]
    minimums=[]
    for number in numbers:
        folder="dh"+number
        file=folder+".csv"
        physics=pd.read_csv(file,header=0,float_precision='round_trip')
        i=0
        for col in physics.columns:
            minsPhysics[i].append(np.min(physics[col]))
            i+=1
    for i in range(5):
        minimums.append(min(minsPhysics[i]))
    return minimums

def data():
    number="03"
    folder="dh"+number
    file=folder+".csv"
    dataTrain=pd.read_csv(file,header=0,float_precision='round_trip')
    number="06"
    folder="dh"+number
    file=folder+".csv"
    dataTest=pd.read_csv(file,header=0,float_precision='round_trip')

    return dataTrain, dataTest

def dataInit():
    dataTrain,dataTest = data()

    numbers=["01","03","05","06"]
    minimums = minPhysics(numbers)
    if(minimums[4]<0):
        dataTrain['pression'] = dataTrain['pression']-minimums[4]+0.1
        dataTest['pression'] = dataTest['pression']-minimums[4]+0.1

    return dataTrain, dataTest

#------------------------------------------ Data for energy regression -----------------------------------------------------------------------------
cstlog=1.1

def linear_energy_Preparation():
    dataTrain,dataTest = dataInit()

    attributes=["density","energy"]
    for atr in attributes:
        dataTrain["log_"+atr]=np.log10(dataTrain[atr]+cstlog)
        dataTest["log_"+atr]=np.log10(dataTest[atr]+cstlog)

    X_train=dataTrain[['log_density']]
    Y_train=dataTrain[['log_energy']]

    X_test=dataTest[['log_density']]
    Y_test=dataTest[['log_energy']]

    return X_train, Y_train, X_test, Y_test

#---------------------------------------------- Log Transform on the translated data ---------------------------------------------------------------------

def datalogPreparation():
    dataTrain,dataTest = dataInit()

    attributes=["density","energy","pression","temperature"]
    for atr in attributes:
        dataTrain["log_"+atr]=np.log10(dataTrain[atr]+cstlog)
        dataTest["log_"+atr]=np.log10(dataTest[atr]+cstlog)

    Y_train=dataTrain[['log_energy',"log_pression"]]
    X_train=dataTrain[["log_density","temperature"]]

    Y_test=dataTest[['log_energy',"log_pression"]]
    X_test=dataTest[["log_density","temperature"]]

    transformerX = MinMaxScaler().fit(X_train)
    transformerY = MinMaxScaler().fit(Y_train)

    X_train_prepared = transformerX.transform(X_train)
    X_test_prepared = transformerX.transform(X_test)
    Y_train_prepared = transformerY.transform(Y_train)
    Y_test_prepared = transformerY.transform(Y_test)

    return X_train_prepared, Y_train_prepared, X_test_prepared, Y_test_prepared, transformerY


def logenergy_Preparation():
    dataTrain,dataTest = dataInit()

    attributes=["density","temperature","energy"]
    for atr in attributes:
        dataTrain["log_"+atr]=np.log10(dataTrain[atr]+cstlog)
        dataTest["log_"+atr]=np.log10(dataTest[atr]+cstlog)

    X_train=dataTrain[['log_density',"temperature"]]
    Y_train=dataTrain[['log_energy']]

    X_test=dataTest[['log_density','temperature']]
    Y_test=dataTest[['log_energy']]

    transformerX = MinMaxScaler().fit(X_train)
    transformerY = MinMaxScaler().fit(Y_train)

    X_train_prepared = transformerX.transform(X_train)
    X_test_prepared = transformerX.transform(X_test)
    Y_train_prepared = transformerY.transform(Y_train)
    Y_test_prepared = transformerY.transform(Y_test)

    return X_train_prepared, Y_train_prepared, X_test_prepared, Y_test_prepared, transformerY

def logpression_Preparation():
    dataTrain,dataTest = dataInit()

    attributes=["density","temperature","pression"]
    for atr in attributes:
        dataTrain["log_"+atr]=np.log10(dataTrain[atr]+cstlog)
        dataTest["log_"+atr]=np.log10(dataTest[atr]+cstlog)

    X_train=dataTrain[["log_density","temperature"]]
    Y_train=dataTrain[["log_pression"]]

    X_test=dataTest[["log_density","temperature"]]
    Y_test=dataTest[["log_pression"]]

    transformerX = MinMaxScaler().fit(X_train)
    transformerY = MinMaxScaler().fit(Y_train)

    X_train_prepared = transformerX.transform(X_train)
    X_test_prepared = transformerX.transform(X_test)
    Y_train_prepared = transformerY.transform(Y_train)
    Y_test_prepared = transformerY.transform(Y_test)

    return X_train_prepared, Y_train_prepared, X_test_prepared, Y_test_prepared, transformerY

def loginputs_Preparation(type_output):
    dataTrain,dataTest = dataInit()

    attributes=["density","temperature","energy","pression"]
    for atr in attributes:
        dataTrain["log_"+atr]=np.log10(dataTrain[atr]+cstlog)
        dataTest["log_"+atr]=np.log10(dataTest[atr]+cstlog)

    X_train=dataTrain[["log_density","temperature"]]
    X_test=dataTest[["log_density","temperature"]]

    Y_train = dataTrain[[type_output]]
    Y_test = dataTest[[type_output]]

    transformerX = MinMaxScaler().fit(X_train)

    X_train_prepared = transformerX.transform(X_train)
    X_test_prepared = transformerX.transform(X_test)

    return X_train_prepared, Y_train, X_test_prepared, Y_test

# data inversion: log+MinMax
def inversion_logscale_outputs(Y,transformerY):
    return 10**(transformerY.inverse_transform(Y))-cstlog

def evaluation_logmodel(model,x_test,y_test,transformerY,sample_weight=None):
    y_pred = model(x_test,training=False)
    y_pred = inversion_logscale_outputs(y_pred,transformerY)
    y_test = inversion_logscale_outputs(y_test,transformerY)

    y_test = tf.convert_to_tensor(y_test)
    y_pred = tf.convert_to_tensor(y_pred)
    if not sample_weight is None:
        sample_weight = tf.convert_to_tensor(sample_weight)

    measures = model.compute_metrics(x=None,y=y_test,y_pred=y_pred,sample_weight=sample_weight)

    for k,v in measures.items():
        measures[k] = float(v)
    return measures

#------------------------------------------------ Quantile transformation on initial data ------------------------------------------------------------

def quantiledata_Preparation():
    dataTrain,dataTest = dataInit()

    X_train=dataTrain[["density","temperature"]]
    Y_train=dataTrain[["energy","pression"]]

    X_test=dataTest[["density","temperature"]]
    Y_test=dataTest[["energy","pression"]]

    output_distribution='uniform'
    transformerX = QuantileTransformer(random_state=0,output_distribution=output_distribution).fit(X_train)
    transformerY = QuantileTransformer(random_state=0,output_distribution=output_distribution).fit(Y_train)

    X_train_prepared = transformerX.transform(X_train)
    X_test_prepared = transformerX.transform(X_test)
    Y_train_prepared = transformerY.transform(Y_train)
    Y_test_prepared = transformerY.transform(Y_test)

    return X_train_prepared, Y_train_prepared, X_test_prepared, Y_test_prepared, transformerY

def quantileenergy_Preparation():
    dataTrain,dataTest = dataInit()

    X_train=dataTrain[["density","temperature"]]
    Y_train=dataTrain[["energy"]]

    X_test=dataTest[["density","temperature"]]
    Y_test=dataTest[["energy"]]

    output_distribution='uniform'
    transformerX = QuantileTransformer(random_state=0,output_distribution=output_distribution).fit(X_train)
    transformerY = QuantileTransformer(random_state=0,output_distribution=output_distribution).fit(Y_train)

    X_train_prepared = transformerX.transform(X_train)
    X_test_prepared = transformerX.transform(X_test)
    Y_train_prepared = transformerY.transform(Y_train)
    Y_test_prepared = transformerY.transform(Y_test)

    return X_train_prepared, Y_train_prepared, X_test_prepared, Y_test_prepared, transformerY

def quantilepression_Preparation():
    dataTrain,dataTest = dataInit()

    X_train=dataTrain[["density","temperature"]]
    Y_train=dataTrain[["pression"]]

    X_test=dataTest[["density","temperature"]]
    Y_test=dataTest[["pression"]]

    output_distribution='uniform'
    transformerX = QuantileTransformer(random_state=0,output_distribution=output_distribution).fit(X_train)
    transformerY = QuantileTransformer(random_state=0,output_distribution=output_distribution).fit(Y_train)

    X_train_prepared = transformerX.transform(X_train)
    X_test_prepared = transformerX.transform(X_test)
    Y_train_prepared = transformerY.transform(Y_train)
    Y_test_prepared = transformerY.transform(Y_test)

    return X_train_prepared, Y_train_prepared, X_test_prepared, Y_test_prepared, transformerY


def augmented_data_pression():
    dataTrain, dataTest = data()
    bornes=[-27.85,-21.463,-4.165,4.38,177,546,1570,4412,12296]
    dataRepart = dataTrain
    ex0 = dataRepart[dataRepart['pression']<=bornes[0]]
    ex1 = dataRepart[(dataRepart['pression']>bornes[0]) & (dataRepart['pression']<=bornes[1])]
    ex2 = dataRepart[(dataRepart['pression']>bornes[1]) & (dataRepart['pression']<=bornes[2])]
    ex3 = dataRepart[(dataRepart['pression']>bornes[2]) & (dataRepart['pression']<=bornes[3])]
    ex4 = dataRepart[(dataRepart['pression']>bornes[3]) & (dataRepart['pression']<=bornes[4])]
    ex5 = dataRepart[(dataRepart['pression']>bornes[4]) & (dataRepart['pression']<=bornes[5])]
    ex6 = dataRepart[(dataRepart['pression']>bornes[5]) & (dataRepart['pression']<=bornes[6])]
    ex7 = dataRepart[(dataRepart['pression']>bornes[6]) & (dataRepart['pression']<=bornes[7])]
    ex8 = dataRepart[(dataRepart['pression']>bornes[7]) & (dataRepart['pression']<=bornes[8])]
    ex9 = dataRepart[(dataRepart['pression']>bornes[8])]
    ex0 = pd.concat([ex0]*1060).sort_index()
    ex1 = pd.concat([ex1]*71).sort_index()
    ex2 = pd.concat([ex2]*19).sort_index()
    ex3 = pd.concat([ex3]).sort_index()
    ex4 = pd.concat([ex4]).sort_index()
    ex5 = pd.concat([ex5]).sort_index()
    ex6 = pd.concat([ex6]).sort_index()
    ex7 = pd.concat([ex7]*3).sort_index()
    ex8 = pd.concat([ex8]*3).sort_index()
    ex9 = pd.concat([ex9]*7).sort_index()
    newTrain = pd.concat([ex0,ex1,ex2,ex3,ex4,ex5,ex6,ex7,ex8,ex9]).sort_index()

    numbers=["01","03","05","06"]
    minimums = minPhysics(numbers)
    if(minimums[4]<0):
        newTrain['pression'] = newTrain['pression']-minimums[4]+0.1
        dataTest['pression'] = dataTest['pression']-minimums[4]+0.1
    
    attributes=["density","temperature","pression"]
    for atr in attributes:
        newTrain["log_"+atr]=np.log10(newTrain[atr]+cstlog)
        dataTest["log_"+atr]=np.log10(dataTest[atr]+cstlog)

    X_train=newTrain[["log_density","temperature"]]
    Y_train=newTrain[["log_pression"]]

    X_test=dataTest[["log_density","temperature"]]
    Y_test=dataTest[["log_pression"]]

    transformerX = MinMaxScaler().fit(X_train)
    transformerY = MinMaxScaler().fit(Y_train)

    X_train_prepared = transformerX.transform(X_train)
    X_test_prepared = transformerX.transform(X_test)
    Y_train_prepared = transformerY.transform(Y_train)
    Y_test_prepared = transformerY.transform(Y_test)

    return X_train_prepared, Y_train_prepared, X_test_prepared, Y_test_prepared, transformerY

#------------------------------------------ Delete extreme data -----------------------------------------------------------------------------------------
def elapsed_logpression_Preparation(bornes):
    dataTrain,dataTest = dataInit()

    dataTrain = dataTrain.drop(dataTrain[(dataTrain['density'] < bornes[0]) | (dataTrain['density'] > bornes[1])].index)
    dataTest = dataTest.drop(dataTest[(dataTest['density'] < bornes[0]) | (dataTest['density'] > bornes[1])].index)

    attributes=["density","temperature","pression","energy"]
    for atr in attributes:
        dataTrain["log_"+atr]=np.log10(dataTrain[atr]+cstlog)
        dataTest["log_"+atr]=np.log10(dataTest[atr]+cstlog)

    X_train=dataTrain[["log_density","temperature"]]
    Y_train=dataTrain[["pression"]]

    X_test=dataTest[["log_density","temperature"]]
    Y_test=dataTest[["pression"]]

    transformerX = MinMaxScaler().fit(X_train)
    transformerY = MinMaxScaler().fit(Y_train)

    X_train_prepared = transformerX.transform(X_train)
    X_test_prepared = transformerX.transform(X_test)
    Y_train_prepared = transformerY.transform(Y_train)
    Y_test_prepared = transformerY.transform(Y_test)

    return tf.convert_to_tensor(X_train_prepared), tf.convert_to_tensor(Y_train), tf.convert_to_tensor(X_test_prepared), tf.convert_to_tensor(Y_test), transformerY
    

