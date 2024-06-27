import numpy as np
from keras import losses
import time

import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import os
os.chdir("/home/bbensaid/Documents/Anabase/NN_shaman") 

import activations_perso
from model import build_model
from metrics_perso import global_dinf, relative_dinf
import tirages

import prepared_eqn
import read
from lds import compute_weights_lds,simple_weights


x_train,y_train,x_test,y_test,transformerY = prepared_eqn.elapsed_logpression_Preparation([-100,100000000]) #8347, 16498, 24649, 100000
#x_train,y_train,x_test,y_test = prepared_eqn.loginputs_Preparation('pression')

#batch_size = x_train.shape[0]
#train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#train_dataset = train_dataset.shuffle(buffer_size=batch_size).batch(batch_size)

""" #weights for lds
kernel="gaussian"; ks=5; sigma=2
sample_weight = compute_weights_lds(y_train,kernel,ks,sigma)
print(np.median(sample_weight.numpy())) """

""" #simple weights directly from the histogram
sample_weight = simple_weights(y_train)
print(np.median(sample_weight)) """

sample_weight=1

# architecture.
name_model="FC"
nbNeurons=[2,50,1]
activations=['tanh','linear']
loss = losses.MeanSquaredError()
#loss = losses.MeanAbsolutePercentageError()
metrics = [tf.keras.metrics.RootMeanSquaredError(), 'mae', global_dinf,'mape',relative_dinf]
name_init="Xavier"
params_init=[-10,10]

#paramètres d'arrêt
eps=2*10**(-2); max_epochs=20000
#paramètres d'entrainement 
lr=0.1
seuil=0.01
f1=30; f2=10000; lambd=0.5; rho=0.9; eps_egd=0.01
beta_1=0.9; beta_2=0.999; epsilon=1e-07
amsgrad=False

#entrainement
tirageMin=0; nbTirages=1
studies = tirages.tirages(tirageMin,nbTirages,name_model,nbNeurons,activations,loss,name_init,params_init,metrics,
x_train,y_train,"LC_EGD2",eps,max_epochs,lr,seuil,f1,f2,rho,eps_egd,lambd,beta_1,beta_2,epsilon,amsgrad,sample_weight,
"simple",x_test,y_test)
print(studies)