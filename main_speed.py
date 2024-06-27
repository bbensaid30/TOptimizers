import numpy as np
from keras import losses
import time

import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import os
os.chdir("/home/bbensaid/Documents/Anabase/NN_shaman") 

import activations_perso
from model import build_model
import tirages

from training import train
from eval import eval_global

from data import Speed

sample_weight=1

x_train,y_train,x_test,y_test = Speed()

# architecture.
name_model="FC"
n_input=4; n_output=1
nbNeurons=[n_input,100, 100, 100, n_output]
activations=['relu','relu', 'relu', 'linear']
#loss = losses.MeanSquaredError()
#loss = losses.MeanAbsoluteError()
loss = losses.MeanAbsolutePercentageError()
metrics = [tf.keras.metrics.RootMeanSquaredError(), 'mape']
name_init="Bengio"
params_init=[-10,10]

#paramètres d'arrêt
eps=10**(-4); max_epochs=5000
#paramètres d'entrainement 
lr=0.1
seuil=0.01
f1=30; f2=10000; lambd=0.5; rho=0.9; eps_egd=0.01
beta_1=0.995; beta_2=0.999; epsilon=1e-07
amsgrad=False

#build the model
model = build_model(name_model,nbNeurons,activations,loss,name_init,params_init,0,metrics)
model.summary()
model_copy = build_model(name_model,nbNeurons,activations,loss,name_init,params_init,0,metrics)

#train the model
model, epochs, norme_grad, cost_final, temps = train("LC_EGD2",model,model_copy,loss,x_train,y_train,eps,max_epochs,
lr,seuil,f1,f2,rho,eps_egd,lambd,beta_1,beta_2,epsilon,amsgrad,sample_weight)

output = model(x_train).numpy()
print(np.min(output)); print(np.max(output))

measures, temps_forward = eval_global('simple',model,x_train,y_train,None,sample_weight)
print(measures)

measures, temps_forward = eval_global('simple',model,x_test,y_test,None,sample_weight)
print(measures)

#np.savetxt('Data/residuals_v1.csv', output-y_train.numpy(), delimiter=',')

