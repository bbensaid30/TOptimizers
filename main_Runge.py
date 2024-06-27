import numpy as np
import tensorflow as tf

from keras import losses, metrics

from data import Runge
import tirages

type='float32'
tf.keras.backend.set_floatx(type)

sample_weight=1

# Prepare the training dataset.
N_train=21; N_test=40
x_train, y_train, x_test, y_test = Runge(type,N_train,N_test)

# architecture.
name_model="FC"
nbNeurons=[1,15,1]
activations=['tanh','linear']
loss = losses.MeanSquaredError()
metrics = ["mean_squared_error"]
name_init="Xavier"
params_init=[-1,1]

#paramètres d'arrêt
eps=10**(-4); max_epochs=10000
#paramètres d'entrainement 
lr=0.1
seuil=0.01
f1=2; f2=10000; lambd=0.5; rho=0.9; eps_egd=0.01
beta_1=0.9; beta_2=0.999; epsilon=1e-07
amsgrad=False

tirageMin=0; nbTirages=1
algo="LC_EGD"
studies = tirages.tirages(tirageMin,nbTirages,name_model,nbNeurons,activations,loss,name_init,params_init,metrics,
x_train,y_train,algo,eps,max_epochs,lr,seuil,f1,f2,rho,eps_egd,lambd,beta_1,beta_2,epsilon,amsgrad,sample_weight,
"simple",x_test,y_test)
print(studies)

fileEnd = tirages.informationFile(tirageMin,nbTirages,name_model, nbNeurons, activations, name_init, params_init,
N_train, N_test,
algo, eps, max_epochs, lr, seuil, f1, f2, rho, eps_egd,lambd, beta_1, beta_2, epsilon)

#folder="Runge"
#tirages.minsRecordRegression(studies,folder,fileEnd,eps)