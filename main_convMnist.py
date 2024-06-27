import numpy as np
import tensorflow as tf

from keras import losses, metrics

from model import build_lenet1_mnist
from data import MNIST,FASHION_MNIST
import tirages

type='float32'
tf.keras.backend.set_floatx(type)

sample_weight=1

# Prepare the training dataset.
x_train, y_train, x_test, y_test = MNIST(type)

# architecture.
name_model="lenet1_mnist"
nbNeurons=[784,24,10]
activations=['tanh','softmax']
loss = losses.MeanSquaredError()
#loss = losses.CategoricalCrossentropy()
metrics = ["categorical_accuracy"]
name_init="Bengio"
params_init=[-1,1]

#paramètres d'arrêt
eps=10**(-4); max_epochs=10000
#paramètres d'entrainement 
lr=0.1
seuil=0.01
f1=30; f2=10000; lambd=0.5; rho=0.9; eps_egd=0.01
beta_1=0.9; beta_2=0.999; epsilon=1e-07
amsgrad=False

tirageMin=0; nbTirages=3
algo="LC_EGD2"
studies = tirages.tirages(tirageMin,nbTirages,name_model,nbNeurons,activations,loss,name_init,params_init,metrics,
x_train,y_train,algo,eps,max_epochs,lr,seuil,f1,f2,rho,eps_egd,lambd,beta_1,beta_2,epsilon,amsgrad,sample_weight,
"simple",x_test,y_test)
print(studies)

""" fileEnd = tirages.informationFile(tirageMin,nbTirages,name_model, nbNeurons, activations, name_init, params_init,
60000, 10000,
algo, eps, max_epochs, lr, seuil, f1, f2, rho, eps_egd,lambd, beta_1, beta_2, epsilon)

folder="FASHION_MNIST"
tirages.minsRecordRegression(studies,folder,fileEnd,eps) """