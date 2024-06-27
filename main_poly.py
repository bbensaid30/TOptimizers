from keras import optimizers
from keras import losses

import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import os
os.chdir("/home/bbensaid/Documents/Anabase/NN_shaman") 

import activations_perso
from model import build_poly
from training import train
import read

batch_size=2
x_train,y_train = read.poly_data()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=batch_size).batch(batch_size)


activation=activations_perso.polyFive
loss = losses.MeanSquaredError()
name_init="Uniform"
w=-2; b=2
params_init=[w,w,b,b]
seed=0

#paramètres d'arrêt
eps=10**(-4); max_epochs=3000

#paramètres d'entrainement 
lr=0.1
seuil=0.01
f1=30; f2=10000; lambd=0.5; rho=0.9; eps_egd=0.01
beta_1=0.9; beta_2=0.999; epsilon=1e-07
amsgrad=False

algo="Adam"
model = build_poly(activation,loss,name_init,params_init,seed); model_copy = build_poly(activation,loss,name_init,params_init,seed)
model, epoch, norme_grad, cost, temps = train(algo,model,model_copy,loss,x_train,y_train,eps,max_epochs,lr,seuil,f1,f2,rho,eps_egd,lambd,beta_1,beta_2,epsilon,amsgrad)

print(model.get_weights())
print("temps: ", temps)


