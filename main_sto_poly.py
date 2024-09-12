import tensorflow as tf
import os 

from keras import losses, backend

type="float64"
backend.set_floatx(type)
os.chdir("/home/bbensaid/Documents/Anabase/NN")

import activations_perso
from model import build_poly
from training import train_sto
import read
from loss_perso import squared_error

PTrain=2
batch_size=1; buffer_size=PTrain
x_train,y_train = read.poly_data(type)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
print(tf.__version__)

activation=activations_perso.polyFive
loss = squared_error
#loss = losses.MeanSquaredError(reduction='sum')
name_init="Uniform"
w=3; b=1
params_init=[w,w,b,b]
seed=0

#paramètres d'arrêt
eps=10**(-4); max_epochs=10000
seed_permut=0

#paramètres d'entrainement 
lr=0.1
seuil=0.01
f1=30; f2=10000; lambd=0.5
beta_1=0.9; beta_2=0.999; epsilon_a=1e-07
amsgrad=False

algo="RAGL"
model = build_poly(activation,loss,name_init,params_init,seed)

if(algo=="RAG" or algo=="RAGL" or algo=="GD_batch"):
    train_dataset = train_dataset.batch(batch_size)
elif(algo=="RRAdam" or algo=="RRGD"):
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size, seed=seed_permut, reshuffle_each_iteration=True).batch(batch_size)

model, epoch, norme_grad, cost, temps = train_sto(algo, model,loss,
train_dataset, PTrain, eps, max_epochs, lr, f1, f2, lambd, beta_1, beta_2, epsilon_a, amsgrad, type)

print(model.get_weights())
print("gradNorm", norme_grad)
print("cost: ", cost)
print("epoch: ", epoch)
print("temps: ", temps)