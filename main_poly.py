from keras import losses, backend
from loss_perso import squared_error

typef="float64"
backend.set_floatx(typef)

import os
os.chdir("/home/bbensaid/Documents/Anabase/NN") 

import activations_perso
from model import build_poly
from training import train
import read

batch_size=2
x_train,y_train = read.poly_data(typef)

activation=activations_perso.polyTwo
loss = losses.MeanSquaredError()
#loss=squared_error
name_init="Uniform"
w=-2; b=4
params_init=[w,w,b,b]
seed=0

#paramètres d'arrêt
eps=10**(-4); max_epochs=10000

#paramètres d'entrainement 
lr=0.1
seuil=0.01
f1=2; f2=10000; lambd=0.5; rho=0.9; eps_egd=0.01
beta_1=0.9; beta_2=0.999; epsilon_a=1e-07
amsgrad=False

algo="LC_EGD"
model = build_poly(activation,loss,name_init,params_init,seed)
model, epoch, norme_grad, cost, temps = train(algo,model,loss,x_train,y_train,eps,max_epochs,lr,seuil,f1,f2,rho,eps_egd,lambd,beta_1,beta_2,epsilon_a,amsgrad,typef)

#print(model.get_weights())
print("temps: ", temps)
print("epoch: ", epoch)