from keras import losses, backend
from loss_perso import squared_error

typef="float64"
backend.set_floatx(typef)

import os
os.chdir("/home/bbensaid/Documents/Anabase/TOptimizers") 

import activations_perso
from model import build_poly
from training import train
from tirages_json import single_sample_json, tirages_json
import read

batch_size=2
x_train,y_train = read.poly_data(typef)

activation=activations_perso.polyThree
loss = losses.MeanSquaredError()
#loss=squared_error
name_init="Uniform"
w=-5; b=7
params_init=[w,w,b,b]
seed=0

#paramètres d'arrêt
eps=10**(-4); max_epochs=10000

#paramètres d'entrainement 
lr=0.01
seuil=0.01
f1=2; f2=10000; lambd=0.5; rho=0.9; eps_egd=0.01
beta_1=0.9; beta_2=0.999; epsilon_a=1e-07
amsgrad=False

algo="LCD_GD"

"""
model = build_poly(activation,loss,name_init,params_init,seed)
model, epoch, norme_grad, cost, temps,_ = train(algo,model,loss,x_train,y_train,eps,max_epochs,lr,seuil,f1,f2,rho,eps_egd,lambd,beta_1,beta_2,epsilon_a,amsgrad,typef)

print(model.get_weights())
print("temps: ", temps)
print("epoch: ", epoch) """

""" dico = single_sample_json("polyTwo.json", "poly", [], [activation], loss, name_init, params_init, seed, ["mae"], x_train, y_train,
algo, eps, max_epochs, lr, seuil, f1, f2, rho, eps_egd, lambd, beta_1, beta_2, 1e-10, amsgrad, typef, None,
"simple",x_train,y_train)

print(dico) """

res = tirages_json("polyThree.jsonl", 0, 10,
    "poly", [], [activation], loss, name_init, params_init, ["mae"], x_train, y_train,
    algo, eps, max_epochs, lr, seuil, f1, f2, rho, eps_egd, lambd, beta_1, beta_2, 1e-10, amsgrad, typef, None,
    "simple",x_train,y_train)