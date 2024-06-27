import numpy as np
import tensorflow as tf

from keras import losses, metrics

from data import IMDB_onehot, IMDB_embedding
import tirages

type='float32'
tf.keras.backend.set_floatx(type)

sample_weight=1

# Prepare the training dataset.
x_train, y_train, x_test, y_test = IMDB_onehot()

# Prepare the training dataset.
#train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#train_dataset = train_dataset.shuffle(buffer_size=batch_size).batch(batch_size)

# architecture.
name_model="FC"
nbNeurons=[10000,1]
activations=['sigmoid']
loss = losses.MeanSquaredError()
#loss = losses.BinaryCrossentropy()
metrics = ["accuracy"]
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

tirageMin=0; nbTirages=1
algo = "LC_EGD2"
studies = tirages.tirages(tirageMin,nbTirages,name_model,nbNeurons,activations,loss,name_init,params_init,metrics,
x_train,y_train,algo,eps,max_epochs,lr,seuil,f1,f2,rho,eps_egd,lambd,beta_1,beta_2,epsilon,amsgrad,sample_weight,
"simple",x_test,y_test)
print(studies)

fileEnd = tirages.informationFile(tirageMin,nbTirages,name_model, nbNeurons, activations, name_init, params_init,
25000, 25000,
algo, eps, max_epochs, lr, seuil, f1, f2, rho, eps_egd,lambd, beta_1, beta_2, epsilon)

""" folder="IMDB"
tirages.minsRecordRegression(studies,folder,fileEnd,eps) """

