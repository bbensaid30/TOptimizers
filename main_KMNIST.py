import tensorflow as tf

from keras import losses, metrics

from data import KMNIST
import tirages_sto

typef='float32'
tf.keras.backend.set_floatx(typef)

# Prepare the training dataset.
x_train, y_train, x_test, y_test = KMNIST(typef)

#batch
PTrain=60000; PTest=10000
buffer_size=60000
batch_size=30000

# architecture and algo.
algo="RAG"
name_model="lenet1_mnist"
nbNeurons=[]
activations=[]
#loss = squared_error
loss = losses.MeanSquaredError(reduction='sum')
metrics = ["categorical_accuracy"]
name_init="Bengio"
params_init=[-1,1]

#paramètres d'arrêt
eps=10**(-4); max_epochs=10
#paramètres d'entrainement 
lr=0.1
seuil=0.01
f1=30; f2=10000; lambd=0.5
beta_1=0.9; beta_2=0.999; epsilon_a=1e-07
amsgrad=False

tirageMin=0; nbTirages=1; nbSeeds=1

studies = tirages_sto.tirages_sto(tirageMin, nbTirages, nbSeeds, 
    name_model, nbNeurons, activations, loss, name_init, params_init, metrics, x_train, y_train, PTrain, 
    algo, batch_size, buffer_size, eps, max_epochs, lr, f1, f2, lambd, beta_1, beta_2, epsilon_a, amsgrad, typef,
    "simple",x_test, y_test, PTest)
print(studies)


""" fileEnd = tirages_sto.informationFileSto(tirageMin,nbTirages,nbSeeds, name_model, nbNeurons, activations, name_init, params_init,
    PTrain, PTest, algo, batch_size, buffer_size, eps, max_epochs, lr, f1, f2, lambd, beta_1, beta_2)

folder="KMNIST"
tirages_sto.minsRecordSto(studies,folder,fileEnd,eps) """