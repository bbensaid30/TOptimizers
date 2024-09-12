from keras import losses, metrics
import tensorflow as tf

from data import HIGGS
import tirages_sto

sample_weight=1

# Prepare the training dataset.
x_train, y_train, x_test, y_test = HIGGS()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))


#batch
PTrain=10500000; PTest=500000
buffer_size=1000000
batch_size=500000

# architecture.
algo="RRAdam"
name_model="FC"
nbNeurons=[21,300,1] 
activation="tanh"
activations=[activation,'sigmoid']
loss = losses.MeanSquaredError()
#loss = losses.BinaryCrossentropy(reduction='sum')
metrics = ["accuracy", metrics.AUC()]
name_init="Bengio"
params_init=[0,0.01]

#paramètres d'arrêt
eps=10**(-4); max_epochs=20
#paramètres d'entrainement 
lr=0.001
seuil=0.01
f1=30; f2=10000; lambd=0.5
beta_1=0.9; beta_2=0.999; epsilon_a=1e-07
amsgrad=False

tirageMin=0; nbTirages=1; nbSeeds=1

studies = tirages_sto.tirages_sto(tirageMin, nbTirages, nbSeeds, 
    name_model, nbNeurons, activations, loss, name_init, params_init, metrics, train_dataset, PTrain, 
    algo, batch_size, buffer_size, eps, max_epochs, lr, f1, f2, lambd, beta_1, beta_2, epsilon_a, amsgrad,"float32",
    "simple",test_dataset, PTest)
print(studies)


""" fileEnd = tirages_sto.informationFileSto(tirageMin,nbTirages,nbSeeds, name_model, nbNeurons, activations, name_init, params_init,
    PTrain, PTest, algo, batch_size, buffer_size, eps, max_epochs, lr, f1, f2, lambd, beta_1, beta_2)

folder="HIGGS"
tirages_sto.minsRecordSto(studies,folder,fileEnd,eps) """