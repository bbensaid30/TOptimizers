import tensorflow as tf

from keras import losses, metrics

from data import CIFAR10, CIFAR10_gray, CIFAR10_gray_flatten, CIFAR10_reduce
import tirages_sto

x_train, y_train, x_test, y_test = CIFAR10_gray_flatten()
PTrain=50000; PTest=10000
buffer_size=50000
batch_size=128

""" num=1000
x_train, y_train, x_test, y_test = CIFAR10_reduce(num)
PTrain=num; PTest=10000
buffer_size=num
batch_size=500 """

# architecture and algo. 
algo="RRAdam"
#name_model="conv_gray_cifar10"
name_model="FC"
#name_model = "lenet1_cifar10"
nbNeurons=[1024,128,64,10]
activations=['tanh', 'tanh', 'softmax']
loss=losses.MeanSquaredError(reduction="sum")
#loss=losses.MeanSquaredError()
#loss=losses.CategoricalCrossentropy(reduction="sum")

metrics = ["accuracy"]
name_init="Xavier"
params_init=[-1,1]

#paramètres d'arrêt
eps=10**(-4); max_epochs=1000
#paramètres d'entrainement 
lr=0.001
seuil=0.01
f1=30; f2=10000; lambd=0.5
beta_1=0.9; beta_2=0.999; epsilon_a=1e-07
amsgrad=False

tirageMin=0; nbTirages=1; nbSeeds=1

studies = tirages_sto.tirages_sto(tirageMin, nbTirages, nbSeeds, 
    name_model, nbNeurons, activations, loss, name_init, params_init, metrics, x_train, y_train, PTrain, 
    algo, batch_size, buffer_size, eps, max_epochs, lr, f1, f2, lambd, beta_1, beta_2, epsilon_a, amsgrad,"float32",
    "simple",x_test, y_test, PTest)
print(studies)


""" fileEnd = tirages_sto.informationFileSto(tirageMin,nbTirages,nbSeeds, name_model, nbNeurons, activations, name_init, params_init,
    PTrain, PTest, algo, batch_size, buffer_size, eps, max_epochs, lr, f1, f2, lambd, beta_1, beta_2)

folder="CIFAR10"
tirages_sto.minsRecordSto(studies,folder,fileEnd,eps) """