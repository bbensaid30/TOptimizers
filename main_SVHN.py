import tensorflow as tf

from keras import losses, metrics

from data import SVHN
import tirages_sto
import tirages

typef="float32"
tf.keras.backend.set_floatx(typef)

x_train, y_train, x_test, y_test = SVHN(typef, True)
PTrain=73257; PTest=26032
buffer_size=73257
batch_size=32

# architecture and algo. 
algo="RAG"
#name_model="conv_gray_cifar10"
name_model="FC"
#name_model = "lenet1_cifar10"
nbNeurons=[3072,300,200,100,10]
activations=['tanh', 'tanh','tanh', 'softmax']
loss=losses.MeanSquaredError(reduction="sum")
#loss=losses.MeanSquaredError()
#loss=losses.CategoricalCrossentropy()

metrics = ["accuracy"]
name_init="Bengio"
params_init=[-1,1]

#paramètres d'arrêt
eps=10**(-4); max_epochs=35
#paramètres d'entrainement 
lr=0.1
seuil=0.01
f1=30; f2=10000; lambd=0.5
beta_1=0.9; beta_2=0.999; epsilon_a=1e-07
amsgrad=False

tirageMin=0; nbTirages=1; nbSeeds=1

studies = tirages_sto.tirages_sto(tirageMin, nbTirages, nbSeeds, 
    name_model, nbNeurons, activations, loss, name_init, params_init, metrics, x_train, y_train, PTrain, 
    algo, batch_size, buffer_size, eps, max_epochs, lr, f1, f2, lambd, beta_1, beta_2, epsilon_a, amsgrad,typef,
    "simple",x_test, y_test, PTest)
""" studies = tirages.tirages(tirageMin,nbTirages,name_model,nbNeurons,activations,loss,name_init,params_init,metrics,
x_train,y_train,algo,eps,max_epochs,lr,seuil,f1,f2,0,0,lambd,beta_1,beta_2,epsilon_a,amsgrad,"float32",1,
"simple",x_test,y_test) """
print(studies)


""" fileEnd = tirages_sto.informationFileSto(tirageMin,nbTirages,nbSeeds, name_model, nbNeurons, activations, name_init, params_init,
    PTrain, PTest, algo, batch_size, buffer_size, eps, max_epochs, lr, f1, f2, lambd, beta_1, beta_2)

folder="CIFAR10"
tirages_sto.minsRecordSto(studies,folder,fileEnd,eps) """