from keras import losses, metrics

from data import FASHION_MNIST
import tirages

sample_weight=1

# Prepare the training dataset.
x_train, y_train, x_test, y_test = FASHION_MNIST()

# architecture.
name_model="lenet1_mnist"
nbNeurons=[]
activations=[]
loss = losses.MeanSquaredError()
#loss = losses.CategoricalCrossentropy()
metrics = ["categorical_accuracy"]
name_init="Bengio"
params_init=[-1,1]

#paramètres d'arrêt
eps=10**(-4); max_epochs=200
#paramètres d'entrainement 
lr=0.1
seuil=0.01
f1=30; f2=10000; lambd=0.5; rho=0.9; eps_egd=0.01
beta_1=0.9; beta_2=0.999; epsilon=1e-07
amsgrad=False

tirageMin=0; nbTirages=1
algo="LC_EGD2"
studies = tirages.tirages(tirageMin,nbTirages,name_model,nbNeurons,activations,loss,name_init,params_init,metrics,
x_train,y_train,algo,eps,max_epochs,lr,seuil,f1,f2,rho,eps_egd,lambd,beta_1,beta_2,epsilon,amsgrad,"float32",sample_weight,
"simple",x_test,y_test)
print(studies)

""" fileEnd = tirages.informationFile(tirageMin,nbTirages,name_model, nbNeurons, activations, name_init, params_init,
60000, 10000,
algo, eps, max_epochs, lr, seuil, f1, f2, rho, eps_egd,lambd, beta_1, beta_2, epsilon)

folder="FASHION_MNIST"
tirages.minsRecordRegression(studies,folder,fileEnd,eps) """


