from keras import losses, metrics

from data import FASHION_MNIST
from tirages_json import tirages_json

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
eps=10**(-4); max_epochs=10000
#paramètres d'entrainement 
lr=0.1
seuil=0.01
f1=30; f2=10000; lambd=0.5; rho=0.9; eps_egd=0.01
beta_1=0.9; beta_2=0.999; epsilon=1e-10
amsgrad=False

tirageMin=0; nbTirages=50
algo="LCD_RMS"

folder="FASHION_MNIST/"
filename=folder+algo+".jsonl"

studies = tirages_json(filename, tirageMin,nbTirages,name_model,nbNeurons,activations,loss,name_init,params_init,metrics,
x_train,y_train,algo,eps,max_epochs,lr,seuil,f1,f2,rho,eps_egd,lambd,beta_1,beta_2,epsilon,amsgrad,"float32",sample_weight,
"simple",x_test,y_test)


