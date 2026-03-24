import argparse
from keras import losses
from tensorflow.experimental import numpy as tnp
import tensorflow as tf

from data import CIFAR10
from tirages_json import tirages_json

from utils import get_memory_usage

def main():
    # 1. Création du parseur
    parser = argparse.ArgumentParser(description="Conv_C1 CIFAR10 avec parallélisation Joblib.")

    # 2. Ajout des arguments
    # On peut définir des valeurs par défaut et des types
    parser.add_argument(
        "--n_jobs", 
        type=int, 
        default=1, 
        help="Nombre de cœurs CPU à utiliser (par défaut: 1)"
    )

    # 3. Récupération des arguments
    args = parser.parse_args()

    # Accès aux variables via args.nom_de_l_argument
    print(f"--- Configuration du Job ---")
    print(f"CPUs alloués      : {args.n_jobs}")
    print(f"----------------------------")

    sample_weight=1


    # Prepare the training dataset.
    x_train, y_train, x_test, y_test = CIFAR10()

    # architecture.
    name_model="conv_C1"
    nbNeurons=[]
    activations=[]
    loss = losses.MeanSquaredError()
    #loss = losses.CategoricalCrossentropy()
    metrics = ["categorical_accuracy"]
    name_init="Bengio"
    params_init=[-1,1]

    #paramètres d'arrêt
    eps=10**(-4); max_epochs=50
    #paramètres d'entrainement 
    lr=0.001
    weight_decay=0.004
    f1=30; f2=10000; lambd=0.5
    beta_1=0.9; beta_2=0.999; epsilon_a=0.01*tf.sqrt(tnp.finfo(tf.float32).eps)
    amsgrad=False

    tirageMin=0; nbTirages=1
    algo="LCD_RMS"

    folder="CIFAR10/"
    filename=folder+algo+".jsonl"

    studies = tirages_json(args.n_jobs, filename, tirageMin,nbTirages,name_model,nbNeurons,activations,loss,name_init,params_init,metrics,
    x_train,y_train,algo,eps,max_epochs,lr,weight_decay,f1,f2,lambd,beta_1,beta_2,epsilon_a,amsgrad,tf.float32,sample_weight,
    "simple",x_test,y_test)
    print(studies)

if __name__ == "__main__":
    start_mem = get_memory_usage()
    main()
    end_mem = get_memory_usage()
    print(f"Consommation RAM : {end_mem:.2f} MB (Augmentation : {end_mem - start_mem:.2f} MB)")

#Lenet1: 6.8GB and 490s for 20 epochs on cpu
#Conv_C1: 5.6GB and 657s for 20 epochs on cpu