import os
os.chdir("/home/bbensaid/Documents/Anabase/NN")

from joblib import Parallel, delayed, parallel_backend
import ray
from math import isnan, isinf
from numpy import format_float_scientific as ffs
import tensorflow as tf

from model import build_model
from training import train_sto
from eval_sto import eval_sto_global

num_cpus = 8; num_gpus=0
n_jobs=-1

def single_sto_sample(name_model, nbNeurons, activations, loss, name_init, params_init, seed, metrics, train_dataset, PTrain,
algo, batch_size, buffer_size, seed_permut, eps, max_epochs, lr, f1, f2, lambd, beta_1, beta_2, epsilon_a, amsgrad, type,
name_eval,test_dataset, PTest, transformerY=None):

    dico = {}

    if(algo=="RRAdam"):
        train_dataset = train_dataset.shuffle(buffer_size=buffer_size, seed=seed_permut, reshuffle_each_iteration=True).batch(batch_size)

    #build the model
    model = build_model(name_model,nbNeurons,activations,loss,name_init,params_init,seed,metrics)

    #train the model
    model, epochs, norme_grad, cost_final, temps = train_sto(algo, model,loss,
    train_dataset, PTrain, eps, max_epochs, lr, f1, f2, lambd, 
    beta_1, beta_2, epsilon_a, amsgrad, type)
    dico['num_tirage'] = seed
    dico['num_permut'] = seed_permut
    dico['epochs'] = epochs
    dico['time_train'] = temps
    dico['norme_grad'] = norme_grad.numpy()
    dico['cost_train'] = cost_final.numpy()

    cost_test=0
    for x_batch_test, y_batch_test in test_dataset:
        prediction_test = model(x_batch_test, training=False)
        Ri_test = loss(y_batch_test, prediction_test)
        cost_test+=Ri_test
    cost_test/=PTest
    dico['cost_test'] = cost_test.numpy()

    #Compute the metrics for train set
    model.reset_metrics()
    measures, temps_forward = eval_sto_global(name_eval,model,train_dataset,transformerY)
    for key in measures.keys():
        dico[key+"_train"] = measures[key]

    #Compute the metrics for test set
    model.reset_metrics()
    measures, temps_forward = eval_sto_global(name_eval,model,test_dataset,transformerY)
    for key in measures.keys():
        dico[key+"_test"] = measures[key]
    dico['temps_forward'] = temps_forward/PTest

    return dico

def tirages_sto(tirageMin, nbTirages, nbSeeds, 
    name_model, nbNeurons, activations, loss, name_init, params_init, metrics, train_dataset, PTrain, 
    algo, batch_size, buffer_size, eps, max_epochs, lr, f1, f2, lambd, beta_1, beta_2, epsilon_a, amsgrad, type,
    name_eval,test_dataset, PTest, transformerY=None):

    if(algo=="RAG" or algo=="RAGL"):
        train_dataset = train_dataset.batch(batch_size)

    test_dataset = test_dataset.batch(batch_size)

    with parallel_backend('threading', n_jobs=n_jobs):
        res = Parallel()(delayed(single_sto_sample)(name_model, nbNeurons, activations, loss, name_init, params_init, i, metrics, train_dataset, PTrain,
    algo, batch_size, buffer_size, j, eps, max_epochs, lr, f1, f2, lambd, beta_1, beta_2, epsilon_a, amsgrad, type,
    name_eval,test_dataset, PTest, transformerY) for i in range(tirageMin, tirageMin+nbTirages) for j in range(nbSeeds))
    
    """ res = [single_sto_sample(name_model, nbNeurons, activations, loss, name_init, params_init, i, metrics, train_dataset, PTrain,
    algo, batch_size, buffer_size, j, eps, max_epochs, lr, f1, f2, lambd, beta_1, beta_2, epsilon_a, amsgrad, type,
    name_eval,test_dataset, PTest, transformerY) for i in range(tirageMin, tirageMin+nbTirages) for j in range(nbSeeds)] """

    return res

def minsRecordSto(studies, folder, fileEnd, eps):
    infoFile = open("Record/"+folder+"/info_"+fileEnd,"w")
    allinfoFile = open("Record/"+folder+"/allinfo_"+fileEnd,"w")
    nonFile = open("Record/"+folder+"/nonConv_"+fileEnd,"w")

    div=0; nonConv=0

    nbTrains = len(studies)
    for i in range(nbTrains):
        if(isnan(studies[i]['norme_grad'])==False and isinf(studies[i]['norme_grad'])==False and studies[i]['norme_grad']<eps):
            for value in studies[i].values():
                infoFile.write(ffs(value) + "\n")
                allinfoFile.write(ffs(value) + "\n")
        else:
            if(studies[i]['norme_grad']>1000 or isnan(studies[i]['norme_grad']) or isinf(studies[i]['norme_grad'])):
                div+=1
                nonFile.write("-3" + "\n")
            else:
                nonConv+=1
                nonFile.write("-2" + "\n")
                for value in studies[i].values():
                    allinfoFile.write(ffs(value) + "\n")
    
    infoFile.write(ffs(nonConv/nbTrains) + "\n")
    infoFile.write(ffs(div/nbTrains) + "\n")

    print("Proportion de divergence: ", div/nbTrains)
    print("Proportion de non convergence: ", nonConv/nbTrains)

    infoFile.close(); allinfoFile.close(); nonFile.close()
        
def informationFileSto(tirageMin,nbTirages,nbSeeds, name_model, nbNeurons, activations, name_init, params_init,
    PTrain, PTest, algo, batch_size, buffer_size, eps, max_epochs, lr, f1, f2, lambd, beta_1, beta_2):

    if(name_model=="FC"):
        arch=""
        for l in range(len(activations)):
            arch += str(nbNeurons[l+1])
            arch += "("+ activations[l] + ")"
    else:
        arch = name_model

    finParameters = ", PTrain=" + str(PTrain) + ", PTest=" + str(PTest) + ", tirageMin=" + str(tirageMin) + ", nbTirages=" + str(nbTirages) + ", nbSeeds=" + str(nbSeeds)+ ", b=" +str(batch_size)+ ", bf=" +str(buffer_size) +", eps=" +str(eps) + ", maxEpoch=" +str(max_epochs) + ")"

    if(algo == "RAG" or algo == "RAGL"):
        parameters = "(eta="+str(lr)+", f1=" + str(f1) + ", f2=" + str(f2) + ", lambd=" + str(lambd)
    elif(algo=="RRAdam"):
        parameters = "(eta="+str(lr)+", b1=" + str(beta_1) + ", b2=" + str(beta_2)
    parameters += finParameters

    if(name_init == "Uniform" or name_init == "Normal"):
        if(len(params_init)==2):
            initialisation = name_init + "(" + str(params_init[0]) + "," + str(params_init[1]) + ")"
        else:
            initialisation = name_init + "(" + str(params_init[0]) + "," + str(params_init[1]) + "," + str(params_init[2]) + "," +str(params_init[3]) + ")"
    else:
        initialisation = name_init

    return algo + "_" + arch + "_" + parameters + "_" + initialisation + ".csv"