import os
os.chdir("/home/bbensaid/Documents/Anabase/NN_shaman")

from joblib import Parallel, delayed, parallel_backend
import ray
from math import isnan, isinf
from numpy import format_float_scientific as ffs

from model import build_model
from training import train
from eval import eval_global

num_cpus = 8; num_gpus=0
n_jobs=-1

def single_sample(name_model, nbNeurons, activations, loss, name_init, params_init, seed, metrics, x_train, y_train,
algo, eps, max_epochs, lr, seuil, f1, f2, rho, eps_egd, lambd, beta_1, beta_2, epsilon, amsgrad, type, sample_weight,
name_eval,x_test,y_test,transformerY=None,sample_weight_eval=None):

    dico = {}

    #build the model
    model = build_model(name_model,nbNeurons,activations,loss,name_init,params_init,seed,metrics)
    model.summary()
    print(model.get_weights()[0].dtype)

    #train the model
    model, epochs, norme_grad, cost_final, temps = train(algo,model, loss,x_train,y_train,eps,max_epochs,
    lr,seuil,f1,f2,rho,eps_egd,lambd,beta_1,beta_2,epsilon,amsgrad,type,sample_weight)
    dico['num_tirage'] = seed
    dico['epochs'] = epochs
    dico['time_train'] = temps
    dico['norme_grad'] = norme_grad.numpy()
    dico['cost_train'] = cost_final.numpy()

    #Compute the test cost
    pred = model(x_test,training=False)
    cost_test = loss(y_test,pred,sample_weight)
    dico['cost_test'] = cost_test.numpy()
    dico["prop_entropie"] = 0

    #Compute the metrics for train set
    model.reset_metrics()
    measures, temps_forward = eval_global(name_eval,model,x_train,y_train,transformerY,sample_weight_eval)
    for key in measures.keys():
        dico[key+"_train"] = measures[key]

    #Compute the metrics for test set
    model.reset_metrics()
    measures, temps_forward = eval_global(name_eval,model,x_test,y_test,transformerY,sample_weight_eval)
    for key in measures.keys():
        dico[key+"_test"] = measures[key]
    dico['temps_forward'] = temps_forward/x_test.shape[0]

    return dico

def tirages(tirageMin, nbTirages,
    name_model, nbNeurons, activations, loss, name_init, params, metrics, x_train, y_train,
    algo, eps, max_epochs, lr, seuil, f1, f2, rho, eps_egd, lambd, beta_1, beta_2, epsilon, amsgrad, type, sample_weight,
    name_eval,x_test,y_test,transformerY=None,sample_weight_eval=None):

    with parallel_backend('loky', n_jobs=n_jobs):
        res = Parallel()(delayed(single_sample)(name_model, nbNeurons, activations, loss, name_init, params, i, metrics, x_train, y_train,
        algo, eps, max_epochs, lr, seuil, f1, f2, rho, eps_egd, lambd, beta_1, beta_2, epsilon, amsgrad, type, sample_weight,
        name_eval,x_test,y_test,transformerY,sample_weight_eval) for i in range(tirageMin, tirageMin+nbTirages))
    
    """res = [single_sample(name_model, nbNeurons, activations, loss, name_init, params, i, metrics, x_train, y_train,
        algo, eps, max_epochs, lr, seuil, f1, f2, rho, eps_egd, lambd, beta_1, beta_2, epsilon, amsgrad, sample_weight,
        name_eval,x_test,y_test,transformerY,sample_weight_eval) for i in range(tirageMin, tirageMin+nbTirages)]"""
    return res


def minsRecordRegression(studies, folder, fileEnd, eps):
    infoFile = open("Record/"+folder+"/info_"+fileEnd,"w")
    allinfoFile = open("Record/"+folder+"/allinfo_"+fileEnd,"w")
    nonFile = open("Record/"+folder+"/nonConv_"+fileEnd,"w")

    div=0; nonConv=0

    nbTirages = len(studies)
    for i in range(nbTirages):
        if(isnan(studies[i]['norme_grad'])==False and isinf(studies[i]['norme_grad'])==False and studies[i]['norme_grad']<eps):
            for value in studies[i].values():
                infoFile.write(ffs(value) + "\n")
                allinfoFile.write(ffs(value) + "\n")
        else:
            if(studies[i]['norme_grad']>1000 or isnan(studies[i]['norme_grad']) or isinf(studies[i]['norme_grad'])):
                div+=1
                nonFile.write("-3" + "\n"); nonFile.write(ffs(studies[i]['prop_entropie']) + "\n")
            else:
                nonConv+=1
                nonFile.write("-2" + "\n"); nonFile.write(ffs(studies[i]['prop_entropie']) + "\n")
                for value in studies[i].values():
                    allinfoFile.write(ffs(value) + "\n")
    
    infoFile.write(ffs(nonConv/nbTirages) + "\n")
    infoFile.write(ffs(div/nbTirages) + "\n")

    print("Proportion de divergence: ", div/nbTirages)
    print("Proportion de non convergence: ", nonConv/nbTirages)

    infoFile.close(); allinfoFile.close(); nonFile.close()
        
def informationFile(tirageMin,nbTirages,name_model, nbNeurons, activations, name_init, params_init,
    PTrain, PTest,
    algo, eps, max_epochs, lr, seuil, f1, f2, rho, eps_egd, lambd, beta_1, beta_2, epsilon):

    if(name_model=="FC"):
        arch=""
        for l in range(len(activations)):
            arch += str(nbNeurons[l+1])
            arch += "("+ activations[l] + ")"
    else:
        arch = name_model

    finParameters = ", PTrain=" + str(PTrain) + ", PTest= " + str(PTest) + ", tirageMin=" + str(tirageMin) + ", nbTirages=" + str(nbTirages) + ", eps=" +str(eps) + ", maxEpoch=" +str(max_epochs) + ")"
    if(algo == "LC_EGD" or algo == "LC_EGD2" or algo == "LC_EGD_Adam"):
        parameters = "(eta="+str(lr)+", f1=" + str(f1) + ", f2=" + str(f2) + ", lambd=" + str(lambd)
    elif(algo=="Momentum"):
        parameters = "(eta="+str(lr)+", b1=" + str(beta_1)
    elif(algo=="RMSProp"):
        parameters = "(eta="+str(lr)+", b2=" + str(beta_2)
    elif(algo=="Adam"):
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


