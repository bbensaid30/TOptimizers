import os
os.chdir("/home/bbensaid/Documents/Anabase/TOptimizers")

import numpy as np
from joblib import Parallel, delayed, parallel_backend
import tensorflow as tf

import io
import datetime
from tqdm import tqdm
import traceback   
import gc

import json
from filelock import FileLock

from model import build_model
from training import train
from eval import eval_global

num_cpus = 4; num_gpus=2
n_jobs=3

def append_to_jsonl(dico, filename="results.jsonl"):
    def safe_convert(obj):
        # 1. Gérer les types numériques NumPy/TF
        if hasattr(obj, "numpy"): # Pour les tenseurs TF
            return obj.numpy().tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # 2. Gérer les fonctions ou objets complexes
        if callable(obj) or "tensorflow" in str(type(obj)):
            return str(obj) # On convertit en texte
            
        return obj

    lock = FileLock(filename + ".lock")
    with lock:
        with open(filename, "a", encoding="utf-8") as f:
            # On utilise le paramètre 'default' pour appliquer safe_convert
            line = json.dumps(dico, default=safe_convert)
            f.write(line + "\n")

#os.environ['OMP_NUM_THREADS'] = '1'
def single_sample_json(filename, name_model, nbNeurons, activations, loss, name_init, params_init, seed, metrics, x_train, y_train,
algo, eps, max_epochs, lr, seuil, f1, f2, rho, eps_egd, lambd, beta_1, beta_2, epsilon, amsgrad, typef, sample_weight,
name_eval,x_test,y_test,transformerY=None,sample_weight_eval=None):
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # Option A: Allocation dynamique (recommandé)
                tf.config.experimental.set_memory_growth(gpu, True)
                # Option B: Fixer une limite stricte (ex: 2Go par process)
                # tf.config.set_logical_device_configuration(gpu, 
                # [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
        except RuntimeError as e:
            print(e)

    try:
        dico = {'name_model': name_model, 'opti': algo}

        dico["name_init"]=name_init
        dico["eps"]=eps
        dico["max_epochs"]=max_epochs
        dico["lr"]=lr; dico["f1"]=f1; dico["f2"]=f2; dico["lambd"]=lambd
        dico["beta1"]=beta_1; dico["beta2"]=beta_2; dico["epsilon_a"]=epsilon
        dico["status"]="pending"
        dico["timestamp"] = datetime.datetime.now().isoformat()


        #build the model
        model = build_model(name_model,nbNeurons,activations,loss,name_init,params_init,seed,metrics)
        #model.summary()
        # On stocke l'architecture ici
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        dico["architecture"] = stream.getvalue()

        #train the model
        results=train(algo,model, loss,x_train,y_train,eps,max_epochs,
        lr,seuil,f1,f2,rho,eps_egd,lambd,beta_1,beta_2,epsilon,amsgrad,typef,sample_weight)
        if results is None:
            raise ValueError("La fonction train() a renvoyé None")
        model, epochs, norme_grad, cost_final, temps, active_security = results
        if np.isnan(cost_final.numpy()):
            raise ValueError("Le coût est devenu NaN - Explosion du gradient")
        
        dico['num_tirage'] = seed
        dico['epochs'] = epochs
        dico['time_train'] = temps
        dico['norme_grad'] = norme_grad.numpy()
        dico['cost_train'] = cost_final.numpy()
        dico['active_security']=active_security

        #Compute the test cost
        pred = model(x_test,training=False)
        cost_test = loss(y_test,pred,sample_weight)
        dico['cost_test'] = cost_test.numpy()

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

        dico['status'] = "success"
        append_to_jsonl(dico, filename)
    
    except Exception as e:
        # En cas de crash, on capture l'erreur sans arrêter les autres jobs
        dico['status'] = "error"
        dico['error_message'] = str(e)
        dico['num_tirage']=seed
        dico['traceback'] = traceback.format_exc()
        
        append_to_jsonl(dico, filename)

    finally:
        # Nettoyage explicite
        tf.keras.backend.clear_session()
        gc.collect()

    return dico

def tirages_json(filename, tirageMin, nbTirages,
    name_model, nbNeurons, activations, loss, name_init, params, metrics, x_train, y_train,
    algo, eps, max_epochs, lr, seuil, f1, f2, rho, eps_egd, lambd, beta_1, beta_2, epsilon, amsgrad, typef, sample_weight,
    name_eval,x_test,y_test,transformerY=None,sample_weight_eval=None):

    """ with parallel_backend('loky', n_jobs=n_jobs):
        res = Parallel()(delayed(single_sample_json)(
            filename, name_model, nbNeurons, activations, loss, name_init, params, i, metrics, x_train, y_train,
            algo, eps, max_epochs, lr, seuil, f1, f2, rho, eps_egd, lambd, beta_1, beta_2, epsilon, amsgrad, typef, sample_weight,
            name_eval,x_test,y_test,transformerY,sample_weight_eval) for i in tqdm(range(tirageMin, tirageMin + nbTirages), desc="Tirages en cours")) """

    # On définit le nombre de tâches par worker avant de le tuer
    with Parallel(n_jobs=n_jobs, max_tasks_per_child=1) as parallel:
        res = parallel(
        delayed(single_sample_json)(filename,
            name_model, nbNeurons, activations, loss, name_init, params, i, 
            metrics, x_train, y_train, algo, eps, max_epochs, lr, seuil, 
            f1, f2, rho, eps_egd, lambd, beta_1, beta_2, epsilon, amsgrad, 
            typef, sample_weight, name_eval, x_test, y_test, 
            transformerY, sample_weight_eval
        ) for i in tqdm(range(tirageMin, tirageMin + nbTirages), desc="Tirages en cours")
    )
    
    return res
