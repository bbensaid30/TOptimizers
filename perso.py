from cmath import isnan
from keras import optimizers
import tensorflow as tf
import time
import numpy as np
import copy

from linesearch import search_full, search_dichotomy_full
from linesearch import search_normalized_full, search_normalized_dichotomy_full
from classic import Adam


def LC_EGD(model, loss_fn,
x,y, eps, max_epochs, lr=0.1, f1=2, f2=10000, lambd=0.5, typef="float32", sample_weight=None):

    optimizer = optimizers.SGD(lr)
    norme_grad=1000; epoch=0; active_Adam=False; epoch_Adam=0
    if(typef=="float32"):
        epsilon_machine = np.finfo(np.float32).eps
    elif(typef=="float64"):
        epsilon_machine = np.finfo(np.float64).eps
    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs):
        if(epoch==0):
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                cost = loss_fn(y, prediction,sample_weight=sample_weight)
                print("cost_init: ", cost)
            grads = tape.gradient(cost, model.trainable_weights)
            norme_grad = tf.linalg.global_norm(grads); V_dot = norme_grad**2
            if(norme_grad<eps):
                break
            print("grad_init: ", norme_grad)
            epoch+=1

        cost_prec = cost
        weight_n = model.get_weights()
        lr, cost, grads, iterLoop = search_full(model, x, y, optimizer, loss_fn, sample_weight, grads, weight_n, cost_prec, lambd, V_dot, lr, f1)
        #update the weights and the gradient

        if(lr*norme_grad<epsilon_machine):
            active_Adam=True
            break

        lr*=f2

        norme_grad= tf.linalg.global_norm(grads); V_dot=norme_grad**2

        epoch+=1    

        if epoch % 1 == 0:
            print("\nStart of epoch %d" % (epoch,))
            print(
                "Training loss (for one batch) at epoch %d: %.8f"
                % (epoch, float(cost))
            )
            print("grad: ", norme_grad)

    end_time = time.time()

    if(active_Adam==True):
        print("Active Adam")
        model,epoch_Adam, norme_grad,cost,_ = Adam(model,loss_fn,x,y,eps,max_epochs-epoch)

    return model, epoch+epoch_Adam, norme_grad, cost, end_time-start_time

def LC_EGD2(model, loss_fn,
x,y, eps, max_epochs, lr=0.1, f1=30, f2=10000, lambd=0.5, type="float32", sample_weight=None):

    optimizer = optimizers.SGD(lr)
    if(type=="float32"):
        epsilon_machine = np.finfo(np.float32).eps
    elif(type=="float64"):
        epsilon_machine = np.finfo(np.float64).eps
    norme_grad=1000; epoch=0; active_Adam=False; epoch_Adam=0
    nbLoops=3
    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs):

        if(epoch==0):
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)

                cost = loss_fn(y, prediction,sample_weight=sample_weight)
                print("cost_init: ", cost)

            grads = tape.gradient(cost, model.trainable_weights)
            norme_grad = tf.linalg.global_norm(grads); V_dot = norme_grad**2;
            if(norme_grad<eps):
                break
            print("grad_init: ", norme_grad)

        cost_prec = cost
        weight_n = model.get_weights()
        
        lr, cost, grads, iterLoop = search_dichotomy_full(model, x, y, optimizer, loss_fn, sample_weight, grads, weight_n, cost_prec, lambd, V_dot, f1, lr, nbLoops)
        if(lr*norme_grad<epsilon_machine):
            active_Adam=True; break

        print("lr: ", lr)
        lr*=f2

        norme_grad= tf.linalg.global_norm(grads); V_dot=norme_grad**2

        epoch+=1    

        if epoch % 1 == 0:
            print("\Epoch %d" % (epoch,))
            print("Loss: ", cost)
            print("grad: ", norme_grad)
        
    if(active_Adam):
        print("Adam débute")
        model,epoch_Adam, norme_grad,cost,time_Adam= Adam(model,loss_fn,x,y,eps,max_epochs,0.001,0.9,0.999,10**(-7),False,sample_weight)

    end_time = time.time()

    return model, epoch+epoch_Adam, norme_grad, cost, end_time-start_time


def LC_NGD(model, loss_fn,
x,y, eps, max_epochs, lr=0.1, f1=30, f2=10000, lambd=0.5, typef="float32", sample_weight=None):

    optimizer = optimizers.SGD(lr)
    if(typef=="float32"):
        epsilon_machine = np.finfo(np.float32).eps
    elif(typef=="float64"):
        epsilon_machine = np.finfo(np.float64).eps
    norme_grad=1000; epoch=0; active_Adam=False; epoch_Adam=0
    nbLoops=3
    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs):

        if(epoch==0):
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)

                cost = loss_fn(y, prediction,sample_weight=sample_weight)
                print("cost_init: ", cost)

            grads = tape.gradient(cost, model.trainable_weights)
            norme_grad = tf.linalg.global_norm(grads)
            if(norme_grad<eps):
                break
            print("grad_init: ", norme_grad)

        cost_prec = cost
        weight_n = copy.deepcopy(model.get_weights())
        
        lr, cost, grads, iterLoop = search_normalized_dichotomy_full(model, x, y, optimizer, loss_fn, sample_weight, grads, weight_n, cost_prec, lambd, norme_grad, f1, lr, nbLoops)

        if(lr*norme_grad<epsilon_machine):
            active_Adam=False; break

        print("lr: ", lr)
        lr*=f2

        norme_grad= tf.linalg.global_norm(grads)

        epoch+=1    

        if epoch % 1 == 0:
            print("\Epoch %d" % (epoch,))
            print("Loss: ", cost)
            print("grad: ", norme_grad)
        
    if(active_Adam):
        print("Adam débute")
        model,epoch_Adam, norme_grad,cost,time_Adam= Adam(model,loss_fn,x,y,eps,max_epochs,0.001,0.9,0.999,10**(-7),False,sample_weight)

    end_time = time.time()

    return model, epoch+epoch_Adam, norme_grad, cost, end_time-start_time

def ER(model,model_inter,loss_fn,
x,y, eps, max_epochs, lr=0.1, seuil=0.01, sample_weight=None):

    optimizer = optimizers.SGD(lr)
    norme_grad=1000; epoch=0
    model_inter.set_weights(model.get_weights())
    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs):

        if(epoch==0):
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)  # Logits for this minibatch
                cost = loss_fn(y, prediction,sample_weight=sample_weight)
                print("cost_init: ", cost)

            grads = tape.gradient(cost, model.trainable_weights)
            norme_grad = tf.linalg.global_norm(grads)
            if(norme_grad<eps):
                break
            #print("grad_init: ", norme_grad)

            optimizer.apply_gradients(zip(grads, model_inter.trainable_weights))
            with tf.GradientTape() as tape:
                prediction = model_inter(x, training=True)  # Logits for this minibatch
                cost = loss_fn(y, prediction,sample_weight=sample_weight)
            grads_inter = tape.gradient(cost, model_inter.trainable_weights)
        
        e = (lr*tf.linalg.global_norm([a-b for a,b in zip(grads,grads_inter)]))/(2*seuil)

        if(e>1):
            lr*=0.9/np.sqrt(e); optimizer.learning_rate=lr
        else:
            optimizer.apply_gradients(zip(grads_inter, model.trainable_weights))
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)  # Logits for this minibatch
                cost = loss_fn(y, prediction,sample_weight=sample_weight)
            grads = tape.gradient(cost, model.trainable_weights)
            lr*=0.9/np.sqrt(e); optimizer.learning_rate=lr
        
        model_inter.set_weights(model.get_weights())
        optimizer.apply_gradients(zip(grads, model_inter.trainable_weights))
        with tf.GradientTape() as tape:
                prediction = model_inter(x, training=True)  # Logits for this minibatch
                cost = loss_fn(y, prediction,sample_weight=sample_weight)
        grads_inter = tape.gradient(cost, model_inter.trainable_weights)
        norme_grad = tf.linalg.global_norm(grads_inter)

        epoch+=1

        if epoch % 2 == 0:
            print("\nStart of epoch %d" % (epoch,))
            print(
                "Training loss (for one batch) at epoch %d: %.8f"
                % (epoch, float(cost))
            )
            print("grad: ", norme_grad)
            print("lr: ", lr)
            #print("Seen so far: %s samples" % ((step + 1) * batch_size))

    print("epochs: ", epoch)
    print("grad_norm: ", norme_grad)
    print("cost_final: ", cost)

    end_time = time.time()

    return model_inter, epoch, norme_grad, cost, end_time-start_time
