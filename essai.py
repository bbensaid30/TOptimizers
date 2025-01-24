from cmath import isnan
from tensorflow.keras import optimizers
import tensorflow as tf
import time
import numpy as np

def LC_EM(model,model_copy,loss_fn,
x,y, eps, max_epochs, lr=0.1, beta_1=0.1, f1=2, f2=10000, lambd=0.5,sample_weight=1):

    optimizer = optimizers.SGD(lr)
    beta_bar = lr/beta_1
    norme_grad=1000; epoch=0
    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs):

        if(epoch==0):
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                cost = loss_fn(y, prediction,sample_weight=sample_weight)
                print("cost_init: ", cost)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(cost, model.trainable_weights); L=len(grads)
            v = [0*grads[i] for i in range(L)]
            norme_grad = tf.linalg.global_norm(grads)
            E = beta_bar*cost; V_dot=0
            if(norme_grad<eps):
                break
            print("grad_init: ", norme_grad)

        E_prec = E
        model_copy.set_weights(model.get_weights()); v_prec=v
        condition=True
        iterLoop=0
        while(condition):
            v = [(1-beta_1)*v_prec[i] + beta_1*grads[i] for i in range(L)]
            optimizer.apply_gradients(zip(v, model.trainable_weights))
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                cost = loss_fn(y, prediction, sample_weight=sample_weight)
            vsquare = tf.linalg.global_norm(v)**2
            E = beta_bar*cost+0.5*vsquare
            condition = E-E_prec>-lambd*lr*V_dot
            if(condition):
                lr/=f1; beta_1/=f1; optimizer.learning_rate=lr
                model.set_weights(model_copy.get_weights()); v=v_prec
            iterLoop+=1
        #print(iterLoop)
        lr*=f2; beta_1*=f2; optimizer.learning_rate=lr

        grads = tape.gradient(cost, model.trainable_weights)
        norme_grad= tf.linalg.global_norm(grads); V_dot=vsquare

        epoch+=1    

        if epoch % 2 == 0:
            print("\nStart of epoch %d" % (epoch,))
            print(
                "Training loss (for one batch) at epoch %d: %.8f"
                % (epoch, float(cost))
            )
            print("grad: ", norme_grad)
            print("lr: ", lr)
            print("dimE: ", E-E_prec)
            print('top: ', -lambd*lr*V_dot)
            #print("Seen so far: %s samples" % ((step + 1) * batch_size))

    print("epochs: ", epoch)
    print("grad_norm: ", norme_grad)
    print("cost_final: ", cost)

    end_time = time.time()

    return model, epoch, norme_grad, cost, end_time-start_time

def LC_EM2(model,model_copy,loss_fn,
x,y, eps, max_epochs, lr=0.1, beta_1=0.1, rho=0.9, epsilon=0.01, lambd=0.5,sample_weight=1):

    optimizer = optimizers.SGD(lr)
    beta_bar = lr/beta_1
    norme_grad=1000; epoch=0
    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs):

        if(epoch==0):
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                cost = loss_fn(y, prediction,sample_weight=sample_weight)
                print("cost_init: ", cost)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(cost, model.trainable_weights); L=len(grads)
            v = [0*grads[i] for i in range(L)]
            norme_grad = tf.linalg.global_norm(grads)
            E = beta_bar*cost; V_dot=0
            if(norme_grad<eps):
                break
            print("grad_init: ", norme_grad)

        E_prec = E
        model_copy.set_weights(model.get_weights()); v_prec=v
        condition=True
        iterLoop=0
        while(condition):
            v = [(1-beta_1)*v_prec[i] + beta_1*grads[i] for i in range(L)]
            optimizer.apply_gradients(zip(v, model.trainable_weights))
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                cost = loss_fn(y, prediction, sample_weight=sample_weight)
            vsquare = tf.linalg.global_norm(v)**2
            E = beta_bar*cost+0.5*vsquare
            condition = E-E_prec>-lambd*lr*V_dot
            if(condition):
                f1 = (rho*(1-lambd)*V_dot)/max((E-E_prec)/lr+V_dot,epsilon*(1-lambd)*V_dot)
                lr*=f1; beta_1*=f1; optimizer.learning_rate=lr
                model.set_weights(model_copy.get_weights()); v=v_prec
            iterLoop+=1
        #print(iterLoop)
        f1 = (rho*(1-lambd)*V_dot)/max((E-E_prec)/lr+V_dot,epsilon*(1-lambd)*V_dot)
        lr*=f1; beta_1*=f1; optimizer.learning_rate=lr

        grads = tape.gradient(cost, model.trainable_weights)
        norme_grad= tf.linalg.global_norm(grads); V_dot=vsquare

        epoch+=1    

        if epoch % 2 == 0:
            print("\nStart of epoch %d" % (epoch,))
            print(
                "Training loss (for one batch) at epoch %d: %.8f"
                % (epoch, float(cost))
            )
            print("grad: ", norme_grad)
            print("lr: ", lr)
            print("dim: ", E-E_prec)
            print('top: ', -lambd*lr*V_dot)
            #print("Seen so far: %s samples" % ((step + 1) * batch_size))

    print("epochs: ", epoch)
    print("grad_norm: ", norme_grad)
    print("cost_final: ", cost)

    end_time = time.time()

    return model, epoch, norme_grad, cost, end_time-start_time