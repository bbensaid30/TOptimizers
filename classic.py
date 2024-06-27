from cmath import isnan
from tensorflow.keras import optimizers
import tensorflow as tf
import time
import numpy as np

def Adam(model,loss_fn,
x,y, eps, max_epochs, lr=0.001, beta_1=0.9, beta_2=0.999,epsilon=1e-07,amsgrad=False,sample_weight=1):

    optimizer = optimizers.Adam(learning_rate=lr,beta_1=beta_1,beta_2=beta_2,epsilon=epsilon,amsgrad=amsgrad)
    norme_grad=1000; epoch=0; cost=1000
    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs):
        # Iterate over the batches of the dataset.

        with tf.GradientTape() as tape:
            prediction = model(x, training=True)
            cost = loss_fn(y, prediction,sample_weight=sample_weight)
        if(epoch==0):
            print("cost_init: ", cost)
            pass

        grads = tape.gradient(cost, model.trainable_weights)
        norme_grad = tf.linalg.global_norm(grads)
        if(norme_grad<eps):
            break
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        epoch+=1

        if epoch % 2 == 0:
            print(
                "Training loss (for one batch) at step %d: %.8f"
                % (epoch, float(cost))
            )
            print("grad: ", norme_grad)
    

    """ print("epochs: ", epoch)
    print("grad_norm: ", norme_grad)
    print("cost_final: ", cost) """

    end_time = time.time()

    return model, epoch, norme_grad, cost, end_time-start_time

def RMSProp(model,loss_fn,
x,y, eps, max_epochs, lr=0.001, beta_2=0.9, epsilon=1e-07, sample_weight=1):

    optimizer = optimizers.RMSprop(learning_rate=lr, rho=beta_2, epsilon=epsilon)
    norme_grad=1000; epoch=0
    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs):
        # Iterate over the batches of the dataset.

        with tf.GradientTape() as tape:

        # Run the forward pass of the layer.
        # The operations that the layer applies.
        # to its inputs are going to be recorded
        # on the GradientTape.
            prediction = model(x, training=True)  # Logits for this minibatch

        # Compute the loss value for this minibatch.
            cost = loss_fn(y, prediction,sample_weight=sample_weight)
        if(epoch==0):
            print("cost_init: ", cost)

        grads = tape.gradient(cost, model.trainable_weights)
        norme_grad = tf.linalg.global_norm(grads)
        if(norme_grad<eps):
            break

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        epoch+=1

        # Log every 100 batches.
        if epoch % 2 == 0:
            print(
                "Training loss (for one batch) at step %d: %.8f"
                % (epoch, float(cost))
            )
            print("grad: ", norme_grad)
            #print("Seen so far: %s samples" % ((step + 1) * batch_size))

    print("epochs: ", epoch)
    print("grad_norm: ", norme_grad)
    print("cost_final: ", cost)

    end_time = time.time()

    return model, epoch, norme_grad, cost, end_time-start_time

def Momentum(model,loss_fn,
x,y, eps, max_epochs, lr=0.01, beta_1=0.9, sample_weight=1):

    optimizer = optimizers.SGD(learning_rate=lr,momentum=beta_1)
    norme_grad=1000; epoch=0
    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs):
        with tf.GradientTape() as tape:
            prediction = model(x, training=True)
            cost = loss_fn(y, prediction,sample_weight=sample_weight)
        if(epoch==0):
            print("cost_init: ", cost)

        grads = tape.gradient(cost, model.trainable_weights)
        norme_grad = tf.linalg.global_norm(grads)
        if(norme_grad<eps):
            break
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        epoch+=1

        if epoch % 2 == 0:
            print(
                "Training loss (for one batch) at step %d: %.8f"
                % (epoch, float(cost))
            )
            print("grad: ", norme_grad)

    print("epochs: ", epoch)
    print("grad_norm: ", norme_grad)
    print("cost_final: ", cost)

    end_time = time.time()

    return model, epoch, norme_grad, cost, end_time-start_time
