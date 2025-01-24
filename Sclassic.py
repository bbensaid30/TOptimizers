from cmath import isnan
from keras import optimizers
import tensorflow as tf
import time
import numpy as np
from utils import gradTotInit, gradSum, gradDiv

def RRAdam(model,loss_fn,
train_dataset, PTrain, eps, max_epochs, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False):
    
    optimizer = optimizers.Adam(learning_rate=lr,beta_1=beta_1,beta_2=beta_2,epsilon=epsilon,amsgrad=amsgrad)
    norme_grad=1000; epoch=0
    start_time = time.time()

    while(norme_grad>eps and epoch<max_epochs):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:

                prediction = model(x_batch_train, training=True)
                Ri = loss_fn(y_batch_train, prediction)
                #Ri = loss_fn(y_batch_train, prediction)/len(x_batch_train)

            grad = tape.gradient(Ri, model.trainable_weights)
            optimizer.apply_gradients(zip(grad, model.trainable_weights))

        g = gradTotInit(grad)
        R=0
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:

                prediction = model(x_batch_train, training=True)
                Ri = loss_fn(y_batch_train, prediction)

            grad = tape.gradient(Ri, model.trainable_weights)
            gradSum(g,grad); R+=Ri
        gradDiv(g,PTrain); R/=PTrain

        norme_grad = tf.linalg.global_norm(g)
        epoch+=1

        if(epoch%1==0):
            print("epoch:", epoch)
            print("gradNorm: ", norme_grad)
            print("R: ", R)

    end_time = time.time()

    return model, epoch, norme_grad, R, end_time-start_time

def RR_rms(model,loss_fn, train_dataset, PTrain, eps, max_epochs, lr=0.01):
    
    optimizer = optimizers.RMSprop(learning_rate=lr)
    norme_grad=1000; epoch=0
    start_time = time.time()

    while(norme_grad>eps and epoch<max_epochs):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:

                prediction = model(x_batch_train, training=True)
                Ri = loss_fn(y_batch_train, prediction)
                #Ri = loss_fn(y_batch_train, prediction)/len(x_batch_train)

            grad = tape.gradient(Ri, model.trainable_weights)
            optimizer.apply_gradients(zip(grad, model.trainable_weights))
            
        g = gradTotInit(grad)
        R=0
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:

                prediction = model(x_batch_train, training=True)
                Ri = loss_fn(y_batch_train, prediction)

            grad = tape.gradient(Ri, model.trainable_weights)
            gradSum(g,grad); R+=Ri
        gradDiv(g,PTrain); R/=PTrain

        norme_grad = tf.linalg.global_norm(g)
        epoch+=1

        if(epoch%1==0):
            print("epoch:", epoch)
            print("gradNorm: ", norme_grad)
            print("R: ", R)

    end_time = time.time()

    return model, epoch, norme_grad, R, end_time-start_time
            
def RRGD(model,loss_fn, train_dataset, PTrain, eps, max_epochs, lr=0.01):
    
    optimizer = optimizers.SGD(learning_rate=lr)
    norme_grad=1000; epoch=0
    start_time = time.time()

    while(norme_grad>eps and epoch<max_epochs):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:

                prediction = model(x_batch_train, training=True)
                Ri = loss_fn(y_batch_train, prediction)
                #Ri = loss_fn(y_batch_train, prediction)/len(x_batch_train)

            grad = tape.gradient(Ri, model.trainable_weights)
            optimizer.apply_gradients(zip(grad, model.trainable_weights))
            
        g = gradTotInit(grad)
        R=0
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:

                prediction = model(x_batch_train, training=True)
                Ri = loss_fn(y_batch_train, prediction)

            grad = tape.gradient(Ri, model.trainable_weights)
            gradSum(g,grad); R+=Ri
        gradDiv(g,PTrain); R/=PTrain

        norme_grad = tf.linalg.global_norm(g)
        epoch+=1

        if(epoch%1==0):
            print("epoch:", epoch)
            print("gradNorm: ", norme_grad)
            print("R: ", R)

    end_time = time.time()

    return model, epoch, norme_grad, R, end_time-start_time

def GD_batch(model,loss_fn, train_dataset, PTrain, eps, max_epochs, lr=0.01):
    
    optimizer = optimizers.SGD(learning_rate=lr)
    norme_grad=1000; epoch=0
    start_time = time.time()

    m=train_dataset.cardinality()
    gradsNorm=np.zeros(m)

    while(norme_grad>eps and epoch<max_epochs):
        R=0
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:

                prediction = model(x_batch_train, training=True)
                Ri = loss_fn(y_batch_train, prediction)
            
            grad = tape.gradient(Ri, model.trainable_weights)
            gradsNorm[step]=tf.linalg.global_norm(grad)**2
            if(step==0):
                g=grad
            else:
                gradSum(g,grad)
            R+=Ri
        
        gradDiv(g,PTrain); R/=PTrain
        optimizer.apply_gradients(zip(g, model.trainable_weights))
       
        """ for i in range(len(grad)):
            print(grad[i].shape) """

        norme_grad = tf.linalg.global_norm(g)
        epoch+=1

        if(epoch%1==0):
            print("epoch:", epoch)
            print("gradNorm: ", norme_grad)
            print("R: ", R)

    end_time = time.time()

    return model, epoch, norme_grad, R, end_time-start_time