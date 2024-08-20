from cmath import isnan
from tensorflow.keras import optimizers
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
                Ri = loss_fn(y_batch_train, prediction)/len(x_batch_train) #normalization not in the loss (no use of mean square) 

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

        if(epoch%5==0):
            print("epoch:", epoch)
            print("gradNorm: ", norme_grad)
            print("R: ", R)

    end_time = time.time()

    return model, epoch, norme_grad, R, end_time-start_time
            
