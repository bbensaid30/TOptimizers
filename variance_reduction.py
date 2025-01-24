import tensorflow as tf
from keras import optimizers
import numpy as np
import time

from utils import gradSum, gradDiv, gradDiff

def Streaming_SVRG(model,loss_fn, train_dataset, PTrain, max_epochs, lr=0.01, typef="float32"):
    
    optimizer = optimizers.SGD(learning_rate=lr)
    m=train_dataset.cardinality()
    b=PTrain/m; facteur=10; B=min(facteur*b,PTrain)
    loop=10

    epoch=0
    
    if(typef=="float32"):
        b=tf.cast(b, tf.float32); B=tf.cast(B, tf.float32)
    if(typef=="float64"):
        b=tf.cast(b, tf.float64); B=tf.cast(B, tf.float64)

    start_time = time.time()
    for step in range(m*max_epochs):

        if(step%loop==0):
            weight_tilde=model.get_weights(); R=0
            for j, (x,y) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    prediction = model(x, training=True)
                    cost = loss_fn(y, prediction)
                R+=cost
                grad = tape.gradient(cost, model.trainable_weights)
                if(j==0):
                    g=grad
                else:
                    gradSum(g,grad)
                if(j>=facteur or j==m):
                    break
            gradDiv(g,B); R/=B

        i=np.random.randint(m)

        for k, (x,y) in enumerate(train_dataset):
            if(k==i):
                weight_prec=model.get_weights()

                model.set_weights(weight_tilde)
                with tf.GradientTape() as tape:
                    prediction = model(x, training=True)
                    cost = loss_fn(y, prediction)
                grad_tilde = tape.gradient(cost, model.trainable_weights)

                model.set_weights(weight_prec)
                with tf.GradientTape() as tape:
                    prediction = model(x, training=True)
                    cost = loss_fn(y, prediction)
                grad = tape.gradient(cost, model.trainable_weights)

                gradDiff(grad, grad_tilde); gradDiv(grad, b)
                gradSum(grad, g)

                optimizer.apply_gradients(zip(grad, model.trainable_weights))

        if(step%m==0):
            epoch+=1
            print("epoch: ", epoch)
            print("cost_batch: ", cost)
            print("R: ", R)
    
    end_time = time.time()

    return model, max_epochs, tf.linalg.global_norm(g), R, end_time-start_time


