import numpy as np
import tensorflow as tf

#modify the weights and the gradient
def search_full(model, x, y, optimizer, loss_fn, sample_weight, grads, weight_n, cost_prec, lambd, V_dot, lr, f1):
    optimizer.learning_rate=lr
    condition=True; iterLoop=0
    while(condition):
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        with tf.GradientTape() as tape:
            prediction = model(x, training=True)
            cost = loss_fn(y, prediction, sample_weight=sample_weight)
        condition = (cost-cost_prec>-lambd*lr*V_dot)
        if(condition):
            lr/=f1; optimizer.learning_rate=lr
            model.set_weights(weight_n)
        iterLoop+=1
    grads = tape.gradient(cost, model.trainable_weights)
    return lr, cost, grads, iterLoop

def search_dichotomy_full(model, x, y, optimizer, loss_fn, sample_weight, grads, weight_n, cost_prec, lambd, V_dot, f1, lr, nbLoops=3):
    optimizer.learning_rate=lr
    condition=True; iterLoop=0
    while(condition):
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        with tf.GradientTape() as tape:
            prediction = model(x, training=True)
            cost = loss_fn(y, prediction, sample_weight=sample_weight)
        condition = cost-cost_prec>-lambd*lr*V_dot
        if(condition):
            lr/=f1; optimizer.learning_rate=lr
            model.set_weights(weight_n)
        iterLoop+=1
    if(iterLoop>1):
        droite = np.log10(lr*f1); gauche = np.log10(lr)
        for k in range(nbLoops):
            milieu = (gauche+droite)/2; lr=10**milieu; optimizer.learning_rate = lr
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                cost = loss_fn(y, prediction, sample_weight=sample_weight)
            if(cost-cost_prec>-lambd*lr*V_dot):
                m_best = gauche
                droite = milieu
                last_pass=False
            else:
                gauche = milieu
                last_pass=True
            if(k<nbLoops-1):
                model.set_weights(weight_n)
            else:
                if(last_pass==False):
                    lr=10**m_best; optimizer.learning_rate = lr
                    model.set_weights(weight_n)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                    with tf.GradientTape() as tape:
                        prediction = model(x, training=True)
                        cost = loss_fn(y, prediction, sample_weight=sample_weight)
            iterLoop+=1
    grads = tape.gradient(cost, model.trainable_weights)
    return lr, cost, grads, iterLoop

#do not modify the weights and the gradients

def search_batch(model, x, y, optimizer, loss_fn, grads, weight_n, cost_prec, lambd, V_dot, f1, lr):
    optimizer.learning_rate=lr
    condition=True; iterLoop=0
    while(condition):
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        prediction = model(x, training=False)
        cost = loss_fn(y, prediction)
        condition = cost-cost_prec>-lambd*lr*V_dot
        if(condition):
            lr/=f1; optimizer.learning_rate=lr
        model.set_weights(weight_n)
        iterLoop+=1
    return lr, iterLoop

def search_dichotomy_batch(model, x, y, optimizer, loss_fn, grads, weight_n, cost_prec, lambd, V_dot, f1, lr, epsilon_machine, nbLoops=2):
    lr, iterLoop = search_batch(model, x, y, optimizer, loss_fn, grads, weight_n, cost_prec, lambd, V_dot, f1, lr)
    if(iterLoop>1 and lr>epsilon_machine):
        droite = np.log10(lr*f1); gauche = np.log10(lr)
        for k in range(nbLoops):
            milieu = (gauche+droite)/2; lr=10**milieu; optimizer.learning_rate = lr
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            prediction = model(x, training=False)
            cost = loss_fn(y, prediction)
            if(cost-cost_prec>-lambd*lr*V_dot):
                m_best = gauche
                droite = milieu
                last_pass=False
            else:
                gauche = milieu
                last_pass=True
            iterLoop+=1
            model.set_weights(weight_n)
            if(not last_pass):
                lr=10**m_best
    return lr, iterLoop
        
