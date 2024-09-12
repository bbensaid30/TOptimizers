from cmath import isnan
from keras import optimizers
import tensorflow as tf
import time
import numpy as np
from utils import gradSum, gradDot, gradDiff, gradDiv, gradMul
from linesearch import search_batch, search_dichotomy_batch


def RAG(model, loss_fn, train_dataset, PTrain, eps, max_epochs, lr=0.1, f1=30, f2=10000, lambd=0.5, type="float32"):
    optimizer = optimizers.SGD(lr)

    m=train_dataset.cardinality()
    coeff_max=2*m-1; heuris_max=2.0
    heuris=1.0

    if(type=="float64"):
        epsilon_machine = np.finfo(np.float64).eps
        coeff_max=tf.cast(coeff_max, tf.float64)
    elif(type=="float32"):
        epsilon_machine = np.finfo(np.float32).eps
        coeff_max=tf.cast(coeff_max, tf.float32)
    coeff=coeff_max

    epoch=0
    eta=lr; eta0=lr; eta1=lr; eta_start=lr
    dist=0
    R=0 
    total_iterLoop=0
    nbLoops=2
    diffs=np.zeros(m)

    start_time = time.time()

    weight_prec = model.get_weights()
    g, grads, gNorm, LTab, R_tab, R, total_iterLoop = init_RAG(model, optimizer, loss_fn, train_dataset, weight_prec, m, PTrain, eps, epsilon_machine, lr, f1, lambd)
    
    epoch+=1
    while((gNorm/PTrain>eps or dist/PTrain>eps) and epoch<=max_epochs):
        R_epoch=R
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            weight_prec = model.get_weights()

            with tf.GradientTape() as tape:
                prediction = model(x_batch_train, training=True)
                cost_prec = loss_fn(y_batch_train, prediction)
            grad = tape.gradient(cost_prec, model.trainable_weights)

            R-=R_tab[step]; R+=cost_prec; R_tab[step]=cost_prec
            gradDiff(g,grads[step]); gradSum(g,grad); grads[step]=grad
            grad_square=tf.linalg.global_norm(grad)**2; gNorm=tf.linalg.global_norm(g)
            prod=gradDot(grad,g,type)

            eta0=eta_start; eta1=eta_start
            condition=(grad_square>eps*eps)
            if(condition):
                eta0, iterLoop = search_dichotomy_batch(model, x_batch_train, y_batch_train, optimizer, loss_fn, grad, weight_prec, cost_prec, lambd, grad_square, f1, eta0, epsilon_machine, nbLoops)
                total_iterLoop+=iterLoop

            imax = np.argmax(LTab)

            if(prod>epsilon_machine and step==imax and prod<gNorm*gNorm):
                eta1, iterLoop = search_dichotomy_batch(model, x_batch_train, y_batch_train, optimizer, loss_fn, g, weight_prec, cost_prec, lambd, prod, f1, eta1, epsilon_machine, nbLoops)
                total_iterLoop+=iterLoop
                eta=max(eta0,eta1)
            else:
                eta=eta0
            
            if(grad_square<eps*eps or eta<epsilon_machine):
                LTab[step]=0
            else:
                LTab[step]=2*(1-lambd)/eta
            LSum=np.sum(LTab); LMax=np.max(LTab)

            if(LSum<epsilon_machine):
                eta=lr; eta_start=lr
            else:
                eta_start=f2*(2*(1-lambd))/LMax
                if(tf.abs(coeff-coeff_max)<epsilon_machine):
                    eta=2/(coeff*LSum)
                else:
                    eta=2/(heuris*coeff*LSum)
            
            optimizer.learning_rate=eta; 
            #print("dtheta: ",  eta*tf.linalg.global_norm(g))
            optimizer.apply_gradients(zip(g, model.trainable_weights))

            dist-=diffs[step]
            if(grad_square>eps*eps and eta0>epsilon_machine):
                diffs[step]=(2*(1-lambd)/eta0)*eta*gNorm
            else:
                diffs[step]=0
            dist+=diffs[step]

            if(gNorm/PTrain<eps and dist/PTrain<eps):
                break
        
        if(dist<gNorm):
            heuris=(heuris+heuris_max)/2
        else:
            heuris=(1+heuris)/2
        if(R-R_epoch<0):
            coeff/=heuris
        else:
            coeff=coeff_max
    
        epoch+=1

        if(epoch%1==0):
            print("epoch:", epoch)
            print("gradNorm: ", gNorm/PTrain)
            print("R: ", R/PTrain)
            print("eta: ", eta)

    end_time=time.time()

    R=0
    for (x_batch_train,y_batch_train) in train_dataset:
        prediction = model(x_batch_train, training=False)
        R += loss_fn(y_batch_train, prediction)

    return model, epoch, gNorm/PTrain, R/PTrain, end_time-start_time

def RAGL(model, loss_fn, train_dataset, PTrain, eps, max_epochs, lr=0.1, f1=30, f2=10000, lambd=0.5, type="float32"):
    optimizer = optimizers.SGD(lr)

    m=train_dataset.cardinality()
    coeff_max=4*m-1; heuris_max=2.0
    heuris=1.0

    if(type=="float64"):
        epsilon_machine = np.finfo(np.float64).eps
        coeff_max=tf.cast(coeff_max, tf.float64)
    elif(type=="float32"):
        epsilon_machine = np.finfo(np.float32).eps
        coeff_max=tf.cast(coeff_max, tf.float32)
    coeff=coeff_max

    epoch=0
    eta=lr; eta0=lr; eta1=lr; eta_start=lr
    dist=0
    R=0
    iterLoop=0; total_iterLoop=0
    nbLoops=2; last_pass=False
    LTab = np.zeros(m); R_tab=np.zeros(m)
    
    start_time = time.time()

    weight_prec = model.get_weights()
    g, weight_inter, gNorm, LTab, R_tab, R, total_iterLoop = init_RAGL(model, optimizer, loss_fn, train_dataset, weight_prec, m, PTrain, eps, epsilon_machine, lr, f1, lambd)
    
    epoch+=1
    while((gNorm/PTrain>eps or dist/PTrain>eps) and epoch<=max_epochs):
        dist=0; R_epoch=R

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            
            weight_prec=model.get_weights()
            
            if(step<m-1):
                model.set_weights(weight_inter)
                with tf.GradientTape() as tape:
                    prediction = model(x_batch_train, training=True)
                    cost_prec = loss_fn(y_batch_train, prediction)
                gs = tape.gradient(cost_prec, model.trainable_weights)
                model.set_weights(weight_prec)

            with tf.GradientTape() as tape:
                prediction = model(x_batch_train, training=True)
                cost_prec = loss_fn(y_batch_train, prediction)
            grad = tape.gradient(cost_prec, model.trainable_weights)
            R-=R_tab[step]; R+=cost_prec; R_tab[step]=cost_prec
            
            if(step==0):
                gSum=grad
            else:
                gradSum(gSum,grad)
            if(step<m-1):
                gradDiff(g,gs); gradSum(g,grad)
            else:
                g=gSum

            grad_square=tf.linalg.global_norm(grad)**2; gNorm=tf.linalg.global_norm(g); prod=gradDot(grad,g,type)

            eta0=eta_start; eta1=eta_start
            condition=(grad_square>eps*eps)
            if(condition):
                eta0, iterLoop = search_dichotomy_batch(model, x_batch_train, y_batch_train, optimizer, loss_fn, grad, weight_prec, cost_prec, lambd, grad_square, f1, eta0, epsilon_machine, nbLoops)
                total_iterLoop+=iterLoop
            total_iterLoop+=iterLoop

            imax = np.argmax(LTab)

            if(prod>epsilon_machine and step==imax and prod<gNorm*gNorm):
                eta1, iterLoop = search_dichotomy_batch(model, x_batch_train, y_batch_train, optimizer, loss_fn, g, weight_prec, cost_prec, lambd, prod, f1, eta1, epsilon_machine, nbLoops)
                total_iterLoop+=iterLoop
                eta=max(eta0,eta1)
            else:
                eta=eta0
            
            if(grad_square<eps*eps or eta<epsilon_machine):
                LTab[step]=0
            else:
                LTab[step]=2*(1-lambd)/eta
            LSum=np.sum(LTab); LMax=np.max(LTab)

            if(LSum<epsilon_machine):
                eta=lr; eta_start=lr
            else:
                eta_start=f2*(2*(1-lambd))/LMax
                if(tf.abs(coeff-coeff_max)<epsilon_machine):
                    eta=2/(coeff*LSum)
                else:
                    eta=2/(heuris*coeff*LSum)

            if(grad_square>eps*eps and eta0>epsilon_machine):
                dist+=(2*(1-lambd)/eta0)*eta*gNorm

            if(step==0):
                LMax_now=LTab[0]; imax_now=0
            else:
                if(LTab[step]>LMax_now):
                    LMax_now=LTab[step]; imax_now=step

            if(step==imax_now):
                weight_memory=model.get_weights()
            
            optimizer.learning_rate=eta; optimizer.apply_gradients(zip(g, model.trainable_weights))

        weight_inter=weight_memory
        
        if(dist<gNorm):
            heuris=(heuris+heuris_max)/2
        else:
            heuris=(1+heuris)/2
        if(R-R_epoch<0):
            coeff/=heuris
        else:
            coeff=coeff_max
    
        epoch+=1

        if(epoch%1==0):
            print("epoch:", epoch)
            print("gradNorm: ", gNorm/PTrain)
            print("R: ", R/PTrain)
            print("eta: ", eta)

    end_time=time.time()

    R=0
    for step, (x_batch_train,y_batch_train) in enumerate(train_dataset):
        prediction = model(x_batch_train, training=False)
        R += loss_fn(y_batch_train, prediction)

    return model, epoch, gNorm/PTrain, R/PTrain, end_time-start_time

def init_RAG(model, optimizer, loss_fn, train_dataset, weight_prec, m, PTrain, eps, epsilon_machine, lr, f1, lambd):
    R=0; total_iterLoop=0
    LTab = np.zeros(m)
    R_tab=np.zeros(m)
    grads=[]

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            prediction = model(x_batch_train, training=True)
            cost_prec = loss_fn(y_batch_train, prediction)
        total_iterLoop+=1
        grad = tape.gradient(cost_prec, model.trainable_weights)
        R_tab[step]=cost_prec; R+=cost_prec

        if(step==0):
            g=grad
        else:
            gradSum(g,grad)
        grads.append(grad[:])
        grad_square = tf.linalg.global_norm(grad)**2

        eta=lr
        condition=(grad_square>eps*eps)
        if(condition):
            eta, iterLoop = search_batch(model, x_batch_train, y_batch_train, optimizer, loss_fn, grad, weight_prec, cost_prec, lambd, grad_square, f1, eta)
            total_iterLoop+=iterLoop
        if(grad_square<eps*eps or eta<epsilon_machine):
            LTab[step]=0
        else:
            LTab[step]=2*(1-lambd)/eta
    gNorm=tf.linalg.global_norm(g)
    if(gNorm/PTrain>eps):
        LSum=np.sum(LTab)
        if(LSum<epsilon_machine):
            eta=lr
        else:
            eta=2*(1-lambd)/LSum
        optimizer.learning_rate=eta; optimizer.apply_gradients(zip(g, model.trainable_weights))

    return g, grads, gNorm, LTab, R_tab, R, total_iterLoop

def init_RAGL(model, optimizer, loss_fn, train_dataset, weight_prec, m, PTrain, eps, epsilon_machine, lr, f1, lambd):
    R=0; total_iterLoop=0
    LTab = np.zeros(m)
    R_tab=np.zeros(m)

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            prediction = model(x_batch_train, training=True)
            cost_prec = loss_fn(y_batch_train, prediction)
        total_iterLoop+=1
        grad = tape.gradient(cost_prec, model.trainable_weights)
        R_tab[step]=cost_prec; R+=cost_prec

        if(step==0):
            g=grad
        else:
            gradSum(g,grad)
        grad_square = tf.linalg.global_norm(grad)**2

        eta=lr
        condition=(grad_square>eps*eps)
        while condition:
            optimizer.learning_rate=eta; optimizer.apply_gradients(zip(grad, model.trainable_weights))
            with tf.GradientTape() as tape:
                prediction = model(x_batch_train, training=False)
                cost = loss_fn(y_batch_train, prediction)
            condition=(cost-cost_prec>-lambd*eta*grad_square)
            if condition:
                eta/=f1
            model.set_weights(weight_prec)
            total_iterLoop+=1
        if(grad_square<eps*eps or eta<epsilon_machine):
            LTab[step]=0
        else:
            LTab[step]=2*(1-lambd)/eta

    weight_inter=model.get_weights()
    gNorm=tf.linalg.global_norm(g)
    if(gNorm/PTrain>eps):
        LSum=np.sum(LTab)
        if(LSum<epsilon_machine):
            eta=lr
        else:
            eta=2*(1-lambd)/LSum
        optimizer.learning_rate=eta; optimizer.apply_gradients(zip(g, model.trainable_weights))

    return g, weight_inter, gNorm, LTab, R_tab, R, total_iterLoop