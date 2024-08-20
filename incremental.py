from cmath import isnan
from tensorflow.keras import optimizers
import tensorflow as tf
import time
import numpy as np
from utils import gradTotInit, gradSum, gradDiv, gradDot, gradDiff

def RAG(model, loss_fn, train_dataset, PTrain, eps, max_epochs, lr=0.1, f1=2, f2=10000, lambd=0.5, type="float32"):
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
    eta=lr; eta0=lr; eta1=lr; eta_start=lr; cost=0; costPrec=0
    LMax=0; LSum=0; dist=0; grad_square=0; gNorm=1000; prod=0
    R=0; R_epoch=0; iterLoop=0; total_iterLoop=0
    imax=0
    gauche=0; droite=0; milieu=0; m_best=0
    nLoops=2; last_pass=False
    LTab = np.zeros(m); diffs=np.zeros(m); R_tab=np.zeros(m)
    grads=[]

    start_time = time.time()

    weight_prec = model.get_weights()
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            prediction = model(x_batch_train, training=True)
            cost_prec = loss_fn(y_batch_train, prediction)
        total_iterLoop+=1
        grad = tape.gradient(cost_prec, model.trainable_weights)
        R_tab[step]=cost_prec; R+=cost_prec; diffs[step]=0

        if(step==0):
            g=grad
        else:
            gradSum(g,grad)
        grads.append(grad[:])
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
        if(grad_square*grad_square<eps*eps or eta<epsilon_machine):
            LTab[step]=0
        else:
            LTab[step]=2*(1-lambd)/eta

    gNorm=tf.linalg.global_norm(g)
    if(gNorm/PTrain>eps):
        LSum=np.sum(LTab); LMax=np.max(LTab)
        if(LSum<epsilon_machine):
            eta=lr
        else:
            eta=2*(1-lambd)/LSum
        optimizer.learning_rate=eta; optimizer.apply_gradients(zip(g, model.trainable_weights))
    
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
            grad_square=tf.linalg.global_norm(grad)**2; gNorm=tf.linalg.global_norm(g); prod=gradDot(grad,g)

            eta0=eta_start; eta1=eta_start
            condition=(grad_square>eps*eps)
            iterLoop=0
            while(condition):
                optimizer.learning_rate=eta0; optimizer.apply_gradients(zip(grad, model.trainable_weights))
                with tf.GradientTape() as tape:
                    prediction = model(x_batch_train, training=False)
                    cost = loss_fn(y_batch_train, prediction)
                condition=(cost-cost_prec>-lambd*eta0*grad_square)
                if condition:
                    eta0/=f1
                model.set_weights(weight_prec)
                iterLoop+=1
            if(iterLoop>1 and eta0>epsilon_machine):
                droite = np.log10(eta0*f1); gauche = np.log10(eta0)
                for k in range(nLoops):
                    milieu = (gauche+droite)/2; eta0=10**milieu; 
                    optimizer.learning_rate = eta0; optimizer.apply_gradients(zip(grad, model.trainable_weights))
                    with tf.GradientTape() as tape:
                        prediction = model(x_batch_train, training=True)
                        cost = loss_fn(y_batch_train, prediction)
                    if(cost-cost_prec>-lambd*eta0*grad_square):
                        m_best = gauche
                        droite = milieu
                        last_pass=False
                    else:
                        gauche = milieu
                        last_pass=True
                    iterLoop+=1
                    model.set_weights(weight_prec)
                if(not last_pass):
                    eta0=10**m_best
            total_iterLoop+=iterLoop

            imax = np.argmax(LTab)

            if(prod>epsilon_machine and step==imax and prod<gNorm*gNorm):
                condition=True
                iterLoop=0
                while(condition):
                    optimizer.learning_rate=eta1; optimizer.apply_gradients(zip(g, model.trainable_weights))
                    with tf.GradientTape() as tape:
                        prediction = model(x_batch_train, training=False)
                        cost = loss_fn(y_batch_train, prediction)
                    condition=(cost-cost_prec>-lambd*eta1*prod)
                    if condition:
                        eta1/=f1
                    model.set_weights(weight_prec)
                    iterLoop+=1
                if(iterLoop>1 and eta1>epsilon_machine):
                    droite = np.log10(eta1*f1); gauche = np.log10(eta1)
                    for k in range(nLoops):
                        milieu = (gauche+droite)/2; eta1=10**milieu; 
                        optimizer.learning_rate = eta1; optimizer.apply_gradients(zip(g, model.trainable_weights))
                        with tf.GradientTape() as tape:
                            prediction = model(x_batch_train, training=False)
                            cost = loss_fn(y_batch_train, prediction)
                        if(cost-cost_prec>-lambd*eta1*prod):
                            m_best = gauche
                            droite = milieu
                            last_pass=False
                        else:
                            gauche = milieu
                            last_pass=True
                        iterLoop+=1
                        model.set_weights(weight_prec)
                    if(not last_pass):
                        eta1=10**m_best
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
            
            optimizer.learning_rate=eta; optimizer.apply_gradients(zip(g, model.trainable_weights))

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

        if(epoch%5==0):
            print("epoch:", epoch)
            print("gradNorm: ", gNorm/PTrain)
            print("R: ", R/PTrain)

    end_time=time.time()

    return model, epoch, gNorm/PTrain, R/PTrain, end_time-start_time
