from keras import optimizers
import tensorflow as tf
from tensorflow.experimental import numpy as tnp
import numpy as np
import time
import copy

from linesearch import search_full, search_dichotomy_full
from linesearch import search_normalized_full, search_normalized_dichotomy_full

from custom_opti import CustomMomentum, CustomRMSProp


def LC_GD(model, loss_fn,
x,y, eps, max_epochs, lr=0.1, f1=2, f2=10000, lambd=0.5, typef=tf.float32, sample_weight=None):

    optimizer = optimizers.SGD(lr)
    norme_grad=tf.constant(1000,dtype=typef); epoch=0; active_security=False
    lr_min=tnp.finfo(typef).eps

    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs and tf.math.is_nan(norme_grad)==False):
        if(epoch==0):
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                cost = loss_fn(y, prediction,sample_weight=sample_weight)
            grads = tape.gradient(cost, model.trainable_weights)
            norme_grad = tf.linalg.global_norm(grads); V_dot = norme_grad**2
            if(norme_grad<eps):
                break
            epoch+=1

        cost_prec = cost
        weight_n = [tf.identity(w) for w in model.trainable_weights]
        lr, cost, grads, _ = search_full(model, x, y, optimizer, loss_fn, sample_weight, grads, weight_n, cost_prec, lambd, V_dot, lr, lr_min, f1)
        #update the weights and the gradient

        if(lr<lr_min):
            if(lambd!=0):
                active_security=True
                lambd=0
            else:
                break

        lr*=f2

        norme_grad= tf.linalg.global_norm(grads); V_dot=norme_grad**2

        epoch+=1    

    end_time = time.time()

    return model, epoch, norme_grad, cost, end_time-start_time,active_security

def LCD_GD(model, loss_fn,
x,y, eps, max_epochs, lr=0.1, f1=30, f2=10000, lambd=0.5, typef=tf.float32, sample_weight=None):

    optimizer = optimizers.SGD(lr)
    norme_grad=tf.constant(1000,dtype=typef); epoch=0; active_security=False
    lr_min=tnp.finfo(typef).eps
    nbLoops=int(np.log10(f2)/np.log10(f1))

    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs and tf.math.is_nan(norme_grad)==False):

        if(epoch==0):
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)

                cost = loss_fn(y, prediction,sample_weight=sample_weight)

            grads = tape.gradient(cost, model.trainable_weights)
            norme_grad = tf.linalg.global_norm(grads); V_dot = norme_grad**2;
            if(norme_grad<eps):
                break

        cost_prec = cost
        weight_n = [tf.identity(w) for w in model.trainable_weights]
        
        lr, cost, grads, _ = search_dichotomy_full(model, x, y, optimizer, loss_fn, sample_weight, grads, weight_n, cost_prec, lambd, V_dot, f1, lr, lr_min, nbLoops)
        if(lr<lr_min):
            if(lambd!=0):
                active_security=True
                lambd=0
            else:
                break

        lr*=f2

        norme_grad= tf.linalg.global_norm(grads); V_dot=norme_grad**2

        epoch+=1    
        
    end_time = time.time()

    return model, epoch, norme_grad, cost, end_time-start_time, active_security

def LC_Mom(model, loss_fn,
x,y, eps, max_epochs, lr=0.1, f1=2, f2=10000, lambd=0.5, beta_1=0.9, typef=tf.float32, sample_weight=None):

    optimizer = CustomMomentum(lr,beta_1)
    optimizer.build(model.trainable_weights)
    norme_grad=tf.constant(1000,dtype=typef); epoch=0; active_security=False
    lr_min=tnp.finfo(typef).eps

    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs and tf.math.is_nan(norme_grad)==False):
        if(epoch==0):
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                cost = loss_fn(y, prediction,sample_weight=sample_weight)
            grads = tape.gradient(cost, model.trainable_weights)
            norme_grad = tf.linalg.global_norm(grads)
            v_norm2 = 0
            V=cost+0.5*v_norm2; V_dot=(1-beta_1)*v_norm2
            if(norme_grad<eps):
                break
            epoch+=1

        #linesearch for Momentum
        optimizer.learning_rate=lr
        V_prec = V
        weight_n = [tf.identity(w) for w in model.trainable_weights]
        v_prev = [tf.identity(v) for v in optimizer.v]
        condition=True; iterLoop=0
        while(condition):
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                cost = loss_fn(y, prediction, sample_weight=sample_weight)
            v_norm2 = tf.linalg.global_norm(optimizer.v)**2
            V=cost+0.5*v_norm2; V_dot=(1-beta_1)*v_norm2
            condition = (V-V_prec>-lambd*V_dot and lr>lr_min)
            if(condition):
                lr/=f1; optimizer.learning_rate=lr
                for w,val in zip(model.trainable_weights, weight_n):
                    w.assign(val)
                for var,val in zip(optimizer.v, v_prev):
                    var.assign(val)
            iterLoop+=1

        if(lr<lr_min):
            if(lambd!=0):
                active_security=True
                lambd=0
            else:
                break

        lr*=f2

        grads = tape.gradient(cost, model.trainable_weights)
        norme_grad= tf.linalg.global_norm(grads)

        epoch+=1    

    end_time = time.time()

    return model, epoch, norme_grad, cost, end_time-start_time,active_security

def LCD_Mom(model, loss_fn,
x,y, eps, max_epochs, lr=0.1, f1=30, f2=10000, lambd=0.5, beta_1=0.9, typef=tf.float32, sample_weight=None):

    optimizer = CustomMomentum(lr,beta_1)
    optimizer.build(model.trainable_weights)
    norme_grad=tf.constant(1000,dtype=typef); epoch=0; active_security=False
    lr_min=tnp.finfo(typef).eps
    nbLoops=int(np.log10(f2)/np.log10(f1))

    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs and tf.math.is_nan(norme_grad)==False):
        if(epoch==0):
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                cost = loss_fn(y, prediction,sample_weight=sample_weight)
            grads = tape.gradient(cost, model.trainable_weights)
            norme_grad = tf.linalg.global_norm(grads)
            v_norm2 = 0
            V=cost+0.5*v_norm2; V_dot=(1-beta_1)*v_norm2
            if(norme_grad<eps):
                break
            epoch+=1

        #linesearch for Momentum
        optimizer.learning_rate=lr
        V_prec = V
        weight_n = [tf.identity(w) for w in model.trainable_weights]
        v_prev = [tf.identity(v) for v in optimizer.v]
        condition=True; iterLoop=0
        while(condition):
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                cost = loss_fn(y, prediction, sample_weight=sample_weight)
            v_norm2 = tf.linalg.global_norm(optimizer.v)**2
            V=cost+0.5*v_norm2; V_dot=(1-beta_1)*v_norm2
            condition = (V-V_prec>-lambd*V_dot and lr>lr_min)
            if(condition):
                lr/=f1; optimizer.learning_rate=lr
                for w,val in zip(model.trainable_weights, weight_n):
                    w.assign(val)
                for var,val in zip(optimizer.v, v_prev):
                    var.assign(val)
            iterLoop+=1
        if(iterLoop>1):
            droite = np.log10(lr*f1); gauche = np.log10(lr)
            for k in range(nbLoops):
                milieu = (gauche+droite)/2; lr=10**milieu; optimizer.learning_rate = lr
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                with tf.GradientTape() as tape:
                    prediction = model(x, training=True)
                    cost = loss_fn(y, prediction, sample_weight=sample_weight)
                v_norm2 = tf.linalg.global_norm(optimizer.v)**2
                V=cost+0.5*v_norm2; V_dot=(1-beta_1)*v_norm2
                condition = (V-V_prec>-lambd*V_dot)
                if(condition):
                    m_best = gauche
                    droite = milieu
                    last_pass=False
                else:
                    gauche = milieu
                    last_pass=True
                if(k<nbLoops-1):
                    for w,val in zip(model.trainable_weights, weight_n):
                        w.assign(val)
                    for var,val in zip(optimizer.v, v_prev):
                        var.assign(val)
                else:
                    if(last_pass==False):
                        lr=10**m_best; optimizer.learning_rate = lr
                        for w,val in zip(model.trainable_weights, weight_n):
                            w.assign(val)
                        for var,val in zip(optimizer.v, v_prev):
                            var.assign(val)
                        optimizer.apply_gradients(zip(grads, model.trainable_weights))
                        with tf.GradientTape() as tape:
                            prediction = model(x, training=True)
                            cost = loss_fn(y, prediction, sample_weight=sample_weight)
                        v_norm2 = tf.linalg.global_norm(optimizer.v)**2
                        V=cost+0.5*v_norm2
                iterLoop+=1

        if(lr<lr_min):
            if(lambd!=0):
                active_security=True
                lambd=0
            else:
                break

        lr*=f2

        grads = tape.gradient(cost, model.trainable_weights)
        norme_grad= tf.linalg.global_norm(grads)

        epoch+=1    

    end_time = time.time()

    return model, epoch, norme_grad, cost, end_time-start_time,active_security

def LC_RMS(model, loss_fn,
x,y, eps, max_epochs, lr=0.1, f1=2, f2=10000, lambd=0.5, beta_2=0.999, eps_a=1e-10, typef=tf.float32, sample_weight=None):

    optimizer = CustomRMSProp(lr,beta_2,eps_a)
    optimizer.build(model.trainable_weights)
    norme_grad=tf.constant(1000,dtype=typef); epoch=0; active_security=False
    lr_min=tnp.finfo(typef).eps

    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs and tf.math.is_nan(norme_grad)==False):
        if(epoch==0):
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                cost = loss_fn(y, prediction,sample_weight=sample_weight)
            grads = tape.gradient(cost, model.trainable_weights)
            norme_grad = tf.linalg.global_norm(grads)
            
            if(norme_grad<eps):
                break
            epoch+=1

        #linesearch for Momentum
        optimizer.learning_rate=lr
        cost_prec = cost
        weight_n = [tf.identity(w) for w in model.trainable_weights]
        s_prev = [tf.identity(s) for s in optimizer.s]
        condition=True; iterLoop=0
        while(condition):
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                cost = loss_fn(y, prediction, sample_weight=sample_weight)
            V_dot = 0
            for i, grad in enumerate(grads):
                V_dot+=tf.norm(grad/tf.pow(eps_a+optimizer.s[i],0.25))**2
            condition = (cost-cost_prec>-lambd*lr*V_dot and lr>lr_min)
            if(condition):
                lr/=f1; optimizer.learning_rate=lr
                for w,val in zip(model.trainable_weights, weight_n):
                    w.assign(val)
                for var,val in zip(optimizer.s, s_prev):
                    var.assign(val)
            iterLoop+=1

        if(lr<lr_min):
            if(lambd!=0):
                active_security=True
                lambd=0
            else:
                break

        lr*=f2

        grads = tape.gradient(cost, model.trainable_weights)
        norme_grad= tf.linalg.global_norm(grads)

        epoch+=1    

    end_time = time.time()

    return model, epoch, norme_grad, cost, end_time-start_time,active_security

def LCD_RMS(model, loss_fn,
x,y, eps, max_epochs, lr=0.1, f1=30, f2=10000, lambd=0.5, beta_2=0.999, eps_a=1e-10, typef=tf.float32, sample_weight=None):

    optimizer = CustomRMSProp(lr,beta_2,eps_a)
    optimizer.build(model.trainable_weights)
    norme_grad=tf.constant(1000,dtype=typef); epoch=0; active_security=False
    lr_min=tnp.finfo(typef).eps
    nbLoops=int(np.log10(f2)/np.log10(f1))

    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs and tf.math.is_nan(norme_grad)==False):
        if(epoch==0):
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                cost = loss_fn(y, prediction,sample_weight=sample_weight)
            grads = tape.gradient(cost, model.trainable_weights)
            norme_grad = tf.linalg.global_norm(grads)
            
            if(norme_grad<eps):
                break
            epoch+=1

        #linesearch for Momentum
        optimizer.learning_rate=lr
        cost_prec = cost
        weight_n = [tf.identity(w) for w in model.trainable_weights]
        s_prev = [tf.identity(s) for s in optimizer.s]
        condition=True; iterLoop=0
        while(condition):
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                cost = loss_fn(y, prediction, sample_weight=sample_weight)
            V_dot = 0
            for i, grad in enumerate(grads):
                V_dot+=tf.norm(grad/tf.pow(eps_a+optimizer.s[i],0.25))**2
            condition = (cost-cost_prec>-lambd*lr*V_dot and lr>lr_min)
            if(condition):
                lr/=f1; optimizer.learning_rate=lr
                for w,val in zip(model.trainable_weights, weight_n):
                    w.assign(val)
                for var,val in zip(optimizer.s, s_prev):
                    var.assign(val)
            iterLoop+=1
        if(iterLoop>1):
            droite = np.log10(lr*f1); gauche = np.log10(lr)
            for k in range(nbLoops):
                milieu = (gauche+droite)/2; lr=10**milieu; optimizer.learning_rate = lr
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                with tf.GradientTape() as tape:
                    prediction = model(x, training=True)
                    cost = loss_fn(y, prediction, sample_weight=sample_weight)
                V_dot = 0
                for i, grad in enumerate(grads):
                    V_dot+=tf.norm(grad/tf.pow(eps_a+optimizer.s[i],0.25))**2
                condition = (cost-cost_prec>-lambd*lr*V_dot)
                if(condition):
                    m_best = gauche
                    droite = milieu
                    last_pass=False
                else:
                    gauche = milieu
                    last_pass=True
                if(k<nbLoops-1):
                    for w,val in zip(model.trainable_weights, weight_n):
                        w.assign(val)
                    for var,val in zip(optimizer.s, s_prev):
                        var.assign(val)
                else:
                    if(last_pass==False):
                        lr=10**m_best; optimizer.learning_rate = lr
                        for w,val in zip(model.trainable_weights, weight_n):
                            w.assign(val)
                        for var,val in zip(optimizer.s, s_prev):
                            var.assign(val)
                        optimizer.apply_gradients(zip(grads, model.trainable_weights))
                        with tf.GradientTape() as tape:
                            prediction = model(x, training=True)
                            cost = loss_fn(y, prediction, sample_weight=sample_weight)
                iterLoop+=1

        if(lr<lr_min):
            if(lambd!=0):
                active_security=True
                lambd=0
            else:
                break

        lr*=f2

        grads = tape.gradient(cost, model.trainable_weights)
        norme_grad= tf.linalg.global_norm(grads)

        epoch+=1    

    end_time = time.time()

    return model, epoch, norme_grad, cost, end_time-start_time,active_security

def LCD_NG(model, loss_fn,
x,y, eps, max_epochs, lr=0.1, f1=30, f2=10000, lambd=0.5, typef=tf.float32, sample_weight=None):

    optimizer = optimizers.SGD(lr)
    norme_grad=tf.constant(1000,dtype=typef); epoch=0; active_security=False
    lr_min=tnp.finfo(typef).eps
    nbLoops=int(np.log10(f2)/np.log10(f1))

    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs and tf.math.is_nan(norme_grad)==False):

        if(epoch==0):
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)

                cost = loss_fn(y, prediction,sample_weight=sample_weight)

            grads = tape.gradient(cost, model.trainable_weights)
            norme_grad = tf.linalg.global_norm(grads)
            if(norme_grad<eps):
                break

        cost_prec = cost
        weight_n = copy.deepcopy(model.get_weights())
        
        lr, cost, grads, _ = search_normalized_dichotomy_full(model, x, y, optimizer, loss_fn, sample_weight, grads, weight_n, cost_prec, lambd, norme_grad, f1, lr, lr_min, nbLoops)

        if(lr<lr_min):
            if(lambd!=0):
                active_security=True
                lambd=0
            else:
                break

        lr*=f2

        norme_grad= tf.linalg.global_norm(grads)

        epoch+=1    
        
    end_time = time.time()

    return model, epoch, norme_grad, cost, end_time-start_time, active_security

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

            grads = tape.gradient(cost, model.trainable_weights)
            norme_grad = tf.linalg.global_norm(grads)
            if(norme_grad<eps):
                break

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

    end_time = time.time()

    return model_inter, epoch, norme_grad, cost, end_time-start_time, False
