import classic
import perso

import Sclassic
import incremental
import variance_reduction

def train(algo,model,loss_fn,
x,y, eps, max_epochs, lr=0.1, weight_decay=0.004, f1=2, f2=10000, lambd=0.5, 
beta_1 = 0.9, beta_2=0.999, epsilon_a=1e-07, amsgrad=False, typef="float32", sample_weight=1):
    if(algo=="LC_GD"):
        return perso.LC_GD(model,loss_fn,x,y,eps,max_epochs,lr,f1,f2,lambd,typef,sample_weight)
    elif(algo=="LCD_NG"):
        return perso.LCD_NG(model,loss_fn,x,y,eps,max_epochs,lr,f1,f2,lambd,typef,sample_weight)
    elif(algo=="LCD_GD"):
        return perso.LCD_GD(model,loss_fn,x,y,eps,max_epochs,lr,f1,f2,lambd,typef,sample_weight)
    elif(algo=="LC_Mom"):
        return perso.LC_Mom(model,loss_fn,x,y,eps,max_epochs,lr,f1,f2,lambd,beta_1,typef,sample_weight)
    elif(algo=="LCD_Mom"):
        return perso.LCD_Mom(model,loss_fn,x,y,eps,max_epochs,lr,f1,f2,lambd,beta_1,typef,sample_weight)
    elif(algo=="LC_RMS"):
        return perso.LC_RMS(model,loss_fn,x,y,eps,max_epochs,lr,f1,f2,lambd,beta_2,epsilon_a,typef,sample_weight)
    elif(algo=="LCD_RMS"):
        return perso.LCD_RMS(model,loss_fn,x,y,eps,max_epochs,lr,f1,f2,lambd,beta_2,epsilon_a,typef,sample_weight)
    elif(algo=="Momentum"):
        return classic.Momentum(model,loss_fn,x,y,eps,max_epochs,lr,beta_1,sample_weight)
    elif(algo=="Adam"):
        return classic.Adam(model,loss_fn,x,y,eps,max_epochs,lr,beta_1,beta_2,epsilon_a,amsgrad,sample_weight)
    elif(algo=="AdamW"):
        return classic.AdamW(model,loss_fn,x,y,eps,max_epochs,lr,weight_decay,beta_1,beta_2,epsilon_a,amsgrad,sample_weight)
    elif(algo=="GD"):
        return classic.GD(model,loss_fn,x,y,eps,max_epochs,lr,sample_weight)
    elif(algo=="GD_clip"):
        return classic.GD_clip(model,loss_fn,x,y,eps,max_epochs,lr,sample_weight)
    
def train_sto(algo, model,loss_fn,
train_dataset, PTrain, eps, max_epochs, lr=0.1, f1=2, f2=10000, lambd=0.5, 
beta_1 = 0.9, beta_2=0.999, epsilon_a=1e-07, amsgrad=False, typef="float32"):
    if(algo=="RRAdam"):
        return Sclassic.RRAdam(model, loss_fn, train_dataset, PTrain, eps, max_epochs, lr, beta_1, beta_2, epsilon_a, amsgrad)
    elif(algo=="RRGD"):
        return Sclassic.RRGD(model, loss_fn, train_dataset, PTrain, eps, max_epochs, lr)
    elif(algo=="RR_rms"):
        return Sclassic.RR_rms(model, loss_fn, train_dataset, PTrain, eps, max_epochs, lr)
    elif(algo=="GD_batch"):
        return Sclassic.GD_batch(model, loss_fn, train_dataset, PTrain, eps, max_epochs, lr)
    elif(algo=="Streaming_SVRG"):
        return variance_reduction.Streaming_SVRG(model, loss_fn, train_dataset, PTrain, max_epochs, lr, typef)
    elif(algo=="RAG"):
        return incremental.RAG(model, loss_fn, train_dataset, PTrain, eps, max_epochs, lr, f1, f2, lambd, typef)
    elif(algo=="RAGL"):
        return incremental.RAGL(model, loss_fn, train_dataset, PTrain, eps, max_epochs, lr, f1, f2, lambd, typef)
    elif(algo=="GD_estim"):
        return incremental.GD_estim(model, loss_fn, train_dataset, PTrain, eps, max_epochs, lr, f1, f2, lambd, typef)
    elif(algo=="IAG"):
        return incremental.IAG(model, loss_fn, train_dataset, PTrain, max_epochs, lr)
