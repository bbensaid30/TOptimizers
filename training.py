import classic
import perso
import essai


def train(algo,model,loss_fn,
x,y, eps, max_epochs, lr=0.1, seuil=0.01, f1=2, f2=10000, rho=0.9, eps_egd=0.01, lambd=0.5, 
beta_1 = 0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, sample_weight=1):
    if(algo=="LC_EGD"):
        return perso.LC_EGD(model,loss_fn,x,y,eps,max_epochs,lr,f1,f2,lambd,sample_weight)
    elif(algo=="LC_EGD2"):
        return perso.LC_EGD2(model,loss_fn,x,y,eps,max_epochs,lr,f1,f2,lambd,sample_weight)
    elif(algo=="Momentum"):
        return classic.Momentum(model,loss_fn,x,y,eps,max_epochs,lr,beta_1,sample_weight)
    elif(algo=="Adam"):
        return classic.Adam(model,loss_fn,x,y,eps,max_epochs,lr,beta_1,beta_2,epsilon,amsgrad,sample_weight)