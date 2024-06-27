from keras import initializers
import numpy as np 

def Uniform(seed,params):
    return initializers.RandomUniform(minval=params[0],maxval=params[1],seed=seed)

def Normal(seed,params):
    initializers.RandomNormal(mean=params[0], stddev=params[1], seed=seed)

def Xavier(n,m,seed):
    sigma = np.sqrt(1/n)
    return initializers.RandomNormal(mean=0,stddev=sigma,seed=seed)

def Bengio(n,m,seed):
    a=-np.sqrt(6/(n+m)); b=-a
    return initializers.RandomUniform(minval=a,maxval=b,seed=seed)

def init(name,n,m,seed,params=[-1,1]):
    if(name=="Uniform"):
        return Uniform(seed,params)
    elif(name=="Normal"):
        return Normal(seed,params)
    elif(name=="Xavier"):
        return Xavier(n,m,seed)
    elif(name=="Bengio"):
        return Bengio(n,m,seed)

