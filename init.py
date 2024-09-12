from keras import initializers
import tensorflow as tf
import numpy as np 

""" class ZerosType(initializers.Initializer):

    def __init__(self, type):
      self.type=type

    def __call__(self, shape, dtype=None):
      if(self.type=="float32"):
        return tf.zeros(shape, dtype=tf.float32)
      elif(self.type=="float64"):
        return tf.zeros(shape, dtype=tf.float64)

    def get_config(self):  # To support serialization
      return {'type': self.type}

class Uniform(initializers.Initializer):

    def __init__(self, params, seed, type):
      self.minval = params[0]
      self.maxval = params[1]
      self.seed=seed
      self.type=type

    def __call__(self, shape, dtype=None):
      if(self.type=="float32"):
        return tf.random.uniform(shape, minval=self.minval, maxval=self.maxval, seed=self.seed, dtype=tf.float32)
      elif(self.type=="float64"):
        return tf.random.uniform(shape, minval=self.minval, maxval=self.maxval, seed=self.seed, dtype=tf.float64)

    def get_config(self): 
      return {'minval': self.minval, 'maxeval': self.maxval, 'seed': self.seed, 'type': self.type}

class Normal(initializers.Initializer):

    def __init__(self, params, seed, type):
      self.mean = params[0]
      self.std = params[1]
      self.seed = seed
      self.type=type

    def __call__(self, shape,dtype=None):
      if(self.type=="float32"):
        return tf.random.normal(shape, mean=self.mean, stddev=self.std, seed=self.seed, dtype=tf.float32)
      elif(self.type=="float64"):
        return tf.random.normal(shape, mean=self.mean, stddev=self.std, seed=self.seed, dtype=tf.float64)

    def get_config(self):  # To support serialization
      return {'mean': self.mean, 'std': self.std, 'seed': self.seed, 'type': self.type}

class Xavier(initializers.Initializer):

    def __init__(self, n, seed, type):
      self.n = n
      self.seed = seed
      self.type=type

    def __call__(self, shape, dtype=None):
      sigma=np.sqrt(1/self.n)
      if(self.type=="float32"):
        return tf.random.normal(shape, mean=0, stddev=sigma, seed=self.seed, dtype=tf.float32)
      elif(self.type=="float64"):
        return tf.random.normal(shape, mean=0, stddev=sigma, seed=self.seed, dtype=tf.float64)

    def get_config(self):
      return {'n': self.n, 'seed': self.seed, 'type': self.type}

class Bengio(initializers.Initializer):

    def __init__(self, n, m, seed, type):
      self.n=n
      self.m=m
      self.seed = seed
      self.type=type

    def __call__(self, shape, dtype=None):
      a=-np.sqrt(6/(self.n+self.m)); b=-a
      if(self.type=="float32"):
        return tf.random.uniform(shape, minval=a, maxval=b, seed=self.seed, dtype=tf.float32)
      elif(self.type=="float64"):
        return tf.random.uniform(shape, minval=a, maxval=b, seed=self.seed, dtype=tf.float64)

    def get_config(self):
      return {'n': self.n, 'm': self.m, 'seed': self.seed, 'type': self.type} """

def init(name, seed, params=[-1,1]):
    if(name=="Uniform"):
        return initializers.RandomUniform(minval=params[0], maxval=params[1], seed=seed)
    elif(name=="Normal"):
        return initializers.RandomNormal(mean=params[0], stddev=params[1], seed=seed)
    elif(name=="Xavier"):
        return initializers.LecunNormal(seed=seed)
    elif(name=="Bengio"):
        return initializers.GlorotUniform(seed=seed)

