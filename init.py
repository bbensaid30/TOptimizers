from keras import initializers
import tensorflow as tf
import numpy as np 

""" class ZerosType(initializers.Initializer):

    def __init__(self, typef):
      self.typef=typef

    def __call__(self, shape, dtype=None):
      if(self.typef=="float32"):
        return tf.zeros(shape, dtype=tf.float32)
      elif(self.typef=="float64"):
        return tf.zeros(shape, dtype=tf.float64)

    def get_config(self):  # To support serialization
      return {'typef': self.typef}

class Uniform(initializers.Initializer):

    def __init__(self, params, seed, typef):
      self.minval = params[0]
      self.maxval = params[1]
      self.seed=seed
      self.typef=typef

    def __call__(self, shape, dtype=None):
      if(self.typef=="float32"):
        return tf.random.uniform(shape, minval=self.minval, maxval=self.maxval, seed=self.seed, dtype=tf.float32)
      elif(self.typef=="float64"):
        return tf.random.uniform(shape, minval=self.minval, maxval=self.maxval, seed=self.seed, dtype=tf.float64)

    def get_config(self): 
      return {'minval': self.minval, 'maxeval': self.maxval, 'seed': self.seed, 'typef': self.typef}

class Normal(initializers.Initializer):

    def __init__(self, params, seed, typef):
      self.mean = params[0]
      self.std = params[1]
      self.seed = seed
      self.typef=typef

    def __call__(self, shape,dtype=None):
      if(self.typef=="float32"):
        return tf.random.normal(shape, mean=self.mean, stddev=self.std, seed=self.seed, dtype=tf.float32)
      elif(self.typef=="float64"):
        return tf.random.normal(shape, mean=self.mean, stddev=self.std, seed=self.seed, dtype=tf.float64)

    def get_config(self):  # To support serialization
      return {'mean': self.mean, 'std': self.std, 'seed': self.seed, 'typef': self.typef}

class Xavier(initializers.Initializer):

    def __init__(self, n, seed, typef):
      self.n = n
      self.seed = seed
      self.typef=typef

    def __call__(self, shape, dtype=None):
      sigma=np.sqrt(1/self.n)
      if(self.typef=="float32"):
        return tf.random.normal(shape, mean=0, stddev=sigma, seed=self.seed, dtype=tf.float32)
      elif(self.typef=="float64"):
        return tf.random.normal(shape, mean=0, stddev=sigma, seed=self.seed, dtype=tf.float64)

    def get_config(self):
      return {'n': self.n, 'seed': self.seed, 'typef': self.typef}

class Bengio(initializers.Initializer):

    def __init__(self, n, m, seed, typef):
      self.n=n
      self.m=m
      self.seed = seed
      self.typef=typef

    def __call__(self, shape, dtype=None):
      a=-np.sqrt(6/(self.n+self.m)); b=-a
      if(self.typef=="float32"):
        return tf.random.uniform(shape, minval=a, maxval=b, seed=self.seed, dtype=tf.float32)
      elif(self.typef=="float64"):
        return tf.random.uniform(shape, minval=a, maxval=b, seed=self.seed, dtype=tf.float64)

    def get_config(self):
      return {'n': self.n, 'm': self.m, 'seed': self.seed, 'typef': self.typef} """

def init(name, seed, params=[-1,1]):
    if(name=="Uniform"):
        return initializers.RandomUniform(minval=params[0], maxval=params[1], seed=seed)
    elif(name=="Normal"):
        return initializers.RandomNormal(mean=params[0], stddev=params[1], seed=seed)
    elif(name=="Xavier"):
        return initializers.LecunNormal(seed=seed)
    elif(name=="Bengio"):
        return initializers.GlorotUniform(seed=seed)

