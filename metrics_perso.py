import tensorflow as tf
import numpy as np

def global_dinf(y_true, y_pred):
    return tf.norm(y_true-y_pred,ord=np.inf)

def relative_dinf(y_true, y_pred):
    return 100*tf.norm((y_pred-y_true)/y_true,ord=np.inf)


