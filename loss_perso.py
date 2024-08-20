import tensorflow as tf

def squared_error(y_true, y_pred):
    return tf.reduce_sum(tf.square(y_true - y_pred))/2