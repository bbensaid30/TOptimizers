import tensorflow as tf

def squared_error(y_true, y_pred, sample_weight=1):
    return tf.nn.l2_loss(y_true-y_pred)

def softmax_cross_entropy(y_true, y_pred,  sample_weight=1, mean=False):
    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits( logits=y_pred, labels=y_true))