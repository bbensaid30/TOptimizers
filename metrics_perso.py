import tensorflow as tf
import numpy as np
import keras

def global_dinf(y_true, y_pred):
    return tf.norm(y_true-y_pred,ord=np.inf)

def relative_dinf(y_true, y_pred):
    return 100*tf.norm((y_pred-y_true)/y_true,ord=np.inf)

class BalancedCategoricalAccuracy(keras.metrics.CategoricalAccuracy):
    def __init__(self, name='balanced_categorical_accuracy', dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true_onehot, y_pred_onehot, sample_weight=None):
        y_true = tf.math.argmax(y_true_onehot, axis=1)
        y_true_int = tf.cast(y_true, tf.int32)

        cls_counts = tf.math.bincount(y_true_int)
        cls_counts = tf.math.reciprocal_no_nan(tf.cast(cls_counts, self.dtype))
        weight = tf.gather(cls_counts, y_true_int)
        return super().update_state(y_true_onehot, y_pred_onehot, sample_weight=weight)


