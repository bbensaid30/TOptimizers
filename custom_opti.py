import tensorflow as tf

class CustomMomentum(tf.keras.optimizers.Optimizer):

    def __init__(self, learning_rate, momentum, name="CustomMomentum", **kwargs):
        super().__init__(name=name, **kwargs)

        # hyperparamètres reconnus par Keras
        self._learning_rate = learning_rate
        self.momentum = momentum

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, lr):
        self._learning_rate = lr

    def build(self, var_list):
        # créer une variable de momentum pour chaque poids
        self.v = [tf.Variable(tf.zeros_like(var), trainable=False) for var in var_list]

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        # itérer sur la liste et mettre à jour les poids et v
        lr=self.learning_rate
        for i, (grad, var) in enumerate(grads_and_vars):
            v = self.v[i]
            v.assign(self.momentum * v + lr * grad)
            var.assign_sub(lr * v)

class CustomRMSProp(tf.keras.optimizers.Optimizer):

    def __init__(self, learning_rate, beta2=0.9, epsilon=1e-10, name="CustomRMSProp"):
        super().__init__(name=name)
        self._learning_rate = learning_rate
        self.beta2 = beta2
        self.epsilon = epsilon
        self.s = []
    
    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, lr):
        self._learning_rate = lr

    def build(self, var_list):
        self.s = [tf.Variable(tf.zeros_like(var), trainable=False) for var in var_list]

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        lr=self.learning_rate
        for i, (grad, var) in enumerate(grads_and_vars):
            s = self.s[i]
            s.assign(self.beta2 * s + (1 - self.beta2) * tf.square(grad))
            var.assign_sub(lr * grad / (tf.sqrt(s + self.epsilon)))