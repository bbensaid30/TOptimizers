import tensorflow as tf
import numpy as np
from keras import datasets
from keras.utils import to_categorical
from keras import layers, Input

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

num_pixels = x_train.shape[1] * x_train.shape[2]
x_train=x_train.reshape(x_train.shape[0], num_pixels)
x_train=x_train / 255.0
x_test = x_test.reshape(x_test.shape[0], num_pixels)
x_test=x_test / 255.0

y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

model = tf.keras.models.Sequential()
activation='tanh'

model.add(Input(shape=(784,)))
model.add(layers.Dense(24))
model.add(layers.Activation(activation))
model.add(layers.Dense(10,activation='softmax'))

batch_size = 64
epochs = 200

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='mse', metrics=['acc'])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)