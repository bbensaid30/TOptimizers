import tensorflow as tf
import numpy as np

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = y_train.flatten()
y_test = y_test.flatten()

input_shape = (32, 32, 3)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
x_train=x_train / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
x_test=x_test / 255.0

y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

model = tf.keras.models.Sequential()
activation='softplus'

model.add(tf.keras.Input(shape=(32,32,3)))
model.add(tf.keras.layers.Conv2D(filters=4,kernel_size=(5,5),padding="valid"))
model.add(tf.keras.layers.Activation(activation))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(filters=12,kernel_size=(5,5),padding="valid"))
model.add(tf.keras.layers.Activation(activation))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10,activation='softmax'))

batch_size = 64
epochs = 50

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse', metrics=['acc'])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)


