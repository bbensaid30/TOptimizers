import tensorflow as tf 
import tensorflow_datasets as tfds

from keras import losses, metrics

from training import train_sto
from model import build_model

train_ds, test_ds = tfds.load('cifar10', split=['train','test'], as_supervised=True)

def normalize_resize(image, label):
    #image = tf.image.rgb_to_grayscale(image)
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255)
    return image, label

def encoding(image, label):
    label = tf.one_hot(label, depth=10)
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.1)
    return image, label 

PTrain=50000; PTest=10000
batch_size=5000
train = train_ds.map(normalize_resize).map(encoding).cache().map(augment).batch(batch_size)
test = test_ds.map(normalize_resize).map(encoding).cache().batch(batch_size)

algo="RRAdam"
#name_model="conv_gray_cifar10"
#name_model="FC"
name_model = "lenet5"
nbNeurons=[3072,64,10]
activations=['softplus', 'softmax']
loss=losses.MeanSquaredError(reduction="sum")
#loss=losses.MeanSquaredError()
#loss=losses.CategoricalCrossentropy(reduction="sum")

metrics = ["accuracy"]
name_init="Xavier"
params_init=[-1,1]

#paramètres d'arrêt
eps=10**(-4); max_epochs=100
#paramètres d'entrainement 
lr=0.1
seuil=0.01
f1=30; f2=10000; lambd=0.5
beta_1=0.9; beta_2=0.999; epsilon_a=1e-07
amsgrad=False

model=build_model(name_model, nbNeurons, activations, loss, name_init, params_init, 0, "float32", metrics)

model, epoch, norme_grad, cost, time_exec = train_sto(algo, model,loss,
train, PTrain, eps, max_epochs, lr, f1, f2, lambd, beta_1, beta_2, epsilon_a, amsgrad, "float32")

