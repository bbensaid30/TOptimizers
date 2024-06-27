import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

from keras import losses
from keras import optimizers
import tensorflow as tf

from Tensorflow.perso import Adam,LC_EGD, Adam_batch, LC_EGD_batch

session_conf = tf.compat.v1.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(config=session_conf)

sample_weight=1
 
# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = cifar10.load_data()
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY
 
# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm
 
def VGG(loss_fn,update,name_init,params_init,seed):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	model.compile(optimizer=update, loss=loss_fn, metrics=['accuracy'])
	return model
 
 
# load dataset
trainX, trainY, testX, testY = load_dataset()
# prepare pixel data
trainX, testX = prep_pixels(trainX, testX)
#batch_size = trainX.shape[0]
batch_size=64
# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.shuffle(buffer_size=batch_size).batch(batch_size)

# define parameters of the model
loss_fn = losses.CategoricalCrossentropy()
name_init='Bengio'; params_init=[-1,1]; seed=0


#paramètres d'arrêt
eps=10**(-4); max_epochs=10

""" #paramètres pour Adam
update = optimizers.Adam(0.001)
lr=0.001; beta_1=0.9; beta_2=0.999; epsilon=1e-07; amsgrad=False
model = VGG(loss_fn,update,name_init,params_init,seed)
model,time = Adam_batch(model,loss_fn,train_dataset,eps,max_epochs,lr,beta_1,beta_2,epsilon,amsgrad,sample_weight) """


#paramètres pour LC_EGD
update = optimizers.SGD(0.1)
lr_init=0.1; f1=2; f2=10000; lambd=0.5
model = VGG(loss_fn,update,name_init,params_init,seed); model_copy = VGG(loss_fn,update,name_init,params_init,seed)
model, time = LC_EGD_batch(model,model_copy,loss_fn,train_dataset,eps,max_epochs,lr_init,f1,f2,lambd,sample_weight)

model.reset_metrics()

print("---------- Train ----------------")
resultsTrain = model.evaluate(trainX,trainY)
print(resultsTrain)

model.reset_metrics()

print("---------- Test ----------------")
resultsTest = model.evaluate(testX,testY)
print(resultsTest)