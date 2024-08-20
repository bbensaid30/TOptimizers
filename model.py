from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import initializers

import tensorflow as tf

from init import init

def build_ex1_5(loss,name_init,params_init, seed):
    model = models.Sequential()
    model.add(layers.Dense(1, activation='linear', input_shape=(1,), use_bias=False, kernel_initializer=init(name_init,1,1,seed,params_init)))
    
    model.compile(loss=loss)
    return model

def build_poly(activation, loss,name_init,params_init, seed):
    model = models.Sequential()
    model.add(layers.Dense(1, activation=activation, input_shape=(1,), kernel_initializer=init(name_init,1,1,seed,params_init[0:2]),
    bias_initializer=init(name_init,1,1,seed,params_init[2:4])))
    
    model.compile(loss=loss)
    return model

def build_FC(nbNeurons, activations, loss, name_init, params, seed, metrics):
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    L=len(nbNeurons)-1
    model = models.Sequential()
    #model.add(Input(shape=(nbNeurons[0],)))
    model.add(layers.Dense(nbNeurons[1], activation=activations[0], input_dim=nbNeurons[0], kernel_initializer=init(name_init,nbNeurons[1],nbNeurons[0],seed,params),
    bias_initializer=initializers.Zeros()))
    if(L>1):
        for l in range(1,len(nbNeurons)-1):
            model.add(layers.Dense(nbNeurons[l+1], activation=activations[l],kernel_initializer=init(name_init,nbNeurons[l+1],nbNeurons[l],seed,params),
            bias_initializer=initializers.Zeros()))
    
    model.compile(loss=loss, metrics=metrics)
    return model

def build_FC_sparse(nbNeurons, activations, loss, name_init, params, seed, metrics):
    L=len(nbNeurons)-1
    inputs = Input(shape=(nbNeurons[0],), sparse=True)
    y = layers.Dense(nbNeurons[1], activation=activations[0], input_dim=nbNeurons[0], kernel_initializer=init(name_init,nbNeurons[1],nbNeurons[0],seed,params),
    bias_initializer=initializers.Zeros())(inputs)
    if(L>1):
        for l in range(1,len(nbNeurons)-1):
            y = layers.Dense(nbNeurons[l+1], activation=activations[l],kernel_initializer=init(name_init,nbNeurons[l+1],nbNeurons[l],seed,params),
            bias_initializer=initializers.Zeros())(y)
    model = Model(inputs,y)
    
    model.compile(loss=loss, metrics=metrics)
    return model

def build_FC_regularizer(nbNeurons, activations, loss, name_init, params, seed, metrics):
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    L=len(nbNeurons)-1
    model = models.Sequential()
    model.add(layers.Dense(nbNeurons[1], activation=activations[0], input_dim=nbNeurons[0], kernel_initializer=init(name_init,nbNeurons[1],nbNeurons[0],seed,params),
    bias_initializer=initializers.Zeros(),kernel_regularizer='l2'))
    if(L>1):
        for l in range(1,len(nbNeurons)-1):
            model.add(layers.Dense(nbNeurons[l+1], activation=activations[l],kernel_initializer=init(name_init,nbNeurons[l+1],nbNeurons[l],seed,params),
            bias_initializer=initializers.Zeros(), kernel_regularizer='l2'))
    
    model.compile(loss=loss, metrics=metrics)
    return model

def build_lenet1_mnist(loss,name_init,params,seed,metrics):
    model = models.Sequential()
    inputShape = (28, 28, 1)
    activation='tanh'

    # define the first set of CONV => ACTIVATION => POOL layers
    model.add(layers.Conv2D(4, 5, padding="valid",input_shape=inputShape, kernel_initializer=init(name_init,2304,784,seed,params)))
    model.add(layers.Activation(activation))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    # define the second set of CONV => ACTIVATION => POOL layers
    model.add(layers.Conv2D(12, 5, padding="valid", kernel_initializer=init(name_init,768,576,seed,params)))
    model.add(layers.Activation(activation))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    # define the first FC => ACTIVATION layers
    model.add(layers.Flatten())
    model.add(layers.Dense(10,kernel_initializer=init(name_init,10,192,seed,params)))
    model.add(layers.Activation("sigmoid"))

    model.compile(loss=loss,metrics=metrics)
    return model

def build_bilstm_embedding(loss,name_init,params,seed,metrics):
    max_features = 2000  # Only consider the top 10k words

    # Input for variable-length sequences of integers
    inputs = Input(shape=(None,), dtype="int32")
    #Embed each integer in a 128-dimensional vector
    x = layers.Embedding(max_features, 128)(inputs)
    # Add 2 bidirectional LSTMs
    #x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(24))(x)
    # Add a classifier
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)

    model.compile(loss=loss,metrics=metrics)
    return model

def build_lstm_embedding(loss,name_init,params,seed,metrics):
    max_features=3500

    model = models.Sequential()
    model.add(layers.Embedding(max_features, 128))
    model.add(layers.LSTM(10))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss=loss,metrics=metrics)
    return model

def build_cnn_embedding(loss,name_init,params,seed,metrics):
    dimension=5000
    maxlen=500

    model = models.Sequential()
    model.add(layers.Embedding(dimension, 16, input_length= maxlen))
    #model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(filters = 4, kernel_size = 5, strides= 1, padding='same', activation= 'relu'))
    model.add(layers.GlobalMaxPooling1D())
    #model.add(layers.Dense(units = 128, activation= 'relu'))
    #model.add(layers.Dropout(0.2))
    # Output layer
    model.add(layers.Dense(1, activation= 'sigmoid'))
    
    # Compile the model
    model.compile(loss=loss,metrics=metrics)
    return model


# nbNeurons and activations have sense for fully connected networks
def build_model(name_model, nbNeurons, activations, loss, name_init, params, seed, metrics):
    if(name_model=="poly"):
        return build_poly(activations[0],loss,name_init,params,seed)
    elif(name_model=="FC"):
        return build_FC(nbNeurons,activations,loss,name_init,params,seed,metrics)
    elif(name_model=="FC_sparse"):
        return build_FC_sparse(nbNeurons,activations,loss,name_init,params,seed,metrics)
    elif(name_model=="FC_regularizer"):
        return build_FC_regularizer(nbNeurons,activations,loss,name_init,params,seed,metrics)
    elif(name_model=='lenet1_mnist'):
        return build_lenet1_mnist(loss,name_init,params,seed,metrics)
    elif(name_model=='bilstm_embedding'):
        return build_bilstm_embedding(loss,name_init,params,seed,metrics)
    elif(name_model=='lstm_embedding'):
        return build_lstm_embedding(loss,name_init,params,seed,metrics)
    elif(name_model=='cnn_embedding'):
        return build_cnn_embedding(loss,name_init,params,seed,metrics)