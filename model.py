from keras import models
from keras import layers
from keras import Input
from keras import initializers

from init import init

def build_ex1_5(loss,name_init,params_init, seed):
    model = models.Sequential()
    model.add(Input((1,)))
    model.add(layers.Dense(1, activation='linear', use_bias=False, kernel_initializer=init(name_init,seed,params_init)))
    
    model.compile(loss=loss)
    return model

def build_poly(activation, loss,name_init,params_init, seed):
    model = models.Sequential()
    model.add(Input((1,)))
    model.add(layers.Dense(1, activation=activation, kernel_initializer=init(name_init,seed,params_init[0:2]),
    bias_initializer=init(name_init,seed,params_init[2:4])))
    
    model.compile(loss=loss)
    return model

def build_FC(nbNeurons, activations, loss, name_init, params, seed, metrics):
    kernel_init = init(name_init, seed, params)
    bias_init = initializers.Zeros()

    L=len(nbNeurons)-1
    model = models.Sequential()
    model.add(Input(shape=(nbNeurons[0],)))
    model.add(layers.Dense(nbNeurons[1], activation=activations[0], kernel_initializer=kernel_init,
    bias_initializer=bias_init))
    if(L>1):
        for l in range(1,len(nbNeurons)-1):
            model.add(layers.Dense(nbNeurons[l+1], activation=activations[l],kernel_initializer=kernel_init,
            bias_initializer=bias_init))
    
    model.compile(loss=loss, metrics=metrics)
    return model

def build_lenet1_mnist(loss,name_init,params,seed,metrics):
    kernel_init = init(name_init, seed, params)
    bias_init = initializers.Zeros()

    model = models.Sequential()
    inputShape = (28, 28, 1)
    activation='tanh'

    model.add(Input(inputShape))
    # define the first set of CONV => ACTIVATION => POOL layers
    model.add(layers.Conv2D(4, 5, padding="valid", kernel_initializer=kernel_init,bias_initializer=bias_init))
    model.add(layers.Activation(activation))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    # define the second set of CONV => ACTIVATION => POOL layers
    model.add(layers.Conv2D(12, 5, padding="valid", kernel_initializer=kernel_init,bias_initializer=bias_init))
    model.add(layers.Activation(activation))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    # define the first FC => ACTIVATION layers
    model.add(layers.Flatten())
    model.add(layers.Dense(10,kernel_initializer=kernel_init,bias_initializer=bias_init))
    model.add(layers.Activation("sigmoid"))

    model.compile(loss=loss,metrics=metrics)
    return model

def build_lenet1_cifar10(loss,name_init,params,seed,metrics):
    kernel_init = init(name_init, seed, params)
    bias_init = initializers.Zeros()

    model = models.Sequential()
    activation='softplus'

    model.add(Input(shape=(32,32,3)))
    # define the first set of CONV => ACTIVATION => POOL layers
    model.add(layers.Conv2D(filters=4,kernel_size=(5,5),padding="valid", kernel_initializer=kernel_init,bias_initializer=bias_init))
    model.add(layers.Activation(activation))
    model.add(layers.AveragePooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(filters=12,kernel_size=(5,5),padding="valid",kernel_initializer=kernel_init,bias_initializer=bias_init))
    model.add(layers.Activation(activation))
    model.add(layers.AveragePooling2D(pool_size=(2,2)))

    model.add(layers.Flatten())
    #model.add(layers.Dense(24,kernel_initializer=kernel_init,bias_initializer=bias_init))
    #model.add(layers.Activation(activation))
    model.add(layers.Dense(10,activation='softmax',kernel_initializer=kernel_init,bias_initializer=bias_init))

    model.compile(loss=loss,metrics=metrics)

    return model

def build_lenet5_cifar10(loss,name_init,params,seed,type,metrics):
    kernel_init = init(name_init, seed, params)
    bias_init = initializers.Zeros()

    model = models.Sequential()
    activation='relu'

    model.add(Input(shape=(32,32,3)))
    # define the first set of CONV => ACTIVATION => POOL layers
    model.add(layers.Conv2D(filters=6,kernel_size=5,padding="valid", kernel_initializer=kernel_init, bias_initializer=bias_init))
    model.add(layers.Activation(activation))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=2))

    model.add(layers.Conv2D(filters=16,kernel_size=5,padding="valid", kernel_initializer=kernel_init, bias_initializer=bias_init))
    model.add(layers.Activation(activation))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(120, kernel_initializer=kernel_init, bias_initializer=bias_init))
    model.add(layers.Activation(activation))
    model.add(layers.Dense(84, kernel_initializer=kernel_init, bias_initializer=bias_init))
    model.add(layers.Activation(activation))
    model.add(layers.Dense(10, activation='softmax', kernel_initializer=kernel_init, bias_initializer=bias_init))

    model.compile(loss=loss,metrics=metrics)

    return model



# nbNeurons and activations have sense for fully connected networks
def build_model(name_model, nbNeurons, activations, loss, name_init, params, seed, metrics):
    if(name_model=="poly"):
        return build_poly(activations[0],loss,name_init,params,seed)
    elif(name_model=="FC"):
        return build_FC(nbNeurons,activations,loss,name_init,params,seed,metrics)
    elif(name_model=='lenet1_mnist'):
        return build_lenet1_mnist(loss,name_init,params,seed,metrics)
    elif(name_model=="lenet1_cifar10"):
        return build_lenet1_cifar10(loss,name_init,params,seed,metrics)
    elif(name_model=="lenet5"):
        return build_lenet5_cifar10(loss,name_init,params,seed,metrics)