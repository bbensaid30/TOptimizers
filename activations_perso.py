from keras import backend as K

def polyTwo(z):
    return K.pow(z,2)  - 1

def polyThree(z):
    return 2*K.pow(z,3)-3*K.pow(z,2)+5


def polyFive(z):
    return K.pow(z,5)-4*K.pow(z,4)+2*K.pow(z,3)+8*K.pow(z,2)-11*z-12