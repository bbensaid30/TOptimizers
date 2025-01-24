from keras import backend as K

def polyTwo(z):
    return z**2 - 1

def polyThree(z):
    return 2*z**3-3*z**2+5

def polyFive(z):
    return z**5-4*z**4+2*z**3+8*z**2-11*z-12