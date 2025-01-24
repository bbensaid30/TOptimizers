import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window

def convert_sparse_matrix_to_sparse_tensor(X_train, X_test):
    coo_train = X_train.tocoo()
    indices_train = np.mat([coo_train.row, coo_train.col]).transpose()
    coo_test = X_test.tocoo()
    indices_test = np.mat([coo_test.row, coo_test.col]).transpose()
    return tf.SparseTensor(indices_train, coo_train.data, coo_train.shape), tf.SparseTensor(indices_test, coo_test.data, (coo_test.shape[0],coo_train.shape[1]))

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1  # set specific indices of results[i] to 1s
    return results.astype('uint8')

def gradTotInit(grad):
    gradTot=[]
    for i in range(len(grad)):
        gradTot.append(tf.zeros_like(grad[i]))
    return gradTot

def gradTotZero(gradTot):
    for i in range(len(gradTot)):
        gradTot[i]=tf.zeros_like(gradTot[i])

def gradSum(gradTot, grad):
    for i in range(len(grad)):
        gradTot[i]+=grad[i]

def gradDiff(gradTot, grad):
    for i in range(len(grad)):
        gradTot[i]-=grad[i]

def gradDiv(gradTot, P):
    for i in range(len(gradTot)):
        gradTot[i]=gradTot[i]/P

def gradMul(gradTot, P):
    for i in range(len(gradTot)):
        gradTot[i]=gradTot[i]*P

def var_compute(grad,g,m,typef):
    if(typef=="float32"):
        result=tf.constant(0.0, dtype=tf.float32)
        m_cast=tf.cast(m,dtype=tf.float32)
    elif(typef=="float64"):
        result=tf.constant(0.0, dtype=tf.float64)
        m_cast=tf.cast(m,dtype=tf.float64)
    for i in range(len(grad)):
        result+=tf.linalg.norm(grad[i]-g[i]/m_cast)**2
    return result

def gradDot(grad1, grad2, typef="float32"):
    if(typef=="float32"):
        result=tf.constant(0.0, dtype=tf.float32)
    elif(typef=="float64"):
        result=tf.constant(0.0, dtype=tf.float64)
    for i in range(len(grad1)):
        result+=tf.tensordot(tf.reshape(grad1[i],[-1]),tf.reshape(grad2[i],[-1]),axes=1)
    return result