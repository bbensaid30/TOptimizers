from keras import losses
import numpy as np
import tensorflow as tf

from collections import Counter
from scipy.ndimage import convolve1d
from utils import get_lds_kernel_window


bins=1
def hist_eqn(y_train):
    hist, bins_edges = np.histogram(a=y_train, bins=bins)
    return hist,bins_edges,np.max(y_train)

def get_bin_idx(label,bins_numbers, bins_edges,maximum):
    if(np.abs(maximum-label)<10**(-8)):
        return bins_numbers-1
    else:
        return np.where(bins_edges > label)[0][0] - 1

def compute_weights_lds(y_train,kernel="gaussian",ks=5,sigma=2):
    # assign each label to its corresponding bin (start from 0)
    # with your defined get_bin_idx(), return bin_index_per_label: [Ns,] 
    hist,bins_edges,maximum = hist_eqn(y_train)
    bins_numbers = len(hist)
    bin_index_per_label = [get_bin_idx(label,bins_numbers,bins_edges,maximum) for label in y_train]

    # calculate empirical (original) label distribution: [Nb,]
    # "Nb" is the number of bins
    Nb = np.max(bin_index_per_label) + 1
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

    lds_kernel_window = get_lds_kernel_window(kernel=kernel, ks=ks, sigma=sigma)
    # calculate effective label distribution: [Nb,]
    eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')

    # Use re-weighting based on effective label distribution, sample-wise weights: [Ns,]
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    weights = [np.float32(1/x) for x in eff_num_per_label]

    return tf.convert_to_tensor(weights)

def simple_weights(y_train):
    P = y_train.shape[0]
    hist,bins_edges,maximum = hist_eqn(y_train)
    bins_numbers = len(hist)
    bin_number_per_label = [hist[get_bin_idx(label,bins_numbers,bins_edges,maximum)]/P for label in y_train]

    weights = [np.float32(1/x) for x in bin_number_per_label]
    total = sum(weights)
    weights_normalized = [P*weight/total for weight in weights]

    return tf.convert_to_tensor(weights_normalized)