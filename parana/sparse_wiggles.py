import numpy as np
from scipy import sparse
from IPython.display import clear_output

def get_mean_activation_difference(model, iterations, data_function, noise_constant = None, noise_vector = None, layers_list = None):
    ''' Gets the mean difference between activated weight values when the input is with and without an added noise vector.
    The noise vector can be adversarial, or random, but created and fed into this function.
    '''
    if not layers_list:
        layers_list = model.layers
    for i in iterations:
        #Forward pass
        batch = data_function(1)[0]
        noisy_batch = batch + noise_vector
        mymodel.sparse_fp(batch)
        activation_value = [abs(j.activate_weights) for j in layers_list]
        mymodel.sparse_fp(noisy_batch)
        noisy_activation_values = [abs(j.activate_weights) for j in layers_list]
        activation_difference = None
            
    return

import time

def get_mean_activations(model, iterations, data_function, noise_constant = None, noise_vector = None, layers_list = None):
    
    if not layers_list:
        layers_list = model.layers
    means = [sparse.csr_matrix(i.weights.shape) for i in layers_list]
    for i in range(iterations):
        clear_output()
        print(i)
        batch = data_function(1)[0]
        if noise_vector:
            batch = batch + noise_vector
        model.sparse_fp(batch)
        activation_values = [abs(j.activate_weights) for j in layers_list]
        means = [j + k for j, k in zip(means, activation_values)]
    means = [j/iterations for j in means]
    
    return means

def get_abs_values(model, iterations = None, data_function = None, noise_constant = None, noise_vector = None, layers_list = None):
    if not layers_list:
        layers_list = model.layers
    abs_values = [i.weights for i in layers_list]
    return abs_values