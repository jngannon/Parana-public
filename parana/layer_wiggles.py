import tensorflow as tf
import numpy as np

def get_mean_activation_difference(session, cost, iterations, data_function, batch_size, X, y, noise_constant = None, noise_vector = None, parameters_list = None, layers_list = None):

    activations_list = []
    for layer in layers_list:
        mean_diff = np.zeros(shape = layer.weights.get_shape())

        for i in range(iterations):
            batch = data_function(batch_size)
            noisy_inputs = np.add(batch[0], noise_vector)
            weight_means = (np.mean(np.abs([session.run(layer.activated_weights, feed_dict={X:[i]}) for i in batch[0]]), axis = 0))
            noisy_means = (np.mean(np.abs([session.run(layer.activated_weights, feed_dict={X:[i]}) for i in noisy_inputs]), axis = 0))
            act_difference = np.subtract(weight_means, noisy_means)
            mean_diff = np.add(mean_diff, act_difference)
        mean_diff = np.divide(mean_diff, iterations)
        activations_list.append(mean_diff)
    return activations_list

def get_mean_activations(session, cost, data_function, batch_size, X, y, iterations = 1, noise_constant = None, noise_vector = None, parameters_list = None, layers_list = None):
    """ Iterations are set to 1, this will be faster to just use a large batch until there are memory problems, if this is the case, batch size can be cut down and iterations increased"""
    activations_list = []
    iterations = 1
    for layer in layers_list:
        mean_act = np.zeros(shape = layer.weights.get_shape())
        
        for i in range(iterations):
            batch = data_function(batch_size)
            if noise_vector is not None:
                inputs = np.add(batch[0], noise_vector)
            if noise_vector is None:
                inputs = batch[0]
            mean_act = np.add(mean_act, (np.mean([session.run(layer.activated_weights, feed_dict={X:[i]}) for i in inputs], axis = 0)))
        mean_act = np.divide(mean_act, iterations)
        activations_list.append(mean_act)
    return activations_list


            