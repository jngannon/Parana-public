# This set of functions are for analysing the wigglyness of parameters

import tensorflow as tf
import numpy as np

def get_absolute_deviation(session, cost, iterations, data_function, batch_size, X, y, noise_constant = None, noise_vector = None, parameters_list = None, layers_list = None):
    #Define a temporary tensorflow session
    temp_sess = tf.Session()
    
    #Get the list of weights
    weight_list = parameters_list
    arrays = range(len(weight_list))
    
    #Placeholders and variables
    n_placeholder = tf.placeholder(tf.float32)
    grads_placeholder = [tf.placeholder(tf.float32, shape = i.shape) for i in weight_list]
    mean = [tf.Variable(tf.zeros(shape = i.shape, dtype = tf.float32)) for i in weight_list]
    sum_abs_diff = [tf.Variable(tf.zeros(shape = i.shape, dtype = tf.float32)) for i in weight_list]
    
    #Initialize variables with temporary session
    init_variables = []
    init_variables.extend(mean)
    init_variables.extend(sum_abs_diff)
    temp_sess.run(tf.variables_initializer(init_variables))
    
    #Operations to calculate mean
    delta = [tf.subtract(grads_placeholder[i], mean[i]) for i in arrays]
    #this updates the mean
    assign_mean = [tf.assign(mean[i], tf.add(mean[i], tf.divide(delta[i], n_placeholder))) for i in arrays]
    #this updates the sum of absolute means with a delta taken from the updated mean
    assign_sum_abs_diff = [tf.assign(sum_abs_diff[i], tf.abs(delta[i])) for i in arrays]
    
    #divide the sum of the absolute differences by the numper of samples
    divide_sum_abs_diff = [tf.divide(i, n_placeholder) for i in sum_abs_diff]
    
    #Operation for getting gradients using main session/graph
    gradients = tf.gradients(cost, weight_list)
    
    for datapoint in range(iterations):
        #get the gradients from the main 
        batch = data_function(batch_size)
        grads_list = session.run(gradients, feed_dict = {X:batch[0], y:batch[1]})
        #Update means
        [temp_sess.run(assign_mean[op], feed_dict={n_placeholder:datapoint+1, grads_placeholder[op]:grads_list[op]}) for op in arrays]
        #update sum of absoluts differences
        [temp_sess.run(assign_sum_abs_diff[op], feed_dict={grads_placeholder[op]:grads_list[op]}) for op in arrays]
    
    output_lists = temp_sess.run(divide_sum_abs_diff, feed_dict={n_placeholder:iterations})
    
    temp_sess.close()
    
    return output_lists

def get_mean_const_label_noise(session, cost, iterations, data_function, batch_size, X, y, noise_constant = None, noise_vector = None, parameters_list = None, layers_list = None):
    #Define a temporary tensorflow session
    temp_sess = tf.Session()
    
    #Get the list of weights
    weight_list = parameters_list
    arrays = range(len(weight_list))
    
    #Placeholders and variables
    n_placeholder = tf.placeholder(tf.float32)
    grads_placeholder = [tf.placeholder(tf.float32, shape = i.shape) for i in weight_list]
    noisy_grads_placeholder = [tf.placeholder(tf.float32, shape = i.shape) for i in weight_list]
    mean = [tf.Variable(tf.zeros(shape = i.shape, dtype = tf.float32)) for i in weight_list]
        
    #Initialize variables with temporary session
    init_variables = []
    init_variables.extend(mean)
    temp_sess.run(tf.variables_initializer(init_variables))
    
    #this updates the mean and delta 
    assign_mean = [tf.assign(mean[i], tf.add(mean[i], tf.subtract(grads_placeholder[i], noisy_grads_placeholder[i]))) for i in arrays]
    
    #Operation for getting gradients using main session/graph
    gradients = tf.gradients(cost, weight_list)
    
    for datapoint in range(iterations):
        #get the gradients from the main 
        batch = data_function(batch_size)
        noisy_labels = np.multiply(batch[1], noise_constant)
        grads_list = session.run(gradients, feed_dict = {X:batch[0], y:batch[1]})
        noisy_grads_list = session.run(gradients, feed_dict = {X:batch[0], y:noisy_labels})
        #Update means
        [temp_sess.run(assign_mean[op], feed_dict={grads_placeholder[op]:grads_list[op], noisy_grads_placeholder[op]:noisy_grads_list[op]}) for op in arrays]

    # Divide through by the number of iterations, only once before killing the graph, 
    output_lists = temp_sess.run([tf.divide(mean[i], iterations) for i in arrays])
    
    temp_sess.close()
    
    return output_lists

def get_mean_gauss_label_noise(session, cost, iterations, data_function, batch_size, X, y, noise_constant = None, noise_vector = None, parameters_list = None, layers_list = None):
    #Define a temporary tensorflow session
    temp_sess = tf.Session()
    
    #Get the list of weights
    weight_list = parameters_list
    arrays = range(len(weight_list))
    
    #Placeholders and variables
    n_placeholder = tf.placeholder(tf.float32)
    grads_placeholder = [tf.placeholder(tf.float32, shape = i.shape) for i in weight_list]
    noisy_grads_placeholder = [tf.placeholder(tf.float32, shape = i.shape) for i in weight_list]
    mean = [tf.Variable(tf.zeros(shape = i.shape, dtype = tf.float32)) for i in weight_list]
        
    #Initialize variables with temporary session
    init_variables = []
    init_variables.extend(mean)
    temp_sess.run(tf.variables_initializer(init_variables))
    
    #this updates the mean and delta 
    assign_mean = [tf.assign(mean[i], tf.add(mean[i], tf.subtract(grads_placeholder[i], noisy_grads_placeholder[i]))) for i in arrays]
    
    #Operation for getting gradients using main session/graph
    gradients = tf.gradients(cost, weight_list)
    
    for datapoint in range(iterations):
        #get the gradients from the main 
        batch = data_function(batch_size)
        noisy_labels = np.add(batch[1], np.random.normal(0, noise_constant, size = batch[1].shape))
        grads_list = session.run(gradients, feed_dict = {X:batch[0], y:batch[1]})
        noisy_grads_list = session.run(gradients, feed_dict = {X:batch[0], y:noisy_labels})
        #Update means
        [temp_sess.run(assign_mean[op], feed_dict={grads_placeholder[op]:grads_list[op], noisy_grads_placeholder[op]:noisy_grads_list[op]}) for op in arrays]

    # Divide through by the number of iterations, only once before killing the graph, 
    output_lists = temp_sess.run([tf.divide(mean[i], iterations) for i in arrays])
    
    temp_sess.close()
    
    return output_lists

def get_absolute_values(session, cost, iterations, data_function, batch_size, X, y, noise_constant = None, noise_vector = None, parameters_list = None, layers_list = None):
    return [session.run(i) for i in parameters_list]

def get_mean_gradients_input_noise(session, cost, iterations, data_function, batch_size, X, y, noise_constant = None, noise_vector = None, parameters_list = None, layers_list = None):
    """ Takes a noise vector, or list of noise vectors. Returns the mean difference between the gradients of each parameter in the list
    with noise added and without. Only working with 1 input vector for now, maybe add an option for a list drawn and added randomly."""
    #Define a temporary tensorflow session
    temp_sess = tf.Session()
    
    #Get the list of weights
    weight_list = parameters_list
    arrays = range(len(weight_list))
    
    #Placeholders and variables
    n_placeholder = tf.placeholder(tf.float32)
    grads_placeholder = [tf.placeholder(tf.float32, shape = i.shape) for i in weight_list]
    noisy_grads_placeholder = [tf.placeholder(tf.float32, shape = i.shape) for i in weight_list]
    mean = [tf.Variable(tf.zeros(shape = i.shape, dtype = tf.float32)) for i in weight_list]
        
    #Initialize variables with temporary session
    init_variables = []
    init_variables.extend(mean)
    temp_sess.run(tf.variables_initializer(init_variables))
    
    #this updates the mean and delta 
    assign_mean = [tf.assign(mean[i], tf.add(mean[i], tf.subtract(grads_placeholder[i], noisy_grads_placeholder[i]))) for i in arrays]
    
    #Operation for getting gradients using main session/graph
    gradients = tf.gradients(cost, weight_list)
    
    for datapoint in range(iterations):
        #get the gradients from the main 
        batch = data_function(batch_size)
        noisy_inputs = np.add(batch[0] , noise_vector)
        grads_list = session.run(gradients, feed_dict = {X:batch[0], y:batch[1]})
        noisy_grads_list = session.run(gradients, feed_dict = {X:noisy_inputs, y:batch[1]})
        #Update means
        [temp_sess.run(assign_mean[op], feed_dict={grads_placeholder[op]:grads_list[op], noisy_grads_placeholder[op]:noisy_grads_list[op]}) for op in arrays]

    # Divide through by the number of iterations, only once before killing the graph, 
    output_lists = temp_sess.run([tf.divide(mean[i], iterations) for i in arrays])
    
    temp_sess.close()
    
    return output_lists
