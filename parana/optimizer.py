import tensorflow as tf
import numpy as np
from parana.parameter_selection import get_maxs

# Need to feed in the model, and can have variables list as an option

class optimizer:
    
    def __init__(self, session, learning_rate, cost_function, model, variables_list = None):
        self._session = session
        self._learning_rate = learning_rate
        self._cost_function = cost_function
        self._model = model
        if variables_list == None:
            self._variables_list = self._model.train_variables                
        if variables_list != None:
            self._variables_list = variables_list
        self._opt = tf.train.GradientDescentOptimizer(self._learning_rate)
        self._grads_and_vars = self._opt.compute_gradients(self._cost_function, var_list = self._variables_list)
        self._gradients_placeholder = [(tf.placeholder(tf.float32, shape = i[1].get_shape()), i[1]) for i in self._grads_and_vars]
        self._opt_min = self._opt.minimize(self._cost_function, var_list = self._variables_list)
        self._apply_grads = self._opt.apply_gradients(self._gradients_placeholder)
    
    
    #@property
    def min_step(self, data):
        self._session.run(self._opt_min, feed_dict = data)
        return 
    
    def min_step_return_gradients(self, data, parameter_list):
        grads_and_vars = self._session.run(self._grads_and_vars, feed_dict = data)
        grads = [i for i,j in grads_and_vars]
        indices = [self._model.train_variables.index(i) for i in parameter_list]
        grads_dict = {}
        for i in range(len(grads)):
            grads_dict[self._gradients_placeholder[i][0]] = grads[i]
        self._session.run(self._apply_grads, feed_dict = grads_dict)
        return [grads[i] for i in indices]
        
class adamopt:
    
    def __init__(self, session, learning_rate, cost_function, model, variables_list = None):
        self._session = session
        self._learning_rate = learning_rate
        self._cost_function = cost_function
        self._model = model
        if variables_list == None:
            self._variables_list = self._model.train_variables                
        if variables_list != None:
            self._variables_list = variables_list
        self._opt = tf.train.AdamOptimizer(self._learning_rate)
        self._grads_and_vars = self._opt.compute_gradients(self._cost_function, var_list = self._variables_list)
        self._gradients_placeholder = [(tf.placeholder(tf.float32, shape = i[1].get_shape()), i[1]) for i in self._grads_and_vars]
        self._opt_min = self._opt.minimize(self._cost_function, var_list = self._variables_list)
        self._apply_grads = self._opt.apply_gradients(self._gradients_placeholder)
    
    
    #@property
    def min_step(self, data):
        self._session.run(self._opt_min, feed_dict = data)
        return 
    
class squeezeopt:
    
    def __init__(self, session, learning_rate, model, logit_cost_function = None):
        self._session = session
        self._model = model
        self._cost_function = logit_cost_function
        self._variables_list = self._model.train_variables
        self._learning_rate = learning_rate
        self._variables_placeholders = [tf.placeholder(tf.float32, shape = i.shape) for i in self._variables_list]
        self._take_step = [tf.assign(i, tf.add(i, j)) for i, j in zip(self._variables_list, self._variables_placeholders)]
        if logit_cost_function == None:
            self._logit_cost = self._model.logit_cost
        self._gradients = tf.gradients(self._logit_cost, self._model.train_variables)
         
        
    def assignop(self, new_parameters_list):
        for i, j in zip(self._assignops, self._variables_placeholders, new_parameters_list):
            session.run(i, j = k)
        return

    def squeezestep(self, data, target_reduction_ratio = None, target_reduction_const = None):
        if target_reduction_ratio and target_reduction_const:
            print ('Set either ratio or constant, not both')
            return
        outputs = self._session.run(self._model.logitoutput, feed_dict = {self._model.inputs:data[self._model.inputs]})
        #initialize
        target = np.zeros_like(outputs)
        # Gets a 1 in the max position of each row in the matrix and zero in the others
        maxs = get_maxs(outputs)
        # Get the correct labels
        labels = data[self._model.get_labels]
        #Gets all of the correct predictions positive and negative
        correct_labels = (np.equal(labels,maxs)).astype(int)
        #Incorrect predictions need further processing
        incorrect_labels = (np.not_equal(labels,maxs)).astype(int)
        # Outputs that flasely predicted a negative label (0 where a 1 is meant to be), set to 1
        false_negative = np.multiply(incorrect_labels, labels)
        # False positive predictions (1 where a 0 is supposed to be) are already set to 0
        if target_reduction_ratio:
            target = np.multiply(correct_labels, outputs)*(1-target_reduction_ratio)
        target = np.add(target, false_negative)
        #Put target back into data dictionary
        data[self._model.get_labels] = target
        #get the gradients
        grads = self._session.run(self._gradients, feed_dict = data)
        grads = [i*self._learning_rate for i in grads]
        #Take a step
        [self._session.run(self._take_step[i], feed_dict = {self._variables_placeholders[i]:grads[i]}) for i in range(len(grads))]
        return 
        