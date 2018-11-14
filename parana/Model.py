import tensorflow as tf
import numpy as np
from parana.bias_consolidation import consolidate_fc_biases
from parana.conv_functions import decouple
from parana.conv_functions import sparse_decouple
from parana.Layers import fc_layer
from parana.Layers import softmax_layer
from parana.Layers import sparse_layer
from scipy.sparse import csr_matrix

class Model():
    
    def __init__(self, inputs, layers, labels, cost_function, logit_cost_function = None, relu_logit_cost_function = None, dropout = 0):
        self.inputs = inputs
        self.layers = layers
        self.labels = labels
        self.cost_function = cost_function
        self.logit_cost_function = logit_cost_function
        self.relu_logit_cost_function = relu_logit_cost_function
        self.dropout = dropout
        
    @property
    def cost(self):
        if self.cost_function == 'cross_entropy':
            self._cost = tf.reduce_mean(-tf.reduce_sum(self.labels * tf.log(self.modeloutput), reduction_indices=[1]))
        if self.cost_function == 'cross_entropy_l2':
            self._cost = tf.reduce_mean(-tf.reduce_sum(self.labels * tf.log(self.modeloutput), reduction_indices=[1])) + tf.reduce_sum([j*tf.nn.l2_loss(i) for i, j in self.decay_rates])
        if self.cost_function == 'quadratic':
            self._cost = tf.reduce_mean(tf.square(self.labels - self.modeloutput)/2)
        if self.cost_function == 'quadratic_l2':
            self._cost = tf.reduce_mean(tf.square(self.labels - self.modeloutput)/2) + tf.reduce_sum([j*tf.nn.l2_loss(i) for i, j in self.decay_rates])
        return self._cost
    
    @property
    def logit_cost(self):
        if self.logit_cost_function == 'quadratic':
            self._logit_cost = tf.reduce_mean(tf.square(self.labels - self.logitoutput)/2)
        if self.logit_cost_function == 'quadratic_l2':
            self._logit_cost = tf.reduce_mean(tf.square(self.labels - self.logitoutput)/2) + tf.reduce_sum([j*tf.nn.l2_loss(i) for i, j in self.decay_rates])
        return self._logit_cost
    
    @property
    def relu_logit_cost(self):
        if self.relu_logit_cost_function == 'cross_entropy':
            self._relu_logit_cost = tf.reduce_mean(-tf.reduce_sum(self.labels * tf.log(self.relu_logit_output), reduction_indices=[1]))
        if self.relu_logit_cost_function == 'cross_entropy_l2':
            self._relu_logit_cost = tf.reduce_mean(-tf.reduce_sum(self.labels * tf.log(self.relu_logit_output), reduction_indices=[1])) + tf.reduce_sum([j*tf.nn.l2_loss(i) for i, j in self.decay_rates])
        if self.relu_logit_cost_function == 'quadratic':
            self._relu_logit_cost = tf.reduce_mean(tf.square(self.labels - self.relu_logit_output)/2)
        if self.relu_logit_cost_function == 'quadratic_l2':
            self._relu_logit_cost = tf.reduce_mean(tf.square(self.labels - self.relu_logit_output)/2) + tf.reduce_sum([j*tf.nn.l2_loss(i) for i, j in self.decay_rates])
        return self._relu_logit_cost
    
    def accuracy(self, session, inputs, labels):
        for i in self.layers:
            i.dropout = 0
        correct = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.modeloutput, 1))
        self._error = session.run(tf.reduce_mean(tf.cast(correct, tf.float32)), feed_dict = {self.inputs: inputs, self.labels: labels})
        for i in self.layers:
            i._dropout = self.dropout
        return self._error
    
    
    def split_accuracy(self, session, stages, inputs, labels):
        mean_accuracy = 0
        total_inputs = len(inputs)
        batch_size = total_inputs/stages
        for i in self.layers:
            i.dropout = 0
        for i in range(stages):
            mean_accuracy += session.run(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.modeloutput, 1)), tf.float32)), feed_dict = {self.inputs: inputs[int(i*batch_size):int((i+1)*batch_size-1)], self.labels: labels[int(i*batch_size):int((i+1)*batch_size-1)]})
        for i in self.layers:
            i._dropout = self.dropout
        return mean_accuracy/stages
    
    def zerodropout(self):
        for i in self.layers:
            i.dropout = 0
        return
    
    @property
    def modeloutput(self):
        return self.layers[-1].activate
    
    @property
    def logitoutput(self):
        return self.layers[-1]._activate_logits
    
    @property
    def relu_logit_output(self):
        return self.layers[-1]._activate_relu_logits
    
    @property
    def get_weights(self):
        allweights = [i._weights for i in self.layers]
        #allweights.pop()
        return allweights
    
    @property
    def get_biases(self):
        allbiases = [i._biases for i in self.layers]
        allbiases.pop()
        return allbiases
    
    @property
    def get_dummies(self):
        return [i._dummies for i in self.layers if i._dummies]
    
    @property
    def train_variables(self):
        #Returns variables to be trained by an optimizer
        #Only weights and biases, not dummies
        variables = []
        for i in self.layers:
            variables.append(i._weights)
            variables.append(i._biases)
        return variables
    
    @property
    def multiply_variables(self):
        variables = []
        for i in self.layers:
            variables.append(i.multiply_weights)
            variables.append(i.multiply_biases)
        return variables
    
    @property
    def assign_variables(self):
        variables = []
        for i in self.layers:
            variables.append(i.assign_weights)
            variables.append(i.assign_biases)
        return variables
    
    def get_last_weights(self):
        return [layers[-1].weights]
    
    @property
    def decay_rates(self):
        rates = []
        for i in self.layers:
            rates.extend(i.get_decay_rates)
        return rates
    
    def consolidate_biases(self, session):
        weight_list, bias_list = consolidate_fc_biases(session = session, layers = self.layers)
        return weight_list, bias_list
    
    def decouple_weights(self, session, layers = None):
        parameter_list = []
        if layers == None:
            layers = self.layers
        for i in layers:
            if i.get_type == 'Convolution':
                conv_weights = session.run(i.weights)
                flat_weights = decouple(conv_weights, input_shape = i.get_input_shape, padding = i.get_padding)
                parameter_list.append(flat_weights)
                conv_biases = session.run(i.biases)
                flat_biases = np.tile(conv_biases, i.get_input_shape[0]*i.get_input_shape[1])
                parameter_list.append(flat_biases)
            if i.get_type != 'Convolution':
                weights = session.run(i.weights)
                weights = csr_matrix(weights)
                parameter_list.append(weights)
                biases = session.run(i.biases)
                parameter_list.append(biases)        
        return parameter_list
     
    def sparse_decouple_weights(self, session, layers = None):
        parameter_list = []
        if layers == None:
            layers = self.layers
        for i in layers:
            if i.get_type == 'Convolution':
                conv_weights = session.run(i.weights)
                flat_weights = decouple(conv_weights, input_shape = i.get_input_shape, padding = i.get_padding)
                flat_weights = csr_matrix(flat_weights)
                parameter_list.append(flat_weights)
                conv_biases = session.run(i.biases)
                flat_biases = np.tile(conv_biases, i.get_input_shape[0]*i.get_input_shape[1])
                parameter_list.append(flat_biases)
            if i.get_type != 'Convolution':
                weights = session.run(i.weights)
                weights = csr_matrix(weights)
                parameter_list.append(weights)
                biases = session.run(i.biases)
                parameter_list.append(biases)        
        return parameter_list
    
    
    @property
    def get_labels(self):
        return self.labels
    
    def smallmodel(self,session, parameters_list):
        # Inputs for the model
        inshape = self.inputs.get_shape()
        self.inputs = tf.reshape(self.inputs, [-1, inshape[1]*inshape[2]*inshape[3]])
        # Rescale the decay rates
        w1_l2 = [session.run(tf.nn.l2_loss(i)) for i in self.train_variables]
        w1_lambda = [i[1] for i in self.decay_rates]
        w2_l2 = [np.sum((i**2)/2) for i in parameters_list]
        w2_lambda = [i*j/k for i, j, k in zip(w1_lambda, w1_l2, w2_l2)]
        session.close()
        for i in range(len(self.layers)-1):
            if i == 0:
                these_inputs = self.inputs
            else:
                these_inputs = self.layers[i-1].activate
            
            self.layers[i] = fc_layer(inputs = these_inputs,
                            size = parameters_list[i*2].shape[1],
                            weight_init = parameters_list[i*2],
                            bias_init = parameters_list[i*2 + 1],
                            weight_decay = w2_lambda[i*2],
                            bias_decay = w2_lambda[i*2 + 1],
                            activation = self.layers[i]._activation)    
        last = len(self.layers)-1
        these_inputs = self.layers[last-1].activate    
        self.layers[last] = softmax_layer(inputs = these_inputs,
                                             size = parameters_list[last*2].shape[1],
                                             weight_init = parameters_list[last*2],
                                             bias_init = parameters_list[last*2 + 1],
                                             weight_decay = w2_lambda[i*2],
                                             bias_decay = w2_lambda[i*2+1],
                                             activation = self.layers[last]._activation)
        if self.logit_cost_function == 'quadratic':
            self._logit_cost = tf.reduce_mean(tf.square(self.labels - self.logitoutput)/2)
        #get l2 w^2 values here, and rescale the regularization in the layer object
        if self.logit_cost_function == 'quadratic_l2':
            self._logit_cost = tf.reduce_mean(tf.square(self.labels - self.logitoutput)/2) + tf.reduce_sum([j*tf.nn.l2_loss(i) for i, j in self.decay_rates])
        session = tf.Session()
        
        return session
    
    def sparsemodel(self, parameters_list):
        for i in range(len(self.layers)):
            self.layers[i] = sparse_layer(weight_init = parameters_list[i*2],
                            bias_init = parameters_list[i*2 + 1])    
        print('Close your tensorflow session, it is no longer useful')
        return
    
    def sparse_fp(self, inputs):
        this_layer_inputs = inputs
        output = None
        for layer in self.layers:
            layer._inputs = this_layer_inputs
            output = layer.activate
            this_layer_inputs = output
        return output
    
    def sparse_accuracy(self, inputs, labels):
        total = 0
        for i in range(len(inputs)):
            if inputs[i].shape[0] != 1:
                this_input = inputs[[i]] 
            else:
                this_input = inputs[i]
            if np.argmax(self.sparse_fp(this_input)) == np.argmax(labels[i]):
                total += 1
        accuracy = total/len(inputs)
        return accuracy