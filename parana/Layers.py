import tensorflow as tf
import numpy as np

class fc_layer:
    
    def __init__(self, inputs, size, weight_decay = 0, bias_decay = 0, dummies = False, weight_init = 0.01, bias_init = 0.1, activation = 'relu', dropout = 0):
        self._inputs = inputs
        self._size = size
        self._input_size = int(self._inputs.get_shape()[1])
        if type(weight_init) == float:
            self._weights = tf.Variable(tf.truncated_normal([self._input_size, self._size], stddev=weight_init), name='Weights')
        if weight_init == 'xavier':
            self._weights = tf.Variable(tf.truncated_normal([self._input_size, self._size], stddev=tf.sqrt(3. / (self._input_size + self._size))), name='Weights')
        if isinstance(weight_init, np.ndarray):
            self._weights = tf.Variable(weight_init, name = 'Weights', dtype = tf.float32)
        self._weight_decay = weight_decay
        self._decoupled_weight_decay = None
        if type(bias_init) == float:
            self._biases = tf.Variable(tf.constant(bias_init, shape = [self._size]), name = 'Biases')
        if isinstance(bias_init, np.ndarray):
            self._biases = tf.Variable(bias_init, name = 'Biases', dtype = tf.float32)
        self._bias_decay = bias_decay
        self._dummies = None
        self._activation = activation
        self._dropout = dropout
        self._dropout_mask = tf.floor(tf.random_uniform(shape = (self._input_size, self._size), maxval = 1) + 1-self._dropout)
        if dummies == True:
            self._dummies = tf.Variable(tf.constant(1.0, shape = [self._size]), name = 'Dummies')
            if activation == 'relu':
                self._activate = tf.multiply(tf.nn.relu(tf.matmul(self._inputs, self._weights) + self._biases), self._dummies)
            if activation == 'sigmoid':
                self._activate = tf.multiply(tf.sigmoid(tf.matmul(self._inputs, self._weights) + self._biases), self._dummies)
        if dummies == False:
            if activation == 'relu':
                self._activate = tf.nn.relu(tf.matmul(self._inputs, tf.multiply(self._weights, self._dropout_mask)) + self._biases)
            if activation == 'sigmoid':
                self._activate = tf.sigmoid(tf.matmul(self._inputs, tf.multiply(self._weights, self._dropout_mask)) + self._biases)
        self._activated_weights = tf.multiply(self._weights, tf.transpose(self._inputs))
        self._weights_placeholder = tf.placeholder(tf.float32, shape = self._weights.shape)
        self._weights_assign_op = tf.assign(self._weights, self._weights_placeholder)
        self._biases_placeholder = tf.placeholder(tf.float32, shape = self._biases.shape)
        self._biases_assign_op = tf.assign(self._biases, self._biases_placeholder)
        self._weights_mask_multiply = tf.assign(self._weights, tf.multiply(self._weights, self._weights_placeholder))
        self._biases_mask_multiply = tf.assign(self._biases, tf.multiply(self._biases, self._biases_placeholder))
        if activation == 'relu':
            self._activate_biases = tf.nn.relu(self.biases)
        if activation == 'sigmoid':
            self._activate_biases = tf.sigmoid(self.biases)
        
    @property    
    def get_inputs(self):
        return self._inputs

    @property    
    def activate(self):
        return self._activate
    
    @property
    def get_decay_rates(self):
        return [(self._weights, self._weight_decay), (self._biases, self._bias_decay)]
    
    @property
    def weights(self):
        return self._weights
    
    @property
    def activated_weights(self):
        return self._activated_weights
    
    @property
    def biases(self):
        return self._biases
    
    @property
    def activate_biases(self):
        return self._activate_biases
    
    @property
    def get_type(self):
        return 'Fully_Connected'
    
    def assign_weights(self, session, weight_matrix):
        session.run(self._weights_assign_op, feed_dict = {self._weights_placeholder: weight_matrix})
        return
    
    def assign_biases(self, session, bias_matrix):
        session.run(self._biases_assign_op, feed_dict = {self._biases_placeholder: bias_matrix})
        return
    
    def multiply_weights(self, session, weight_matrix):
        session.run(self._weights_mask_multiply, feed_dict = {self._weights_placeholder: weight_matrix})
        return
    
    def multiply_biases(self, session, bias_matrix):
        session.run(self._biases_mask_multiply, feed_dict = {self._biases_placeholder: bias_matrix})

class softmax_layer:
    
    def __init__(self, inputs, size, weight_init = 0.01, weight_decay = 0, bias_init = 0.1, bias_decay = 0, dummies = False, activation = None, dropout = 0):
        self._inputs = inputs
        self._size = size
        self._input_size = int(self._inputs.get_shape()[1])
        if type(weight_init) == float:
            self._weights = tf.Variable(tf.truncated_normal([self._input_size, self._size], stddev=weight_init), name='Softmax_Weights')
        if isinstance(weight_init, np.ndarray):
            self._weights = tf.Variable(weight_init, name = 'Softmax_Weights', dtype = tf.float32)
        self._weight_decay = weight_decay
        self._decoupled_weight_decay = None
        if type(bias_init) == float:
            self._biases = tf.Variable(tf.constant(bias_init, shape = [self._size]), name = 'Softmax_Biases')
        if isinstance(bias_init, np.ndarray):
            self._biases = tf.Variable(bias_init, name = 'Softmax_Biases', dtype = tf.float32)
        self._bias_decay = bias_decay
        self._dummies = None
        #self._dummies = tf.Variable(tf.constant(1.0, shape = [self._size], name = 'Dummies'))
        self._activation = None
        self._dropout = dropout
        self._dropout_mask = tf.floor(tf.random_uniform(shape = (self._input_size, self._size), maxval = 1) + 1-self._dropout)
        self._activate = tf.nn.softmax(tf.matmul(self._inputs, tf.multiply(self._weights, self._dropout_mask)) + self._biases)
        self._activate_logits = tf.matmul(self._inputs, self._weights) + self._biases
        self._activate_relu_logits = tf.nn.relu(self._activate_logits)
        self._activated_weights = tf.multiply(self._weights, tf.transpose(self._inputs))
        self._weights_placeholder = tf.placeholder(tf.float32, shape = self._weights.shape)
        self._weights_assign_op = tf.assign(self._weights, self._weights_placeholder)
        self._biases_placeholder = tf.placeholder(tf.float32, shape = self._biases.shape)
        self._biases_assign_op = tf.assign(self._biases, self._biases_placeholder)
        self._weights_mask_multiply = tf.multiply(self._weights, self._weights_placeholder)
        self._biases_mask_multiply = tf.assign(self._biases, tf.multiply(self._biases, self._biases_placeholder))
        
    @property    
    def activate(self):
        return self._activate
    
    @property    
    def activate_logits(self):
        return self._activate_logits
    
    @property
    def activate_relu_logits(self):
        return self._activate_relu_logits
    
    @property
    def get_decay_rates(self):
        return [(self._weights, self._weight_decay), (self._biases, self._bias_decay)]
    
    @property
    def weights(self):
        return self._weights
    
    @property
    def activated_weights(self):
        return self._activated_weights
    
    @property
    def biases(self):
        return self._biases
    
    @property
    def get_type(self):
        return 'Softmax'
    
    def assign_weights(self, session, weight_matrix):
        session.run(self._weights_assign_op, feed_dict = {self._weights_placeholder: weight_matrix})
        return
    
    def assign_biases(self, session, bias_matrix):
        session.run(self._biases_assign_op, feed_dict = {self._biases_placeholder: bias_matrix})
        return
    
    def multiply_weights(self, session, weight_matrix):
        session.run(self._weights_mask_multiply, feed_dict = {self._weights_placeholder: weight_matrix})
        return
    
    def multiply_biases(self, session, bias_matrix):
        session.run(self._biases_mask_multiply, feed_dict = {self._biases_placeholder: bias_matrix})
        
class conv_layer:
    
    def __init__(self, inputs, height, width, filters, padding, stride, channels = 1, weight_init = 0.01, weight_decay = 0, bias_init = 0.1, bias_decay = 0, activation = 'relu', flatten = False, dropout = 0):
        self._inputs = inputs
        self._input_size = int(self._inputs.get_shape()[1])
        self._size = filters
        # Filter dimensions
        self._height = height
        self._width = width
        self._filters = filters
        self._channels = channels
        self._size = self._height*self._width*self._filters*self._channels
        # Filter parameters
        self._padding = padding
        self._stride = stride
        # Input dimensions
        self._batch_size = inputs.get_shape()[0]
        self._input_height = inputs.get_shape()[1]
        self._input_width = inputs.get_shape()[2]
        self._input_channels = int(inputs.get_shape()[3])
        # Output dimensions
        self._output_height = (self._input_height + 2*padding - self._height)//self._stride + 1
        self._output_width = (self._input_width + 2*padding - self._width)//self._stride + 1
        self._flatten = flatten
        self._activation = activation
        self._dropout = dropout
        self._dropout_mask = tf.floor(tf.random_uniform(shape = (self._height, self._width, self._input_channels, self._filters), maxval = 1) + 1-self._dropout)
        if type(weight_init) == float:
            self._weights = tf.Variable(tf.truncated_normal([self._height, self._width, self._input_channels, self._filters], stddev=weight_init), name='Filter_Weights')
        if weight_init == 'xavier':
            self._weights = tf.Variable(tf.truncated_normal([self._height, self._width, self._input_channels, self._filters], stddev=tf.sqrt(3. / (self._input_size + self._size))), name='Weights')
        
        if isinstance(weight_init, np.ndarray):
            self._weights = tf.Variable(weight_init, name = 'Filter_Weights', dtype = tf.float32)
        self._weight_decay = weight_decay
        if type(bias_init) == float:
            self._biases = tf.Variable(tf.constant(bias_init, shape = [self._filters]), name = 'Filter_Biases')
        if isinstance(bias_init, np.ndarray):
            self._biases = tf.Variable(bias_init, name = 'Filter_Biases', dtype = tf.float32)
        self._bias_decay = bias_decay
        if flatten == False:
            self._activate = tf.nn.relu(tf.add(tf.nn.conv2d(self._inputs, tf.multiply(self._weights, self._dropout_mask), strides = [1,1,1,1], padding = "SAME"), self._biases))
        if flatten == True:
            self._activate = tf.reshape(tf.nn.relu(tf.add(tf.nn.conv2d(self._inputs, self._weights, strides = [1,1,1,1], padding = "SAME"), self._biases)), [-1 , self._output_height*self._output_width*self._filters])
        self._weights_placeholder = tf.placeholder(tf.float32, shape = self._weights.shape)
        self._weights_assign_op = tf.assign(self._weights, self._weights_placeholder)
        self._biases_placeholder = tf.placeholder(tf.float32, shape = self._biases.shape)
        self._biases_assign_op = tf.assign(self._biases, self._biases_placeholder)
            
    @property    
    def get_inputs(self):
        return self._inputs
    
    @property
    def get_input_shape(self):
        return [int(i) for i in self._inputs.get_shape()[1:]]
    
    @property
    def get_padding(self):
        padding = [0, 0, 0, 0]
        if self._height %2 != 0:
            padding[0] = int((self._height - 1)/2)
            padding[1] = padding[0]
        if self._width %2 != 0:
            padding[2] = int((self._width - 1)/2)
            padding[3] = padding[2]
        if self._height %2 == 0:
            padding[0] = int((self._height - 1)//2)
            padding[1] = int(self._height/2)
        if self._width %2 == 0:
            padding[2] = int((self._width - 1)//2)
            padding[3] = int(self._width/2)
        if self._height == 1:
            padding[:2] = 0
        if self._width == 1:
            padding[2:] = 0
        return padding
    
    @property    
    def activate(self):
        return self._activate
    
    @property
    def get_decay_rates(self):
        return [(self._weights, self._weight_decay), (self._biases, self._bias_decay)]
    
    @property
    def weights(self):
        return self._weights
    
    @property
    def biases(self):
        return self._biases
    
    @property
    def get_type(self):
        return 'Convolution'
    
    def assign_weights(self, session, weight_matrix):
        session.run(self._weights_assign_op, feed_dict = {self._weights_placeholder: weight_matrix})
        return
    
    def assign_biases(self, session, bias_matrix):
        session.run(self._biases_assign_op, feed_dict = {self._biases_placeholder: bias_matrix})
        return
    
    def multiply_weights(self, session, weight_matrix):
        session.run(self._weights_mask_multiply, feed_dict = {self._weights_placeholder: weight_matrix})
        return
    
    def multiply_biases(self, session, bias_matrix):
        session.run(self._biases_mask_multiply, feed_dict = {self._biases_placeholder: bias_matrix})
        return

        
class sparse_layer:
    
    def __init__(self, weight_init, bias_init, inputs = None):
        self._inputs = inputs
        self._activation = 'relu'
        self._weights = weight_init
        self._biases = bias_init
        self._weight_nnz = self._weights.nnz
    
    @property    
    def get_inputs(self):
        return self._inputs

    @property    
    def activate(self):
        dot_product = self._inputs@self._weights
        add_biases = dot_product + self._biases
        relu = add_biases*(add_biases>0)
        return relu

    @property
    def softmax_activate(self):
        dot_product = self._inputs@self._weights
        add_biases = dot_product + self._biases
        softmax = np.array([[np.exp(i)/sum([np.exp(j) for j in k]) for i in k] for k in add_biases])
        return softmax
        
    @property
    def weights(self):
        return self._weights
    
    @property
    def activate_weights(self):
        act_weights = self._weights.multiply((np.transpose(self._inputs)))
        return act_weights
    
    @property
    def biases(self):
        return self._biases
    
    @property
    def activate_biases(self):
        return self._activate_biases
    
    @property
    def get_type(self):
        return 'Sparse'
    
