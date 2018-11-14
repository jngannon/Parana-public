import tensorflow as tf
import numpy as np
import pickle

class saver:
    
    def __init__(self, model, session):
        self._model = model
        self._session = session
        self._variables_list = model.train_variables
        self._parameter_list = None
        self._accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self._model.labels, 1), tf.argmax(self._model.modeloutput, 1)), tf.float32))
        
    def store_parameters(self):
        self._parameter_list = [self._session.run(i) for i in self._variables_list]
        return
    
    def split_accuracy(self, session, stages, inputs, labels):
        mean_accuracy = 0
        total_inputs = len(inputs)
        batch_size = total_inputs/stages
        for i in self._model.layers:
            i.dropout = 0
        for i in range(stages):
            mean_accuracy += session.run(self._accuracy, feed_dict = {self._model.inputs: inputs[int(i*batch_size):int((i+1)*batch_size-1)], self._model.labels: labels[int(i*batch_size):int((i+1)*batch_size-1)]})
        for i in self._model.layers:
            i._dropout = self._model.dropout
        return mean_accuracy/stages
    
    def return_variables(self):
        return self._parameter_list
    
    def return_parameters(self):
        return self._parameter_list
    
    def pickle_parameters(self, filename):
        if not self._parameter_list:
            print('You need to store some parameters first')
            return
        pickle.dump(self._parameter_list, open(filename, 'wb'))
        print('pickled')
        return
    
    def load_parameters(self, filename):
        loaded_parameters = pickle.load(open(filename, 'rb'))
        self._parameter_list = loaded_parameters
        for i in enumerate(self._model.layers):
            i[1].assign_weights(self._session, self._parameter_list[i[0]*2])
            i[1].assign_biases(self._session, self._parameter_list[i[0]*2+1])
        print('Parameters loaded from ', filename)
        return
    
    def restore_parameters(self):
        for i in enumerate(self._model.layers):
            i[1].assign_weights(self._session, self._parameter_list[i[0]*2])
            i[1].assign_biases(self._session, self._parameter_list[i[0]*2+1])
        return
    
    def store_layer(self, layer):
        for i in enumerate(self._model.layers):
            if layer == i[1]:
                self._parameter_list[i[0]] = self._session.run(self._variables_list[i[0]*2])
                self._parameter_list[i[0]+1] = self._session.run(self._variables_list[i[0]*2+1])                                     
        return
    
    def restore_layer(self, layer):
        for i in enumerate(self._model.layers):
            if layer == i[1]:
                i[1].assign_weights(self._session, self._parameter_list[i[0]*2])
                i[1].assign_biases(self._session, self._parameter_list[i[0]*2+1])
        return
    
    def get_model_parameters():
        
        #
        
        return
    
    def store_sparse(self):
        self._parameter_list = self._model.train_variables
        return
    
    def restore_sparse(self):
        for i in enumerate(self._model.layers):
            i[1]._weights =  self._parameter_list[i[0]*2]
            i[1]._biases = self._parameter_list[i[0]*2+1]
        return