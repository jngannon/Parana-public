import tensorflow as tf
import numpy as np
from parana.wiggles import get_absolute_deviation
from parana.wiggles import get_mean_const_label_noise
from parana.wiggles import get_mean_gauss_label_noise
from scipy import sparse
# get_mean_const_noise(noise_constant, session, parameters_list, cost, iterations, data_function, batch_size, X, y)

class lobotomizer:
    
    """ Removes parameters by setting their value to 0, based on a selected wigglyness metric"""
    
    def __init__(self, session, model, wigglyness, parameter_selection, cost, data_function, X, y, batch_size = 200, iterations = 100, parameters_list = None, layers_list = None, noise_constant = None, noise_vector = None):
        self._session = session
        self._model = model
        self._parameters_list = parameters_list
        self._layers_list = layers_list
        self._wigglyness = wigglyness
        self._noise_constant = noise_constant
        self._noise_vector = noise_vector
        self._parameter_selection = parameter_selection
        self._cost = cost
        self._data_function = data_function
        self._batch_size = batch_size
        self._X = X
        self._y = y
        if parameters_list:
            self._parameter_means = [np.mean(self._session.run(i)) for i in parameters_list]
            self._parameter_stds = [np.std(self._session.run(i)) for i in parameters_list]
            self._mask_list = [np.ones(shape=i.get_shape()) for i in parameters_list]
        if layers_list:
            self._parameter_means = []
            self._parameter_stds = []
            self._mask_list = []
        self._arrays = None
        self._iterations = iterations
    
    def reset_masks(self):
        self._mask_list = [np.ones(shape=i.get_shape()) for i in self._parameters_list]
    
    def get_wigglyness(self, wigglyness = None, parameters_list = None, iterations = None):
        if not iterations:
            iterations = self._iterations
        if not wigglyness:
            wigglyness = self._wigglyness
        if not parameters_list:
            parameters_list = self._parameters_list
        self._arrays = wigglyness(noise_constant = self._noise_constant,
                                  noise_vector = self._noise_vector,
                                  session = self._session, 
                                  parameters_list = self._parameters_list,
                                  layers_list = self._layers_list,
                                  cost = self._cost,
                                  iterations = iterations,
                                  data_function = self._data_function,
                                  batch_size = self._batch_size,
                                  X = self._X, y = self._y)
        return self._arrays
    
    
    def prune_step(self, parameter_ratio, layers_list = None, parameters_list = None, iterations = 100):
        if not self._arrays:
            self.get_wigglyness(self._wigglyness)
        if self._layers_list:
            parameters_list = [i.weights for i in self._layers_list]
        if layers_list:
            parameters_list = [i.weights for i in layers_list]
        if not parameters_list:
            parameters_list = self._parameters_list
        mask_list = [np.ones(shape=i.get_shape()) for i in self._parameters_list]
        #get_indices
        indices = self._parameter_selection(self._arrays, parameter_ratio)                    
        #update mask
        for i in range(len(parameters_list)):
            # get the index 
            for parameter in enumerate(self._parameters_list):
                if parameter[1] == parameters_list[i]:
                    mask_index = parameter[0]
            for j in indices[mask_index]:
                mask_list[mask_index][j[0],j[1]] = 0
        self._mask_list = mask_list
        #multiply mask
        for i in enumerate(self._parameters_list):
            self._model.multiply_variables[self._model.train_variables.index(i[1])](self._session, mask_list[i[0]])
        parameters_list = None
    
    def reinitialize_step(self, parameter_ratio, parameters_list = None, iterations = 100):
        """ Get mean and var for the parameter array, reinitialize instead of set to 0"""
        if not self._arrays:
            get_wigglyness(self._wigglyness)
        if not parameters_list:
            parameters_list = self._parameters_list
        mask_list = [np.ones(shape=i.get_shape()) for i in parameters_list]
        means_list = [self._parameter_means[self._parameters_list.index(i)] for i in parameters_list]
        stds_list = [self._parameter_stds[self._parameters_list.index(i)] for i in parameters_list]
        #get_indices
        indices = self._parameter_selection(self._arrays, parameter_ratio)                    
        #update mask
        for i in range(len(parameters_list)):
            for j in indices[i]:
                mask_list[i][j[0],j[1]] = np.random.normal(means_list[i], stds_list[i])
        #multiply mask
        for i in enumerate(parameters_list):
            self._model.multiply_variables[self._model.train_variables.index(i[1])](self._session, mask_list[i[0]])
            
    def split_reinitialize_step(self, parameter_ratio, parameters_list, iterations = 100):
        """ Get mean and var for the parameter array, set half to 0 and half to rand"""
        if not self._arrays:
            get_wigglyness(self._wigglyness)
        if not parameters_list:
            parameters_list = self._parameters_list
        mask_list = [np.ones(shape=i.get_shape()) for i in parameters_list]
        #get_indices
        indices = self._parameter_selection(self._arrays, parameter_ratio)                    
        #update mask
        for i in range(len(parameters_list)):
            for j in indices[i]:
                mask_list[i][j[0],j[1]] = 0
        #multiply mask
        for i in enumerate(parameters_list):
            self._model.multiply_variables[self._model.train_variables.index(i[1])](self._session, mask_list[i[0]])
            
    def multiply_mask(self):
        for i in enumerate(self._parameters_list):
            self._model.multiply_variables[self._model.train_variables.index(i[1])](self._session, self._mask_list[i[0]])
            
    def get_mask(self):
        return self._mask_list    
    
class lobotomizer_2w:
    
    """ Removes parameters by setting their value to 0, based on a metric based on a difference between 2 wigglyness metrics"""
        
    def __init__(self, session, model, wigglyness, wigglyness2, metric, parameter_selection, cost, data_function, X, y, data_function2 = None, batch_size = 200, noise_constant = None, noise_vector = None, parameters_list = None, layers_list = None, noise_constant2 = None, noise_vector2 = None):
        self._session = session
        self._model = model
        self._parameters_list = parameters_list
        self._layers_list = layers_list
        self._wigglyness = wigglyness
        self._wigglyness2 = wigglyness2
        self._metric = metric
        self._noise_constant = noise_constant
        self._noise_vector = noise_vector
        self._noise_constant2 = noise_constant2
        self._noise_vector2 = noise_vector2
        self._parameter_selection = parameter_selection
        self._cost = cost
        self._data_function = data_function
        self._data_function2 = data_function2
        if not data_function2:
            self._data_function2 = data_function
        self._batch_size = batch_size
        self._X = X
        self._y = y
        if parameters_list:
            self._parameter_means = [np.mean(self._session.run(i)) for i in parameters_list]
            self._parameter_stds = [np.std(self._session.run(i)) for i in parameters_list]
            self._mask_list = [np.ones(shape=i.get_shape()) for i in parameters_list]
        if layers_list:
            self._parameter_means = []
            self._parameter_stds = []
            self._mask_list = []
        self._arrays = None
        self._wigglyness_arrays = None
        self._wigglyness_arrays2 = None
    
    def get_wigglyness(self, wigglyness = None, parameters_list = None, iterations = 100):
        if not iterations:
            iterations = self._iterations
        if self._layers_list:
            parameters_list = [i.weights for i in self._layers_list]
        if not parameters_list:
            parameters_list = self._parameters_list
        self._wigglyness_arrays = self._wigglyness(noise_constant = self._noise_constant,
                                             noise_vector = self._noise_vector,
                                             session = self._session,
                                             parameters_list = self._parameters_list,
                                             layers_list = self._layers_list,
                                             cost = self._cost,
                                             iterations = iterations,
                                             data_function = self._data_function,
                                             batch_size = self._batch_size,
                                             X = self._X, y = self._y)
        self._wigglyness_arrays2 = self._wigglyness2(noise_constant = self._noise_constant2,
                                               noise_vector = self._noise_vector2,
                                               session = self._session,
                                               parameters_list = self._parameters_list,
                                               layers_list = self._layers_list,
                                               cost = self._cost,
                                               iterations = iterations,
                                               data_function = self._data_function2,
                                               batch_size = self._batch_size,
                                               X = self._X, y = self._y)
        self._arrays = self._metric(self._wigglyness_arrays, self._wigglyness_arrays2)
        return self._arrays
    
    def prune_step(self, parameter_ratio, parameters_list = None, iterations = 100):
        if not self._arrays:
            self.get_wigglyness()
        if self._layers_list:
            parameters_list = [i.weights for i in self._layers_list]
        if not parameters_list:
            parameters_list = self._parameters_list
        mask_list = [np.ones(shape=i.get_shape()) for i in parameters_list]
        #get_indices
        indices = self._parameter_selection(self._arrays, parameter_ratio)                    
        #update mask
        for i in range(len(parameters_list)):
            for j in indices[i]:
                mask_list[i][j[0],j[1]] = 0
        self._mask_list = mask_list
        #multiply mask
        for i in enumerate(parameters_list):
            self._model.multiply_variables[self._model.train_variables.index(i[1])](self._session, mask_list[i[0]])
    
class sparse_lobotomizer:
    
    def __init__(self, model, wigglyness, iterations, data_function, parameter_selection, layers_list = None, noise_constant = None, noise_vector = None):
        self._model = model
        self._wigglyness = wigglyness
        self._iterations = iterations
        self._data_function = data_function
        self._parameter_selection = parameter_selection
        self._noise_constant = noise_constant
        self._noise_vector = noise_vector
        if not layers_list:
            self._layers_list = model.layers
        if layers_list:
            self._layers_list = layers_list
        self._weights_list = [i.weights for i in layers_list]
        self._arrays = None
        
    def get_wigglyness(self, iterations = None, layers_list = None, noise_constant = None, noise_vector = None):
        if not layers_list:
            layers_list = self._layers_list
        if not iterations:
            iterations = self._iterations
        self._arrays = self._wigglyness(model = self._model, 
                                        iterations = iterations, 
                                        data_function = self._data_function, 
                                        noise_constant = self._noise_constant, 
                                        noise_vector = self._noise_vector, 
                                        layers_list = layers_list)
        return self._arrays
    
    def prune_step(self, prune_ratio, layers_list = None):
        if not layers_list:
            layers_list = self._layers_list
        prune_ratio = 1-prune_ratio
        weights_list = [i.weights for i in layers_list]
        original_parameter_size = [i._weight_nnz for i in layers_list]
        if not self._arrays:
            self.get_wigglyness()
        indices = self._parameter_selection(array_list = weights_list,
                                            k_ratio = prune_ratio,
                                            original_parameter_size = original_parameter_size)
        
        for i in range(len(weights_list)):
                        row, col, data = sparse.find(weights_list[i])
                        rowcol = (np.delete(row, indices[i]), np.delete(col, indices[i]))
                        data = np.delete(data, indices[i])
                        layers_list[i]._weights = sparse.csr_matrix((data, rowcol), shape = weights_list[i].shape)
        
        return
     