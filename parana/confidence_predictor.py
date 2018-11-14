""" A module that stores the mean and standard deviation of the correct outputs of a model. Need to think about what to return... 

will need a class module to store things """
import tensorflow as tf
import numpy as np

class confidence_predictor:
    
    def __init__(self, model, session, X, y):
        self._model = model
        self._session = session
        self._X = X
        self._y = y
        self._means = []
        self._stds = []
        
    def get_distributions(self, data):
        all_predictions = self._session.run(self._model.logitoutput, feed_dict = {self._X : data[0]})
        correct_preds_only = [[] for i in range(data[1].shape[1])]
        for i in zip(data[1], all_predictions):
            if np.argmax(i[0]) == np.argmax(i[1]):
                correct_preds_only[np.argmax(i[0])].append(max(i[1]))
        means = [np.mean(i) for i in correct_preds_only]
        stds = [np.std(i) for i in correct_preds_only]
        self._means = means
        self._stds = stds
        return means, stds
    
    def predict_stds(self, data):
        """Returns a value in the column of the max value that is the standard deviations from the mean, and zeros everywhere else."""
        predictions = []
        if self._means == []:
            print('Get distributions first')
            return
        all_predictions = self._session.run(self._model.logitoutput, feed_dict = {self._X : data[0]})
        for i in all_predictions:
            new_output = np.zeros(shape = i.shape)
            amax = np.argmax(i)
            new_output[amax] = (np.abs(max(i) - self._means[amax]))/self._stds[amax]
            predictions.append(new_output)
        
        return predictions

    def get_outputs(self, data):
        all_predictions = self._session.run(self._model.logitoutput, feed_dict = {self._X : data[0]})
        correct_preds_only = [[] for i in range(data[1].shape[1])]
        for i in zip(data[1], all_predictions):
            if np.argmax(i[0]) == np.argmax(i[1]):
                correct_preds_only[np.argmax(i[0])].append(max(i[1]))
        return correct_preds_only
    