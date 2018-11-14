import tensorflow as tf
import numpy as np

class sign_tracker:
    
    """Tracks sign changes"""
    
    def __init__(self, parameter_list, cutoff = None):
        self._maxs = [np.zeros(i.shape) for i in parameter_list]
        self._sums = [np.zeros(i.shape) for i in parameter_list]
        self._total = [np.zeros(i.shape) for i in parameter_list]
        self._itterations = range(len(parameter_list))
        self._cutoff = cutoff
        
        
    def update(self, parameter_list):
        current_sign = [np.sign(i) for i in parameter_list]
        self._total = [np.add(self._total[i], current_sign[i]) for i in self._itterations]
        self._sums = [np.where(np.sign(self._sums[i]) == current_sign[i],self. _sums[i], 0) for i in self._itterations]
        if self._cutoff:
            current_sign = [np.multiply(current_sign[i], np.abs(parameter_list[i])>self._cutoff) for i in self._itterations]
        self._sums = [np.add(self._sums[i], current_sign[i]) for i in self._itterations]
        self._maxs = [np.where(np.abs(self._sums[i])>np.abs(self._maxs[i]), self._sums[i], self._maxs[i]) for i in self._itterations]
    
    def get_max(self):
        return self._maxs
    
    def get_total(self):
        return self._total
    
    