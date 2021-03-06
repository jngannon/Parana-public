import numpy as np
from parana.metrics import rankmatrix

def get_k_min(array_list, k_ratio):
    """ Returns the indices of the K lowest values in a matrix"""
    return_list = []
    for i in array_list:
        k_parameters = int(i.size*k_ratio)
        indices =  np.argsort(i.flatten())[:k_parameters]
        update = np.vstack(np.unravel_index(indices, i.shape)).tolist()
        update = np.transpose(update)
        return_list.extend([update])
    return return_list

def get_k_smallest(array_list, k_ratio):
    """ Returns the indices of the K lowest absolute values in a matrix"""
    return_list = []
    for i in array_list:
        k_parameters = int(i.size*k_ratio)
        indices =  np.argsort(np.abs(i).flatten())[:k_parameters]
        update = np.vstack(np.unravel_index(indices, i.shape)).tolist()
        update = np.transpose(update)
        return_list.extend([update])
    return return_list

def get_k_biggest(array_list, k_ratio):
    """ Returns the indices of the K highest values in a matrix"""
    return_list = []
    for i in array_list:
        k_parameters = int(i.size*k_ratio)
        indices =  np.argsort(np.abs(i).flatten())[-k_parameters:]
        update = np.vstack(np.unravel_index(indices, i.shape)).tolist()
        update = np.transpose(update)
        return_list.extend([update])
    return return_list
    

def get_k_max(array_list, k_ratio):
    """ Returns the indices of the K highest values in a matrix"""
    return_list = []
    for i in array_list:
        k_parameters = int(i.size*k_ratio)
        indices =  np.argsort(i.flatten())[-k_parameters:]
        update = np.vstack(np.unravel_index(indices, i.shape)).tolist()
        update = np.transpose(update)
        return_list.extend([update])
    return return_list
    
def get_uniform(matrix, k_parameters):
    """ Returns K indices randomly chosen from the matrix"""
    return
    
def get_normal_low(array_list, k_ratio, std_d_ratio):
    """ Returns K indices from the matrix chosen randomly with a higher probability given to
    lowest values, called normal because that is the numpy function used, std_d(standard deviation) 
    represents a sort of energy parameter controlling how many indices further away from the lowest values
    will be chosen. In its limits small values of std_d will return get_k_min, and high will return get_uniform."""
    return_list = []
    for i in array_list:
        k_parameters = int(i.size*k_ratio)
        std_d = i.size*std_d_ratio
        sorted_indices =  np.argsort(np.abs(i).flatten())
        sorted_indices = np.vstack(np.unravel_index(sorted_indices, i.shape)).tolist()
        sorted_indices = np.transpose(sorted_indices).tolist()
        random_list = []
        while len(random_list) < k_parameters:
            # dividing the standard deviation by 2 because taking the absolute value doubles the number of values within 1 SD.
            rand = int(np.abs(np.random.normal(0, (std_d/2))))
            if rand not in random_list and rand < len(sorted_indices):
                random_list.extend([rand])
        return_list.extend([[sorted_indices[i] for i in random_list]])
    return return_list

    
def get_normal_high(array_list, k_ratio, std_d_ratio):
    """ Returns K indices from the matrix chosen randomly with a higher probability given to
    highest values, called normal because that is the numpy function used, std_d(standard deviation) 
    represents a sort of energy parameter controlling how many indices further away from the highest values
    will be chosen. In its limits small values of std_d will return get_k_max, and high will return get_uniform."""
    return_list = []
    for i in array_list:
        k_parameters = int(i.size*k_ratio)
        std_d = i.size*std_d_ratio
        sorted_indices =  np.argsort(-np.abs(i).flatten())
        sorted_indices = np.vstack(np.unravel_index(sorted_indices, i.shape)).tolist()
        sorted_indices = np.transpose(sorted_indices).tolist()
        random_list = []
        while len(random_list) < k_parameters:
            # dividing the standard deviation by 2 because taking the absolute value doubles the number of values within 1 SD.
            rand = int(np.abs(np.random.normal(0, (std_d/2))))
            if rand not in random_list and rand < len(sorted_indices):
                random_list.extend([rand])
        return_list.extend([[sorted_indices[i] for i in random_list]])
    return return_list
    
def get_normal_split(matrix, k_parameters, std_d):
    """ Returns K indices chosen from 2 distributions half favouring the lowest values, and hald favoring the highest
    Std_d(standard_deviation) represents the spread of these distributions giving a sort of energy parameter controlling
    how many parameters further away from the highest/lowest are chosen. At its limits, small values of std_d will return 
    an even split of get_k_max and get_k_min, and at high values will return get_uniform."""
    return

def get_maxs(matrix):
    """Returns a 1 in every row with the rest zeros"""
    maxs = np.zeros(shape=matrix.shape)
    for i in enumerate(np.argmax(matrix, axis = 1)):
        maxs[i[0]][i[1]] = 1
    return maxs

def get_zeros(array_list, k_ratio = None):
    """Returns the indices of all zero values"""
    zeros = []
    for i in array_list:
        zeros.append(np.argwhere(np.array(i) == 0))
    return zeros

def get_negatives(array_list, k_ratio = None):
    negatives = []
    for i in array_list:
        negatives.append(np.argwhere(np.array(i) < 0))
    return negatives

def get_positives(array_list, k_ratio = None):
    positives = []
    for i in array_list:
        positives.append(np.argwhere(np.array(i) > 0))
    return positives

#I know this will be stupid slow, but for now it works.
def inside_out_negative(array_list, k_ratio):
    return_list = []
    for i in array_list:
        i = np.multiply(i, (i<1))
        i = (rankmatrix(i))
        maxs = (np.max(i)+1)
        i = np.add(i, np.multiply((i==0), maxs))
        k_parameters = int(i.size*k_ratio)
        indices =  np.argpartition(i.flatten(), k_parameters)[:k_parameters]
        update = np.vstack(np.unravel_index(indices, i.shape)).tolist()
        update = np.transpose(update)
        return_list.extend([update])
    return return_list