import numpy as np
from scipy import sparse


def get_k_min(array_list, k_ratio, original_parameter_size):
    # Returns a list of indices for a sparse array 
    return_list = []
    for i in enumerate(array_list):
        
        _, _, data = sparse.find(i[1])
        
        remove_ammount = int(len(data) - original_parameter_size[i[0]]*k_ratio)
        
        sort_data = np.argsort(abs(data))
        
        return_list.append(sort_data[:remove_ammount])
        
    return return_list