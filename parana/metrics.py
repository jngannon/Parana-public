import numpy as np
from scipy.stats import rankdata

def rankmatrix(matrix):
    matrixshape = matrix.shape
    matrixranked = rankdata(np.abs(matrix), method = 'dense')
    return matrixranked.reshape(matrix.shape)-1

def get_rank_diff(param_list_1, param_list_2):
    arrays = len(param_list_1)
    newlist = []
    if arrays != len(param_list_2):
        print ('Lists arent the same')
        return
    for i in range(arrays):
        rank_matrix_1 = rankmatrix(param_list_1[i])
        rank_matrix_2 = rankmatrix(param_list_2[i])
        diff_matrix = np.subtract(rank_matrix_1, rank_matrix_2)
        newlist.append(diff_matrix)
    return newlist

def get_rank_metric(param_list_1, param_list_2):
    diff_matrix_list = np.abs(get_rank_diff(param_list_1, param_list_2))
    return [np.mean(i)/i.size for i in diff_matrix_list]

def get_abs_diff(param_list_1, param_list_2):
    arrays = len(param_list_1)
    newlist = []
    if arrays != len(param_list_2):
        print ('Lists arent the same')
        return
    for i in range(arrays):
        diff_matrix = np.subtract(np.abs(param_list_1[i]), np.abs(param_list_2[i]))
        newlist.append(diff_matrix)
    return newlist

def get_abs_diff_metric(param_list_1, param_list_2):
    diff_matrix_list = get_abs_diff(param_list_1, param_list_2)
    return [np.mean(np.abs(i))/i.size for i in diff_matrix_list]