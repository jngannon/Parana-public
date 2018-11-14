import tensorflow as tf
import numpy as np
from scipy.sparse import csr_matrix


def decouple(weights, input_shape, padding, stride = 1):
    
    height = input_shape[0] + padding[0] + padding[1]
    width = input_shape[1] + padding[2] + padding[3]
    channels = input_shape[2]
    v_strides = int((height+1 - weights.shape[0])/stride)
    h_strides = int((width+1 - weights.shape[1])/stride)
    decoupled_weight_matrix = []
    weights = np.transpose(weights, [3, 0, 1, 2])
    for v in range(v_strides):
        for h in range(h_strides):
            for fil in weights:
                weight_row = np.zeros(height*width*channels)
                r = 0
                for row in fil:
                    weight_row[v*width*channels + h*channels + r*width*channels: v*width*channels + h*channels + r*width*channels + weights.shape[1]*channels] = np.reshape(row, -1)
                    r += 1
                decoupled_weight_matrix.append(weight_row)
    #remove the cols that correspond to padding
    #top
    delete_indices = [i for i in range(0, width*channels*padding[0])]
    #left and right
    for i in range(input_shape[0]):
        #left
        start = padding[0]*width*channels + i*width*channels
        delete_indices.extend([j for j in range(start,start+padding[2]*channels)])
        #right
        start = padding[0]*width*channels + i*width*channels + padding[2]*channels + input_shape[1]*channels
        delete_indices.extend([j for j in range(start,start+padding[3]*channels)])
    #bottom
    start = padding[0]*width*channels + input_shape[0]*width*channels 
    delete_indices.extend([i for i in range(start, start+width*channels*padding[1])])
    #delete them here
    decoupled_weight_matrix = np.delete(decoupled_weight_matrix, delete_indices, 1)
    return np.transpose(decoupled_weight_matrix)

def sparse_decouple(weights, input_shape, padding, stride = 1):
    height = input_shape[0] + padding[0] + padding[1]
    width = input_shape[1] + padding[2] + padding[3]
    channels = input_shape[2]
    v_strides = int((height+1 - weights.shape[0])/stride)
    h_strides = int((width+1 - weights.shape[1])/stride)
    row = []
    column = []
    data = []
    weights = np.transpose(weights, [3, 0, 1, 2])
    
    #columns that correspond to padding
    #top
    delete_indices = [i for i in range(0, width*channels*padding[0])]
    #left and right
    for i in range(input_shape[0]):
        #left
        start = padding[0]*width*channels + i*width*channels
        delete_indices.extend([j for j in range(start,start+padding[2]*channels)])
        #right
        start = padding[0]*width*channels + i*width*channels + padding[2]*channels + input_shape[1]*channels
        delete_indices.extend([j for j in range(start,start+padding[3]*channels)])
    #bottom
    start = padding[0]*width*channels + input_shape[0]*width*channels 
    delete_indices.extend([i for i in range(start, start+width*channels*padding[1])])
    delete_indices = np.array(delete_indices)
    for v in range(v_strides):
        print('v',v)
        for h in range(h_strides):
            for fil in weights:
                r = 0
                weight_row = np.zeros(height*width*channels)
                for filter_row in fil:
                    weight_row[v*width*channels + h*channels + r*width*channels: v*width*channels + h*channels + r*width*channels + weights.shape[1]*channels] = np.reshape(filter_row, -1)
                    r += 1
                    
                    ## Still havn't thought this through. also dont put in 0 values, and also put the correct 
                filter_index = 0    
                for point in enumerate(weight_row):
                    if point[0] not in delete_indices and point[1]  != 0:
                        row.append(filter_index)
                        column.append(v)
                        data.append(point[1])
                        filter_index +=1
    decoupled_weight_matrix = csr_matrix((data, (row, column)), shape = (v_strides, h_strides*width*channels))
    return