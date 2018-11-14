import numpy as np
import tensorflow as tf 

def consolidate_fc_biases(session, layers):
    
    weights = []
    biases = []
    
    for i in layers:
        if i.get_type == 'Fully_Connected' or i.get_type == 'Softmax':
            weights.append(i.weights)
            biases.append(i.biases)
    
    depth = range(len(weights)-1)
    #Get tensors into numpy arrays.
    weight_list = [session.run(i) for i in weights]
    print([i.shape for i in weight_list])
    bias_list = [session.run(i) for i in biases]
    for i in depth:
        #Sum along the first axis to find which activations have all zero weights
        zeros = np.sum(np.abs(weight_list[i]), axis=0)
        #Get the indices of all zero weight activations
        zero_indices = np.where(zeros == 0)[0]
        if len(zero_indices)>0:
            #Indices of parameters to keep
            keep_indices = np.where(zeros != 0)[0]
            #Weights to be kept, with non zero values in their rows
            weight_list[i] = weight_list[i][:, keep_indices]
            #Biases to be consolidated into the biases of the next layer
            zero_biases = bias_list[i][zero_indices]
            #Biases to be kept
            bias_list[i] = bias_list[i][keep_indices]
            #Values of the biases that would have been passed to the next layer
            activated_biases = session.run(layers[i].activate_biases)[zero_indices]
            #Dot product of biases to be consolidated and associated weight columns of following layer
            add_to_next = np.dot(activated_biases, weight_list[i+1][zero_indices,:])
            #Remove columns associated with consolidated biases from following layer
            weight_list[i+1] = weight_list[i+1][keep_indices,:]
            #Add the dot product, which remains constant regardless of input, to the following layers biases.
            bias_list[i+1] = np.add(add_to_next, bias_list[i+1])
        else:
            print ('layer',i+1,'No activations removed')
    print('Bias consolidation complete, final layer sizes:')
    print([i.shape for i in weight_list])
    return weight_list, bias_list
        