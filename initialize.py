import numpy as np

# initialize weights for a kernal, use a initializion technique to initialize the weights of each kernal
# a kernal is a square matrix made up of weights, these weights are adjusted by algorythms as the algorythm learns

# input channels are the depth of number of feature maps in the input data, ie rgb images have 3 input channels while a greyscale image will have 1 input channel
# output channels are the depth or number of feature maps produced by applying filters to the input data, each kernal produces one output layer, also known as a feature map
# each output channel captures different aspects or features of the input data learned bu the corresponding filter.

# filter size (also known as kernal size) is a hyperparamater that determines the spatial extent of the local region that the filter considers during the convolution operation.
# The filter size is usually a square matrix, larger filter sizes capture more spatial information which can be good for detecting larger patters/features
# smaller sizes are good at capturing finer details and more localized patters, common image classifier sizes are 3x3, 5x5 and 7x7

# This function outputs a 4 dimensional array of random values in the format filters size x input channels x output channnels, if you had put in the function as follows
# initialize_weights(3, 3, 16, 'random') you would get a 4d array of 3x3x16 and to access the elements needed for your 3x3 kernel size you would access it as follows
# array[layer, row, :, kernal number], ie if you wanted to initialise your first 3x3 kernal for an RGB image with 16 outputs you would need to array[0-2, 0-2, :, 0] 

def initialize_weights(filter_size, input_channels, output_channels, initialization='random'):
    if initialization == 'random':
        return np.random.randn(input_channels, filter_size, filter_size, output_channels)
    elif initialization == 'xavier':
        stddev = np.sqrt(2 / (input_channels + output_channels))
        return np.random.randn(filter_size, filter_size, input_channels, output_channels) * stddev
    elif initialization == 'he':
        stddev = np.sqrt(2 / input_channels)
        return np.random.randn(filter_size, filter_size, input_channels, output_channels) * stddev
    else:
        raise ValueError("Invalid initialization method")
    
#initialize biases for a layer

# a bias is a value applied to the value obtained by the kernal calculation, this allows the activation of neurons even when all inputs are zero
# they allow the neural network to represent functions that do not pass through the origin
# essentially allowing the neural network to learn and represent patterns that are independent of the input values
# biases are usually randonly initialized to small constants or zero values randomly and updated by back propagation during training

def initialize_biases(output_channels):
    return np.zeros((output_channels,))


