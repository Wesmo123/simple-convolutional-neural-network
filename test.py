import numpy as np
import initialize
import cnnKernal
from dataloader import load_data

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

one = np.array([[1,2,3], [4,5,6]])
two = np.array([[0,1,0], [1,0,1]])
#print(np.sum(one*two))
#print(train_x_orig[0])

kern = cnnKernal.cnnKernal(initialize.initialize_weights(3, 3,  16,'random'), initialize.initialize_biases(16), 3 , 3 , 0, 'true')

#print(kern.print())


def layerCut(inputMap, currentstep, kSize):
    slice = inputMap[0 + currentstep : kSize + currentstep]
    return slice

def channelSeparator(inputmap, channels):   # seperate the input channels into seperate arrays and stores them in a list

    channelArrays = [inputmap[:, :, i] for i in range(channels)]

    return channelArrays

def convolve(inputMap, mapdim1, mapdim2):

    bias = 1 # change this to self.bias
    testarr = np.array([[[1,0,0],[0,1,0], [0,0,1]], [[1,0,0],[0,1,0], [0,0,1]], [[1,0,0],[0,1,0], [0,0,1]]]) # array for testing change to self.kArray
    kSize = 3   # change to self.size
    depth = 3 # the amount of layers the kernal has (also determined by the amount of inputchannels)
    stop = 0
    container = np.zeros((1, mapdim2 - (kSize - 1))) # container for outputmap, will NOT work with striding (3d, rows , coloumns)
    for steps in range(0, mapdim1 - (kSize - 1)): # calculates the number of rows in the output map and uses them to tell the kernal when to "step down" to the next layer, WILL NOT WORK WITH STRIDING
        rowcontainer = np.array([])
        for coloumnL in range(0, mapdim2 - (kSize - 1)):    # tracks the back end of the kernal to know when to stop and step down
            if depth != 1:
                convolve = 0
                for iter in range(0, depth):
                    convolve = convolve + np.sum(inputMap[iter][0 + steps : kSize + steps, 0 + coloumnL : kSize + coloumnL] * testarr[iter])
                convolve = convolve + bias
                rowcontainer = np.append(rowcontainer, convolve)
        container = np.vstack((container, rowcontainer))

    container = np.delete(container, (0), axis=0)
    return container

#print(kern.convolve(kern.channelSeparator(train_x_orig[0]), 64, 64))




