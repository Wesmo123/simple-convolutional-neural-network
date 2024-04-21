import numpy as np
import cnnKernal
import initialize

class layer:
    def __init__(self, pooling='false', noKernals=16, kernalSize=3, inputChannels=3, outputChannels=16, initialization='random'):
        self.pooling = pooling
        self.noKernals = noKernals
        self.kernalSize = kernalSize
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.initialweights = initialize.initialize_weights(kernalSize, inputChannels, outputChannels, initialization)
        self.bias = initialize.initialize_biases(outputChannels)
        self.kernalList = []

        for iter in range(0, noKernals):
            self.kernalList.append(cnnKernal.cnnKernal(self.initialweights, self.bias, self.kernalSize, self.inputChannels, iter, self.pooling))
