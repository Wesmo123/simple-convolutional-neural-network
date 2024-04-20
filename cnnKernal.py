import numpy as np
import initialize

class cnnKernal:    #class for the cnn's kernals, allows easy access to an kernals entire structure easily and stores all the layers in one place
    def __init__(self, initArr=0, bias=0, ksize=0, ichannel=0, kernalNo=0, pool='false'): 

        if pool == 'false':
            self.bias = bias
            self.size = ksize   # size of the kernal represented by an integer, ie ksize = 3 means that the kernal is 3x3
            self.kernalNo = kernalNo    #number of the kernal in the layer
            self.depth = ichannel   # how many input layers are expected and the subsequent depth of the kernal matrix

            if(ichannel == 1):  # edge case check for if there is only 1 input channel
                self.kArray = np.zeros((1,ksize))   #creates a place holder zero array to allow the array to not automatically flatten
            else:   # else carry on as usual
                self.kArray = np.zeros((1, ksize, ksize))   #creates a place holder zero array to allow the array to not automatically flatten

            for iterz in range(ichannel):   # function is carried out for however many input channels there are
                newKernalRow = np.zeros((1, ksize)) #creates a place holder zero array to allow vstacking and for the array to not automatically flatten
                if(iterz == ichannel - 1):  #check when the function is about to complete and deletes the place holder zero array if it is true
                    self.kArray = np.delete(self.kArray, (0), axis=0)   

                for itery in range(ksize):  #iterates over the arrays within the initarray 2d section and stacks them with vstack
                    newKernalRow = np.vstack([newKernalRow, initArr[iterz, itery, :, kernalNo]])    

                    if(itery == ksize - 1 and ichannel != 1): # check if function is ready to complete and if there is only 1 input channel
                        newKernalRow = np.delete(newKernalRow, (0), axis=0) #deletes the placeholder 0 array that is used to keep the shape of newKernalRow
                        newKernalRow = newKernalRow[np.newaxis, :] # makes the newKernalArray 3d
                        self.kArray = np.concatenate((self.kArray, newKernalRow))   # appends the newKernalArray to the existing kArray

                    elif(itery == ksize -1 and ichannel == 1):  # check if function is ready to complete and if there is only 1 input channel
                        newKernalRow = np.delete(newKernalRow, (0), axis=0)     #deletes the placeholder 0 array that is used to keep the shape of newKernalRow
                        self.kArray = newKernalRow[np.newaxis, :]   # in the case where there is only 1 input channel simply sets the kArray to the newKernalRow

        elif pool == 'true':
            self.size = ksize
            self.depth = ichannel
            self.kernalNo = kernalNo

    def print(self):    #simply prints the stored 
        print(self.kArray)

    def channelSeparator(self, inputmap):   # seperate the input channels into seperate arrays and stores them in a list

        channelArrays = [inputmap[:, :, i] for i in range(self.depth)]

        return channelArrays

    def convolve(self, inputMap, mapdim1, mapdim2): # convolution operation, HAS NO STRIDING OR PADDING OPTIONS

        container = np.zeros((1, mapdim2 - (self.size - 1))) # container for outputmap, uses a array of zeros to keep its shape
        for steps in range(0, mapdim1 - (self.size - 1)): # calculates the number of rows in the output map and uses them to tell the kernal when to "step down" to the next layer, WILL NOT WORK WITH STRIDING
            rowcontainer = np.array([]) # container for the convolved rows before they are added to the final feature map array
            for coloumnL in range(0, mapdim2 - (self.size - 1)):    # tracks the back end of the kernal to know when to stop and step down
                if self.depth != 1: # makes sure that there is atleast 2 layers
                    convolve = 0
                    for iter in range(0, self.depth):   #performs the convolution operation for however many layers there are
                        convolve = convolve + np.sum(inputMap[iter][0 + steps : self.size + steps, 0 + coloumnL : self.size + coloumnL] * self.kArray[iter])    #the convolution operation done by elementwise multiplication
                    convolve = convolve + self.bias[self.kernalNo]  # adds bias into the convolution
                    rowcontainer = np.append(rowcontainer, convolve) # appends the convolved element to the new row

            container = np.vstack((container, rowcontainer)) # adds the new row onto the container array, which is our new featuremap

        container = np.delete(container, (0), axis=0)   # deletes the placeholder zero array used to keep the containers shape
        return container
    
    def maxPool(self, inputMap, mapdim1, mapDim2):
        print("pool stub")

    def activationFunction(self):
        print("activationFunction")

        #TODO:  finish maxPool method
        #       finish activationFunction method
        #       Implement Stride and Padding capabilities
        #       Implement 1x1 kernal functionality
