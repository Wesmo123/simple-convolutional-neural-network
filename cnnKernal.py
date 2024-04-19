import numpy as np
import initialize

class cnnKernal:    #class for the cnn's kernals, allows easy access to an kernals entire structure easily and stores all the layers in one place
    def __init__(self, initArr, ksize, ichannel, kernalNo): 
        
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

    def print(self):    #simply prints the stored kArray
        print(self.kArray)
                    
        