import numpy as np
import random
import math

#
# new neural network. Uses numpy library for fast matrix multiplication. Supports an arbitrary amount of layers.
# type n=NN() to initialize.
# l parameter specifies layers.
# a parameter specifies the activation function.
# weights are automatically initialized. To set your own, use setWeights(). To make sure you don't break the dimensions, use getWeights() and modify the vals there.
#

class NN(object):
    
    def __init__(self, l=[2,5,4,1], a='sigmoid'): # l[0] is the input layer. L[1] through L[n-1] are hidden layers. L[n] is the output layer. Arbitrarily deep ANNs can now be initialized.
        self.a = a # activation function options: 'sigmoid', 'step', 'rect', 'softplus', 'tanh'
        self.l = l
        self.initWeights()
        
    def initWeights(self):
        self.w = []
        for i in range(len(self.l)-1):
            n = np.random.random((self.l[i],self.l[i+1]))
            print(n)
            self.w.append(n)
            print("")
        
    def getWeights(self):
        return self.w
        
    def setWeights(self,new): # work has to be done to make sure that l and the weights match
    # if columnar lengths are different, flag error
        self.w = new
        
    def feedForward(self, inputs):
        try:
            a = np.dot(inputs,self.w[0])
            for i in range(len(a)):
                a[i] = sigmoid(a[i])
        except ValueError:
            print("Input dimensions are incorrect")
            return
            
        for i in range(1,len(self.w)):
            a = np.dot(a, self.w[i])
            for i in range(len(a)):
                if self.a == 'step':
                    a[i] = step(a[i])
                elif self.a == 'rect':
                    a[i] = rect(a[i])
                elif self.a == 'softplus':
                    a[i] = softplus(a[i])
                elif self.a == 'tanh':
                    a[i] = tanh(a[i])
                else:
                    a[i] = sigmoid(a[i])
            
        return a
#
# activation functions
#
        
def sigmoid(finalSum): # if sum is positive, return 1. else, return 0
    if finalSum > 5:
        return 1
    if finalSum < -5:
        return 0
    else:
        return 1/(1+((math.e) ** (-finalSum))) # sigmoid function

def step(finalSum): # binary activation
    if finalSum <= 0:
        return 0
    else:
        return 1
        
def rect(finalSum): # rectilinear function.
    if finalSum <= 0:
        return 0
    else:
        return finalSum
        
def softplus(finalSum): # soft rectilinear function
    return math.log(1+math.e**finalSum)
    
def tanh(self, finalSum):
    return math.tanh(finalSum)
    
# demo
n=NN()