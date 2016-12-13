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
    
    def __init__(self, l=[2,9,8,1], a='sigmoid', bias=1): # l[0] is the input layer. L[1] through L[n-1] are hidden layers. L[n] is the output layer. Arbitrarily deep ANNs can now be initialized.
        self.a = a # activation function options: 'sigmoid', 'step', 'rect', 'softplus', 'tanh'
        self.bias = bias # bias node input. To turn bias nodes off, set to 0.
        self.l = l
        self.initWeights()
        
    def initWeights(self):
        self.w = []
        for i in range(len(self.l)-1):
            n = 2*np.random.random_sample((self.l[i],self.l[i+1]))-1 # weights are initialized to random values between -1 and 1
            self.w.append(n)
        self.b = 2*np.random.random_sample((len(self.l)-1))-1 # bias nodes get appended to the end
        self.w.append(self.b)
                
    def getWeights(self):
        return self.w
        
    def setWeights(self,new):
        for i in range(len(new)):
            if len(new[i]) != self.l[i]: # if the new weight setup is different 
                print("Warning! Network configuration is different than original specification. Continuing regardless")
                # self.l[i] = len(new[i]) # for making sure l is consistent with new weights
        self.w = new
        
    def feedForward(self, inputs, brk=False):
        try:
            a = np.dot(inputs,self.w[0]) + self.w[len(self.w)-1][0]*self.bias # the second addition is the bias. Their index is at the end of the array. Their values are added to the activations. I verified this with hand calculations
            for i in range(len(a)):
                a[i] = sigmoid(a[i])
        except ValueError:
            print("Input dimensions are incorrect")
            return
            
        for i in range(1,len(self.w)-1):
            a = np.dot(a, self.w[i]) + self.w[len(self.w)-1][i]*self.bias # dot product of activations against neuron weights
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
        
    def bp(self,inp,target): # inp is an array or arrays of values. target is the matching intended output. If you only want to do bp on a single input, you still need to have nested brackets, like [[i_1, i_2, ..., i_n]], where n is the no. of neurons in the input layer.
        print("Don't use this yet")
        for i in range(len(inp)):
            out = self.feedForward(inp[i])
            print("Out: ", out)
            error = (1/2)*((np.subtract(target[i],out))**2) # squared error
            print(error)

#
# activation functions
#

def sigmoid(finalSum): # if sum is positive, return 1. else, return 0
    if finalSum > 8:
        return 1
    if finalSum < -8:
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
    
def tanh(finalSum):
    return math.tanh(finalSum)

def sigmoidDeriv(finalSum):
    p = sigmoid(finalSum)
    return p*(1-p)
    
def deltaOuter(o_i, o_j, t_j, c=0.02):
    return -c*o_i*(o_j - t_j)*o_j*(1 - o_j)
    
def deltaInner(o_i, o_j, j, l, c=0.02):
    s = 0.0
    for i in len(range(l)):
        s += deltaInner(l)*j[i]
    return -c*o_i*(s)*o_j*(1-o_j)


def demo1(): # or table demonstration
    b=NN([2,1],a='sigmoid',bias=1)
    b.setWeights([np.array([[10],[10]]), np.array([0])])
    print("Perceptron that demonstrates correct responses to OR truth table inputs")
    print("")
    print("Weights: ", b.getWeights())
    print("")
    print("In: [0][0]. Out: ", b.feedForward([0,0]))
    print("In: [1][0]. Out: ", b.feedForward([1,0]))
    print("In: [0][1]. Out: ", b.feedForward([0,1]))
    print("In: [1][1]. Out: ", b.feedForward([1,1]))
    
demo1()
