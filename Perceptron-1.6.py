from sineFunction import *

import random
import math

class Neuron(object):

    objID = 0
    
    def __init__(self, inputList=[], weightList=[], nType='h', bias=1): # initialization
        self.id = Neuron.objID
        Neuron.objID = Neuron.objID+1
        self.weightList = weightList
        self.inputList = inputList
        self.bias = bias

    def getType(self):
        return self.nType
        
    def setWeights(self, weightList=[]):
        self.weightList = weightList

    def getWeights(self):
        return self.weightList

    def setInputs(self, inputList=[], bias=1):
        self.inputList = inputList
        self.bias = bias

    def getInputs(self):
        return self.inputList

    def feedForward(self, weightList, inputList, biasNeuron): # sums all input*weight. Bias input. Calculates activation function. Returns summation.
        
        finalSum = 0.0

        if type(weightList) == int or type(weightList) == float:
            weightList = [weightList]

        if type(inputList) == int or type(inputList) == float:
            inputList = [inputList]
                
        for c in range(len(inputList)):
            finalSum += weightList[c] * inputList[c]
            
        finalSum += (biasNeuron.getInputs() * biasNeuron.getWeights()) # add bias neuron
            
        return self.activation(finalSum)

    def activation(self, finalSum): # if sum is positive, return 1. else, return 0

        if finalSum > 4.6:
            return 1
        elif finalSum < -4.6:
            return -1
        else:
            return 1/(1+((math.e) ** (-finalSum))) # sigmoid function

    def trainer(self, inputList=[], weightList=[], summation=0, desired=0, biasNeuron): # takes the value that comes out of feedforward. calculates error. updates weights

        if  type(weightList) == float or type(weightList) == int:
            weightList = [weightList]

        l = 0.025 # learning constant, slows down rate of learning
        error = desired - summation
        print(error)
        perturbation = 0.01 * error * random.randint * round(random.uniform(-1,1),5)

        for c in range(len(inputList)):
            weightList[c] = round(weightList[c] + (error * inputList[c] * l) + perturbation, 4)
            biasNeuron.setWeights(round(weightList[c] + (error * inputList[c] * l), + perturbation, 4))

        self.weightList = weightList
        return weightList

    def isConnected(self, c, n2):
        if [(c.getConnection())[0], (c.getConnection())[1]] == [self, n2]:
            return True
        else:
            return False

    def __repr__(self):
        return ('<Neuron {} with weights {} and bias {}>'.format(self.id, self.weightList, self.bias))

    def __str__(self):
        return ('<Neuron {}>'.format(self.id))
    

class Input(Neuron):
    def __init__(self, nType='i'):
        Neuron.__init__(self, inputList=[], weightList=[])
        self.nType = nType

class Hidden(Neuron):
    def __init__(self, nType='h'):
        Neuron.__init__(self, inputList=[], weightList=[])
        self.nType = nType

class Output(Neuron):
    def __init__(self, nType='o'):
        Neuron.__init__(self, inputList=[], weightList=[])
        self.nType = nType

class Bias(Neuron):
    def __init__(self, nType='b'): # inputList must always equal 1
        Neuron.__init__(self, inputList=1, weightList=0)
        self.nType = nType

def getRandomWeight(inputNum=None): # returns list of random floats from (-1,1). default argument returns single float. otherwise, returns float(s) inside a list

    if inputNum == None:
        return round(random.uniform(-1,1),4)
    elif type(inputNum) == int:
        j = 0
        randomWeightsList = []
        while j < inputNum:
            randomWeightsList.append(round(random.uniform(-1,1),4))
            j += 1
        j = 0
        return randomWeightsList

class Connection(): # sets up connection between two neurons
    def __init__(self, a=None, b=None, weight=0.0):
        self.a = a
        self.b = b
        self.weight = weight
        self.connectionList = [a,b,weight]

    def setConnection(self, a, b, weight = getRandomWeight()):
        self.a = a
        self.b = b
        self.weight = weight
        self.connectionList = [a,b,weight]
        
    def __repr__(self):
        return('<Connection between {} and {} with weight {})>'.format(self.a, self.b, self.weight))
        
def train(inputVal, desired): #inputVal = -12, desired = -1
    n0.setInputs([inputVal])
    s0 = n0.feedForward(n0.getWeights(), n0.getInputs(), b0)                # input
    n0.trainer(n0.getInputs(), n0.getWeights(), s0, desired, b0)
    
    s1 = n1.feedForward(n1.getWeights(), s0, b1)                        # hidden 1
    n1.trainer(n0.getInputs(), n1.getWeights(), s1, desired, b1)
    
    s2 = n2.feedForward(n2.getWeights(), s0, b1)                        # hidden 2
    n2.trainer(n0.getInputs(), n2.getWeights(), s2, desired, b1)
    
    s3 = n3.feedForward(n3.getWeights(), s0, b1)                        # hidden 3
    n3.trainer(n0.getInputs(), n3.getWeights(), s3, desired, b1)
    
    s4 = n4.feedForward(n4.getWeights(), [s1,s2,s3], b2)                # output (weight 1)
    n4.trainer(n0.getInputs(), n4.getWeights(), s4, desired, b2)

def feedThrough(inputVal):

    n0.setInputs([inputVal])
    s0 = n0.feedForward(n0.getWeights(), n0.getInputs())                # input
    s1 = n1.feedForward(n1.getWeights(), s0, b1)                        # hidden 1
    s2 = n2.feedForward(n2.getWeights(), s0, b1)                        # hidden 2
    s3 = n3.feedForward(n3.getWeights(), s0, b1)                        # hidden 3
    s4 = n4.feedForward(n4.getWeights(), s0, b1)                        # hidden 3
    s5 = n4.feedForward(n4.getWeights(), [s1,s2,s3,s4], b2)             # output (weight 1)

    return s4

def test():
    c = 0
    while c < len(sineX):
        train(sineX[c], sineY[c])
        train(sineX[c], sineY[c])
        c += 1

n0 = Input()  # input 1
n1 = Input()  # input 2
n2 = Hidden() # hidden
n3 = Hidden() # hidden
n4 = Hidden() # hidden
n5 = Hidden() # hidden
n6 = Output() # output

b0 = Bias(1, 0) # layer 0 bias
b1 = Bias() # layer 1 bias
b2 = Bias() # layer 2 bias

for layer in neuronArray:
    for item in layer:
        item.setWeights(getRandomWeight()) # initialize random weights for every neuron

n4.setWeights(getRandomWeight(3))

for item in biasArray:
    item.setWeights(getRandomWeight())

