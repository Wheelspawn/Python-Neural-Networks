# A single perceptron. To use, define lists of inputs and random weights and give them as arguments to sumInputs, or use getRandomWeights to randomize the weights for your input
# This perceptron is now capable of learning, as it adjusts its weights based on past results.
# In TestFunc(), I use two inputs and train the neuron to detect whether a point is to the left or right of the line x=0.
# If the input is to the left, the neuron outputs a -1. If it is to the right, it outputs a +1.
# With multiple neurons, we can solve linearly nonseparable problems.


import random
import pylab


class Neuron():

    objID = 0
    
    def __init__(self, inputList=[], weightList=[], bias=1, nType=0): # initialization
        self.id = Neuron.objID
        Neuron.objID = Neuron.objID+1
        
        self.inputList = inputList
        self.weightList = weightList
        self.bias=bias

    def setType(self, nType): # 0: standard  1: input  2: output
        self.nType = nType

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

    def feedForward(self, WeightList, inputList, bias): # sums all input*weight. Bias input. Calculates activation function. Returns summation.
        
        finalSum = 0.0
        for c in range(len(inputList)):
            finalSum += WeightList[c] * inputList[c]

        finalSum += bias * WeightList[-1]
            
        return self.getActivationFunc(finalSum)

    def getActivationFunc(self, finalSum): # if sum is positive, return 1. else, return 0

        e = 2.718281828

        if finalSum > 5:
            return 1
        elif finalSum < -5:
            return -1
        else:
            return 1/(1+(e ** (-finalSum)))

    def trainerFunc(self, inputList=[], weightList=[], summation=0, desired=0):
    
        l = 0.04 # learning constant

        error = desired - summation

        for c in range(len(inputList)-1):
            weightList[c] = round(weightList[c] + (error * inputList[c] * l),4)

        self.weightList = weightList
        pylab.plot(weightList)
        pylab.show()
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


def getRandomWeights(inputList=[0]): # returns list of random floats from -1 to 1
    j = 0
    randomWeightsList = []
    while j < len(inputList):
        randomWeightsList.append(round(random.uniform(-1,1),4))
        j += 1
    j = 0
    return randomWeightsList
    

def getSingleWeight():
    return (round(random.uniform(-1,1),4))


class Connection(): # sets up connection between two neurons
    def __init__(self, a=None, b=None, weight=0.0):
        self.a = a
        self.b = b
        self.weight = weight
        self.connectionList = [a,b,weight]

    def setConnection(self, a, b, weight = getSingleWeight()):
        self.a = a
        self.b = b
        self.weight = weight
        self.connectionList = [a,b,weight]
        
    def __repr__(self):
        return('<Connection between {} and {} with weight {})>'.format(self.a, self.b, self.weight))
        
def train(inputVals, desired):
    n0.feedForward()

n0 = Neuron() # input
n1 = Neuron() # hidden
n2 = Neuron() # hidden
n3 = Neuron() # hidden
n4 = Neuron() # output

c01 = Connection(n0,n1, getSingleWeight())
c02 = Connection(n0,n2, getSingleWeight())
c03 = Connection(n0,n3, getSingleWeight())
c14 = Connection(n1,n4, getSingleWeight())
c24 = Connection(n2,n4, getSingleWeight())
c34 = Connection(n3,n4, getSingleWeight())

neuronArray = [ [n0] , [n1,n2,n3], [n4] ] # input layers 1, 2, 3
connectionArray = [ [c01, c02, c03], [c14,c24,c34] ] # connections from layer 1 to layer 2, and connections from layer 2 to layer 3

for layer in neuronArray:
    for item in layer:
        item.setWeights(getRandomWeights([0,0])) # initialize random weights for every neuron
        
sineX = [-12,-10,-8,-6,-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,4,5,6,8,10,12]
sineY = [0.5,0.5,-1.0,0.3,1.0,0.8,-0.1,-0.6,-0.9,-1.0,-0.8,-0.5,0.0,0.5,0.8,1.0,0.9,0.6,0.1,-0.8,-1.0,-0.3,1.0,-0.5,-0.5]