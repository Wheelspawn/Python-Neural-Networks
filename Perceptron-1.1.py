# A single perceptron. To use, define lists of inputs and random weights and give them as arguments to sumInputs, or use getRandomWeights to randomize the weights for your input

import random

class Neuron():
    def __init__(self, inputList=[]): # initialization
        self.inputList = inputList

    def setInputs(self, inputList=[], bias=1):
        self.inputList = inputList
        self.bias = bias

    def getInputs(self):
        return self.inputList

    def feedForward(self, WeightsList, inputList, bias): # sums all input*weight. 1 bias input. Calculates activation function. Returns summation.
        
        finalSum = 0.0
        for c in range(len(inputList)):
            finalSum += WeightsList[c] * inputList[c]

        finalSum += bias * WeightsList[-1]
            
        return self.getActivationFunc(finalSum)

    def getActivationFunc(self, finalSum): # if sum is positive, return 1. else, return 0

        if finalSum < 0:
            return -1
        elif finalSum >= 0:
            return 1

        # sigmoidal activation function
        '''
        e = 2.718281828

        if finalSum > 5:
            return 1
        elif finalSum < -5:
            return -1
        else:
            return 1/(1+(e ** (-finalSum)))
        '''

def getRandomWeights(self, inputList): # returns list of floats from -1 to 1
    j = 0
    randomWeightsList = []
    while j < len(self.inputList)+1:
        randomWeightsList.append(random.uniform(-1,1))
        j += 1
    j = 0
    return randomWeightsList
        
def testFunc():
    n = Neuron([4,6])
    i1 = n.getInputs()
    print(i1)
    r1 = n.getRandomWeights(i1)
    print(r1)
    f1 = n.feedForward(r1, i1)
    print(f1)

def testFunc2():
    n = Neuron([1,2,3])
    i1 = n.getInputs()
    print(i1)
    r1 = n.getRandomWeights(i1)
    print(r1)
    f1 = n.feedForward(r1, i1)
    print(f1)

def testFunc3():
    n = Neuron([5,5])
    i1 = n.getInputs()
    print(i1)
    r1 = n.getRandomWeights(i1)
    print(r1)
    f1 = n.feedForward(r1, i1)
    print(f1)

def testFunc4():
    n = Neuron([-5,-5])
    i1 = n.getInputs()
    print(i1)
    r1 = n.getRandomWeights(i1)
    print(r1)
    f1 = n.feedForward(r1, i1)
    print(f1)
