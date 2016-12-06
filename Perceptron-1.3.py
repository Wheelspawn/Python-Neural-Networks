# A single perceptron. To use, define lists of inputs and random weights and give them as arguments to sumInputs, or use getRandomWeights to randomize the weights for your input
# This perceptron is now capable of learning, as it adjusts its weights based on past results.
# In TestFunc(), I use two inputs and train the neuron to detect whether a point is to the left or right of the line y=-x.
# If the input is to the left, the neuron outputs a -1. If it is to the right, it outputs a +1.

import random

class Neuron():
    def __init__(self, inputList=[], weightList=[], bias=1): # initialization
        self.inputList = inputList
        self.weightList = weightList
        self.bias=bias

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

    def trainerFunc(self, inputList=[], weightList=[], summation=0, desired=0):
    
        l = 0.025 # learning constant

        error = desired - summation

        for c in range(len(inputList)-1):
            weightList[c] = round(weightList[c] + (error * inputList[c] * l),4)

        self.weightList = weightList
        return weightList

def getRandomWeights(inputList): # returns list of random floats from -1 to 1
    j = 0
    randomWeightsList = []
    while j < len(inputList)+1:
        randomWeightsList.append(round(random.uniform(-1,1),4))
        j += 1
    j = 0
    return randomWeightsList
        
def testFunc():

    # n:    object instance
    # i1:   inputs
    # w1:   weights
    # s1:   summation

    n = Neuron([4,4])
    
    i1 = n.getInputs() # enter new inputs
    print(i1)
    
    w1 = getRandomWeights(i1) # randomize weights (for the first iteration)
    print(w1)
    
    s1 = n.feedForward(w1, i1, n.bias) # feedfoward (i.e. get summation)
    print(s1)

    w1 = n.trainerFunc(i1, w1, s1, 1) # retrain (because (4,4) is positive, we want 1 to be positive too)
    print(w1)

    s1 = n.feedForward(w1, i1, n.bias) # feedforward
    print(s1)

    w1 = n.trainerFunc(i1, w1, s1, 1) # retrain
    print(w1)

    n.setInputs([1,2])
    print(n.getInputs())

    s1 = n.feedForward(w1, i1, n.bias) # feedforward
    print(s1)

    w1 = n.trainerFunc(i1, w1, s1, 1) # retrain
    print(w1)

    s1 = n.feedForward(w1, i1, n.bias) # feedforward
    print(s1)

    w1 = n.trainerFunc(i1, w1, s1, 1) # retrain
    print(w1)

    s1 = n.feedForward(w1, i1, n.bias) # feedforward
    print(s1)

    n.setInputs([-4,-4]) # try a new set of inputs [-4,-4]
    print(n.getInputs())

    w1 = n.trainerFunc(i1, w1, s1, -1) # train
    print(w1)

    s1 = n.feedForward(w1, i1, n.bias) #feedforward
    print(s1)

    n.setInputs([-1,-1])
    print(n.getInputs())

    w1 = n.trainerFunc(i1, w1, s1, -1) # train
    print(w1)

    s1 = n.feedForward(w1, i1, n.bias) # feedforward
    print(s1)

    w1 = n.trainerFunc(i1, w1, s1, -1) # train
    print(w1)

    s1 = n.feedForward(w1, i1, n.bias) # final feedforward test
    print(s1)

    n.setInputs([-8,1]) # test a random point
    print(n.getInputs())
    s1 = n.feedForward(w1, i1, n.bias)
    print(s1)

    n.setInputs([-8,10])

    s1 = n.feedForward(w1, i1, n.bias) #feedforward
    print(s1)

    w1 = n.trainerFunc(i1, w1, s1, 1) # train
    print(w1)

    s1 = n.feedForward(w1, i1, n.bias) #feedforward
    print(s1)

    w1 = n.trainerFunc(i1, w1, s1, 1) # train
    print(w1)

    s1 = n.feedForward(w1, i1, n.bias) #feedforward
    print(s1)

    w1 = n.trainerFunc(i1, w1, s1, 1) # train
    print(w1)
    
    print(n.getInputs())
    s1 = n.feedForward(w1, i1, n.bias)
    print(s1)
