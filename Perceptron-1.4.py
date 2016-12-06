# A single perceptron. To use, define lists of inputs and random weights and give them as arguments to sumInputs, or use getRandomWeights to randomize the weights for your input
# This perceptron is now capable of learning, as it adjusts its weights based on past results.
# In TestFunc(), I use two inputs and train the neuron to detect whether a point is to the left or right of the line x=0.
# If the input is to the left, the neuron outputs a -1. If it is to the right, it outputs a +1.
# Because there is only one neuron, the neuron's understanding of the function will always be linear. Therefore, the neuron's only use is in linear regression.

import random

class Neuron():
    def __init__(self, inputList=[], weightList=[], bias=1): # initialization
        self.inputList = inputList
        self.weightList = weightList
        self.bias=bias

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

        print(finalSum)
        if finalSum < 0:
            return -1
        elif finalSum >= 0:
            return 1
        
    def trainerFunc(self, inputList=[], weightList=[], summation=0, desired=0):
    
        l = 0.04 # learning constant

        error = desired - summation

        for c in range(len(inputList)-1):
            weightList[c] = round(weightList[c] + (error * inputList[c] * l),4)

        self.weightList = weightList
        return weightList

def getRandomWeights(inputList=[0,0]): # returns list of random floats from -1 to 1
    j = 0
    randomWeightsList = []
    while j < len(inputList)+1:
        randomWeightsList.append(round(random.uniform(-1,1),4))
        j += 1
    j = 0
    return randomWeightsList

def genPoints(n): # generate a list of n x-y coords with the sign of -1 or 1 telling whether the point is to the left or right of line y=x

    testPoints = []
    i = 0
    while i < n:
        x = (round(random.uniform(-50,50),2))
        y = (round(random.uniform(-50,50),2))
        sign = 0

        if x < y:
            sign = -1
        else:
            sign = 1
        
        testPoints.append( [x,y,sign] )
        i+=1

    return testPoints
    

    # n:    object instance
    # i1:   inputs
    # w1:   weights
    # s1:   summation

testPoints = [ [-1,1,-1],[-1,-1,-1],[-2,-2,-1],[-2,3,-1],[1,1,1],[1,2,1],[2,-3,1],[2,-4,1] ] # x=0
    
n = Neuron([-1,1])
    
i1 = n.getInputs() # get new inputs
print(i1)
    
w1 = getRandomWeights(i1) # randomize weights (for the first iteration)
print(w1)
    
s1 = n.feedForward(w1, i1, n.bias) # feedfoward (i.e. get summation)
print(s1)
w1 = n.trainerFunc(i1, w1, s1, -1) # retrain (because (-1,1) is negative, we want 1 to be negative too)
print(w1)
s1 = n.feedForward(w1, i1, n.bias) # feedforward
print(s1)
w1 = n.trainerFunc(i1, w1, s1, -1) # retrain
print(w1)
s1 = n.feedForward(w1, i1, n.bias) # feedforward
print(s1)


n.setInputs([-1,-1])
i1 = n.getInputs()
print(i1)

s1 = n.feedForward(w1, i1, n.bias) # feedfoward
print(s1)
w1 = n.trainerFunc(i1, w1, s1, -1) # retrain
print(w1)
s1 = n.feedForward(w1, i1, n.bias) # feedforward
print(s1)
w1 = n.trainerFunc(i1, w1, s1, -1) # retrain
print(w1)
s1 = n.feedForward(w1, i1, n.bias) # feedforward
print(s1)


n.setInputs([-2,-2])
i1 = n.getInputs()
print(i1)

s1 = n.feedForward(w1, i1, n.bias) # feedfoward
print(s1)
w1 = n.trainerFunc(i1, w1, s1, -1) # retrain
print(w1)
s1 = n.feedForward(w1, i1, n.bias) # feedforward
print(s1)
w1 = n.trainerFunc(i1, w1, s1, -1) # retrain
print(w1)
s1 = n.feedForward(w1, i1, n.bias) # feedforward
print(s1)


n.setInputs([-2,4])
i1 = n.getInputs()
print(i1)

s1 = n.feedForward(w1, i1, n.bias) # feedfoward
print(s1)
w1 = n.trainerFunc(i1, w1, s1, -1) # retrain
print(w1)
s1 = n.feedForward(w1, i1, n.bias) # feedforward
print(s1)
w1 = n.trainerFunc(i1, w1, s1, -1) # retrain
print(w1)
s1 = n.feedForward(w1, i1, n.bias) # feedforward
print(s1)


n.setInputs([1,1])
i1 = n.getInputs()
print(i1)

s1 = n.feedForward(w1, i1, n.bias) # feedfoward
print(s1)
w1 = n.trainerFunc(i1, w1, s1, 1) # retrain
print(w1)
s1 = n.feedForward(w1, i1, n.bias) # feedforward
print(s1)
w1 = n.trainerFunc(i1, w1, s1, 1) # retrain
print(w1)
s1 = n.feedForward(w1, i1, n.bias) # feedforward
print(s1)


n.setInputs([1,2])
i1 = n.getInputs()
print(i1)

s1 = n.feedForward(w1, i1, n.bias) # feedfoward
print(s1)
w1 = n.trainerFunc(i1, w1, s1, 1) # retrain
print(w1)
s1 = n.feedForward(w1, i1, n.bias) # feedforward
print(s1)
w1 = n.trainerFunc(i1, w1, s1, 1) # retrain
print(w1)
s1 = n.feedForward(w1, i1, n.bias) # feedforward
print(s1)


n.setInputs([2,-3])
i1 = n.getInputs()
print(i1)

s1 = n.feedForward(w1, i1, n.bias) # feedfoward
print(s1)
w1 = n.trainerFunc(i1, w1, s1, 1) # retrain
print(w1)
s1 = n.feedForward(w1, i1, n.bias) # feedforward
print(s1)
w1 = n.trainerFunc(i1, w1, s1, 1) # retrain
print(w1)
s1 = n.feedForward(w1, i1, n.bias) # feedforward
print(s1)


n.setInputs([2,-4])
i1 = n.getInputs()
print(i1)

s1 = n.feedForward(w1, i1, n.bias) # feedfoward
print(s1)
w1 = n.trainerFunc(i1, w1, s1, 1) # retrain
print(w1)
s1 = n.feedForward(w1, i1, n.bias) # feedforward
print(s1)
w1 = n.trainerFunc(i1, w1, s1, 1) # retrain
print(w1)
s1 = n.feedForward(w1, i1, n.bias) # feedforward
print(s1)

n.setInputs([-1,4]) # untrained point 1
i1 = n.getInputs()
print(i1)
s1 = n.feedForward(w1, i1, n.bias) # feedfoward
print(s1)

n.setInputs([3,-3]) # untrained point 2
i1 = n.getInputs()
print(i1)
s1 = n.feedForward(w1, i1, n.bias) # feedfoward
print(s1)

n.setInputs([-4,18]) # untrained point 3
i1 = n.getInputs()
print(i1)
s1 = n.feedForward(w1, i1, n.bias) # feedfoward
print(s1)

n.setInputs([0.2,0]) # untrained point 4
i1 = n.getInputs()
print(i1)
s1 = n.feedForward(w1, i1, n.bias) # feedfoward
print(s1)
