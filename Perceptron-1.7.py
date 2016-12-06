from SineFunction import *

import random
import math
import pylab

global errorList
global allWeights
errorList = []
allWeights = []

class Neuron(object):

    #objID = 0
    
    def __init__(self, inputList=[], weightList=[], nType='h', bias=1): # initialization
        #self.id = Neuron.objID
        #Neuron.objID = Neuron.objID+1
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

    def feedForward(self, weightList, inputList, biasNeuron=None): # sums all input*weight. Bias input. Calculates activation function. Returns summation.

        #print("Weight list: ", weightList)
        #print("Input list: ", inputList)
        
        finalSum = 0.0

        if type(weightList) == int or type(weightList) == float:
            weightList = [weightList]

        if type(inputList) == int or type(inputList) == float:
            inputList = [inputList]
                
        for c in range(len(inputList)):
            finalSum += weightList[c] * inputList[c]
            
        if biasNeuron != None:
            finalSum += (biasNeuron.getInputs() * biasNeuron.getWeights()) # add bias neuron
            
        return self.activation(finalSum)

    def activation(self, finalSum): # if sum is positive, return 1. else, return 0

        if finalSum > 4.6:
            return 1
        elif finalSum < -4.6:
            return -1
        else:
            return 1/(1+((math.e) ** (-finalSum))) # sigmoid function

    def trainer(self, biasNeuron, inputList=[], weightList=[], summation=0, desired=0): # takes the value that comes out of feedforward. calculates error. updates weights

        if  type(weightList) == float or type(weightList) == int:
            weightList = [weightList]
            
        print(inputList)

        l = 0.05 # learning constant, slows down rate of learning
        error = desired - summation
        #print(self, " Error: ", error)

        if type(inputList) == float or type(inputList) == int:
            inputList = [inputList]

        for c in range(len(inputList)):
            weightList[c] = round((weightList[c] + (error * inputList[c] * l) + perturbation(error)), 5)
            
        allWeights.append(weightList[c])
        errorList.append(error)

        self.weightList = weightList
        return weightList

    def isConnected(self, c, n2):
        if [(c.getConnection())[0], (c.getConnection())[1]] == [self, n2]:
            return True
        else:
            return False

    '''
    def __str__(self):
        return ('<Neuron {}>'.format(self.id))
    '''
        
class NeuralNetwork(object):
    instance=0
    def __init__(self, inputs=2, hidden1=3, hidden2=2, outputs=1, bias=3, nList=[ [], [], [], [], [] ]):
        self.id = NeuralNetwork.instance
        NeuralNetwork.instance = NeuralNetwork.instance + 1
        self.inputs = inputs
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.outputs = outputs
        self.bias = bias
        self.nList=nList
        
    def initWeights(self): # initialize random weights
        for i in range(self.inputs):
            self.nList[0].append(Input(str(self.id)+'i'+str(i)))
            self.nList[0][i].setWeights(getRandomWeight())
            print(self.nList[0][i])
        for i in range(self.hidden1):
            self.nList[1].append(Hidden(str(self.id)+'h'+str(i)))
            self.nList[1][i].setWeights(getRandomWeight(len(self.nList[0])))
        for i in range(self.hidden2):
            self.nList[2].append(Hidden(str(self.id)+'h'+str(i)))
            self.nList[2][i].setWeights(getRandomWeight(len(self.nList[1])))
        for i in range(self.outputs):
            self.nList[3].append(Output(str(self.id)+'o'+str(i)))
            self.nList[3][i].setWeights(getRandomWeight(len(self.nList[2])))
        for i in range(self.bias):
            self.nList[4].append(Bias(str(self.id)+'b'+str(i)))
            self.nList[4][i].setWeights(getRandomWeight())
            
    def getNeuronList(self):
        return self.nList
        
    def getNeuron(self, layer, loc):
        if layer < len(self.nList) and loc < len(self.nList[layer]):
            return self.nList[layer][loc]
        else:
            print("Out of bounds")
        
    def getWeights(self):
        allWeights = []
        for layer in self.nList:
            for neuron in layer:
                allWeights.append(neuron.getWeights())
        return allWeights
        
    def __repr__(self):
        return('<NN {} with {} input, {}, hidden (1), {} hidden (2), {} outputs and {} bias )>'.format(self.id, self.inputs, self.hidden1, self.hidden2, self.outputs, self.bias))
     
'''
def backPropagation(outputList, desiredList, weight):
        
    for output in outputList:
        for desired in desiredList:
            error += (1/2)*(desired-output)**2
            
    errorSSE = (1/2)*(desired-output)**2 # sum of squares error
    errorMSE = errorSSE/(p*n) # mean-squared index of error
    
    derivError = (desired-output) #dE/dw
    derivSigmoid = (e**x)/((e**x+1)**2) # sigmoid functio derivative
'''
    
class Input(Neuron):
    def __init__(self, nType='i'):
        Neuron.__init__(self, inputList=[], weightList=[])
        self.nType = nType
        
    def __repr__(self):
        return ('<{} Neuron with weights {} and bias {}>'.format(self.nType, self.weightList, self.bias))

class Hidden(Neuron):
    def __init__(self, nType='h'):
        Neuron.__init__(self, inputList=[], weightList=[])
        self.nType = nType
        
    def __repr__(self):
        return ('<{} Neuron with weights {} and bias {}>'.format(self.nType, self.weightList, self.bias))

class Output(Neuron):
    def __init__(self, nType='o'):
        Neuron.__init__(self, inputList=[], weightList=[])
        self.nType = nType
        
    def __repr__(self):
        return ('<{} Neuron with weights {} and bias {}>'.format(self.nType, self.weightList, self.bias))

class Bias(Neuron):
    def __init__(self, nType='b'): # inputList must always equal 1
        Neuron.__init__(self, inputList=1, weightList=0) # make the initial weight 0 because this will render the default argument bias inert.
        self.nType = nType
        
    def __repr__(self):
        return ('<{} Neuron with weights {} and bias {}>'.format(self.nType, self.weightList, self.bias))
        
def breed(a,b):
    if len(a) == len(b):
        child = []
        c=0
        while c < len(a):
            if random.choice([0,1]) == 1:
                child.append(a[c])
            else:
                child.append(b[c])
            c += 1
        return child
    else:
        print("The two sequences have different lengths")
        
def sequence(a,b):
    if len(a) == len(b):
        seqStart = random.randint(0,len(a))
        seqEnd = random.randint(0,len(a))
        
        if seqStart > seqEnd: # check to make sure the sequence goes from low to high
            z = seqStart
            seqStart = seqEnd
            seqEnd = z
        if seqStart != seqEnd:
            child = a[:seqStart] + b[seqStart:seqEnd] + a[seqEnd:]
            return child
        else:
            return a
    else:
        print("The two sequences have different lengths")
        
def mutate(p):
    seq = random.randint(0,len(p)) # random place in genome
    mutation = round(random.random(),2)
    child = p[:(seq-1)] + [mutation] + p[seq:]
    return child

def perturb(p, place):
    mutation = random.choice([place,-place])
    loc = random.randint(0, len(p)-1)
    p[loc] += mutation
    p[loc] = round(p[loc],2)
    return p

def getRandomWeight(inputNum=None): # returns list of random floats from (-1,1). default argument returns single float. otherwise, returns float(s) inside a list

    if inputNum == None:
        return round(random.uniform(-1,1),5)
    elif type(inputNum) == int:
        j = 0
        randomWeightsList = []
        while j < inputNum:
            randomWeightsList.append(round(random.uniform(-1,1),5))
            j += 1
        j = 0
        return randomWeightsList
        
def perturbation(error): # fuzziness to help escape from local maxima
    p = (0.005 * error * round(random.uniform(-1,1),5))
    return p

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
        
def train(inputVal1, inputVal2, desired): #inputVal = -12, desired = -1
    
    '''
    n0.setInputs(inputVal1)
    n1.setInputs(inputVal2)
    
    s0 = n0.feedForward(n0.getWeights(), n0.getInputs(), b0)            # input 1
    n0.trainer(b0, n0.getInputs(), n0.getWeights(), s0, desired)
    
    s1 = n1.feedForward(n1.getWeights(), s0, b0)                        # hidden 2
    n1.trainer(b0, n1.getInputs(), n1.getWeights(), s1, desired)
    
    s2 = n2.feedForward(n2.getWeights(), s0, b1)                        # hidden 1
    n2.trainer(b1, [s0,s1], n2.getWeights(), s2, desired)
    
    s3 = n3.feedForward(n3.getWeights(), s0, b1)                        # hidden 2
    n3.trainer(b1, [s0,s1], n3.getWeights(), s3, desired)

    s4 = n4.feedForward(n2.getWeights(), s0, b1)                        # hidden 3
    n4.trainer(b1, [s0,s1], n4.getWeights(), s4, desired)
    
    s5 = n5.feedForward(n3.getWeights(), s0, b1)                        # hidden 4
    n5.trainer(b1, [s0,s1], n5.getWeights(), s5, desired)
    
    s6 = n6.feedForward(n6.getWeights(), [s1,s2,s3,s4], b2)                # output (weight 1)
    n6.trainer(b2, [s2,s3,s4,s5], n6.getWeights(), s6, desired)
    '''

def feedThrough(inputVal1, inputVal2): # can only take two inputs at the moment, such as x and y

    n0.setInputs(inputVal1)
    n1.setInputs(inputVal2)
    
    s0 = n0.feedForward(n0.getWeights(), n0.getInputs())            # input 1
    s1 = n1.feedForward(n1.getWeights(), n1.getInputs())            # input 2
    
    s2 = n1.feedForward(n1.getWeights(), [s0,s1], b1)               # hidden 1
    s3 = n2.feedForward(n2.getWeights(), [s0,s1], b1)               # hidden 2
    s4 = n3.feedForward(n3.getWeights(), [s0,s1], b1)               # hidden 3
    s5 = n4.feedForward(n4.getWeights(), [s0,s1], b1)               # hidden 4
    
    s6 = n4.feedForward(n4.getWeights(), [s2,s3,s4,s5], b2)         # output (weight 1)

    return s6

def test():

    valsRight = []
    valsLeft = []

    i = 0
    while i < 4000:
        x = random.random()*100
        y = random.random()*100

        if x > abs(y):
            valsLeft.append([x,y])
        else:
            valsRight.append([x,y])
        i += 1

    stopVal = 0
    
    c = 0
    if len(valsLeft) < len(valsRight):
        stopVal = len(valsLeft)
    else:
        stopVal = len(valsRight)
        
    while c < stopVal:
        train(valsLeft[c][0], valsLeft[c][1], 1)
        train(valsLeft[c][0], valsLeft[c][1], 1)
        train(valsRight[c][0], valsRight[c][1], -1)
        train(valsRight[c][0], valsRight[c][1], -1)
        c += 1
    
    # do this later:
    '''
    c = 0
    while c < len(sineX):
        train(xAbove[c], yAbove[c], 1)
        train(xBelow[c], yBelow[c], -1)
        c += 1
    '''

test()

pylab.plot(allWeights,'.')
pylab.plot(errorList,'.')
# pylab.xlim(xmin=0, xmax=3000)
pylab.ylim(ymin=-20, ymax=20)
pylab.show
