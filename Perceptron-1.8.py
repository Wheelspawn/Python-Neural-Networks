# from SineFunction import *

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
            return 0
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
    def __init__(self, inputs=2, hidden1=12, hidden2=10, outputs=1, bias=3, nList=[ [], [], [], [], [] ]):
        self.id = NeuralNetwork.instance
        NeuralNetwork.instance = NeuralNetwork.instance + 1
        self.inputs = inputs
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.outputs = outputs
        self.bias = bias
        self.nList=nList
        
    def initWeights(self): # initialize random weights
        self.nList= [ [], [], [], [], [] ]
        for i in range(self.inputs):
            self.nList[0].append(Input(str(self.id)+'i'+str(i)))
            self.nList[0][i].setWeights(getRandomWeight(1))
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
        return self.nList
        
    def setWeights(self, nList):
        self.nList = nList
        
    def __repr__(self):
        return('<NN {} with {} input, {}, hidden(1), {} hidden(2), {} outputs and {} bias )>'.format(self.id, self.inputs, self.hidden1, self.hidden2, self.outputs, self.bias))
     
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
        
class GameBoard():
    def __init__(self, count=0, board=[[" "," "," "],[" "," "," "],[" "," "," "]]):
        self.count = count
        self.board=board
        
    def display(self):
        print(" ------------- \n | {} | {} | {} | \n ------------- \n | {} | {} | {} | \n ------------- \n | {} | {} | {} | \n -------------".format(self.board[0][0],self.board[0][1],self.board[0][2],self.board[1][0],self.board[1][1],self.board[1][2],self.board[2][0],self.board[2][1],self.board[2][2]))

    def add(self, char, n):
        self.count += 1
        n-=1
        a = n//3
        b = n%3
        if n < 9:
            if self.board[a][b] != 'x' and self.board[a][b] != 'o':
                self.board[a][b] = char
            else:
                return False
        else:
            return False
            
        if self.count > 4:
            if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ' or self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
                print(char, " win")
            
            c=0
            while c < 3:
                if self.board[c][0] == self.board[c][1] == self.board[c][2] != ' ' or self.board[0][c] == self.board[1][c] == self.board[2][c] != ' ':
                    print(char, " win")
                    break
                else:
                    c+=1
        if self.count == 9:
            print('tie')

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
    if type(p) == int or type(p) == float:
        p += random.choice([place,-place])
        return p
    else:
        mutation = random.choice([place,-place])
        loc = random.randint(0, len(p)-1)
        p[loc] += mutation
        p[loc] = round(p[loc],2)
        return p
        
def genomeToList(n):
    l = [ [], [], [], [], [] ]
    i=0
    for layer in n.getWeights():
        for neuron in layer:
                l[i].append(neuron.getWeights())
        i+=1
    return l
    
def listToGenome(l, n): # assumes weights match neurons properly
    nList= [ [], [], [], [], [] ]
    for i in range(n.inputs):
        nList[0].append(Input(str(n.id)+'i'+str(i)))
        nList[0][i].setWeights(l[0][i])
    for i in range(n.hidden1):
        nList[1].append(Hidden(str(n.id)+'h'+str(i)))
        nList[1][i].setWeights(l[1][i])
    for i in range(n.hidden2):
        nList[2].append(Hidden(str(n.id)+'h'+str(i)))
        nList[2][i].setWeights(l[2][i])
    for i in range(n.outputs):
        nList[3].append(Output(str(n.id)+'o'+str(i)))
        nList[3][i].setWeights(l[3][i])
    for i in range(n.bias):
        nList[4].append(Bias(str(n.id)+'b'+str(i)))
        nList[4][i].setWeights(l[4][i])
        
    return nList
    

def getRandomWeight(inputNum=None): # returns list of random floats from (-1,1). default argument returns single float. otherwise, returns float(s) inside a list

    if inputNum == None:
        return round(random.uniform(-1,1),2)
    elif type(inputNum) == int:
        j = 0
        randomWeightsList = []
        while j < inputNum:
            randomWeightsList.append(round(random.uniform(-1,1),2))
            j += 1
        j = 0
        return randomWeightsList
        
def perturbation(error): # fuzziness
    p = (0.005 * error * round(random.uniform(-1,1),2))
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

def feedThrough(n, inputs): # (inputs, inputList, hidden1, hidden2, outputs, bias):
    
    inputList = n.getNeuronList()[0]
    hidden1 = n.getNeuronList()[1]
    hidden2 = n.getNeuronList()[2]
    outputs = n.getNeuronList()[3]
    bias = n.getNeuronList()[4]
    
    FFInputs = []
    FFHidden1 = []
    FFHidden2 = []
    FFOutputs = []

    if len(inputList) == len(inputs):
        c=0
        for i in inputList: # for every input node
            i.setInputs(inputs[c]) # give node an input
            FFInputs.append(i.feedForward(i.getWeights(), i.getInputs())) # feedfoward and append the results
            c+=1
    else:
        print("Inputs and inputs nodes do not match")
    
    for hidden in hidden1:
        FFHidden1.append(hidden.feedForward(hidden.getWeights(), FFInputs, bias[0]))
    for hidden in hidden2:
        FFHidden2.append(hidden.feedForward(hidden.getWeights(), FFHidden1, bias[1]))
    for out in outputs:
        FFOutputs.append(out.feedForward(out.getWeights(), FFHidden2, bias[2]))
        
    if len(FFOutputs) == 1:
        return FFOutputs[0]
    else:
        return FFOutputs
    


def trainSine(n):
    xValsAbove = []
    yValsAbove = []
    xValsBelow = []
    yValsBelow = []
    a = 0
    b = 0
    while a+b < 300:
        x = random.random()*20*random.choice([-1,1])
        y = random.random()*2*random.choice([-1,1])
        if y > math.sin(x) and a <= 150:
            xValsAbove.append(x)
            yValsAbove.append(y)
            a += 1
        else:
            xValsBelow.append(x)
            yValsBelow.append(y)
            b += 1
                
    d=0
    error = 0
    while d < (len(yValsAbove) - abs(len(yValsAbove) - len(yValsBelow))):
        output1 = feedThrough(n, [xValsAbove[d], yValsAbove[d]])
        output2 = feedThrough(n, [xValsBelow[d], yValsBelow[d]])
        
        if output1 < 0.98:
            error += 1 - output1
        if output2 > 0.02:
            error += output2
        '''
        output1 = feedThrough(n, [xValsAbove[d], yValsAbove[d]])
        error += (abs(output1-1))
        output2 = feedThrough(n, [xValsBelow[d], yValsBelow[d]])
        error += (abs(output2+1))
        '''
        d+=1
                
    error += round((random.random()*random.choice([-1,1]))/1000,8)
    return error
    
def plotter():
    xValsAbove = []
    yValsAbove = []
    xValsBelow = []
    yValsBelow = []
    a = 0
    b = 0
    while a+b < 300:
        x = random.random()*15*random.choice([-1,1])
        y = random.random()*2*random.choice([-1,1])
        if y > math.sin(x) and a <= 150:
            xValsAbove.append(x)
            yValsAbove.append(y)
            a += 1
        elif y < math.sin(x):
            xValsBelow.append(x)
            yValsBelow.append(y)
            b += 1
    
    pylab.plot(xValsAbove,yValsAbove,'.')
    pylab.plot(xValsBelow,yValsBelow,'.')
    pylab.xlim(xmin=-15, xmax=15)
    pylab.ylim(ymin=-3, ymax=3)
    pylab.show

def evolve():
    n = NeuralNetwork()
    n.initWeights()
    errorList = []
    errorDict = {}
    survivors = {}
    c=0
    while c < 300:
        n.initWeights()
        error = trainSine(n)
        errorList.append(error) # append error
        errorDict[error] = n.getWeights() # append list of weights that return the respective error
        print("Error: ", error)
        error = 0
        c+=1
    c=0
        
    errorList.sort()
    
    e=0
    while e < 10:
        survivors[errorList[e]] = errorDict[errorList[e]]
        print(errorList[e])
        e+=1
        
    loop = 0
    while loop < 999:
        errorList = []
        survivors = generate(survivors, n)
        for survivor in survivors:
            errorList.append(survivor)
        if errorList[0] < 1:
            return survivors
        loop += 1
    
    print(survivors) #, survivors[3], survivors[4], survivors[5]]
    return survivors
        
def generate(survivors, n):
    errorList = []
    errorDict = {}
    survivorDict = {}
    
    for survivor in survivors:
        errorList.append(survivor) # survivor error
        errorDict[survivor] = survivors[survivor] # dictionary of errors with values as weight lists
        
    errorList.sort()
        
    a=0
    while a < 400:
        # print("Error List: ", errorList)
        rand = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9],2)
        n.setWeights(survivors[errorList[rand[0]]])
        copy1 = genomeToList(n)
        for layer in copy1:
            for weights in layer:
                weights = 
        n.setWeights(survivors[errorList[rand[1]]])
        copy2 = genomeToList(n)
        n.nList = listToGenome(breed(copy1, copy2), n)
        error = trainSine(n)
        if error not in errorList:
            errorList.append(error)
            errorDict[error] = n.getWeights()
        else:
            a -= 1
        a +=1
        
    c=0
    while c < 100:
        n.setWeights(survivors[errorList[random.randint(0,9)]])
        for layer in n.getWeights():
            # print("Layer: ", layer)
            for neuron in layer:
                temp = []
                # print("Neuron: ", neuron)
                if type(neuron.getWeights()) == float or type(neuron.getWeights()) == int:
                    neuron.setWeights(round(perturb(neuron.getWeights(), random.choice([0.0,0.0,0.1,0.5,1.0,3.0,5.0,7.0])),2)) #([0.0,0.0,0.0,0.0,0.1,0.1,0.1,0.1,0.5])),2))
                else:
                    for weight in neuron.getWeights():
                        # print("Weights: ", weight)
                        temp.append(round(perturb(weight, 0.1),2))
                        neuron.setWeights(temp)
        error = trainSine(n)
        errorList.append(error)
        errorDict[error] = n.getWeights()
        c+=1
        
        
    # print(secondGen)
    errorList.sort()
        
    e=0
    print("Survivors: ")
    while e < 10:
        survivorDict[errorList[e]] = errorDict[errorList[e]]
        print("Error: ", errorList[e])
        e+=1
        
    return survivorDict