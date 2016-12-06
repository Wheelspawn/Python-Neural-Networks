# from SineFunction import *

import random
import math
import pylab
import copy

global errorList
global allWeights
errorList = []
allWeights = []

class Neuron(object):

    #objID = 0
    
    def __init__(self, inputList=[], weightList=[], nType='h'): # initialization
        #self.id = Neuron.objID
        #Neuron.objID = Neuron.objID+1
        self.weightList = weightList
        self.inputList = inputList

    def getType(self):
        return self.nType
        
    def setWeights(self, weightList=[]):
        self.weightList = weightList

    def getWeights(self):
        return self.weightList

    def setInputs(self, inputList=[]):
        self.inputList = inputList

    def getInputs(self):
        return self.inputList

    def feedForward(self, weightList, inputList, biasNeuron=None): # sums all input*weight. Bias input. Calculates activation function. Returns summation.
        
        finalSum = 0.0

        if type(weightList) == int or type(weightList) == float:
            weightList = [weightList]

        if type(inputList) == int or type(inputList) == float:
            inputList = [inputList]
            
        if len(weightList) != len(inputList):
            print(weightList)
            print(inputList)
            # print("Lengths not equal: ", len(weightList), len(inputList))
            raise ValueError("Lengths not equal: ", len(weightList), len(inputList))
                
        for c in range(len(inputList)):
            finalSum += weightList[c] * inputList[c]
            
        if biasNeuron != None:
            finalSum += (biasNeuron.getInputs() * biasNeuron.getWeights()) # add bias neuron
            
        return self.sigmoid(finalSum)

    def sigmoid(self, finalSum): # if sum is positive, return 1. else, return 0
        if finalSum > 4.6:
            return 1
        if finalSum < -4.6:
            return 0
        else:
            return 1/(1+((math.e) ** (-finalSum))) # sigmoid function
            
    def tanh(self, finalSum):
        return math.tanh(finalSum)
            
    def step(self, finalSum):
        
        if finalSum <= 0:
            return 0
        else:
            return 1
            
    def rectifier(self, finalSum):
        
        if finalSum <= 0:
            return 0
        else:
            return finalSum
            
    def softplus(self, finalSum):
        return math.log(1+math.e**finalSum)

    def trainer(self, biasNeuron, inputList=[], weightList=[], summation=0, desired=0): # takes the value that comes out of feedforward. calculates error. updates weights

        if  type(weightList) == float or type(weightList) == int:
            weightList = [weightList]
            
        # print(inputList)

        l = 0.03 # learning constant, slows down rate of learning
        error = desired - summation
        #print(self, " Error: ", error)

        if type(inputList) == float or type(inputList) == int:
            inputList = [inputList]

        for c in range(len(inputList)):
            weightList[c] = round((weightList[c] + (error * inputList[c] * l) + perturbation(error)), 4)
            
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
    def __init__(self, inputs=2, hidden1=9, hidden2=8, outputs=1, bias=3, nList=[ [], [], [], [], [] ]):
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
            self.nList[0].append(Input('i'+str(i)+'n'+str(self.id)))
            self.nList[0][i].setWeights(getRandomWeight(1))
        for i in range(self.hidden1):
            self.nList[1].append(Hidden1('h'+str(i)+'l0'+'n'+str(self.id)))
            self.nList[1][i].setWeights(getRandomWeight(len(self.nList[0])))
        for i in range(self.hidden2):
            self.nList[2].append(Hidden2('h'+str(i)+'l1'+'n'+str(self.id)))
            self.nList[2][i].setWeights(getRandomWeight(len(self.nList[1])))
        for i in range(self.outputs):
            self.nList[3].append(Output('o'+str(i)+'n'+str(self.id)))
            self.nList[3][i].setWeights(getRandomWeight(len(self.nList[2])))
        for i in range(self.bias):
            self.nList[4].append(Bias('b'+str(i)+'n'+str(self.id)))
            self.nList[4][i].setWeights(0) # self.nList[4][i].setWeights(getRandomWeight())
            
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
        
    def sigmoidDeriv(self, inputSum):
        
        return 1 - (inputSum)**2
        '''
        if inputSum > 6 or inputSum < -6:
            return 0
        else:
            return round((math.e**inputSum)/(math.e**inputSum+1)**2,3)
        '''
        
    def error(self, desired, out):
        return round((1/2)*(desired-out)**2,3)
        
    def derivError(self, desired, out):
        return round((desired-out),3)
        
    def backPropagation(self, desired, inputs, l):
        
        ffAll = feedForward(self, inputs, 4)
        '''
        print(self.nList)
        print("")
        print(ffAll)
        print("")
        '''
        
        actual = ffAll[3]
        # print("ffAll: ", ffAll)
        
        outputDelta = [] # output deltas
        hidden2Delta = [] # hidden 2 deltas
        hidden1Delta = [] # hidden 1 deltas
        inputDelta = [] # input deltas

        for out in range(len(self.nList[3])): # for every output neuron
            outputDelta.append(self.sigmoidDeriv(actual[out]) * self.derivError(desired, actual[out])) # delta calculated as difference between desired and actual
            
        num = 0
        
        for hidden2 in self.nList[2]:
            error = 0.0
            for out in range(len(self.nList[3])):
                error += outputDelta[out]*self.nList[3][out].getWeights()[out]
            hidden2Delta.append(self.sigmoidDeriv(ffAll[2][num])*error)
            num += 1
        
        num = 0
        for hidden1 in self.nList[1]:
            error = 0.0
            for hidden2 in range(len(self.nList[2])):
                error += hidden2Delta[hidden2] * self.nList[2][hidden2].getWeights()[hidden2]
            hidden1Delta.append(self.sigmoidDeriv(ffAll[1][num]*error))
            num += 1
            # hidden1.getWeights()[w] += hidden1Delta[w] * l
            
        for hidden2 in range(len(self.nList[2])):
            for out in range(len(self.nList[3])):
                self.nList[3][out].getWeights()[hidden2] += outputDelta[out]*ffAll[2][hidden2]*l # delta * input value to out * learning constant
        
        for hidden1 in range(len(self.nList[1])):
            for hidden2 in range(len(self.nList[2])):
                self.nList[2][hidden2].getWeights()[hidden1] += hidden2Delta[hidden2]*ffAll[1][hidden1]*l
        
        for inp in range(len(self.nList[0])):
            for hidden1 in range(len(self.nList[1])):
                self.nList[1][hidden1].getWeights()[inp] += hidden1Delta[hidden1]*ffAll[0][inp]*l
        
        # for inp in range(len(self.nList[0])): # do input node update here
        
        # print("Total error: ", sum(outputDelta)+sum(hidden2Delta)+sum(hidden1Delta))
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
        return ('<{} Neuron with weights {}>'.format(self.nType, self.weightList))

class Hidden1(Neuron):
    def __init__(self, nType='h(1)'):
        Neuron.__init__(self, inputList=[], weightList=[])
        self.nType = nType
        
    def __repr__(self):
        return ('<{} Neuron with weights {}>'.format(self.nType, self.weightList))
        
class Hidden2(Neuron):
    def __init__(self, nType='h(2)'):
        Neuron.__init__(self, inputList=[], weightList=[])
        self.nType = nType
        
    def __repr__(self):
        return ('<{} Neuron with weights {}>'.format(self.nType, self.weightList))
        
class Hidden3(Neuron):
    def __init__(self, nType='h(3)'):
        Neuron.__init__(self, inputList=[], weightList=[])
        self.nType = nType
        
    def __repr__(self):
        return ('<{} Neuron with weights {}>'.format(self.nType, self.weightList))

class Output(Neuron):
    def __init__(self, nType='o'):
        Neuron.__init__(self, inputList=[], weightList=[])
        self.nType = nType
        
    def __repr__(self):
        return ('<{} Neuron with weights {}>'.format(self.nType, self.weightList))

class Bias(Neuron):
    def __init__(self, nType='b'): # inputList must always equal 1
        Neuron.__init__(self, inputList=1, weightList=0) # make the initial weight 0 because this will render the default argument bias inert.
        self.nType = nType
        
    def __repr__(self):
        return ('<{} Neuron with weights {}>'.format(self.nType, self.weightList))
        
class GameBoard():
    def __init__(self, count=0, board=[[" "," "," "],[" "," "," "],[" "," "," "]], data = [0,0,0,0,0,0,0,0,0], win='n/a'):
        self.count = count
        self.board=board
        self.data = data
        self.win = win
        
    def display(self):
        print(" ------------- \n | {} | {} | {} | \n ------------- \n | {} | {} | {} | \n ------------- \n | {} | {} | {} | \n -------------".format(self.board[0][0],self.board[0][1],self.board[0][2],self.board[1][0],self.board[1][1],self.board[1][2],self.board[2][0],self.board[2][1],self.board[2][2]))

    def getBoard(self):
        return self.data
    
    def add(self, char, n):
        self.count += 1
        n-=1
        a = n//3
        b = n%3
        if n < 9:
            if self.board[a][b] != 'x' and self.board[a][b] != 'o':
                self.board[a][b] = char
                if char == 'x':
                    self.data[n] = 1
                elif char == 'o':
                    self.data[n] = -1
            else:
                return False
        else:
            return False
            
        if self.count > 4:
            if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ' or self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
                print(char, " win")
                self.win = char
            
            c=0
            while c < 3:
                if self.board[c][0] == self.board[c][1] == self.board[c][2] != ' ' or self.board[0][c] == self.board[1][c] == self.board[2][c] != ' ':
                    print(char, " win")
                    self.win = char
                    break
                else:
                    c+=1
        if self.count == 9:
            print('t')
            self.win = 't'
            
    def reset(self):
        self.board = [[" "," "," "],[" "," "," "],[" "," "," "]]
        self.data = [0,0,0,0,0,0,0,0,0]
        
def playGame(n, g, numTimes):
    turn = 0
    for i in range(numTimes):
        coinFlip = random.choice([0,1])
        
        if coinFlip == 0:
            ideas = feedForward(n, g.getBoard())
            best = max(ideas)
            move = ideas.index(best)+1
            g.add('x', move)
            turn += 1
        else:
            g.add('o', random.randrange(0,8))
        
        
    g.reset()
    return wins

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
        
def mutate(iList, error): # only takes lists of weights, as opposed to classes of neurons with weights. If you want to convert an NN's genome, use genomeToList
    wList = copy.deepcopy(iList)
    for i in range(len(wList)):
        for j in range(len(wList[i])):
            if type(wList[i][j]) == float or type(wList[i][j]) == int:
                wList[i][j] = round(wList[i][j] + random.choice([0.0,0.0,0.0,random.uniform(-1,1),random.uniform(-1,1)*random.randint(1,3),random.uniform(-1,1)*error]),3)
            else:
                for k in range(len(wList[i][j])):
                    wList[i][j][k] = round(wList[i][j][k] + random.choice([0.0,0.0,0.0,0.0,random.uniform(-1,1),random.uniform(-1,1)*random.randint(1,5),random.uniform(-1,1)*error]),3)
    return wList

def perturb(p, place):
    if type(p) == int or type(p) == float:
        p += random.choice([place,-place])
        return p
    else:
        mutation = random.choice([place,-place])
        loc = random.randint(0, len(p)-1)
        p[loc] += mutation
        p[loc] = round(p[loc],4)
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
        nList[1].append(Hidden1(str(n.id)+'h'+str(i)))
        nList[1][i].setWeights(l[1][i])
    for i in range(n.hidden2):
        nList[2].append(Hidden2(str(n.id)+'h'+str(i)))
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
        return round(random.uniform(-8,8),3)
    elif type(inputNum) == int:
        j = 0
        randomWeightsList = []
        while j < inputNum:
            randomWeightsList.append(round(random.uniform(-0.8,0.8),3))
            j += 1
        j = 0
        return randomWeightsList
        
def perturbation(error): # fuzziness
    p = (0.005 * error * round(random.uniform(-1,1),4))
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

def feedForward(n, inputs, choice=3): # (inputs, inputList, hidden1, hidden2, outputs, bias):
    
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
            c+=1
    else:
        print("Inputs and inputs nodes do not match")
        
    for inp in inputList:
        FFInputs.append(inp.feedForward(inp.getWeights(), inp.getInputs()))
        '''
        print(inp.feedForward(inp.getWeights(), inp.getInputs()))
        print("stuff that got printed out")
        '''
        
    if choice == 0:
        if len(inputs) == 1:
            return inputs[0]
        else:
            return inputs
            
    for hidden in hidden1:
        FFHidden1.append(hidden.feedForward(hidden.getWeights(), inputs, bias[0]))
        
    if choice == 1:
        if len(FFHidden1) == 1:
            return FFHidden1[0]
        else:
            return FFHidden1
            
    for hidden in hidden2:
        FFHidden2.append(hidden.feedForward(hidden.getWeights(), FFHidden1, bias[1]))
        
    if choice == 2:
        if len(FFHidden2) == 1:
            return FFHidden2[0]
        else:
            return FFHidden2
                
    for out in outputs:
        FFOutputs.append(out.feedForward(out.getWeights(), FFHidden2, bias[2]))
        
    if choice == 3:
        if len(FFOutputs) == 1:
            return FFOutputs[0]
        else:
            return FFOutputs
            
    if choice == 4: # when choice is 4, return the list of all layer output lists
        return [inputs, FFHidden1, FFHidden2, FFOutputs]
        
fullErrorList = []
errorPlot = []
                
def trainSine(n): # generates training data for the neural network
    batch = 2000 # batch size, must be even or bad things will happen
    
    xValsAbove = []
    xValsBelow = []
    yValsAbove = []
    yValsBelow = []
    
    a = 0
    b = 0
    while a+b < batch:
        
        x = random.random()
        y = random.random()
        
        if (y > x*x): # right-hand side is the function we want to teach the network.
            if a > (batch/2)-1:
                continue
            else:
                xValsAbove.append(x)
                yValsAbove.append(y)
                a += 1
        else:
            if b > (batch/2)-1:
                continue
            else:
                xValsBelow.append(x)
                yValsBelow.append(y)
                b += 1
                
    d=0
    error = 0
    
    while d < (batch/2):
        
        '''
        n.backPropagation(1,[valsAbove[d][0], valsAbove[d][1]],0.025)
        n.backPropagation(0,[valsBelow[d][0], valsBelow[d][1]],0.025)
        '''
        
        d += 1
        
        output1 = feedForward(n, [xValsAbove[d], yValsAbove[d]])
        output2 = feedForward(n, [xValsBelow[d], yValsBelow[d]])
        
        if output1 < 0.99:
            error += (1-output1)**2
        if output2 > 0.01:
            error += (output2)**2
        d+=1
        
    sineX = []
    sineY = []
            
    e=-100
    while e < 100:
        sineX.append(e)
        sineY.append(math.sin(e))
        e += 0.1
    
    pylab.plot(xValsAbove,yValsAbove,'.',color='b')
    pylab.plot(xValsBelow,yValsBelow,'.',color='r')
    pylab.plot(sineX,sineY,color='k')
    pylab.xlim(xmin=-20, xmax=20)
    pylab.ylim(ymin=-2, ymax=2)
    pylab.show
    
    return error/batch
    
def plotter(): # test for showing how the data points should be classified
    xValsAbove = []
    yValsAbove = []
    xValsBelow = []
    yValsBelow = []
    a = 0
    b = 0
    
    amount = 2000 # amount of points to plot (should be high, so we can see what the classification should look like)
    
    while a+b < amount: # making sure there is an even number on both sides
        # x = random.random()*20*random.choice([-1,1])
    
        x = random.random()
        y = random.random()
        
        # y = random.choice([random.random()*20,random.random()*20,random.random()*20,random.random()*20,random.random()*20,random.random()*20,random.random()*20,random.random()*20,random.random()*random.choice([-1,1])]) # y = random.choice([random.random()*5*random.choice([-1,1]),random.random()*2*random.choice([-1,1]),random.random()*random.choice([-1,1]),random.random()*random.choice([-1,1])])
        
        if y > x*x:
            if a > (amount/2)-1:
                continue
            else:
                xValsAbove.append(x)
                yValsAbove.append(y)
                a += 1
        else:
            if b > (amount/2)-1:
                continue
            else:
                xValsBelow.append(x)
                yValsBelow.append(y)
                b += 1
            
    pylab.plot(xValsAbove,yValsAbove,'.',color='b')
    pylab.plot(xValsBelow,yValsBelow,'.',color='r')
    pylab.xlim(xmin=0, xmax=1)
    pylab.ylim(ymin=0, ymax=1)
    
    pylab.show

def evolve(): # evolutionary algorithm for evolving the weights of the neural network
    n = NeuralNetwork()
    n.initWeights()
    errorList = []
    errorDict = {}
    survivors = []
    c=0
    while c < 100:
        n.initWeights()
        error = trainSine(n)
        errorList.append(copy.deepcopy(error)) # append error
        errorDict[error] = copy.deepcopy(genomeToList(n)) # append list of weights that return the respective error
        print("Error: ", error)
        error = 0
        c+=1
    c=0
        
    errorList.sort()
    
    e=0
    while e < 10:
        survivors.append((errorList[e], errorDict[errorList[e]]))
        print(errorList[e])
        e+=1
        
    loop = 0
    while loop < 400: # call it quits after n generations
        survivors = generate(survivors, n, loop+1) # use "loop" as a var that controls the rate of mutations. As the generations get deeper, lessen the rate of change.
        if survivors[0][0] < 0.005:
            break
        loop += 1
    
    pylab.plot(errorPlot,'.',color='k')
    pylab.show
    
    print(survivors) #, survivors[3], survivors[4], survivors[5]]
    return survivors
        
def generate(survivors, n, genNum): # creates a generation of neural network weights, tests them and keeps the best ones
    errorList = []
    survivorList = []
    
    for i in range(len(survivors)):
        errorList.append(survivors[i]) # survivors
    
    errorList.sort()
        
    breedLoop=0
    while breedLoop < 10:
        rand = random.sample([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],2)
        
        n.setWeights(listToGenome(errorList[rand[0]][1],n))
        copy1 = genomeToList(n)
        n.setWeights(listToGenome(errorList[rand[1]][1],n))
        copy2 = genomeToList(n)
        
        child = breed(copy1, copy2)
        n.setWeights(listToGenome(child, n))
        # print(n.getWeights())
        error = trainSine(n)
        
        listCheck = 0
        for l in range(len(errorList)):
            if error in errorList[l]:
                listCheck += 1
                break
        if listCheck == 0:
            errorList.append((copy.deepcopy(error), copy.deepcopy(child)))
            
        breedLoop +=1
    
    mutateLoop=0
    while mutateLoop < 15:
        mList = errorList[random.randint(0,4)][1]
        newList = mutate(mList, errorList[0][0])
        n.setWeights(listToGenome(newList,n))
        error = trainSine(n)
        
        listCheck = 0
        for l in range(len(errorList)):
            if error in errorList[l]:
                # print("Y")
                listCheck += 1
                break
        if listCheck == 0:
            errorList.append((copy.deepcopy(error), copy.deepcopy(newList)))
            # print("N")
        mutateLoop += 1
        
    errorList.sort()
    # print(errorList[:3])
    # errorList = sorted(errorList, key=lambda errors: errors[0])

    e=0
    print("Gen. " + str(genNum) + " survivors: ")
    while e < 10:
        survivorList.append(errorList[e])
        print(errorList[e][0])
        e+=1
        
    print("Top: ", survivorList[0])
    print("")
        
    errorPlot.append(errorList[0])
        
    return survivorList
    
def plotterNN(n): # use this function to plot the values of the NN

    xValsAbove = []
    yValsAbove = []
    xValsBelow = []
    yValsBelow = []
    xValsUncertain = []
    yValsUncertain = []
    
    # n.getNeuron(0,0).setWeights([10, 0])
    # n.getNeuron(0,0).setWeights([random.random()*random.choice([-1,1])*10, random.random()*random.choice([-1,1])*10])

    a=0
    while a < 4000:
        x = random.random()
        y = random.random()
        
        # out = n.getNeuron(0,0).feedForward(n.getNeuron(0,0).getWeights(),[x,y],n.getNeuron(4,0))
        
        out = feedForward(n,[x,y])
        
        if out > 0.75:
            xValsAbove.append(x)
            yValsAbove.append(y)
        elif out <= 0.75 and out >= 0.25:
            xValsUncertain.append(x)
            yValsUncertain.append(y)
        elif out < 0.25:
            xValsBelow.append(x)
            yValsBelow.append(y)
        a+=1
    
    pylab.plot(xValsAbove,yValsAbove,'.',color='b')
    pylab.plot(xValsBelow,yValsBelow,'.',color='r')
    pylab.plot(xValsUncertain,yValsUncertain,'.',color='y')
    pylab.xlim(xmin=0, xmax=1)
    pylab.ylim(ymin=0, ymax=1)
    pylab.show

print("To see a demonstration, type runDemo() in the console. This requires Pylab for plotting.")
print("If you do not have Pylab, type runDemo2() for the non-graphical version.")
# print("To read the documentation, type doc().")

def runDemo():
    print("Demo showing a feedforward neural network classifying points above and below the function y=x^2.")
    n = NeuralNetwork(2,9,8,1,3)
    print("Neural network initialized. Generating weights.")
    n.initWeights()
    print(n.getWeights())
    loop = 0
    print("Running through training cycles...")
    while loop < 10: # run through 10 training epochs
        trainSine(n)
        loop += 1
    print("Finished.")
    plotterNN(n)
    print(n.nList)

def runDemo2():
    print("Demo showing a feedforward neural network classifying points above and below the function y=x^2.")
    n = NeuralNetwork(2,9,8,1,3)
    print("Neural network initialized. Generating weights.")
    n.initWeights()
    print(n.getWeights())
    loop = 0
    print("Running through training cycles...")
    while loop < 100: # run through 10 training epochs
        trainSine(n)
        loop += 1
    print("Finished.")
    print("")
    
    print("Testing [0.9,0.9]. Target value is 1. Actual value is: ", feedForward(n, [0.9,0.9]))
    print("Testing [0.9,0.1]. Target value is 0. Actual value is: ", feedForward(n, [0.9,0.1]))
    print("Testing [0.4,0.6]. Target value is 1. Actual value is: ", feedForward(n, [0.4,0.6]))
    print("Testing [0.6,0.0]. Target value is 0. Actual value is: ", feedForward(n, [0.6,0.0]))
    print("Testing [0.2,0.2]. Target value is 1. Actual value is: ", feedForward(n, [0.2,0.2]))
    print("Testing [0.7,0.3]. Target value is 0. Actual value is: ", feedForward(n, [0.7,0.3]))