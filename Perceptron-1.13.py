# from SineFunction import *

import random
import math
import pylab
import copy
import time

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
    def __init__(self, inputs=9, hidden1=15, hidden2=24, hidden3=21, hidden4=12, outputs=9, bias=5):
        self.id = NeuralNetwork.instance
        NeuralNetwork.instance = NeuralNetwork.instance + 1
        self.inputs = inputs
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.outputs = outputs
        self.bias = bias
        
    def initWeights(self): # initialize random weights
        self.nList = [ [], [], [], [], [], [] ]
        for i in range(self.hidden1):
            self.nList[0].append(Hidden1('h'+str(i)+'l1'+'n'+str(self.id)))
            self.nList[0][i].setWeights(getRandomWeight(self.inputs))
            
        for i in range(self.hidden2):
            self.nList[1].append(Hidden2('h'+str(i)+'l2'+'n'+str(self.id)))
            self.nList[1][i].setWeights(getRandomWeight(len(self.nList[0])))
            
        for i in range(self.hidden3):
            self.nList[2].append(Hidden3('h'+str(i)+'l3'+'n'+str(self.id)))
            self.nList[2][i].setWeights(getRandomWeight(len(self.nList[1])))
            
        for i in range(self.hidden4):
            self.nList[3].append(Hidden4('h'+str(i)+'l4'+'n'+str(self.id)))
            self.nList[3][i].setWeights(getRandomWeight(len(self.nList[2])))
            
        for i in range(self.outputs):
            self.nList[4].append(Output('o'+str(i)+'n'+str(self.id)))
            self.nList[4][i].setWeights(getRandomWeight(len(self.nList[3])))
            
        for i in range(self.bias):
            self.nList[5].append(Bias('b'+str(i)+'n'+str(self.id)))
            self.nList[5][i].setWeights(0) # self.nList[4][i].setWeights(getRandomWeight())
            
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
        
class Hidden4(Neuron):
    def __init__(self, nType='h(4)'):
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
    def __init__(self, count=0, board=[[" "," "," "],[" "," "," "],[" "," "," "]], data = [0,0,0,0,0,0,0,0,0], winner=None):
        self.count = count
        self.board=board
        self.data = data
        self.winner = winner
        
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
            if self.check(char) == True:
                self.winner = char
                # print(char, " wins")
            
        if self.count == 9:
            print("Count: ", self.count)
            self.winner = 't'
            
    def check(self, char): # is game over? If so, return true. Otherwise, return false
    
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ' or self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
                self.winner = char
                return True
            
        c=0
        while c < 3:
            if self.board[c][0] == self.board[c][1] == self.board[c][2] != ' ' or self.board[0][c] == self.board[1][c] == self.board[2][c] != ' ':
                self.winner = char
                return True
            else:
                c+=1
        return False
            
    def reset(self):
        self.board = [[" "," "," "],[" "," "," "],[" "," "," "]]
        self.data = [0,0,0,0,0,0,0,0,0]
        self.count = 0
        self.winner = None
        
def playGame(g, n1, n2):
    whoseTurn = 0
    coinFlip = random.choice([0,1])
        
    if coinFlip == 0:
        # print("X's turn")
        moves = outputToBoard(feedForward(n1,g.data))
        g.add('x', moves[0])
        whoseTurn += 1
    else:
        whoseTurn = 1
        # print("O's turn")
        if n2 == 'r':
            moves = randomAgent()
        else:
            moves = outputToBoard(feedForward(n2,g.data))
        g.add('o',moves[0])
        whoseTurn -= 1
    
    g.display()
    # time.sleep(2)
            
    while g.winner == None:
        if whoseTurn == 0:
            # print("")
            # print("X's turn")
            moves = outputToBoard(feedForward(n1,g.data))
            for i in range(len(moves)):
                if g.data[moves[i]] == 0: # if position in board has not already been played
                    g.add('x',moves[i]+1)
                    break
            whoseTurn += 1
            
        elif whoseTurn == 1:
            # print("")
            # print("O's turn")
            if n2 == 'r':
                moves = randomAgent()
            else:
                moves = outputToBoard(feedForward(n2,g.data))
            # print("")
            for i in range(len(moves)):
                if g.data[moves[i]] == 0:
                    g.add('o',moves[i]+1)
                    break
            whoseTurn -= 1
            
        g.display()
        
        time.sleep(0.1)
        
def randomAgent():
    y = [0,1,2,3,4,5,6,7,8]
    random.shuffle(y)
    return y
    
def championShip():
    garry = NeuralNetwork()
    garry.initWeights()
    
    josh = NeuralNetwork()
    josh.initWeights()
    
    adam = NeuralNetwork()
    adam.initWeights()
    
    richard = NeuralNetwork()
    richard.initWeights()
    
    a=gameTester(garry,'r')
    b=gameTester(josh,'r')
    c=gameTester(adam,'r')
    d=gameTester(richard,'r')
    
    print("Garry's games: ", a)
    print("Josh's games: ", b)
    print("Adam's games: ", c)
    print("Richard's games: ", d)
        
def gameTester(p1,p2):
    w = 0 # george's wins
    l = 0 # george's losses (john's wins)
    t = 0 # ties
    g = GameBoard()
    
    for game in range(0,16):
        p1.initWeights()
        playGame(g,p1,p2)
        
        if g.winner == 'x':
            w += 1
        elif g.winner == 'o':
            l += 1
        else:
            t += 1
        g.reset()
        
    return [w,l,t]
        
    # print('George wins: ', w)
    # print('John wins: ', l)
    # print('Ties: ', t)
    
def outputToBoard(o):
    newVal = []
    finalVal = []
    for i in range(len(o)):
        newVal.append([o[i], i])
        
    newVal.sort(reverse=True)
    
    for j in range(len(newVal)):
        finalVal.append(newVal[j][1])
        
    return finalVal
    
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
                wList[i][j] = round(wList[i][j] + random.gauss(0,1),3)
            else:
                for k in range(len(wList[i][j])):
                    wList[i][j][k] = round(wList[i][j][k] + random.gauss(0,0.7),3)
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
    l = [ [], [], [], [], [], [] ]
    i=0
    for layer in n.getWeights():
        for neuron in layer:
                l[i].append(neuron.getWeights())
        i+=1
    return l
    
def listToGenome(l, n): # assumes weights match neurons properly
    nList= [ [], [], [], [], [], [] ]
    for i in range(n.hidden1):
        nList[0].append(Hidden1(str(n.id)+'h'+str(i)))
        nList[0][i].setWeights(l[0][i])
        
    for i in range(n.hidden2):
        nList[1].append(Hidden2(str(n.id)+'h'+str(i)))
        nList[1][i].setWeights(l[1][i])
        
    for i in range(n.hidden3):
        nList[2].append(Hidden3(str(n.id)+'h'+str(i)))
        nList[2][i].setWeights(l[2][i])
        
    for i in range(n.hidden4):
        nList[3].append(Hidden4(str(n.id)+'h'+str(i)))
        nList[3][i].setWeights(l[3][i])
        
    for i in range(n.outputs):
        nList[4].append(Output(str(n.id)+'o'+str(i)))
        nList[4][i].setWeights(l[4][i])
        
    for i in range(n.bias):
        nList[5].append(Bias(str(n.id)+'b'+str(i)))
        nList[5][i].setWeights(l[5][i])
        
    return nList
    

def getRandomWeight(inputNum=None): # returns list of random floats from (-1,1). default argument returns single float. otherwise, returns float(s) inside a list

    if inputNum == None:
        return round(random.uniform(-4,4),3)
    elif type(inputNum) == int:
        j = 0
        randomWeightsList = []
        while j < inputNum:
            randomWeightsList.append(round(random.uniform(-4,4),3))
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
        
def feedForward(n, inputs, choice=5): # (inputs, inputList, hidden1, hidden2, outputs, bias):
    
    hidden1 = n.getNeuronList()[0]
    hidden2 = n.getNeuronList()[1]
    hidden3 = n.getNeuronList()[2]
    hidden4 = n.getNeuronList()[3]
    outputs = n.getNeuronList()[4]
    bias = n.getNeuronList()[5]
    
    FFHidden1 = []
    FFHidden2 = []
    FFHidden3 = []
    FFHidden4 = []
    FFOutputs = []
        
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
        n
    if choice == 2:
        if len(FFHidden2) == 1:
            return FFHidden2[0]
        else:
            return FFHidden2
        
    for hidden in hidden3:
        FFHidden3.append(hidden.feedForward(hidden.getWeights(), FFHidden2, bias[1]))
        
    if choice == 3:
        if len(FFHidden3) == 1:
            return FFHidden3[0]
        else:
            return FFHidden3
        
    for hidden in hidden4:
        FFHidden4.append(hidden.feedForward(hidden.getWeights(), FFHidden3, bias[1]))
        
    if choice == 4:
        if len(FFHidden4) == 1:
            return FFHidden4[0]
        else:
            return FFHidden4
                
    for out in outputs:
        FFOutputs.append(out.feedForward(out.getWeights(), FFHidden4, bias[2]))
        
    if choice == 5:
        if len(FFOutputs) == 1:
            return FFOutputs[0]
        else:
            return FFOutputs
            
    if choice == 6: # when choice is 4, return the list of all layer output lists
        return [inputs, FFHidden1, FFHidden2, FFHidden3, FFHidden4, FFOutputs]
        
fullErrorList = []
errorPlot = []
                
def trainGame(n): # generates training data for the neural network
    print('hi')
    
    
def evolve(): # evolutionary algorithm for evolving the weights of the neural network
    n = NeuralNetwork()
    n.initWeights()
    errorList = []
    errorDict = {}
    survivors = []
    c=0
    while c < 40:
        n.initWeights()
        error = trainGame(n)
        errorList.append(copy.deepcopy(error)) # append error
        errorDict[error] = copy.deepcopy(genomeToList(n)) # append list of weights that return the respective error
        print("Error: ", error)
        error = 0
        c+=1
    c=0
        
    errorList.sort()
    
    e=0
    while e < 40:
        survivors.append((errorList[e], errorDict[errorList[e]]))
        print(errorList[e])
        e+=1
        
    loop = 0
    while loop < 300: # call it quits after n generations
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
    while breedLoop < 80:
        rand = random.sample([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39],2)
        
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
    while mutateLoop < 20:
        mList = errorList[random.randint(0,10)][1]
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
    while e < 40:
        survivorList.append(errorList[e])
        print(errorList[e][0])
        e+=1
        
    print("Top: ", survivorList[0])
    print("")
        
    errorPlot.append(errorList[0])
        
    return survivorList
    
    
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
    
def sett(t):
    n=NeuralNetwork()
    z=listToGenome(t,n)
    n.setWeights(z)
    plotterNN(n)

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