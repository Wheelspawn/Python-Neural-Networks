import numpy as np
import random
import math
import pylab

#
# written by Nathaniel Le Sage 2017
#

#
# new neural network. Uses numpy library for fast matrix multiplication. Supports an arbitrary amount of layers.
# type n=NN() to initialize.
# l parameter specifies layers.
# a parameter specifies the activation function.
# weights are automatically initialized. To set your own, use setWeights(). To make sure you don't break the dimensions, use getWeights() and modify the vals there.
#

global step
step = 0

class NN(object):
    
    def __init__(self, l=[2,3,1], act='sigmoid', c=0.03, bias=1): # l[0] is the input layer. L[1] through L[n-1] are hidden layers. L[n] is the output layer. Arbitrarily deep ANNs can now be initialized.
        self.act = act # activation function
        self.bias = bias # bias node input. To turn bias nodes off, set to 0.
        self.l = l # number of neurons in each layer
        self.c = c # learning constant
        self.initWeights()
        
    def initWeights(self):
        self.w = []
        for i in range(len(self.l)-1):
            bit = ((i==len(self.l)-2)!=True).real # bit that tells if loop is at its end
            n = 2*np.random.random_sample((self.l[i]+1,self.l[i+1]))-1 # weights are initialized to random values between -1 and 1. Add an extra weight for the bias node
            self.w.append(n)
                
    def getWeights(self):
        return self.w
        
    def setWeights(self,new):
        for i in range(len(new)):
            if len(new[i]) != self.l[i]: # if the new weight setup is different 
                print("Warning! Network configuration is different than original specification. Continuing regardless...")
                self.l[i] = len(new[i])
                # self.l[i] = len(new[i]) # for making sure l is consistent with new weights
        self.w = new
        
    def feedForward(self, inputs, brk=False): # the brk argument returns all the activations
        inputs = inputs[:]
        inputs.append(self.bias) # bias in the input layer
        allVals = [inputs] 
        
        try:
            a = np.dot(inputs,self.w[0]) # activations
            for i in range(len(a)): # for the inputs into the network
                a[i] = eval((self.act+'({})').format(a[i])) # send each value through the specified activation function
                
            allVals.append(a)
            
            for i in range(1,len(self.w)): # for the processing of the hidden layers
                a = np.append(a, self.bias) # activations
                a = np.dot(a, self.w[i]) # integration of inputs in the next layer
                for i in range(len(a)):
                    a[i] = eval((self.act+'({})').format(a[i])) # activations
                allVals.append(a)
                
            if brk == True:

                for i in range(1,len(allVals)-1):
                    allVals[i] = np.append(allVals[i],self.bias)
                
                return allVals # return every activation
            return a
                
        except ValueError:
            print()
            print("Input dimensions are incorrect")
            return
        except NameError:
            print()
            print("Activation function not recognized")
            return
            
    def connection(self, layer, send, receive): # get connection weight from neuron in specified layer to neuron in adjacent layer
        return self.w[layer][send][receive]
        
    def neuronWeights(self, layer, index): # get the weights of a neuron from specified layer and index
        end = self.l[layer]+1
        return self.w[layer][0:end,index:index+1]
        
    def bp(self,inputs,target): # inp is an array or arrays of values. target is the matching intended output.
                                # The pattern follows [ [i_1, i_2, ..., i_n] ] and [ [t_1, t_2, ..., t_n] ]
                                # where i_1, i_2, ..., i_n and t_1, t_2, ..., t_n are input and target vectors, respectively
    
        outputError = []
        # perceptron
        for i in range(len(inputs)): # for each input
            out = self.feedForward(inputs[i],True) # calculate the output
            for j in range(len(self.w[-1][0])): # for every output node
                
                d = deltaError(out[-1][j], target[i][j]) # calculate error
                
                for k in range(len(self.w[i])): # for every node in the hidden layer (+1 for bias)
                    # print("W: ", self.w[-1][k][j])
                    self.w[-1][k][j] += self.c * out[-2][k] * d
                
                outputError.append(d)
            
            '''
            print("Weights after output node update: ", self.w)
            print("Output error: ", outputError)
            '''
            
            if len(self.l) > 2: # if there are one or more hidden layers
                totalError = []
                for i in range(len(self.w)-2,-1,-1): # for each layer
                    for j in range(len(self.w[i][0])):
                        for k in range(len(self.w[i])):
                            
                            npWeights = np.dot(outputError,self.neuronWeights(1,0)[j])
                            
                            '''
                            print("Layer out: ", out[i+1][j])
                            print("Output error: ", outputError)
                            
                            print("Output neuron weights: ", m.neuronWeights(1,0)[j])
                            
                            print("npWeights: ", npWeights)
                            print("New error: ", hiddenError(out[i+1][j], npWeights))
                            
                            print("Final delta: ", self.w[i+1][k][0] * hiddenError(out[i+1][j], npWeights))
                            '''
                            
                            totalError.append(self.w[i][k][j] * hiddenError(out[i+1][j], outputError[0]))
                            self.w[i][k][j] += self.c * hiddenError(out[i+1][j], npWeights) * out[i][k]
                            
                    outputError = totalError[:]
                    totalError = []
             
def deltaError(o, t): # output, target
    error = o*(1 - o)*(t - o)
    return error
    
def hiddenError(o, e):
    error = -o*(1 - o)*e
    # print(error)
    return error
    
    
def evolve(n, data, trial='error'):
    for i in range(80000):
        n.initWeights()
        totalError = 0.0
        if trial == 'error':
            for j in range(100):
                result = n.feedForward(data[j][0][0])
                totalError += 1/200 * (result - data[j][1][0])**2
                print(totalError)
        elif trial == 'score':
            print('score')
        
def breed(m,n):
    a = m.getWeights()
    b = n.getWeights()
    o = NN(m.l, m.act, m.bias)
    c = o.getWeights()
    for i in range(len(a)):
        for j in range(len(a[i])):
            for k in range(len(a[i][j])):
                c[i][j][k] = random.choice([a[i][j][k],b[i][j][k]])
    return c
    
def mutate(n):
    l = n.getWeights()
    for i in range(len(l)):
        for j in range(len(l[i])):
            for k in range(len(l[i][j])):
                l[i][j][k] += random.choice([0,0,random.gauss(0,0.5)])
#
# activation functions
#
                
def helloThere():
    print("Hello there")

def generateData(n): # generates input/output pairs
    vals = []
    while len(vals) < n:
        x = round(random.random(),3)
        y = round(random.random(),3)
        
        lineUpper = "(y > -x+1)"
        lineLower = "(y <= -x+1)"
        parUpper = "y > ( -4*( x**2 )+( 4*x ) )"
        parLower = "y <= ( -4*( x**2 )+( 4*x ) )"
        circleUpper = "( ( x - 1/2 )**2 + ( y - 1/2 )**2 > 1/16 )"
        circleLower = "( ( x - 1/2 )**2 + ( y - 1/2 )**2 <= 1/16 )"
        
        if eval(parUpper):
            vals.append([[x, y],[1]]) # input, desired output
        elif eval(parLower):
            vals.append([[x, y],[0]]) # input, desired output
            
    return vals

def plotData(x1,y1,x2,y2):
    pylab.plot(x1,y1,'.',color='r')
    pylab.plot(x2,y2,'.',color='b')
    pylab.xlim(xmin=0, xmax=1)
    pylab.ylim(ymin=0, ymax=1)
    pylab.show
    
    print("hi")
    
def plotNetwork(n):
    xValsRed = []
    yValsRed = []
    xValsUncertain = []
    yValsUncertain = []
    xValsBlue = []
    yValsBlue = []
    
    # xLine = []
    # yLine = []
    
    for i in range(0,500):
        x = random.random()
        y = random.random()
        f = n.feedForward([x,y])
        # if ( ( x - 1/2 )**2 + ( y - 1/2 )**2 < 1/16 ):
        if f[0] > 0.55:
            xValsRed.append(x)
            yValsRed.append(y)
        elif f[0] <= 0.55 and f[0] > 0.45:
            xValsUncertain.append(x)
            yValsUncertain.append(y)
        else:
            xValsBlue.append(x)
            yValsBlue.append(y)
    '''        
    a=0.0
    b=0.0
    for j in range(10000):
        a += 0.0001
        b = a+1/3
        xLine.append(a)
        yLine.append(b)
    '''
            
    pylab.plot(xValsRed,yValsRed,'.',color='r')
    pylab.plot(xValsUncertain,yValsUncertain,'.',color='m')
    pylab.plot(xValsBlue,yValsBlue,'.',color='b')
    # pylab.plot(xLine,yLine,'.',color='k')
    pylab.xlim(xmin=0, xmax=1)
    pylab.ylim(ymin=0, ymax=1)
    pylab.show

def sigmoid(finalSum): # if sum is positive, return 1. else, return 0
    if finalSum > 8:
        return 1
    if finalSum < -8:
        return 0
    else:
        return 1/(1+((math.e) ** (-finalSum))) # sigmoid function
        
def sigmoidSimple(finalSum):
    if finalSum < -4:
        return 0
    elif finalSum >= -4 and finalSum < -1:
        return finalSum/12 + 1/3
    elif finalSum >= -1 and finalSum < 1:
        return finalSum
    elif finalSum >= 1 and finalSum < 4:
        return finalSum/12 + 2/3
    else:
        return 1
        
def timeTest(n=1000):
    import time
    
    n1 = NN([2,1],a='ss')

    start1 = time.time()
    for i in range(0,n):
        n1.feedForward([random.random()*random.randrange(-10,10),random.random()*random.randrange(-10,10)])
    end1 = time.time()
    print(end1 - start1)

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
    
# evolutionary algorithm
# generate
# test (error/score, world)
# select

def demo1(): # or table demonstration
    print("Nonlinear classification. Requires matplotlib.")
    o=NN([2,2,1],act='sigmoid',c=0.03,bias=1.0)
    print("Weights: ", o.getWeights())
    print("")
    
    z=generateData(40)
    print("Data: ", z)
    
    print("Training...")
    for i in range(0,500):
        for j in range(0,len(z)-1):
            # print(j)
            # print([z[j][0]], [z[j][1]])
            o.bp([z[j][0]], [z[j][1]])

    print("Weights: ", o.getWeights())
    print("")
    print("In: [0.1][0.1]. Out: ", o.feedForward([0.1,0.1]))
    print("In: [0.8][0.9]. Out: ", o.feedForward([0.8,0.9]))
    print("In: [0.5][0.5]. Out: ", o.feedForward([0.5,0.5]))
    print("In: [0.6][0.4]. Out: ", o.feedForward([0.6,0.4]))
    
    plotNetwork(o)
    
def demo2():
    print("Perceptron that demonstrates correct responses to OR truth table inputs.")
    o=NN([2,1],act='sigmoid',bias=1)
    print("Weights: ", o.getWeights())
    print("")
    print("Training",end='')
    for i in range(0,20000):
        if i%1000==0:
            print('.',end='')
        o.bp([[0.0,0.0]],[[0]])
        o.bp([[1.0,0.0]],[[1]])
        o.bp([[0.0,0.0]],[[0]])
        o.bp([[0.0,1.0]],[[1]])
        o.bp([[0.0,0.0]],[[0]])
        o.bp([[1.0,1.0]],[[1]])
    print("")
    print("Weights: ", o.getWeights())
    print("")
    print("In: [0][0]. Out: ", o.feedForward([0.0,0.0]))
    print("In: [1][0]. Out: ", o.feedForward([1.0,0.0]))
    print("In: [0][1]. Out: ", o.feedForward([0.0,1.0]))
    print("In: [1][1]. Out: ", o.feedForward([1.0,1.0]))
        
def demo3():
    print("Demonstrates word-property association")
    z=generateData(15000)
    p=NN([3,5],act='sigmoid',bias=1)
    print("Weights: ", p.getWeights())
    print("")
    print("Training", end="")
    for i in range(0,len(z)-1):
        if i%1000==0:
            print(p.getWeights())
        #     print(".", end="")
        p.bp([[1.0,0.0,0.0]],([[0,1,0,0,1]]))
        p.bp([[0.0,1.0,0.0]],([[1,0,0,1,0]]))
        p.bp([[0.0,0.0,1.0]],([[0,0,1,0,1]]))
        # print(p.getWeights())
        
    print("")
    print("Objects: Banana, Apple, Blueberry")
    print("Qualities: Red, Yellow, Blue, Hard, Soft")
    
    p.act='step'
    
    print(p.feedForward([1.0,0.0,0.0]))
    print(p.feedForward([0.0,1.0,0.0]))
    print(p.feedForward([0.0,0.0,1.0]))

# print("Type demo1() for a graphical table with linear classification (requires matplotlib)")
# print("Type demo2() for a non-graphical demo with or tables")

'''
m=NN([2,2,1],c=1.0,bias=0.0)
m.w = [np.array([ [0.1, 0.4],[0.8, 0.6],[0.0, 0.0] ]), np.array([ [0.3],[0.9],[0.0] ])]
print("Weights: ", m.w)
o=m.feedForward([0.35,0.9],brk=True)
print("Ouput: ", o)
m.bp([[0.35,0.9]],[[0.5]])
print("New weights: ", m.w)
'''