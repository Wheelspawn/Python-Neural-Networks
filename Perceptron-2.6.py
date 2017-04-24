import numpy as np
import random
import math
import pylab

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
    
    def __init__(self, l=[2,3,1], act='sigmoid', bias=1): # l[0] is the input layer. L[1] through L[n-1] are hidden layers. L[n] is the output layer. Arbitrarily deep ANNs can now be initialized.
        self.act = act # activation function options: 'sigmoid', 'step', 'rect', 'softplus', 'tanh'
        self.bias = bias # bias node input. To turn bias nodes off, set to 0.
        self.l = l
        self.initWeights()
        
    def initWeights(self):
        self.w = []
        for i in range(len(self.l)-1):
            n = 2*np.random.random_sample((self.l[i]+1,self.l[i+1]))-1 # weights are initialized to random values between -1 and 1. Add an extra weight for the bias node
            self.w.append(n)
        # self.b = 2*np.random.random_sample((len(self.l)-1))-1 # bias nodes get appended to the end2
        # self.w.append(self.b)
                
    def getWeights(self):
        return self.w
        
    def setWeights(self,new):
        for i in range(len(new)):
            if len(new[i]) != self.l[i]: # if the new weight setup is different 
                print("Warning! Network configuration is different than original specification. Continuing regardless")
                # self.l[i] = len(new[i]) # for making sure l is consistent with new weights
        self.w = new
        
        
    def feedForward(self, inputs, brk=False): # the brk argument returns the feedforward at every layer
        inputs.append(1.0)
        allVals = [inputs]
        # print(allVals)
        # print(self.w[0])
        try:
            # print("In: ", inputs+[1])
            # print("W: ", self.w[0])
            a = np.dot(inputs,self.w[0])
            for i in range(len(a)):
                if self.act == 'step':
                    a[i] = step(a[i])
                elif self.act == 'rect':
                    a[i] = rect(a[i])
                elif self.act == 'softplus':
                    a[i] = softplus(a[i])
                elif self.act == 'tanh':
                    a[i] = tanh(a[i])
                elif self.act == 'ss':
                    a[i] = sigmoidSimple(a[i])
                else:
                    a[i] = sigmoid(a[i])
            allVals.append(a)
            # print("Processed: ", a)
        except ValueError:
            print()
            print("Input dimensions are incorrect")
            return
            
        for i in range(1,len(self.w)):
            a = np.append(a, self.bias)
            a = np.dot(a, self.w[i]) # + self.w[len(self.w)-1][i]*self.bias # dot product of activations against neuron weights, with bias input
            for i in range(len(a)): # activation functions
                if self.act == 'step':
                    a[i] = step(a[i])
                elif self.act == 'rect':
                    a[i] = rect(a[i])
                elif self.act == 'softplus':
                    a[i] = softplus(a[i])
                elif self.act == 'tanh':
                    a[i] = tanh(a[i])
                elif self.act == 'ss':
                    a[i] = sigmoidSimple(a[i])
                else:
                    a[i] = sigmoid(a[i])
            # print("Processed: ", a)
            allVals.append(a)
        
        if brk == True:
            # print("allVals: ", allVals)
            return allVals
        return a
        
    def bp(self,inputs,target): # inp is an array or arrays of values. target is the matching intended output. If you only want to do bp on a single input, you still have to use nested brackets, like [[i_1, i_2, ..., i_n]], where n is the no. of neurons in the input layer.
        # totalError = []
    
        ''' # perceptron
        for i in range(len(inputs)): # for each input
            out = self.feedForward(inputs[i],True) # calculate the output
            for j in range(self.l[len(self.l)-1]): # for every output node
                for k in range(self.l[len(self.l)-2]+1): # for every node in the hidden layer
                    d = deltaError(out[len(out)-1][j], target[i][j]) # calculate error
                    self.w[len(self.w)-2][k] += 0.03 * out[-2][k] * d # learning constant times hidden layer output * error
        '''
        
        # no hidden layer
        
        print("Input len: ", len(inputs))
        print("Inputs: ", inputs)
        
        for i in range(len(inputs)): # for each input
            out = self.feedForward(inputs[i],True) # calculate the output
            for j in range(self.l[len(self.l)-1]): # for every output node
                for k in range(self.l[len(self.l)-2]+1): # for every node in the hidden layer
                    print("weight getting changed: ", self.w[len(self.w)-2][k][j])
                    print("out: ", out[len(out)-1][j])
                    print("target: ", target[i][j])
                    print("every output: ", out)
                    print("index:", k)
                    print("hidden output? ", out[-2][k])
                    d = deltaError(out[len(out)-1][j], target[i][j]) # calculate error
                    print("delta: ", d)
                    print("change: ", 0.03 * out[-2][k] * d)
                    print("")
                    self.w[len(self.w)-2][k] += 0.03 * out[-2][k] * d # learning constant times hidden layer output * error
                    # totalError.append(d)
            # hidden layer(s)
        '''
            if len(self.l) > 2:
                for i in range(self.l[-2]):
                    print(d[i])
        print("Total error: ", totalError)
        '''
        
        
def deltaError(o_b, t_b):
    error = -o_b*(1 - o_b)*(t_b - o_b)
    # print("Error: ", error)
    return error
    
def deltaInner(o_a, e):
    return o_a * (1-o_a) * e
    
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

def generateData(n): # generates input/output pairs for [2,1] perceptron
    vals = []
    while len(vals) < n:
        x = round(random.random(),3)
        y = round(random.random(),3)
        # if ( ( x - 1/2 )**2 + ( y - 1/2 )**2 < 1/16 ):
        if y >= x:  # ( ( x - 1/2 )**2 + ( y - 1/2 )**2 < 1/16 ) and len(xValsRed) < n/2:
            # print([[x, y],[1]])
            vals.append([[x, y],[1]]) # input, desired output
        elif y < x: # ( ( x - 1/2 )**2 + ( y - 1/2 )**2 > 1/16 ) and len(xValsBlue) < n/2:
            # print([[x, y],[0]])
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
    
    for i in range(0,6000):
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
    print("Perceptron that does nonlinear classification. Requires matplotlib.")
    o=NN([2,1],act='sigmoid',bias=1)
    print("Weights: ", o.getWeights())
    print("")
    # o.setWeights([np.array([[10],[10]]), np.array([0])])
    z=generateData(400000)
    print("Training...")
    for i in range(0,len(z[0])):
        o.bp([ [z[0][i], z[1][i] ] ],([[1]]))
        o.bp([ [z[2][i], z[3][i] ] ],([[0]]))
    print("Weights: ", o.getWeights())
    print("")
    print("In: [0.1][0.1]. Out: ", o.feedForward([0.1,0.1]))
    print("In: [0.8][0.9]. Out: ", o.feedForward([0.8,0.9]))
    print("In: [0.5][0.5]. Out: ", o.feedForward([0.5,0.5]))
    print("In: [0.6][0.4]. Out: ", o.feedForward([0.6,0.4]))
    
    plotNetwork(o)
    del o # end resource
    
def demo2():
    print("Perceptron that demonstrates correct responses to OR truth table inputs.")
    o=NN([2,1],act='sigmoid',bias=1)
    print("Weights: ", o.getWeights())
    print("")
    print("Training...")
    for i in range(0,10000):
        o.bp([[0.0,0.0]],([[0]]))
        o.bp([[1.0,0.0]],([[1]]))
        o.bp([[0.0,0.0]],([[0]]))
        o.bp([[0.0,0.1]],([[1]]))
        o.bp([[0.0,0.0]],([[0]]))
        o.bp([[1.0,1.0]],([[1]]))
    print("Weights: ", o.getWeights())
    print("")
    print("In: [0][0]. Out: ", o.feedForward([0.0,0.0]))
    print("In: [1][0]. Out: ", o.feedForward([1.0,0.0]))
    print("In: [0][1]. Out: ", o.feedForward([0.0,1.0]))
    print("In: [1][1]. Out: ", o.feedForward([1.0,1.0]))
    
    del o # end resource
    
def demo3():
    print("Demonstrates word-property association")
    p=NN([3,5],act='sigmoid',bias=1)
    print("Weights: ", p.getWeights())
    print("")
    print("Training", end="")
    for i in range(0,30000):
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
    
    print(p.feedForward([1.0,0.0,0.0]))
    print(p.feedForward([0.0,1.0,0.0]))
    print(p.feedForward([0.0,0.0,1.0]))
    
# print("Type demo1() for a graphical table with linear classification (requires matplotlib)")
# print("Type demo2() for a non-graphical demo with or tables")
    
'''

n=NN([2,2])
print(n.getWeights())
print(n.feedForward([0,1]))
print(n.feedForward([1,0]))

for xiojaqu in range(2):
    print("")
    print("Step ", xiojaqu)
    n.bp([[0,1]],[[1,0]])
    n.bp([[1,0]],[[0,1]])
    
print(n.getWeights())
print(n.feedForward([0,1]))
print(n.feedForward([1,0]))
'''
