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

class NN(object):
    
    def __init__(self, l=[2,3,1], a='sigmoid', bias=1): # l[0] is the input layer. L[1] through L[n-1] are hidden layers. L[n] is the output layer. Arbitrarily deep ANNs can now be initialized.
        self.a = a # activation function options: 'sigmoid', 'step', 'rect', 'softplus', 'tanh'
        self.bias = bias # bias node input. To turn bias nodes off, set to 0.
        self.l = l
        self.initWeights()
        
    def initWeights(self):
        self.w = []
        for i in range(len(self.l)-1):
            n = 2*np.random.random_sample((self.l[i]+1,self.l[i+1]))-1 # weights are initialized to random values between -1 and 1. Add an extra weight for the bias node
            self.w.append(n)
        # self.b = 2*np.random.random_sample((len(self.l)-1))-1 # bias nodes get appended to the end
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
                if self.a == 'step':
                    a[i] = step(a[i])
                elif self.a == 'rect':
                    a[i] = rect(a[i])
                elif self.a == 'softplus':
                    a[i] = softplus(a[i])
                elif self.a == 'tanh':
                    a[i] = tanh(a[i])
                elif self.a == 'ss':
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
        for i in range(len(inputs)): # for each input
            out = self.feedForward(inputs[i],True) # calculate the output
            for j in range(self.l[len(self.l)-1]): # for every output node
                for k in range(self.l[len(self.l)-2]+1): # for every node in the hidden layer
                    # print("Target: ", target[i])
                    # print("Out: ", out)
                    d = deltaError(out[len(out)-1][j], target[i][j]) # calculate error
                    # print("out[len(out-2)]", out[-2])
                    self.w[len(self.w)-2][k] += 0.2 * out[-2][k] * d # learning constant times hidden layer output * error
             
def deltaError(o_b, t_b):
    error = o_b*(1 - o_b)*(t_b - o_b)
    # print("Error: ", error)
    return error

#
# activation functions
#

def generateData(n):
    xValsRed = []
    yValsRed = []
    xValsBlue = []
    yValsBlue = []
    
    while len(xValsRed)+len(xValsBlue) < n:
        x = random.random()
        y = random.random()
        # if ( ( x - 1/2 )**2 + ( y - 1/2 )**2 < 1/16 ):
        if y >= (x+1/3) and len(xValsRed) < n/2:
            xValsRed.append(x)
            yValsRed.append(y)
        elif y < (x+1/3) and len(xValsBlue) < n/2:
            xValsBlue.append(x)
            yValsBlue.append(y)
            
    return [xValsRed,yValsRed,xValsBlue,yValsBlue]

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
    
    xLine = []
    yLine = []
    
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
            
    a=0.0
    b=0.0
    for j in range(10000):
        a += 0.0001
        b = a+1/3
        xLine.append(a)
        yLine.append(b)
            
    pylab.plot(xValsRed,yValsRed,'.',color='r')
    pylab.plot(xValsUncertain,yValsUncertain,'.',color='m')
    pylab.plot(xValsBlue,yValsBlue,'.',color='b')
    pylab.plot(xLine,yLine,'.',color='k')
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
    
def deltaInner(o_i, o_j, j, l, c=0.025):
    s = 0.0
    for i in range(l):
        s += deltaInner(l)*j[i]
    return -c*o_i*(s)*o_j*(1-o_j)
    
# evolutionary algorithm
# generate
# test (error/score, world)
# select


def demo1(): # or table demonstration
    print("Perceptron that does linear classification. Requires matplotlib.")
    o=NN([2,1],a='sigmoid',bias=1)
    print("Weights: ", o.getWeights())
    print("")
    # o.setWeights([np.array([[10],[10]]), np.array([0])])
    z=generateData(10000)
    print("Training...")
    for i in range(0,len(z[0])):
        o.bp([[z[0][i],z[1][i]]],([[1]]))
        o.bp([[z[2][i],z[3][i]]],([[0]]))
    print("Weights: ", o.getWeights())
    print("")
    print("In: [1.0][0.0]. Out: ", o.feedForward([1.0,0.0]))
    print("In: [0.8][0.2]. Out: ", o.feedForward([0.8,0.2]))
    print("In: [0.3][0.1]. Out: ", o.feedForward([0.3,0.1]))
    print("")
    print("In: [0.5][0.5]. Out: ", o.feedForward([0.5,0.5]))
    print("")
    print("In: [0.0][1.0]. Out: ", o.feedForward([0.0,1.0]))
    print("In: [0.5][0.7]. Out: ", o.feedForward([0.5,0.7]))
    print("In: [0.7][0.9]. Out: ", o.feedForward([0.7,1.9]))
    
    plotNetwork(o)
    del o # end resource
    
def demo2():
    print("Perceptron that demonstrates correct responses to OR truth table inputs. Requires matplotlib.")
    o=NN([2,1],a='sigmoid',bias=1)
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
    
# demo1()

print("Type demo1() for a graphical table with linear classification (requires matplotlib)")
print("Type demo2() for a non-graphical demo with or tables")