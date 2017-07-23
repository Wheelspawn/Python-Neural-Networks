
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
            bit =(i//(len(self.l)-2)^1) # bit that tells if loop is at its end, for bias
            # print(bit)
            n = 2*np.random.random_sample((self.l[i+1],self.l[i]+bit))-1 # weights are initialized to random values between -1 and 1. Add an extra weight for hidden layer bias node
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
        allVals = [inputs] # this is our first level of activations
            
        for i in range(len(self.w)):
            allVals.append([]) # add a list to represent the activations for the next layer
            for j in range(len(self.w[i])):
                a = sigmoid(np.dot(allVals[i],self.w[i][j])) # dot product of activations and weights for each neuron in layer
                allVals[i+1].append(a)
                
            if i < len(self.w)-2:
                allVals[i+1].append(self.bias) # we must add the bias as the last activation for each layer, for all but the output node and final activations
                
        if brk==False:
            return allVals[-1]
        else:
            return allVals
        
        '''
        try:
        except ValueError:
            print()
            print("Input dimensions are incorrect")
            return
        except NameError:
            print()
            print("Activation function not recognized")
            return
        '''
            
    def connection(self, layer, send, receive): # get connection weight from neuron in specified layer to neuron in adjacent layer
        return self.w[layer][send][receive]
        
    def neuronWeights(self, layer, index): # get the weights of a neuron from specified layer and index
        return self.w[layer][index]
    
    def bp(self,inputs,targets,batch=True):
        
        if len(inputs[0]) != self.l[0]:
            print("No-go. Length of input vectors must be equal to length of target vectors.")
        else:
            if batch==True:      
                for i in range(len(inputs)):
                    out = self.feedForward(inputs[i],True)
                    print(inputs[i])
                    print(targets[i])
                    errors = self.calculateErrors(out, inputs[i],targets[i])
                    self.updateWeights(out, errors)
                    
        
        '''
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
            
            if len(self.l) > 2: # if there are one or more hidden layers
                totalError = []
                for i in range(len(self.w)-2,-1,-1): # for each layer
                    for j in range(len(self.w[i][0])):
                        for k in range(len(self.w[i])):
                            
                            npWeights = np.dot(outputError,self.neuronWeights(1,0)[j])
                            
                            print("Layer out: ", out[i+1][j])
                            print("Output error: ", outputError)
                            
                            print("Output neuron weights: ", m.neuronWeights(1,0)[j])
                            
                            print("npWeights: ", npWeights)
                            print("New error: ", hiddenError(out[i+1][j], npWeights))
                            
                            print("Final delta: ", self.w[i+1][k][0] * hiddenError(out[i+1][j], npWeights))
                            
                            totalError.append(self.w[i][k][j] * hiddenError(out[i+1][j], outputError[0]))
                            self.w[i][k][j] += self.c * hiddenError(out[i+1][j], npWeights) * out[i][k]
                            
                    outputError = totalError[:]
                    totalError = []
            '''
            
    def calculateErrors(self, out, inputVec, targetVec):
        
        errors = [[]]
        
        # inputVec = errors[0], I think???
        
        # print("")
        # print("Output layer")
        # print("")
        
        for i in range(len(self.w[-1])): # for every output node
            
            d = deltaError(out[-1][i], targetVec[i]) # calculate error
            
            # for j in range(len(self.w[-1][i])):
                
                # print("Weight: ", self.w[-1][i][j])
                # print("Output: ", out[-2][j])
                # print("Delta: ", d)
        
                # self.w[-1][i][j] += self.c * out[-2][j] * d
                
                # print("Final: ", self.w[-1][i][j])
            
            errors[0].append(d)
            
        # print("Output errors: ", errors)
        
        # print("")
        # print("Hidden layer")
        # print("")
        
        if len(self.l) > 2: # if there are one or more hidden layers
            for i in range(len(self.w)-2,-1,-1): # for each layer
                errors = [[]] + errors
                for j in range(len(self.w[i])): # for each neuron in layer
                    
                    error = 0
                    for k in range(len(self.w[i+1])):
                        # print("Weight: ", self.neuronWeights(i+1,k)[j])
                        # print("Prev error: ", errors)
                        
                        error += self.neuronWeights(i+1,k)[j] * errors[1][k]
                    
                    # print("Output: ", out[i+1][j])
                    errors[0].append(out[i+1][j]*(1-out[i+1][j])*error)
                
        # print("Final errors: ", errors)
        return errors
            
    def updateWeights(self, out, errors):
        print("and we arrive at the final destination")
        print("")
        print(self.w)
        print("")
        print(out)
        print("")
        print(errors)
        
        for i in range(len(self.w[-1])): # for every output node
            for j in range(len(self.w[-1][i])):
                self.w[-1][i][j] += self.c * out[-2][j] * errors[-1][i]
                
        if len(self.l) > 2: # if there are one or more hidden layers
            for i in range(len(self.w)-2,-1,-1): # for each layer
                for j in range(len(self.w[i])): # for each neuron:
                    for k in range(len(self.w[i][j])): # for each weight:
                        print("Weight: ", self.w[i][j][k])
                        print("Output: ", out[i+1][j])
                        print("Errors: ", errors[i][j])
                        print("")
                        self.w[i][j][k] += self.c * out[i+1][j] * errors[i][j]
                
        print("")
        print(self.w)
             
def deltaError(o, t): # output, target
    error = o*(1 - o)*(t - o)
    return error
    
def hiddenError(o, e):
    error = -o*(1 - o)*e
    # print(error)
    return error

#
# activation functions
#
                
def generateData(n): # generates input/output pairs

    inputs = []
    labels= []
    
    while len(inputs) < n:
        x = round(random.random(),3)
        y = round(random.random(),3)
        
        lineUpper = "(y > -x+1)"
        lineLower = "(y <= -x+1)"
        parUpper = "y > ( -4*( x**2 )+( 4*x ) )"
        parLower = "y <= ( -4*( x**2 )+( 4*x ) )"
        circleUpper = "( ( x - 1/2 )**2 + ( y - 1/2 )**2 > 1/16 )"
        circleLower = "( ( x - 1/2 )**2 + ( y - 1/2 )**2 <= 1/16 )"
        
        inputs.append([x,y])
        
        if eval(circleUpper):
            labels.append([1]) # input, desired output
        elif eval(circleLower):
            labels.append([0]) # input, desired output
            
    return [inputs,labels]

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
    
    for i in range(0,800):
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


def demo1(): # or table demonstration
    print("Nonlinear classification. Requires matplotlib.")
    o=NN([2,5,1],act='sigmoid',c=0.03,bias=1.0)
    print("Weights: ", o.getWeights())
    print("")
    
    z=generateData(800)
    inputs = z[0]
    labels = z[1]
    
    
    print("Training...")
    for i in range(0,200):
        for j in range(0,len(z)-1):
            # print(j)
            # print([z[j][0]], [z[j][1]])
            o.bp(inputs,labels)

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


m=NN([2,2,1],c=1.0,bias=0.0)
m.w = [np.array( [ [ 0.1, 0.8, 0.0 ], [ 0.4, 0.6, 0.0] ] ), np.array( [ [ 0.3, 0.9 ] ] ) ]
print("Weights: ", m.w)
o=m.feedForward([0.35,0.9],brk=True)
print("Ouput: ", o)
m.bp([[0.35,0.9]],[[0.5]])
o=m.feedForward([0.35,0.9],brk=True)
print("New ouput: ", o)
'''

m=NN([2,3,2],c=1.0,bias=0.0)
print("Weights: ", m.w)
o=m.feedForward([0.35,0.9],brk=True)
print("Ouput: ", o)
m.bp([[0.35,0.9]],[[0.5,0.5]])

'''