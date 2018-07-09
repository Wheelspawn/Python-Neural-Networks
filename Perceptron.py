
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
            bit = (1 if (i < len(self.l)-2 or len(self.l) == 2) else 0) # bit that tells if loop is at its end, for bias
            # print(bit)
            n = 2*np.random.random_sample((self.l[i+1],self.l[i]+bit))-1 # weights are initialized to random values between -1 and 1. Add an extra weight for hidden layer bias node
            self.w.append(n)
        # print(np.array(self.w))
        self.w=np.array(self.w)
                
    def getWeights(self):
        return self.w
        
    def setWeights(self,newW):
        
        newL = [len(newW[0][0])-1]
        for i in range(len(newW)):
            newL.append(len(newW[i]))
        
        self.w=np.array(newW)
        self.l=newL
        
    def feedForward(self, inputs, brk=False): # the brk argument returns all the activations
        
        inputs = inputs[:]
        inputs.append(self.bias) # bias in the input layer
        allVals = [inputs] # this is our first level of activations
            
        for i in range(len(self.w)):
            allVals.append([]) # add a list to represent the activations for the next layer
            for j in range(len(self.w[i])):
                a = eval(self.act)(np.dot(allVals[i],self.w[i][j])) # dot product of activations and weights for each neuron in layer
                allVals[i+1].append(a)
                
            if i < len(self.w)-2:
                allVals[i+1].append(self.bias) # we must add the bias as the last activation for each layer, for all but the output node and final activations
                
        if brk==False:
            return np.array(allVals[-1])
        else:
            return np.array(allVals)
        
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
        if len(inputs[0]) != self.l[0]: # cheap error handling printout, must fix later
            print(inputs[0])
            print(self.l[0])
            print("No-go. Length of input vectors must be equal to length of target vectors.")
            return
            
        else:
            if batch==True:
                allErrors = []
                allOutputs = []
                
                for i in range(len(inputs)):
                    allOutputs.append(self.feedForward(inputs[i],True))
                    allErrors.append(self.calculateErrors(allOutputs[i], inputs[i], targets[i]))
                    
                for i in range(len(allErrors)):
                    self.updateWeights(allOutputs[i],allErrors[i])
                    
            else: # these lines are just a'trick' to do stochastic learning
                for i in range(len(inputs)):
                    self.bp([inputs[i]],[targets[i]],batch=True)
                    
    def calculateErrors(self, out, inputVec, targetVec):
        
        errors = [[]]
        
        for i in range(len(self.w[-1])): # for every output node
            
            
            d = deltaError(out[-1][i], targetVec[i]) # calculate error
            
            errors[0].append(d)
        
        if len(self.l) > 2: # if there are one or more hidden layers
            for i in range(len(self.w)-2,-1,-1): # for each layer
                errors = [[]] + errors
                for j in range(len(self.w[i])): # for each neuron in layer
                    
                    a = []
                    b = []
                    
                    error = 0
                    for k in range(len(self.w[i+1])):
                        
                        a.append(self.neuronWeights(i+1,k)[j])
                        b.append(errors[1][k])
                        
                        error += self.neuronWeights(i+1,k)[j] * errors[1][k]
                        
                    errors[0].append(hiddenError(out[i+1][j], error))
                
        return errors
            
    def updateWeights(self, out, errors):
        
        '''
        print("")
        print("Errors: ",errors)
        print("")
        '''
        
        for i in range(len(self.w[-1])): # for every output neuron
            for j in range(len(self.w[-1][i])): # for every weight in neuron
                
                '''
                print("Weight: ", self.w[-1][i][j])
                print("Output: ", out[-2][j])
                print("Error: ", errors[-1][i])
                print("")
                '''
                
                last = self.w[-1][i][j]
                
                self.w[-1][i][j] += self.c * out[-2][j] * errors[-1][i] # update
                # print(last)
                # self.w[-1][i][j] += (1-sigmoid(last))*self.c*out[-2][j]*errors[-1][i]+(sigmoid(last)*last) # inertia term
                
        if len(self.l) > 2: # if there are one or more hidden layers
            for i in range(len(self.w)-2,-1,-1): # for each layer
                for j in range(len(self.w[i])): # for each neuron:
                    for k in range(len(self.w[i][j])): # for each weight:
                        
                        '''
                        print("Weight: ", self.w[i][j][k])
                        print("Output: ", out[i][k])
                        print("Errors ", errors[i][j])
                        print("")
                        '''
                        
                        last = self.w[i][j][k] # keeping the pre-update weight in memory for inertia term
                        
                        self.w[i][j][k] += self.c * out[i][k] * errors[i][j] # update
                        # print(last)
                        # self.w[i][j][k] += (1-sigmoid(last))*self.c*out[i][k]*errors[i][j]+(sigmoid(last)*last) # inertia term
    
    def hebbian(self, inputs):
        o = self.feedForward(inputs,brk=True);
        for i in range(1,len(o)-1):
            for j in range(len(o[i])):
                print("i,j: ", i,",",j)
                print(self.w[i][j])
                print(o[i][j])
                print("fuck you")
                self.w[i][j] *= (1+self.c) * o[i][j]
                        
def deltaError(o, t): # output, target
    error = o*(1 - o)*(t - o)
    return error
    
def hiddenError(o, e):
    error = o*(1 - o)*e
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

def linear(finalSum):
    return finalSum
    
def tanh(finalSum):
    return math.tanh(finalSum)

global d
d = 1.0

def decayer(i):
    global d
    
    d = d*0.99
    
    if d <= i:
        return d
    else:
        return i*d

def oneHot(v):
    c = [decayer(i) if i==max(v) else 0.1 for i in v]
    return c

def euclidDist(a,b):
    from scipy.spatial.distance import euclidean
    z = euclidean(a,b)
    return z

 # returns an array of euclidean distances between the second to last output and the output weights
def outputEuclids(o,w):
    r=[]
    
    for weights in w[-1]:
        r.append(euclidDist(o[-2],weights))
    return r

# converts the result of outputEuclids into the values to be backpropagated
def unsupervisedVec(o,w):
    z = np.argmax(o)
    u = np.zeros(len(o))
    for i in range(len(o)):
        u[i] = np.random.uniform(0,max(o[z]-0.25,0))
        # u.append(1/(4**z[i]))
    u[z] = 1
        
    return u

def circleGen(xMed, yMed, stdev):
    return [random.gauss(xMed,stdev),random.gauss(yMed,stdev)]

def demo1(): # or table demonstration
    print("Nonlinear classification. Requires matplotlib.")
    o=NN([2,9,1],act='sigmoid',c=0.03,bias=1.0)
    print("Weights: ", o.getWeights())
    print("")
    
    for a in range(0,10):
        z=generateData(150)
        inputs = z[0]
        labels = z[1]
        
        print("Training...")
        for i in range(0,150):
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
    for i in range(0,3000):
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
        p.bp([[1.0,0.0,0.0]],[[0,1,0,0,1]])
        p.bp([[0.0,1.0,0.0]],[[1,0,0,1,0]])
        p.bp([[0.0,0.0,1.0]],[[0,0,1,0,1]])
        # print(p.getWeights())
        
    print("")
    print("Objects: Banana, Apple, Blueberry")
    print("Qualities: Red, Yellow, Blue, Hard, Soft")
    
    p.act='step'
    
    print(p.feedForward([1.0,0.0,0.0]))
    print(p.feedForward([0.0,1.0,0.0]))
    print(p.feedForward([0.0,0.0,1.0]))

def demo4():
    p=NN([2,4,1],act='sigmoid',bias=1)
    p.w = [np.array( [ [ 0.5,-0.5, 0.1 ], [ 0.1, 0.8, -0.8 ], [ -0.3, -0.3, -0.9 ], [ -0.5, 0.3, -0.3 ] ] ), np.array( [ [ 0.3, 0.7, -0.7, 0.7 ] ] ) ]
    '''
    print("XOR function to test nonlinearity of neural network")
    print("")
    print("Weights: ", p.getWeights())
    print("")
    '''
    for i in range(0,22500):
        p.bp([ [0.0,0.0], [1.0,0.0], [0.0,1.0], [1.0,1.0] ], [ [0], [1], [1], [0] ] )
    ''' print("")
    print("Weights: ", p.getWeights())
    '''
    print("")
    print("In: [0][0]. Out: ", p.feedForward([0.0,0.0]))
    print("In: [1][0]. Out: ", p.feedForward([1.0,0.0]))
    print("In: [0][1]. Out: ", p.feedForward([0.0,1.0]))
    print("In: [1][1]. Out: ", p.feedForward([1.0,1.0]))

'''
setosa = [ [ 5.1, 3.5, 1.4, 0.2 ],
            [ 4.9, 3, 1.4, 0.2 ],
            [ 4.7, 3.2, 1.3, 0.2 ],
            [ 4.6, 3.1, 1.5, 0.2 ],
            [ 5, 3.6, 1.4, 0.3 ],
            [ 5.4, 3.9, 1.7, 0.4 ],
            [ 4.6, 3.4, 1.4, 0.3 ],
            [ 5, 3.4, 1.5, 0.2 ],
            [ 4.4, 2.9, 1.4, 0.2 ],
            [ 4.9, 3.1, 1.5, 0.1 ],
            [ 5.4, 3.7, 1.5, 0.2 ],
            [ 4.8, 3.4, 1.6, 0.2 ],
            [ 4.8, 3, 1.4, 0.1 ],
            [ 4.3, 3, 1.1, 0.1 ],
            [ 5.8, 4, 1.2, 0.2 ],
            [ 5.7, 4.4, 1.5, 0.4 ],
            [ 5.4, 3.9, 1.3, 0.4 ],
            [ 5.1, 3.5, 1.4, 0.3 ],
            [ 5.7, 3.8, 1.7, 0.3 ],
            [ 5.1, 3.8, 1.5, 0.3 ],
            [ 5.4, 3.4, 1.7, 0.2 ],
            [ 5.1, 3.7, 1.5, 0.4 ],
            [ 4.6, 3.6, 1, 0.2 ],
            [ 5.1, 3.3, 1.7, 0.5 ],
            [ 4.8, 3.4, 1.9, 0.2 ],
            [ 5, 3, 1.6, 0.2 ],
            [ 5, 3.4, 1.6, 0.4 ],
            [ 5.2, 3.5, 1.5, 0.2 ],
            [ 5.2, 3.4, 1.4, 0.2 ],
            [ 4.7, 3.2, 1.6, 0.2 ],
            [ 4.8, 3.1, 1.6, 0.2 ],
            [ 5.4, 3.4, 1.5, 0.4 ],
            [ 5.2, 4.1, 1.5, 0.1 ],
            [ 5.5, 4.2, 1.4, 0.2 ],
            [ 4.9, 3.1, 1.5, 0.2 ],
            [ 5, 3.2, 1.2, 0.2 ],
            [ 5.5, 3.5, 1.3, 0.2 ],
            [ 4.9, 3.6, 1.4, 0.1 ],
            [ 4.4, 3, 1.3, 0.2 ],
            [ 5.1, 3.4, 1.5, 0.2 ],
            [ 5, 3.5, 1.3, 0.3 ],
            [ 4.5, 2.3, 1.3, 0.3 ],
            [ 4.4, 3.2, 1.3, 0.2 ],
            [ 5, 3.5, 1.6, 0.6 ],
            [ 5.1, 3.8, 1.9, 0.4 ],
            [ 4.8, 3, 1.4, 0.3 ],
            [ 5.1, 3.8, 1.6, 0.2 ],
            [ 4.6, 3.2, 1.4, 0.2 ],
            [ 5.3, 3.7, 1.5, 0.2 ],
            [ 5, 3.3, 1.4, 0.2 ] ]

virginica = [ [ 6.3, 3.3, 6, 2.5 ],
            [ 5.8, 2.7, 5.1, 1.9 ],
            [ 7.1, 3, 5.9, 2.1 ],
            [ 6.3, 2.9, 5.6, 1.8 ],
            [ 6.5, 3, 5.8, 2.2 ],
            [ 7.6, 3, 6.6, 2.1 ],
            [ 4.9, 2.5, 4.5, 1.7 ],
            [ 7.3, 2.9, 6.3, 1.8 ],
            [ 6.7, 2.5, 5.8, 1.8 ],
            [ 7.2, 3.6, 6.1, 2.5 ],
            [ 6.5, 3.2, 5.1, 2 ],
            [ 6.4, 2.7, 5.3, 1.9 ],
            [ 6.8, 3, 5.5, 2.1 ],
            [ 5.7, 2.5, 5, 2 ],
            [ 5.8, 2.8, 5.1, 2.4 ],
            [ 6.4, 3.2, 5.3, 2.3 ],
            [ 6.5, 3, 5.5, 1.8 ],
            [ 7.7, 3.8, 6.7, 2.2 ],
            [ 7.7, 2.6, 6.9, 2.3 ],
            [ 6, 2.2, 5, 1.5 ],
            [ 6.9, 3.2, 5.7, 2.3 ],
            [ 5.6, 2.8, 4.9, 2 ],
            [ 7.7, 2.8, 6.7, 2 ],
            [ 6.3, 2.7, 4.9, 1.8 ],
            [ 6.7, 3.3, 5.7, 2.1 ],
            [ 7.2, 3.2, 6, 1.8 ],
            [ 6.2, 2.8, 4.8, 1.8 ],
            [ 6.1, 3, 4.9, 1.8 ],
            [ 6.4, 2.8, 5.6, 2.1 ],
            [ 7.2, 3, 5.8, 1.6 ],
            [ 7.4, 2.8, 6.1, 1.9 ],
            [ 7.9, 3.8, 6.4, 2 ],
            [ 6.4, 2.8, 5.6, 2.2 ],
            [ 6.3, 2.8, 5.1, 1.5 ],
            [ 6.1, 2.6, 5.6, 1.4 ],
            [ 7.7, 3, 6.1, 2.3 ],
            [ 6.3, 3.4, 5.6, 2.4 ],
            [ 6.4, 3.1, 5.5, 1.8 ],
            [ 6, 3, 4.8, 1.8 ],
            [ 6.9, 3.1, 5.4, 2.1 ],
            [ 6.7, 3.1, 5.6, 2.4 ],
            [ 6.9, 3.1, 5.1, 2.3 ],
            [ 5.8, 2.7, 5.1, 1.9 ],
            [ 6.8, 3.2, 5.9, 2.3 ],
            [ 6.7, 3.3, 5.7, 2.5 ],
            [ 6.7, 3, 5.2, 2.3 ],
            [ 6.3, 2.5, 5, 1.9 ],
            [ 6.5, 3, 5.2, 2 ],
            [ 6.2, 3.4, 5.4, 2.3 ],
            [ 5.9, 3, 5.1, 1.8 ] ]

def logit(x):
    return np.log(x/(1-x))

# autoencoder
auto = NN(l=[4,3,2,3,4])

train_setosa=setosa[0:40]
train_virginica=virginica[0:40]

test_setosa=setosa[40:]
test_virginica=virginica[40:]

for i in range(0,10000):
    r1 = train_setosa[random.randint(0,len(train_setosa)-1)].copy()
    r2 = train_virginica[random.randint(0,len(train_virginica)-1)].copy()
    
    for j in range(0,5):
        auto.bp([r1],[[sigmoid(r) for r in r1]])
        
    for k in range(0,5):
        auto.bp([r2],[[sigmoid(r) for r in r2]])

mse = 0

print("Setosa encoding:")
for i in range(len(test_setosa)):
    print(auto.feedForward(test_setosa[i],brk=True)[2][0:2])

print("")
print("Virginica encoding:")
for j in range(len(test_virginica)):
    print(auto.feedForward(test_virginica[j],brk=True)[2][0:2])

'''    
