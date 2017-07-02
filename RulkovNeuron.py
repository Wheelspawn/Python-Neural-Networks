import matplotlib.patches as mpatches
import pylab
import random
import numpy as np

class RulkovNeuron(object):
    
    def __init__(self, fast=-1.0, slow=-3.93, act=0.0, mu=0.001, control=4.5, longAvg=[], medAvg=[], leak=-0.01):
        self.fast = fast
        self.slow = slow
        self.act = act
        self.mu = mu
        self.control = control
        self.longAvg = longAvg
        self.medAvg = medAvg
        self.leak = leak

    def update(self,current,inputs):
        dFast = self.membraneFunc(self.fast,self.slow+current,self.control)
        dSlow = self.slowFunc(self.slow,self.fast,self.mu,inputs+self.leak)
        
        self.fast = dFast
        self.slow = dSlow

    def membraneFunc(self, x, y, a):
        if x <= 0.0:
            self.act=0.0
            return (a/(1.0-x))+y
        elif 0.0 < x < (a+y):
            return a+y
        elif x >= (a+y):
            self.act=1.0
            return -1.0
            
    def membraneDeriv(self, x, a):
        return a/(x-1)**2
        
    def slowFunc(self, slow, fast, mu, inputs):
        return slow - mu*(fast+1)+mu*inputs

def test():
    
    r1=RulkovNeuron()
    r1.control=4
    
    firstLayer = []
    
    for k in range(0,60):
        firstLayer.append(RulkovNeuron(control=4))
    
    f1=[]
    f2=[]
    tStep=[]
    
    for n in range(0,4000):
        
        avg=0
        
        for neuron in firstLayer:
            neuron.update(0.0,0.03+random.uniform(-0.01,0.01))
            avg += neuron.fast
            
        f1.append(avg/60)
        
        if n < 1000:
            r1.update(0.0,0.03+random.uniform(-0.01,0.01))
        else:
            r1.update(0.0,-0.12+sum(list(neuron.act for neuron in firstLayer)))
        
        tStep.append(n)
        f2.append(r1.fast)
        
    pylab.plot(tStep,f1,color='r')
    pylab.plot(tStep,f2,color='b')
    
    layer1Plt = mpatches.Patch(color='red', label='Avg potential of (60) sending neurons')
    layer2Plt = mpatches.Patch(color='blue', label='Potential of receiving neuron')
    
    pylab.legend(handles=[layer1Plt,layer2Plt])
    
    pylab.xlim(xmin=500, xmax=4000)
    pylab.ylim(ymin=-1.6, ymax=1.6)
    pylab.show()
    
test()