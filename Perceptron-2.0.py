import numpy as np
import random
import math

class MLP(object):
    
    def __init__(self, inputs=9, hidden1=10, hidden2=10, hidden3=None, hidden4=None, outputs=9, bias=2):
        self.inputs = inputs
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.outputs = outputs
        
    def initWeights(self):
        self.nList = np.ndarray([])