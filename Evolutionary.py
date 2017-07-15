# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 14:36:26 2017

@author: Nathaniel
"""

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