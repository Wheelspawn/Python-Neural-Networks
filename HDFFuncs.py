

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 00:26:03 2017

@author: wheelspawn
"""

import csv
import numpy as np
from h5py import File,Group

from Perceptron import NN

def hdfToWeights(path):
    w=[]
    f=File(path)
    for key in f["Weights"].keys():
        w.append(np.array(f["Weights"][key]))
        
    return np.array(w)

def weightsToHDF(w, name):
    f=File(name+".h5","w")
    
    weights=f.create_group("Weights")
    
    for i in range(len(w[:-1])):
        weights.create_dataset("Hidden "+str(i+1),data=w[i])
    weights.create_dataset("Output",data=w[-1])
    
    f.close()

def vectorsToArray(path):
    listy=[]
    with open(path) as csvfile:
        myReader = csv.reader(csvfile, delimiter=',')
        for row in myReader:
            n=[]
            for item in row:
                p = item.split(',')
                p = (list(float(i) for i in p if i != ''))
                if p != []:
                    n.extend(p)
            listy.append(n)
        csvfile.close()
    return listy
