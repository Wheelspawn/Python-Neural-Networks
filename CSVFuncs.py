#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 00:26:03 2017

@author: wheelspawn
"""

import csv
import pandas
import numpy as np

def csvToWeights(path):
    listy=[]
    with open(path) as csvfile:
        myReader = csv.reader(csvfile, delimiter=',')
        for row in myReader:
            n=[]
            for item in row:
                p = item.split(',')
                p = (list(float(i) for i in p if i != ''))
                if p != []:
                    n.append(p)
            listy.append(np.array(n))
    return listy

def weightsToCsv(weights, name):
    with open(name, "w+") as csvfile:
        myWriter = csv.writer(csvfile,delimiter=',')
        print(weights)
        for row in weights:
            myWriter.writerow(row)

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
    return listy
