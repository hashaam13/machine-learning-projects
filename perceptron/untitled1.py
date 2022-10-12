#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 14:28:31 2021

@author: hashaam
"""
import numpy as np
th0=np.array([[1]])
th=np.array([[1]])
x=np.array([[1,2,3,3],[1,1,1,7]])
y=np.array([[3,1,2,6]])
#yr=np.dot(x,th)+th0
k=np.random.randint(4)
print(x[:,k:k+1])
#grth=-2*x*(y-yr)
#grth0=-2*(y-yr)
#print(grth,grth0)