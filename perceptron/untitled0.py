#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 22:01:42 2021

@author: hashaam
"""

import numpy as np
def hinge(v):
    return np.where(v<1,1-v,33)
    pass

# x is dxn, y is 1xn, th is dx1, th0 is 1x1
def hinge_loss(x, y, th, th0):
    return hinge(y*(np.dot(th.T,x)+th0))
    pass

# x is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
def svm_obj(x, y, th, th0, lam):
    loss=hinge_loss(x,y,th,th0)
    loss1=loss.mean()
    return loss1+lam*(np.linalg.norm(th)**2)
    pass
x=np.array([[1,2,-1],[1,2,0]])
y=np.array([[1,1,-1]])
th=np.array([[1,0]]).T
th0=np.array([[0]])
lam=0
print(x[:,0:1])