# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 18:39:38 2022

@author: Grace
"""

import random
import torch
from torch import nn, optim
import math
from IPython import display
from res.plot_lib import plot_data, plot_model, set_default
set_default()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)