# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:14:26 2019

@author: atecan00
"""

import numpy as np
import numpy.random as rd


def pick_element(Set, num):
    N = len(Set)
    i = (num%N)
    num = np.floor(num/N)
    return int(i), num