# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 11:23:14 2025

@author: tomke
"""
import random
studlist = ['Ansa', 'Antoine', 'Andrew', 'Hunter', 'Jayant', 'Quincy']

def RandomChoice(alist):
    random.shuffle(alist)
    print(alist)
    
RandomChoice(studlist)