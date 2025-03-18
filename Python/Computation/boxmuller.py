# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 20:58:58 2025

@author: tomke
"""

import numpy as np
from matplotlib import pyplot as plt

def box_muller(u1, u2):
    z0 = np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)
    z1 = np.sqrt(-2*np.log(u1))*np.sin(2*np.pi*u2)
    return z0, z1

u1, u2 = np.random.rand(100000), np.random.rand(100000)

z0, z1 = box_muller(u1, u2)

fig, axs = plt.subplots(1,2, figsize=(12,5))

axs[0].hist(u1,bins=30,density=True)
axs[0].set_title('Uniform Distribution')
axs[0].set_xlim([0,1])

axs[1].hist(z0, bins=30, density=True, alpha=0.4, color='pink', edgecolor='black', label='z0')
axs[1].hist(z1, bins=30, density=True, alpha=0.4, color='lightgreen', edgecolor='black', label='z1')
axs[1].set_title('Normal Distribution')
axs[1].legend()

plt.tight_layout()
plt.show()