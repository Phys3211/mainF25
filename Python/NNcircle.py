# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:06:37 2025

@author: tomke
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import jax.numpy as jnp
from jax import grad, vmap, jit
import math

def model(theta, x): 
    w0, b0, w1, b1 = theta
    z = jnp.einsum('ji,ci->cj', w0, x) + b0
    z = jnp.maximum(0,z) # ReLU activation function (also referred to as threshold function)
    output = jnp.einsum('ji,bi->bj', w1, z) + b1
    return output 

def unit_circle_loss(theta,x):
    fx = model(theta,x) 
    return jnp.mean((jnp.linalg.norm(fx, axis=1) - 1)**2)

def update(theta, x, lr = 0.001):
    w0, b0, w1, b1 = theta
    loss = unit_circle_loss(theta, x)
    grad_loss_w0, grad_loss_b0, grad_loss_w1, grad_loss_b1 = grad(unit_circle_loss)(theta, x)
    new_params = [w0 - lr*grad_loss_w0, b0 - lr*grad_loss_b0, w1 - lr*grad_loss_w1, b1 - lr*grad_loss_b1]
    return loss, new_params

num_points = 100
xs = np.random.normal(size=(num_points,2))
d, width = xs.shape[1], 1024
theta = [np.random.normal(scale=1,size=(width,d)), np.zeros(width), np.random.normal(scale=1,size=(2,width)), np.zeros(2)]
 
plt.scatter(xs[:,0], xs[:,1])
plt.show()

fx = model(theta, xs)
plt.scatter(fx[:,0], fx[:,1])

num_epochs, losses, thetas = 10000, [], []
for epoch in range(num_epochs+1):
    loss, theta = update(theta,xs)
    thetas.append(theta)
    losses.append(loss)
    if epoch % 1000 == 0:
        fx = model(thetas[-1], xs)
        plt.scatter(fx[:,0], fx[:,1])
        plt.title(f"Epoch {epoch}")
        plt.show()