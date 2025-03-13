# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:46:37 2025

@author: tomke and Chat-GPT
"""
import numpy as np

def is_iterable(obj):
    """Check if the object is an iterable but not a string."""
    return hasattr(obj, "__iter__") and not isinstance(obj, str)

def trapezoidrule(f_or_data, a=None, b=None, N=10):
    if is_iterable(f_or_data):
        x = np.linspace(0, len(f_or_data) - 1, len(f_or_data))
        y = np.array(f_or_data)
    else:
        x = np.linspace(a, b, N + 1)
        y = np.array([f_or_data(xi) for xi in x])
    
    h = (x[-1] - x[0]) / N
    return h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))

def trapezoiderr(f, a, b, N=10):
    I1 = trapezoidrule(f, a, b, int(N // 2))
    I2 = trapezoidrule(f, a, b, N)
    return abs((I2 - I1) / 3)

def simpsonrule(f_or_data, a=None, b=None, N=10):
    if is_iterable(f_or_data):
        x = np.linspace(0, len(f_or_data) - 1, len(f_or_data))
        y = np.array(f_or_data)
    else:
        x = np.linspace(a, b, N + 1)
        y = np.array([f_or_data(xi) for xi in x])
    
    h = (x[-1] - x[0]) / N
    return (h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))

def simpsonerr(f, a, b, N=10):
    I1 = simpsonrule(f, a, b, int(N // 2))
    I2 = simpsonrule(f, a, b, N)
    return abs((I2 - I1) / 15)

def cubicrule(f_or_data, a=None, b=None, N=10):
    if is_iterable(f_or_data):
        x = np.linspace(0, len(f_or_data) - 1, len(f_or_data))
        y = np.array(f_or_data)
    else:
        x = np.linspace(a, b, N + 1)
        y = np.array([f_or_data(xi) for xi in x])
    
    h = (x[-1] - x[0]) / N
    return (3 * h / 8) * (y[0] + y[-1] + 9 * np.sum(y[1:-1:3]) + 9 * np.sum(y[2:-1:3]) + 6 * np.sum(y[3:-1:3]))

def quarticrule(f_or_data, a=None, b=None, N=10):
    if is_iterable(f_or_data):
        x = np.linspace(0, len(f_or_data) - 1, len(f_or_data))
        y = np.array(f_or_data)
    else:
        x = np.linspace(a, b, N + 1)
        y = np.array([f_or_data(xi) for xi in x])
    
    h = (x[-1] - x[0]) / N
    return (h / 45) * (14 * (y[0] + y[-1]) + 64 * np.sum(y[1:-1:4]) + 24 * np.sum(y[2:-1:4]) + 64 * np.sum(y[3:-1:4]) + 28 * np.sum(y[4:-1:4]))

def rombergrule(f, a, b, max_iters=5, accuracy=1e-9):
    R = np.zeros((max_iters, max_iters))  # Dynamically sized matrix
    err = np.zeros(max_iters)

    # Compute first trapezoidal approximation
    for i in range(max_iters):
        N = 2**i  # Ensure proper step sizes
        R[i, 0] = trapezoidrule(f, a, b, N)

        if i > 0:
            for k in range(1, i + 1):
                R[i, k] = R[i, k - 1] + (R[i, k - 1] - R[i - 1, k - 1]) / (4**k - 1)

            err[i] = abs(R[i, i] - R[i - 1, i - 1])
            if err[i] < accuracy:
                return R[i, i]  # Stop if accuracy is reached

    return R[max_iters - 1, max_iters - 1]  # Return most refined estimate

