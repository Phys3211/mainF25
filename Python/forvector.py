# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 19:25:24 2025

@author: tomke
Illustration of time saving with vectorized calculations
"""

import numpy as np
import time

# Create a large array
size = 1_000_000  # 1 million elements
arr = np.random.rand(size)

# Using a for loop
start_time = time.time()
squared_loop = np.zeros(size)
for i in range(size):
    squared_loop[i] = arr[i] ** 2
end_time = time.time()
print(f"For loop time: {end_time - start_time:.5f} seconds")

# Using NumPy vectorization
start_time = time.time()
squared_vectorized = arr ** 2  # Vectorized operation
end_time = time.time()
print(f"Vectorized time: {end_time - start_time:.5f} seconds")

