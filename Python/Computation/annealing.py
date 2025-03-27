# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 15:45:47 2025

@author: tomke
"""

L = 10
J = 1.
T = 5.0
T_min = 0.1
alpha = 0.95  # cooling factor (will be adjusted)
target_acceptance = 0.4
smoothing = 0.05  # to stabilize adaptation

S = Ising(L, J, T)

energies = []
temps = []

while T > T_min:
    S.set_temp(T)

    naccept = 0
    nsweeps = 100
    for _ in range(nsweeps * S.N):
        naccept += S.metropolis()

    acc_ratio = naccept / (nsweeps * S.N)

    # Record energy and temp
    energies.append(S.E / S.N)
    temps.append(T)

    # Adjust cooling rate based on acceptance ratio
    if acc_ratio < 0.2:
        alpha = max(0.90, alpha - smoothing)
    elif acc_ratio > 0.5:
        alpha = min(0.99, alpha + smoothing)

    T *= alpha  # update temp

import matplotlib.pyplot as plt

plt.plot(temps, energies)
plt.xlabel("Temperature")
plt.ylabel("Energy per spin")
plt.title("Auto-Tuned Simulated Annealing")
plt.gca().invert_xaxis()
plt.grid()
plt.show()