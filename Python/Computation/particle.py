# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 18:42:14 2025

@author: tomke
"""

import numpy as np

class Particle2D:
    def __init__(self, mass, x, y, vx, vy):
        self.mass = mass
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([vx, vy], dtype=float)
        self.previous_position = self.position - self.velocity * 0.01  # Small initial step

    def euler(self, force_func, dt):
        acceleration = force_func(self.position) / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

    def verlet(self, force_func, dt):
        acceleration = force_func(self.position) / self.mass
        new_position = 2 * self.position - self.previous_position + acceleration * dt**2
        self.velocity = (new_position - self.previous_position) / (2 * dt)
        self.previous_position = self.position.copy()
        self.position = new_position

    def velocity_verlet(self, force_func, dt):
        acceleration = force_func(self.position) / self.mass
        self.position += self.velocity * dt + 0.5 * acceleration * dt ** 2
        new_acceleration = force_func(self.position) / self.mass
        self.velocity += 0.5 * (acceleration + new_acceleration) * dt

    def leapfrog(self, force_func, dt):
        acceleration = force_func(self.position) / self.mass
        half_velocity = self.velocity + 0.5 * acceleration * dt
        self.position += half_velocity * dt
        new_acceleration = force_func(self.position) / self.mass
        self.velocity = half_velocity + 0.5 * new_acceleration * dt


    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def vx(self):
        return self.velocity[0]

    @property
    def vy(self):
        return self.velocity[1]