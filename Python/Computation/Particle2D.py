# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:54:31 2025

@author: tomke
"""

import numpy as np

class Particle2D:
    def __init__(self, mass=1., x=0., y=0., vx=0., vy=0.):
        """
        Initializes the particle with mass, position, and velocity.
        """
        self.mass = mass
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([vx, vy], dtype=float)
        # this is needed for the verlet integration method (needs previous time step at beginning)
        self.previous_position = self.position - self.velocity * 0.01
        # additional properties for the leapfrog algorithm
        self.acceleration = np.array([0.0, 0.0], dtype=float)
        self.half_velocity = self.velocity - 0.5 * self.acceleration
        
    def __str__(self):
        """Returns a string representation of the particle."""
        return f"Mass: {self.mass}, Position: {self.position}, Velocity: {self.velocity}"
    
    def euler(self, force, dt):
        """Euler method for updating state."""
        self.velocity += (force / self.mass) * dt
        self.position += self.velocity * dt

    def rk2(self, force_func, dt):
        """Second-order Runge-Kutta method."""
        a1 = force_func(self.position, self.velocity) / self.mass
        v_half = self.velocity + 0.5 * dt * a1
        x_half = self.position + 0.5 * dt * self.velocity
        a2 = force_func(x_half, v_half) / self.mass
        self.velocity += dt * a2
        self.position += dt * v_half

    def rk4(self, force_func, dt):
        """Fourth-order Runge-Kutta method."""
        k1_v = (force_func(self.position, self.velocity) / self.mass) * dt
        k1_x = self.velocity * dt

        k2_v = (force_func(self.position + 0.5 * k1_x, self.velocity + 0.5 * k1_v) / self.mass) * dt
        k2_x = (self.velocity + 0.5 * k1_v) * dt

        k3_v = (force_func(self.position + 0.5 * k2_x, self.velocity + 0.5 * k2_v) / self.mass) * dt
        k3_x = (self.velocity + 0.5 * k2_v) * dt

        k4_v = (force_func(self.position + k3_x, self.velocity + k3_v) / self.mass) * dt
        k4_x = (self.velocity + k3_v) * dt

        self.velocity += (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
        self.position += (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6


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
