# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 15:41:23 2025

@author: tomke and ChatGPT for Hexdecimal stuff
"""

class MersenneTwister:
    def __init__(self, seed):
        # Parameters for MT19937
        self.n = 624         # State size
        self.m = 397         # Offset for scrambling
        self.a = 0x9908B0DF  # Constant used in the transformation
        self.upper_mask = 0x80000000  # Most significant bit mask
        self.lower_mask = 0x7FFFFFFF  # Least significant 31 bits mask

        # Initialize state array
        self.mt = [0] * self.n
        self.index = self.n
        self.mt[0] = seed

        # Seed initialization with linear congruence
        for i in range(1, self.n):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & 0xFFFFFFFF

    def twist(self):
        """Performs the twisting transformation on the state array."""
        for i in range(self.n):
            x = (self.mt[i] & self.upper_mask) + (self.mt[(i + 1) % self.n] & self.lower_mask)
            xA = x >> 1
            if x % 2 != 0:
                xA ^= self.a
            self.mt[i] = self.mt[(i + self.m) % self.n] ^ xA
        self.index = 0

    def extract_number(self):
        """Extracts a tempered random number based on the internal state."""
        if self.index >= self.n:
            self.twist()

        y = self.mt[self.index]
        self.index += 1

        # Tempering transformation
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= (y >> 18)

        return y & 0xFFFFFFFF  # Ensure a 32-bit output

# Example usage:
mt = MersenneTwister(seed=42)

print("First 10 Mersenne Twister random numbers:")
for _ in range(10):
    print(mt.extract_number())
